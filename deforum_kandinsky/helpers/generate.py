import os

# Related third-party imports%
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat
from PIL import Image
# Local application/library specific imports
from .k_samplers import sampler_fn, make_inject_timing_fn
from .callback import SamplerCallback
from .conditioning import exposure_loss, make_mse_loss, get_color_palette, make_clip_loss_fn
from .conditioning import make_rgb_color_match_loss, blue_loss_fn, threshold_by, make_aesthetics_loss_fn, mean_loss_fn, var_loss_fn, exposure_loss
from .model_wrap import CFGDenoiserWithGrad
from .load_images import load_img, prepare_mask, prepare_overlay_mask
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser

from ..pipelines import KandinskyImg2ImgPipeline, KandinskyV22Img2ImgPipeline
import torch 


def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    batch_size = args.n_samples

    # cond prompts
    cond_prompt = args.cond_prompt
    assert cond_prompt is not None
    cond_data = [batch_size * [cond_prompt]]

    # uncond prompts
    uncond_prompt = args.uncond_prompt
    assert uncond_prompt is not None
    uncond_data = [batch_size * [uncond_prompt]]
    
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    init_pil = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        init_pil = args.init_sample * 0.5 + 0.5
        init_pil = init_pil.clamp(0, 1)
        init_pil = init_pil.cpu().permute(0, 2, 3, 1).float().numpy()

        init_pil = root.model.decoder_img2img.numpy_to_pil(init_pil)[0]
    elif args.use_init and frame == 0:
        if isinstance(args.init_image, Image.Image): 
            init_pil = args.init_image
        else:
            init_pil = Image.open(args.init_image)
            
        init_pil = init_pil.resize((args.W, args.H))
        
    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        #print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        #print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                            init_latent.shape,
                            args.mask_contrast_adjust, 
                            args.mask_brightness_adjust,
                            args.invert_mask)
        
        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
        
        mask = mask.to(root.device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

    # Init MSE loss image
    init_mse_image = None
    if args.init_mse_scale and args.init_mse_image != None and args.init_mse_image != '':
        init_mse_image, mask_image = load_img(args.init_mse_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_mse_image = init_mse_image.to(root.device)
        init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=batch_size)

    assert not ( args.init_mse_scale != 0 and (args.init_mse_image is None or args.init_mse_image == '') ), "Need an init image when init_mse_scale != 0"

    # Noise schedule for the k-diffusion samplers (used for masking)

    if args.colormatch_scale != 0:
        assert args.colormatch_image is not None, "If using color match loss, colormatch_image is needed"
        colormatch_image, _ = load_img(args.colormatch_image)
        colormatch_image = colormatch_image.to('cpu')
        del(_)
    else:
        colormatch_image = None

    # Loss functions
    if args.init_mse_scale != 0:
        if args.decode_method == "linear":
            mse_loss_fn = make_mse_loss(root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(init_mse_image.to(root.device)))))
        else:
            mse_loss_fn = make_mse_loss(init_mse_image)
    else:
        mse_loss_fn = None

    if args.colormatch_scale != 0:
        _,_ = get_color_palette(root, args.colormatch_n_colors, colormatch_image, verbose=True) # display target color palette outside the latent space
        if args.decode_method == "linear":
            grad_img_shape = (int(args.W/args.f), int(args.H/args.f))
            colormatch_image = root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(colormatch_image.to(root.device))))
            colormatch_image = colormatch_image.to('cpu')
        else:
            grad_img_shape = (args.W, args.H)
        color_loss_fn = make_rgb_color_match_loss(root,
                                                  colormatch_image, 
                                                  n_colors=args.colormatch_n_colors, 
                                                  img_shape=grad_img_shape,
                                                  ignore_sat_weight=args.ignore_sat_weight)
    else:
        color_loss_fn = None

    if args.clip_scale != 0:
        clip_loss_fn = make_clip_loss_fn(root, args)
    else:
        clip_loss_fn = None

    if args.aesthetics_scale != 0:
        aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
    else:
        aesthetics_loss_fn = None

    if args.exposure_scale != 0:
        exposure_loss_fn = exposure_loss(args.exposure_target)
    else:
        exposure_loss_fn = None

    loss_fns_scales = [
        [clip_loss_fn,              args.clip_scale],
        [blue_loss_fn,              args.blue_scale],
        [mean_loss_fn,              args.mean_scale],
        [exposure_loss_fn,          args.exposure_scale],
        [var_loss_fn,               args.var_scale],
        [mse_loss_fn,               args.init_mse_scale],
        [color_loss_fn,             args.colormatch_scale],
        [aesthetics_loss_fn,        args.aesthetics_scale]
    ]

    # Conditioning gradients not implemented for ddim or PLMS
    assert not( any([cond_fs[1]!=0 for cond_fs in loss_fns_scales]) and (args.sampler in ["ddim","plms"]) ), "Conditioning gradients not implemented for ddim or plms. Please use a different sampler."

    generator = torch.Generator(device="cuda").manual_seed(43)
    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            for cond_prompts, uncond_prompts in zip(cond_data,uncond_data):
                    if isinstance(cond_prompts, tuple):
                        cond_prompts = list(cond_prompts)
                    if isinstance(uncond_prompts, tuple):
                        uncond_prompts = list(uncond_prompts)

                    c = root.model.prior(cond_prompts, guidance_scale=1.0, generator=generator).image_embeds
                    uc = root.model.prior(uncond_prompts, guidance_scale=1.0, generator=generator).negative_image_embeds

                    model_kwargs = dict(
                         image_embeds=c,
                         negative_image_embeds=uc,
                         strength=args.strength,
                         height=args.H,
                         width=args.W,
                         num_inference_steps=args.steps
                    )
                    
                    if init_pil is None:
                        model_kwargs["strength"] = 1.0
                        imarray = np.random.normal(args.H, args.W, 3) * 255
                        init_pil = Image.fromarray(imarray.astype('uint8')).convert('RGB')
                    
                    model_kwargs["image"] = init_pil
                        
                    if isinstance(root.model.decoder_img2img, KandinskyImg2ImgPipeline): 
                        model_kwargs["prompt"] = ""
                    
                    x_samples, samples = root.model.decoder_img2img(**model_kwargs)
                    
                    if return_latent:
                        results.append(samples.clone())


                    if args.use_mask and args.overlay_mask:
                        # Overlay the masked image after the image is generated
                        if args.init_sample_raw is not None:
                            img_original = args.init_sample_raw
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception("Cannot overlay the masked image without an init image to overlay")

                        if args.mask_sample is None or args.using_vid_init:
                            args.mask_sample = prepare_overlay_mask(args, root, img_original.shape)

                        x_samples = img_original * args.mask_sample + x_samples * ((args.mask_sample * -1.0) + 1)

                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        def uint_number(datum, number):
                            if number == 8:
                                datum = Image.fromarray(datum.astype(np.uint8))
                            elif number == 32:
                                datum = datum.astype(np.float32)
                            else:
                                datum = datum.astype(np.uint16)
                            return datum
                        if args.bit_depth_output == 8:
                            exponent_for_rearrange = 1
                        elif args.bit_depth_output == 32:
                            exponent_for_rearrange = 0
                        else:
                            exponent_for_rearrange = 2
                        x_sample = 255.**exponent_for_rearrange * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = uint_number(x_sample, args.bit_depth_output)
                        results.append(image)
    return results
