import os
import torch
from tqdm import tqdm
import requests
import sys

# Decodes the image without passing through the upscaler. The resulting image will be the same size as the latent
# Thanks to Kevin Turner (https://github.com/keturn) we have a shortcut to look at the decoded image!
def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode


def download_model(model_map,root):
    
    url = model_map[root.model_checkpoint]['url']

    # CLI dialogue to authenticate download
    if model_map[root.model_checkpoint]['requires_login']:
        print("This model requires an authentication token")
        print("Please ensure you have accepted the terms of service before continuing.")

        username = input("[What is your huggingface username?]: ")
        token = input("[What is your huggingface token?]: ")

        _, path = url.split("https://")

        url = f"https://{username}:{token}@{path}"

    # contact server for model
    print(f"..attempting to download {root.model_checkpoint}...this may take a while")
    ckpt_request = requests.get(url,stream=True)
    request_status = ckpt_request.status_code

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(os.path.join(root.models_path, root.model_checkpoint), 'wb') as model_file:
        file_size = int(ckpt_request.headers.get("Content-Length"))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=root.model_checkpoint) as pbar:
            for chunk in ckpt_request.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    model_file.write(chunk)
                    pbar.update(len(chunk))





