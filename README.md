# Deforum Kandinsky
### install dependencies
```bash
git clone https://gitlab.aicloud.sbercloud.ru/sberai-rnd/deforum-kandinsky.git
cd deforum-kandinsky
pip install -r requirements.txt
```

### Import dependencies
```python
from IPython.display import Video
from deforum_kandinsky import KandinskyV22Img2ImgPipeline, DeforumKandinsky
from diffusers import KandinskyV22PriorPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
import imageio.v2 as iio
from PIL import Image
import numpy as np
import torch
import datetime
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython import display
```

### convert list frames to mp4 video
```python
#  create video from generated frames
def frames2video(frames, output_path="video.mp4", fps=24, display=False):
    writer = iio.get_writer(output_path, fps=fps)
    for frame in tqdm(frames):
        writer.append_data(np.array(frame))
    writer.close()
    if display:
        display.Video(url=output_path)
```

### Kandinsky 2.2
```python
from deforum_kandinsky import KandinskyV22Img2ImgPipeline, DeforumKandinsky
from diffusers import KandinskyV22PriorPipeline

# load models
device = "cuda"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior', 
    subfolder='image_encoder'
    ).to(torch.float16).to(device)

unet = UNet2DConditionModel.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder', 
    subfolder='unet'
    ).to(torch.float16).to(device)

prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior', 
    image_encoder=image_encoder, 
    torch_dtype=torch.float16
    ).to(device)
decoder = KandinskyV22Img2ImgPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder', 
    unet=unet, 
    torch_dtype=torch.float16
    ).to(device)
```

### Do the same using Kandinsky 2.1
```python 
from deforum_kandinsky import KandinskyImg2ImgPipeline, DeforumKandinsky
from diffusers import KandinskyPriorPipeline

device = "cuda"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", 
    subfolder='image_encoder',
    torch_dtype=torch.float16
    ).to(device)
unet = UNet2DConditionModel.from_pretrained(
    "kandinsky-community/kandinsky-2-1", 
    subfolder='unet',
    torch_dtype=torch.float16
    ).to(device)
prior = KandinskyPriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", 
    torch_dtype=torch.float16
    ).to(device)
decoder = KandinskyImg2ImgPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-1', 
    unet=unet, 
    torch_dtype=torch.float16
    ).to(device)
```
### Easily create animation constructor
#### define instance of Deforum
```python
deforum = DeforumKandinsky(
    prior=prior,
    decoder_img2img=decoder,
    device='cuda'
)
```

```python   
animation = deforum(
    prompts=[
        "winter forest, snowflakes, Van Gogh style",
        "spring forest, flowers, sun rays, Van Gogh style",
        "summer forest, lake, reflections on the water, summer sun, Van Gogh style",
        "autumn forest, rain, Van Gogh style",
        "winter forest, snowflakes, Van Gogh style",
    ], 
    animations=['live', 'right', 'right', 'right', 'live'], 
    prompt_durations=[1, 1, 1, 1, 1],
    H=640,
    W=640,
    fps=24,
    save_samples=False,
)

frames = []

out = widgets.Output()
pbar = tqdm(animation, total=len(deforum))
display.display(out)
for frame, current_params in pbar:
    frames.append(frame)
    with out:
        image = item.pop('image', None)
        frames.append(image)
        display.clear_output(wait=True) 
        for key, value in current_params.items():
            print(f"{key}: {value}")

# save and display video
display.clear_output(wait=True) 
frames2video(frames, "output_2_2.mp4", fps=24)
display.Video(url="output_2_2.mp4")
```
