# Deforum-Kandinsky
<p align="left">
    <a target="_blank" href="https://colab.research.google.com/drive/1V0E_nM8bxOhVPBXP-J9jCbeAaNAYsI9v?usp=sharing">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p> 

## Introduction
<p>
In the past few years, there has been a marked increase in the popularity of generative models that utilize various data modalities. One of the most challenging undertakings in this regard is synthesizing videos from text, which is both time-consuming and resource-intensive. The core of proposed solution/animation approach is Kandinsky extension with Deforum features. This leads to new generative opportunities of text2image model.
</p>

## Examples
<p>
    <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/afe7885c-fb3c-479d-9f1a-8427f2636555"/>
    <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/6f2b74f3-d290-4b26-b7c9-ccfe239e54e6"/>
<p>
     <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/4cc93196-5146-4889-9b9d-0b87079c643f"/>
     <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/03a0d489-7138-42fe-9089-48c63bed9911"/>
</p>
<p>
    <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/f7588587-d307-4eb6-a610-b5618c5c0be5">
    <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/1d58d9f6-da3b-4cca-9240-ff196478f1dd>">
</p>                                  
<p>
    <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/58f8e66d-1d42-40b5-bbd1-9f593237c6aa"/>
    <img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/725b5165-e6cf-4aaf-a51f-afc382c2a5f9"/>
</p>

## Getting Started

### 1. Clone repository
```bash
git clone https://github.com/ai-forever/deforum-kandinsky.git
cd deforum-kandinsky
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Running Deforum
### 1. Import dependencies
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

### 2. Convert list frames to mp4 video
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

### 3. Load Kandinsky 2.1 or 2.2 
```python
from diffusers import KandinskyV22PriorPipeline
from deforum_kandinsky import (
    KandinskyV22Img2ImgPipeline, 
    DeforumKandinsky,  
    KandinskyImg2ImgPipeline, 
    DeforumKandinsky
)

# load models
model_version = 2.2
device = "cuda"

if model_version == 2.2:
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

elif model_version == 2.1: 

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

### 4. Define instance of Kandinsky with Deforum
```python
deforum = DeforumKandinsky(
    prior=prior,
    decoder_img2img=decoder,
    device='cuda'
)
```

### 5. Create text prompt and set up configs for animation
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
with out:
    for item in pbar:
        frame = item.pop('image', None)
        frames.append(frame)
        display.clear_output(wait=True) 
        display.display(frame)
        for key, value in item.items():
            print(f"{key}: {value}")

# save and display video
display.clear_output(wait=True) 
frames2video(frames, "output_2_2.mp4", fps=24)
display.Video(url="output_2_2.mp4")
```

## References
<a href="https://deforum.github.io/">Deforum web-page</a>
</br>
<a href="https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit#heading=h.7z6glzthkva2)https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit#heading=h.7z6glzthkva2">Quick Guide to Deforum v06</a>
</br>
<a href="https://github.com/deforum-art/deforum-stable-diffusion">GitHub repository: deforum-stable-diffusion</a>
</br>
