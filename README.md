# Deforum-Kandinsky
<p align="left">
    <a target="_blank" href="https://colab.research.google.com/drive/1V0E_nM8bxOhVPBXP-J9jCbeAaNAYsI9v?usp=sharing">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p> 

## Examples


<img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/1f40a5a1-c02b-4aac-a937-eee0aa721062" height="800" width="800"/>

## Introduction
<p>
In the past few years, there has been a marked increase in the popularity of generative models that utilize various data modalities. One of the most challenging undertakings in this regard is synthesizing videos from text, which is both time-consuming and resource-intensive. The core of proposed solution/animation approach is Kandinsky extension with Deforum features. This leads to new generative opportunities of text2image model.
</p>

## Description

<img src="https://github.com/ai-forever/deforum-kandinsky/assets/69383296/2869904e-f980-4553-8660-7baa0ab47a12" height="800" width="800"/>

<p>
The idea of animating a picture is quite simple: from the original 2D image we obtain a pseudo-3D scene and then simulate a camera flyover of this scene. The pseudo-3D effect occurs due to the human eye’s perception of motion dynamics through spatial transformations. Using various motion animation mechanics, we get a sequence of frames that look like they were shot with a first-person camera. The process of creating such a set of personnel is divided into 2 stages:
<ol>
  <li>creating a pseudo-3D scene and simulating a camera flyby (obtaining successive frames);</li>
  <li>application of the image-to-image approach for additional correction of the resulting images.</li>
</ol>

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
    for index, item in enumerate(pbar):
        frame = item["image"]
        frames.append(frame)
        display.clear_output(wait=True) 
        display.display(frame)
        for key, value in item.items():
            if not isinstance(value, (np.ndarray, torch.Tensor, Image.Image)):
                print(f"{key}: {value}")
            

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
