# Deforum Kandinsky

## Getting started...
### install dependencies
```bash
git clone https://gitlab.aicloud.sbercloud.ru/sberai-rnd/deforum-kandinsky.git
cd deforum-kandinsky
pip install -r requirements.txt
```

### Import dependencies
```python
from IPython.display import Video
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
import imageio.v2 as iio
from tqdm.autonotebook import tqdm
from PIL import Image
import numpy as np
import torch
```

### convert list frames to mp4 video
```python
#  create video from generated frames
def frames2video(frames, output_path="video.mp4"):
    writer = iio.get_writer(output_path, fps=24)
    for frame in tqdm(frames):
        writer.append_data(np.array(frame))
    writer.close()
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
### Easily create animation using default modes

#### define instance of Deforum
```python
deforum = DeforumKandinsky(
    prior=prior,
    decoder_img2img=decoder,
    device='cuda'
)
```

#### One-liner
```python
frames = deforum("Van Gogh's Starry Night", "live")
```
#### Or
```python
frames = deforum(
    prompts=[
        "Van Gogh's Starry Night", 
        'Space, stars, galaxies', 
        'Milky Way'
    ], 
    animations=[
        'live', 
        'up', 
        'spin_clockwise'
    ], 
    prompt_durations=[4, 3, 2],
    negative_prompts=[
        'low quility', 
        'bad image, cropped', 
        'cropped'
    ],
    verbose=True,
    H=640,
    W=640,
)
```