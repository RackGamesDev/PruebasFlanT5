#SCRIPT AJENO

import torch
import requests
import random
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline
    )

# Set device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Load the pipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

#@markdown comparing guidance scales:
cfg_scales = [1.1, 8, 12] #@param
prompt = input("prompt: ") #@param
fig, axs = plt.subplots(1, len(cfg_scales), figsize=(16, 5))
tamagno = int(input("tamagno (480): "))
for i, ax in enumerate(axs):
  im = pipe(prompt, height=tamagno, width=tamagno,
    guidance_scale=cfg_scales[i], num_inference_steps=35,
    generator=torch.Generator(device=device).manual_seed(random.randint(1,99))).images[0]
  ax.imshow(im)
  ax.set_title(f'CFG Scale {cfg_scales[i]}')
plt.tight_layout()
plt.show()

#print(list(pipe.components.keys())) # List components