#SCRIPT AJENO

import torch
import requests
import random
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import LMSDiscreteScheduler


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

# We'll use a couple of demo images later in the notebook
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")
init_image = download_image(input("url imagen (https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png): ")).resize((512, 512))


model_id = "stabilityai/stable-diffusion-2-1-base"
# Loading an Img2Img pipeline
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)

# Apply Img2Img
result_image = img2img_pipe(
    prompt=input("prompt: "),
    image=init_image, # The starting image
    strength=float(input("fuerza (0.1-0.9): ")), # 0 for no change, 1.0 for max strength
).images[0]

# View the result
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(result_image);axs[1].set_title('Result')
result_image.show()