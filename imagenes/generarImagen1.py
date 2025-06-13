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

# We'll use a couple of demo images later in the notebook
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Download images for inpainting example
#img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
#mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
#img_url = input("url de la imagen original: ")
#mask_url = input("url de la mascara: ")

#init_image = download_image(img_url).resize((512, 512))
#mask_image = download_image(mask_url).resize((512, 512))

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

tamagnox = int(input("tamagno x (640): "))
tamagnoy = int(input("tamagno y (480): "))
escala_guia = int(input("guidance scale (8): "))
pasos_inferencia = int(input("num inference steps (35): "))
for i in range(0,int(input("cantidad de imagenes: "))):
    # Set up a generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(random.randint(1,99)) #42

    # Run the pipeline, showing some of the available arguments
    pipe_output = pipe(
        prompt=input("prompt: "), # What to generate
        negative_prompt=input("prompt negativo: "), # What NOT to generate
        height=tamagnoy, width=tamagnox,     # Specify the image size
        guidance_scale=escala_guia,          # How strongly to follow the prompt
        num_inference_steps=pasos_inferencia,    # How many steps to take
        generator=generator        # Fixed random seed
    )

    # View the resulting image
    pipe_output.images[0].show()


