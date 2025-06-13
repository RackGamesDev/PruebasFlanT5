#SCRIPT AJENO

import torch
import random
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

# Set device
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load pipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Optionally replace the scheduler
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

# Plot the scheduler's alphas_cumprod
plt.plot(pipe.scheduler.alphas_cumprod, label=r'$\bar{\alpha}$')
plt.xlabel('Timestep (high noise to low noise ->)')
plt.title('Noise schedule')
plt.legend()
plt.show()

# Get prompt and image size from user
tamagno = int(input("tamagno (480): "))
user_prompt = input("prompt: ")

# Generate a random seed for reproducibility
seed = random.randint(1, 99)
generator = torch.Generator(device=device).manual_seed(seed)

# Generate the image
output = pipe(
    prompt=user_prompt,
    height=tamagno,
    width=tamagno,
    generator=generator
)

# Show the resulting image
output.images[0].show()
# Or with matplotlib:
# plt.imshow(output.images[0])
# plt.axis("off")
# plt.show()