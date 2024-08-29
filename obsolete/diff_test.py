from diffusers import AutoPipelineForText2Image
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    print("importing finished")

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True).to("cuda")

    prompt = "astronaut climbing medieval castle walls"
    image = pipeline_text2image(prompt=prompt).images[0]
    print(type(image))
    
    plt.imshow(np.asarray(image))
    plt.show()

