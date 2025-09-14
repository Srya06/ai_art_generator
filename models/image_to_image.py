"""
Image-to-Image generation
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

class ImageToImageGenerator:
    def __init__(self):
        """Initialize img2img pipeline"""
        print("🔄 Loading Image-to-Image model...")
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            self.pipe.to("cuda")
        
        print("✅ Image-to-Image model ready!")
    
    def generate(self, image_path, prompt, strength=0.75, num_steps=20):
        """Generate modified image"""
        try:
            # Load and preprocess image
            init_image = Image.open(image_path).convert("RGB")
            init_image = init_image.resize((512, 512))
            
            # Generate modified version
            result = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps
            ).images[0]
            
            return result
        except Exception as e:
            print(f"❌ Image-to-image generation failed: {e}")
            return None
    
    def batch_modify(self, image_paths, prompts, **kwargs):
        """Modify multiple images"""
        results = []
        for img_path, prompt in zip(image_paths, prompts):
            result = self.generate(img_path, prompt, **kwargs)
            if result:
                results.append(result)
        return results