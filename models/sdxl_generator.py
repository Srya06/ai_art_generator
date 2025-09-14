"""
SDXL Generator for high-quality images
"""

import torch
from diffusers import DiffusionPipeline
from PIL import Image

class SDXLGenerator:
    def __init__(self):
        """Initialize SDXL model"""
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available. SDXL will be slow on CPU.")
        
        print("🔄 Loading SDXL model (this may take a while)...")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None
        )
        
        if torch.cuda.is_available():
            self.pipe.to("cuda")
        
        print("✅ SDXL model loaded!")
    
    def generate(self, prompt, num_steps=20, guidance_scale=7.5):
        """Generate high-quality image with SDXL"""
        try:
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            return image
        except Exception as e:
            print(f"❌ SDXL generation failed: {e}")
            return None
    
    def generate_batch(self, prompts, **kwargs):
        """Generate multiple images"""
        results = []
        for prompt in prompts:
            image = self.generate(prompt, **kwargs)
            if image:
                results.append(image)
        return results