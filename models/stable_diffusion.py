"""
Stable Diffusion Generator
Basic implementation with optimization
"""

import keras_cv
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image

class StableDiffusionGenerator:
    def __init__(self, img_width=512, img_height=512, optimize=True):
        """Initialize Stable Diffusion model with optimizations"""
        print("🔄 Loading Stable Diffusion model...")
        
        if optimize:
            # Enable mixed precision for better performance
            keras.mixed_precision.set_global_policy("mixed_float16")
            
        # Create model with optimizations
        self.model = keras_cv.models.StableDiffusion(
            img_width=img_width, 
            img_height=img_height,
            jit_compile=optimize  # XLA compilation
        )
        
        # Warm up the model
        print("🔥 Warming up model...")
        self.model.text_to_image("warmup", batch_size=1)
        print("✅ Stable Diffusion ready!")
    
    def generate(self, prompt, batch_size=1, num_steps=50):
        """Generate images from text prompt"""
        try:
            images = self.model.text_to_image(
                prompt, 
                batch_size=batch_size,
                num_steps=num_steps
            )
            return images
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    def generate_with_settings(self, prompt, **kwargs):
        """Generate with custom settings"""
        return self.model.text_to_image(prompt, **kwargs)
    
    def clear_session(self):
        """Clear TensorFlow session to free memory"""
        keras.backend.clear_session()