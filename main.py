"""
AI Art Generator - Main Application
Complete implementation of Stable Diffusion models
"""

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from models.stable_diffusion import StableDiffusionGenerator
from models.sdxl_generator import SDXLGenerator
from models.image_to_image import ImageToImageGenerator
from utils.image_utils import save_image, display_image

def setup_directories():
    """Create necessary directories"""
    directories = ['generated_images', 'reference_images']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Project directories created")

def main():
    print("🎨 AI Art Generator - VS Code Edition")
    print("=" * 50)
    
    setup_directories()
    
    # Initialize generators
    print("Loading models...")
    sd_generator = StableDiffusionGenerator()
    
    while True:
        print("\nChoose an option:")
        print("1. Generate with Stable Diffusion")
        print("2. Generate with SDXL (High Quality)")
        print("3. Image-to-Image Generation")
        print("4. Start Web Interface")
        print("5. Start API Server")
        print("6. Benchmark Models")
        print("7. Exit")
        
        choice = input("Enter choice (1-7): ")
        
        if choice == '1':
            prompt = input("Enter your prompt: ")
            style = input("Enter style (or press Enter for default): ") or "digital art"
            generate_basic(sd_generator, prompt, style)
            
        elif choice == '2':
            prompt = input("Enter your prompt: ")
            generate_sdxl(prompt)
            
        elif choice == '3':
            image_path = input("Enter path to reference image: ")
            prompt = input("Enter modification prompt: ")
            generate_img2img(image_path, prompt)
            
        elif choice == '4':
            start_gradio_interface()
            
        elif choice == '5':
            start_api_server()
            
        elif choice == '6':
            benchmark_models()
            
        elif choice == '7':
            print("Thanks for using AI Art Generator!")
            break
            
        else:
            print("Invalid choice. Please try again.")

def generate_basic(generator, prompt, style):
    """Generate with basic Stable Diffusion"""
    full_prompt = f"{prompt}, {style}, high quality, detailed"
    print(f"Generating: {full_prompt}")
    
    images = generator.generate(full_prompt, batch_size=1)
    save_path = save_image(images[0], prompt)
    display_image(images[0], f"Generated: {full_prompt}")
    print(f"✅ Image saved to: {save_path}")

def generate_sdxl(prompt):
    """Generate with SDXL for higher quality"""
    print("Loading SDXL model...")
    sdxl_gen = SDXLGenerator()
    
    print(f"Generating with SDXL: {prompt}")
    image = sdxl_gen.generate(prompt)
    save_path = save_image(image, f"SDXL_{prompt}")
    display_image(image, f"SDXL Generated: {prompt}")
    print(f"✅ SDXL Image saved to: {save_path}")

def generate_img2img(image_path, prompt):
    """Generate image-to-image"""
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        return
    
    print("Loading Image-to-Image model...")
    img2img_gen = ImageToImageGenerator()
    
    print(f"Modifying image with prompt: {prompt}")
    result_image = img2img_gen.generate(image_path, prompt)
    save_path = save_image(result_image, f"img2img_{prompt}")
    display_image(result_image, f"Modified: {prompt}")
    print(f"✅ Modified image saved to: {save_path}")

def start_gradio_interface():
    """Launch Gradio web interface"""
    from api.gradio_app import launch_gradio_app
    print("🌐 Starting Gradio web interface...")
    launch_gradio_app()

def start_api_server():
    """Launch FastAPI server"""
    try:
        from api.fastapi_server import start_server
    except ImportError:
        print("❌ Could not import 'api.fastapi_server'. Please ensure 'api/fastapi_server.py' exists and is in the correct location.")
        return
    print("🚀 Starting FastAPI server...")
    start_server()

def benchmark_models():
    """Compare different models performance"""
    print("🔥 Running model benchmarks...")
    
    test_prompt = "a beautiful landscape with mountains and lakes"
    
    # Test basic SD
    sd_gen = StableDiffusionGenerator()
    import time
    start = time.time()
    sd_image = sd_gen.generate(test_prompt)[0]
    sd_time = time.time() - start
    
    print(f"Stable Diffusion: {sd_time:.2f}s")
    
    # Display comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sd_image)
    plt.title(f"Stable Diffusion ({sd_time:.1f}s)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()