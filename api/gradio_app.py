"""
Gradio web interface for AI Art Generator
"""

import gradio as gr
from models.stable_diffusion import StableDiffusionGenerator
from models.sdxl_generator import SDXLGenerator
from utils.image_utils import save_image

# Global models (loaded once)
sd_generator = None
sdxl_generator = None

def load_models():
    """Load models once at startup"""
    global sd_generator, sdxl_generator
    if sd_generator is None:
        sd_generator = StableDiffusionGenerator()
    print("✅ Models loaded for web interface")

def generate_sd_image(prompt, style, model_type):
    """Generate image with selected model"""
    global sd_generator, sdxl_generator
    
    full_prompt = f"{prompt}, {style}, high quality, detailed"
    
    try:
        if model_type == "Stable Diffusion":
            if sd_generator is None:
                sd_generator = StableDiffusionGenerator()
            images = sd_generator.generate(full_prompt, batch_size=1)
            result_image = images[0]
        
        elif model_type == "SDXL (High Quality)":
            if sdxl_generator is None:
                sdxl_generator = SDXLGenerator()
            result_image = sdxl_generator.generate(full_prompt)
        
        # Save the generated image
        save_path = save_image(result_image, prompt)
        return result_image, f"✅ Generated and saved to: {save_path}"
        
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"

def launch_gradio_app():
    """Launch the Gradio web interface"""
    load_models()
    
    # Create interface
    interface = gr.Interface(
        fn=generate_sd_image,
        inputs=[
            gr.Textbox(
                label="Prompt", 
                placeholder="a magical forest with glowing mushrooms",
                lines=2
            ),
            gr.Dropdown(
                ["digital art", "oil painting", "watercolor", "realistic photo", 
                 "cartoon style", "fantasy art", "cyberpunk", "minimalist"],
                label="Style",
                value="digital art"
            ),
            gr.Radio(
                ["Stable Diffusion", "SDXL (High Quality)"],
                label="Model Type",
                value="Stable Diffusion"
            )
        ],
        outputs=[
            gr.Image(label="Generated Art", type="pil"),
            gr.Textbox(label="Status")
        ],
        title="🎨 AI Art Generator",
        description="Generate amazing artwork from text descriptions using AI!",
        theme="soft",
        examples=[
            ["a beautiful sunset over mountains", "digital art", "Stable Diffusion"],
            ["a cute robot in a garden", "cartoon style", "Stable Diffusion"],
            ["an ancient castle in the clouds", "fantasy art", "SDXL (High Quality)"]
        ]
    )
    
    # Launch with public sharing
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True
    )