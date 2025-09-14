import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def save_image(image, name_or_prompt, output_dir="generated_images"):
    """Save generated image with timestamp and safe filename"""
    Path(output_dir).mkdir(exist_ok=True)

    # Create filename (limit length, clean unsafe chars)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in str(name_or_prompt) if c.isalnum() or c in (" ", "_")).rstrip()[:50]
    filename = f"{timestamp}_{safe_name.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Save PIL image
    if isinstance(image, Image.Image):
        image.save(filepath)
    else:
        raise ValueError("Unsupported image type passed to save_image")

    return filepath

def display_image(image, title="Generated Image"):
    """Display image with matplotlib"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def create_image_grid(images, titles=None, rows=1):
    """Create a grid of images"""
    cols = len(images) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def resize_image(image_path, size=(512, 512)):
    """Resize image to specified dimensions"""
    image = Image.open(image_path).convert("RGB")
    return image.resize(size)
