"""
Setup script for AI Art Generator
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    print("🔄 Installing requirements...")
    
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    print("✅ Requirements installed successfully!")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "generated_images",
        "reference_images", 
        "models/__pycache__",
        "api/__pycache__",
        "utils/__pycache__"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Project directories created!")

def main():
    print("🚀 Setting up AI Art Generator...")
    install_requirements()
    setup_directories()
    print("🎉 Setup complete! Run 'python main.py' to start!")

if __name__ == "__main__":
    main()