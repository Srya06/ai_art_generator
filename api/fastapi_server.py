# api/fastapi_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import your model + utils
from models.stable_diffusion import StableDiffusionGenerator
from utils.image_utils import save_image

# Load Stable Diffusion model once (not every request!)
sd_generator = StableDiffusionGenerator()

app = FastAPI(title="AI Art Generator API")

class GenerateRequest(BaseModel):
    prompt: str
    style: str = "digital art"

class GenerateResponse(BaseModel):
    prompt: str
    style: str
    image_path: str

@app.get("/")
def home():
    return {"message": "🚀 AI Art Generator API is running"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        full_prompt = f"{req.prompt}, {req.style}, high quality, detailed"
        images = sd_generator.generate(full_prompt, batch_size=1)
        image = images[0]
        save_path = save_image(image, full_prompt.replace(" ", "_"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return GenerateResponse(
        prompt=req.prompt,
        style=req.style,
        image_path=save_path
    )

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)
