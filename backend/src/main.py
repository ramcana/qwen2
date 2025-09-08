import io
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add backend directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qwen_generator import QwenImageGenerator

# --- Pydantic Models for Request Bodies ---

class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 50
    cfg_scale: float = 7.0
    seed: int = -1
    language: str = "en"
    enhance_prompt: bool = True
    style: str = "default"  # For the new style picker

# --- Global Objects ---

# This dictionary will hold our model instance
model_cache = {}

# --- FastAPI Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    print("INFO:     Starting up and loading model...")
    generator = QwenImageGenerator()
    model_loaded = generator.load_model()
    if model_loaded:
        model_cache["generator"] = generator
        print("INFO:     Model loaded successfully.")
    else:
        print("ERROR:    Failed to load model. API will not be functional.")
    yield
    # Shutdown: Clean up resources (if any)
    print("INFO:     Shutting down...")
    model_cache.clear()

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Qwen Image Generation API",
    description="A modern API for professional image generation using the Qwen-Image model, with a React frontend.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Helper Functions ---

def get_generator() -> QwenImageGenerator:
    """Get the generator instance from the cache, handle if not loaded."""
    generator = model_cache.get("generator")
    if not generator:
        raise RuntimeError("Model is not loaded. The application failed to start correctly.")
    return generator

def image_to_streaming_response(image, message):
    """Converts a PIL image to a FastAPI streaming response."""
    if image is None:
        return {"error": message}, 400

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    headers = {"X-Generation-Message": message}
    return StreamingResponse(buffer, media_type="image/png", headers=headers)

# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def read_root():
    """Root endpoint to check API status."""
    return {"message": "Welcome to the Qwen Image Generation API. The model is " + ("loaded." if "generator" in model_cache else "not loaded.")}

@app.post("/api/generate/text-to-image", tags=["Generation"])
async def generate_text_to_image(request: TextToImageRequest):
    """
    Generate an image from a text prompt.
    This endpoint uses the core text-to-image functionality of the Qwen-Image model.
    """
    generator = get_generator()

    # Placeholder for style-based prompt enhancement
    final_prompt = request.prompt
    if request.style != "default":
        # In a real implementation, this would look up a style dictionary
        final_prompt = f"{request.prompt}, in the style of {request.style}"

    image, message = generator.generate_image(
        prompt=final_prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.steps,
        cfg_scale=request.cfg_scale,
        seed=request.seed,
        language=request.language,
        enhance_prompt_flag=request.enhance_prompt,
    )

    return image_to_streaming_response(image, message)

@app.post("/api/generate/image-to-image", tags=["Generation"])
async def generate_image_to_image(
    init_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.8),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(50),
    cfg_scale: float = Form(7.0),
    seed: int = Form(-1),
    language: str = Form("en"),
    enhance_prompt: bool = Form(True),
):
    """
    Generate an image based on an initial image and a text prompt.
    Requires the Qwen-Image-Edit model.
    """
    generator = get_generator()
    from PIL import Image

    image_data = await init_image.read()
    pil_image = Image.open(io.BytesIO(image_data))

    image, message = generator.generate_img2img(
        prompt=prompt,
        init_image=pil_image,
        strength=strength,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        language=language,
        enhance_prompt_flag=enhance_prompt,
    )

    return image_to_streaming_response(image, message)

@app.post("/api/generate/inpainting", tags=["Generation"])
async def generate_inpainting(
    init_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(50),
    cfg_scale: float = Form(7.0),
    seed: int = Form(-1),
    language: str = Form("en"),
    enhance_prompt: bool = Form(True),
):
    """
    Perform inpainting on an image using a mask.
    Requires the Qwen-Image-Edit model.
    """
    generator = get_generator()
    from PIL import Image

    init_image_data = await init_image.read()
    pil_init_image = Image.open(io.BytesIO(init_image_data))

    mask_image_data = await mask_image.read()
    pil_mask_image = Image.open(io.BytesIO(mask_image_data))

    image, message = generator.generate_inpaint(
        prompt=prompt,
        init_image=pil_init_image,
        mask_image=pil_mask_image,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        language=language,
        enhance_prompt_flag=enhance_prompt,
    )

    return image_to_streaming_response(image, message)

@app.post("/api/generate/super-resolution", tags=["Generation"])
async def generate_super_resolution(
    input_image: UploadFile = File(...),
    scale_factor: int = Form(2),
):
    """
    Enhance the resolution of an image.
    """
    generator = get_generator()
    from PIL import Image

    image_data = await input_image.read()
    pil_image = Image.open(io.BytesIO(image_data))

    image, message = generator.super_resolution(
        image=pil_image,
        scale_factor=scale_factor,
    )

    return image_to_streaming_response(image, message)

# To run this app:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# The working directory should be backend/src
