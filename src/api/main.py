#!/usr/bin/env python3
"""
FastAPI Backend for Qwen-Image Generator
Memory-optimized endpoints with advanced image generation capabilities
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add parent directories to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.qwen_generator import QwenImageGenerator
from src.qwen_image_config import ASPECT_RATIOS

# Initialize FastAPI app
app = FastAPI(
    title="Qwen-Image API",
    description="Professional text-to-image generation with memory optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator: Optional[QwenImageGenerator] = None
generation_queue: List[Dict[str, Any]] = []
is_generating = False


# Pydantic models for API requests/responses
class TextToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid")
    width: int = Field(1664, ge=512, le=2048, description="Image width")
    height: int = Field(928, ge=512, le=2048, description="Image height")
    num_inference_steps: int = Field(
        50, ge=10, le=100, description="Number of inference steps"
    )
    cfg_scale: float = Field(4.0, ge=1.0, le=20.0, description="CFG scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    language: str = Field("en", pattern="^(en|zh)$", description="Prompt language")
    enhance_prompt: bool = Field(True, description="Enable prompt enhancement")
    aspect_ratio: Optional[str] = Field(None, description="Preset aspect ratio")


class ImageToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    init_image_path: str = Field(..., description="Path to input image")
    strength: float = Field(0.7, ge=0.1, le=1.0, description="Transformation strength")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    width: int = Field(1024, ge=512, le=2048, description="Output width")
    height: int = Field(1024, ge=512, le=2048, description="Output height")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Inference steps")
    cfg_scale: float = Field(4.0, ge=1.0, le=20.0, description="CFG scale")
    seed: int = Field(-1, description="Random seed")
    language: str = Field("en", pattern="^(en|zh)$", description="Language")
    enhance_prompt: bool = Field(True, description="Enhance prompt")


class GenerationResponse(BaseModel):
    success: bool
    image_path: Optional[str] = None
    message: str
    generation_time: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None


class StatusResponse(BaseModel):
    model_loaded: bool
    gpu_available: bool
    memory_info: Optional[Dict[str, Any]] = None
    queue_size: int
    is_generating: bool


class AspectRatioResponse(BaseModel):
    ratios: Dict[str, tuple]


# Memory management utilities
async def clear_gpu_memory():
    """Clear GPU memory asynchronously"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        torch.cuda.empty_cache()


async def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    allocated = torch.cuda.memory_allocated(0)
    total = torch.cuda.get_device_properties(0).total_memory

    return {
        "gpu_available": True,
        "allocated_gb": round(allocated / 1e9, 2),
        "total_gb": round(total / 1e9, 2),
        "usage_percent": round(100 * allocated / total, 1),
        "device_name": torch.cuda.get_device_name(0),
    }


# Generator management
async def get_generator() -> QwenImageGenerator:
    """Get or initialize the generator instance"""
    global generator
    if generator is None:
        generator = QwenImageGenerator()
        success = generator.load_model()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
    return generator


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Qwen-Image API",
        "version": "2.0.0",
        "description": "Professional text-to-image generation with memory optimization",
        "endpoints": {
            "status": "/status",
            "generate": "/generate/text-to-image",
            "img2img": "/generate/image-to-image",
            "aspect-ratios": "/aspect-ratios",
            "queue": "/queue",
            "docs": "/docs",
        },
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API and model status"""
    memory_info = await get_memory_info()

    return StatusResponse(
        model_loaded=generator is not None and generator.pipe is not None,
        gpu_available=torch.cuda.is_available(),
        memory_info=memory_info,
        queue_size=len(generation_queue),
        is_generating=is_generating,
    )


@app.post("/initialize")
async def initialize_model():
    """Initialize the Qwen-Image model"""
    try:
        await clear_gpu_memory()
        _ = await get_generator()  # Initialize generator, don't need to store reference
        return {"success": True, "message": "Model initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model initialization failed: {str(e)}"
        )


@app.get("/aspect-ratios", response_model=AspectRatioResponse)
async def get_aspect_ratios():
    """Get available aspect ratio presets"""
    return AspectRatioResponse(ratios=ASPECT_RATIOS)


@app.post("/generate/text-to-image", response_model=GenerationResponse)
async def generate_text_to_image(
    request: TextToImageRequest, background_tasks: BackgroundTasks
):
    """Generate image from text prompt"""
    global is_generating

    if is_generating:
        # Add to queue instead of rejecting
        job_id = (
            f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(generation_queue)}"
        )
        generation_queue.append(
            {
                "job_id": job_id,
                "type": "text_to_image",
                "request": request.dict(),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return GenerationResponse(
            success=True,
            message=f"Request queued. Position: {len(generation_queue)}",
            job_id=job_id,
        )

    try:
        is_generating = True
        await clear_gpu_memory()

        gen = await get_generator()

        # Apply aspect ratio if provided
        if request.aspect_ratio and request.aspect_ratio in ASPECT_RATIOS:
            request.width, request.height = ASPECT_RATIOS[request.aspect_ratio]

        # Generate image
        start_time = datetime.now()

        # Use the actual generator method
        image, message = gen.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            language=request.language,
            enhance_prompt_flag=request.enhance_prompt,
        )

        generation_time = (datetime.now() - start_time).total_seconds()

        if image is None:
            raise HTTPException(status_code=500, detail=message)

        # Save image and get path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_generated_{timestamp}.png"
        image_path = os.path.join("generated_images", filename)
        image.save(image_path)

        # Schedule memory cleanup
        background_tasks.add_task(clear_gpu_memory)

        return GenerationResponse(
            success=True,
            image_path=image_path,
            message=message,
            generation_time=generation_time,
            parameters=request.dict(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        is_generating = False


@app.post("/generate/image-to-image", response_model=GenerationResponse)
async def generate_image_to_image(
    request: ImageToImageRequest, background_tasks: BackgroundTasks
):
    """Generate image from image + text prompt"""
    global is_generating

    if is_generating:
        job_id = (
            f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(generation_queue)}"
        )
        generation_queue.append(
            {
                "job_id": job_id,
                "type": "image_to_image",
                "request": request.dict(),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return GenerationResponse(
            success=True,
            message=f"Request queued. Position: {len(generation_queue)}",
            job_id=job_id,
        )

    try:
        is_generating = True
        await clear_gpu_memory()

        gen = await get_generator()

        # Load input image
        from PIL import Image

        if not os.path.exists(request.init_image_path):
            raise HTTPException(status_code=404, detail="Input image not found")

        init_image = Image.open(request.init_image_path)

        start_time = datetime.now()

        image, message = gen.generate_img2img(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            init_image=init_image,
            strength=request.strength,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            language=request.language,
            enhance_prompt_flag=request.enhance_prompt,
        )

        generation_time = (datetime.now() - start_time).total_seconds()

        if image is None:
            raise HTTPException(status_code=500, detail=message)

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_img2img_{timestamp}.png"
        image_path = os.path.join("generated_images", filename)
        image.save(image_path)

        background_tasks.add_task(clear_gpu_memory)

        return GenerationResponse(
            success=True,
            image_path=image_path,
            message=message,
            generation_time=generation_time,
            parameters=request.dict(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Image-to-image generation failed: {str(e)}"
        )
    finally:
        is_generating = False


@app.get("/queue")
async def get_queue():
    """Get current generation queue status"""
    return {
        "queue_size": len(generation_queue),
        "is_generating": is_generating,
        "queue": generation_queue[:5],  # Show first 5 items
    }


@app.delete("/queue/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job"""
    global generation_queue

    original_size = len(generation_queue)
    generation_queue = [job for job in generation_queue if job["job_id"] != job_id]

    if len(generation_queue) < original_size:
        return {"success": True, "message": f"Job {job_id} cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found in queue")


@app.get("/images/{filename}")
async def get_image(filename: str):
    """Serve generated images"""
    image_path = os.path.join("generated_images", filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/png", filename=filename)


@app.get("/memory/clear")
async def clear_memory():
    """Manual memory clearing endpoint"""
    await clear_gpu_memory()
    memory_info = await get_memory_info()
    return {
        "success": True,
        "message": "GPU memory cleared",
        "memory_info": memory_info,
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    memory_info = await get_memory_info()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": generator is not None,
        "gpu_available": torch.cuda.is_available(),
        "memory_info": memory_info,
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting Qwen-Image FastAPI server...")
    print("📊 Memory optimization enabled")
    print("🔗 CORS configured for React frontend")

    # Pre-warm the generator (optional)
    try:
        await get_generator()
        print("✅ Model pre-loaded successfully")
    except Exception as e:
        print(f"⚠️ Model pre-loading failed: {e}")
        print("💡 Model will be loaded on first request")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await clear_gpu_memory()
    print("🛑 Qwen-Image API server stopped")


if __name__ == "__main__":
    print(
        """
    🎨 Qwen-Image FastAPI Server

    Features:
    ✅ Memory-optimized endpoints
    ✅ Request queuing system
    ✅ CORS enabled for React
    ✅ Background memory cleanup
    ✅ Real-time status monitoring

    Endpoints:
    📍 Health: /health
    📍 Status: /status
    📍 Generate: /generate/text-to-image
    📍 Img2Img: /generate/image-to-image
    📍 Queue: /queue
    📍 Docs: /docs

    🌐 Access at: http://localhost:8000
    📚 API Docs: http://localhost:8000/docs
    """
    )

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
