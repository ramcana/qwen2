#!/usr/bin/env python3
"""
FastAPI Server for Qwen-Image Generator
Modern REST API backend for React frontend
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_generator import QwenImageGenerator
from qwen_image_config import ASPECT_RATIOS

# Initialize FastAPI app
app = FastAPI(
    title="Qwen-Image API",
    description="Professional AI Image Generation API",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend.localhost", "http://api.localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator: Optional[QwenImageGenerator] = None
generation_queue: Dict[str, Dict] = {}
current_generation: Optional[str] = None

# Pydantic models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1664
    height: int = 928
    num_inference_steps: int = 50
    cfg_scale: float = 4.0
    seed: int = -1
    language: str = "en"
    enhance_prompt: bool = True
    aspect_ratio: str = "16:9"

class ImageToImageRequest(GenerationRequest):
    init_image_path: str
    strength: float = 0.7

class StatusResponse(BaseModel):
    model_loaded: bool
    device: str
    memory_info: Dict
    current_generation: Optional[str] = None
    queue_length: int = 0
    initialization: Optional[Dict] = None

class GenerationResponse(BaseModel):
    success: bool
    message: str
    image_path: Optional[str] = None
    generation_time: Optional[float] = None
    parameters: Optional[Dict] = None
    job_id: Optional[str] = None

# Global state for initialization
initialization_status = {
    "status": "starting",
    "message": "API server starting...",
    "progress": 0,
    "model_loaded": False,
    "error": None
}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the generator on startup"""
    global generator, initialization_status
    
    try:
        print("🚀 Starting Qwen-Image API Server...")
        initialization_status.update({
            "status": "initializing",
            "message": "Initializing Qwen-Image Generator...",
            "progress": 10
        })
        
        generator = QwenImageGenerator()
        
        initialization_status.update({
            "status": "creating_directories",
            "message": "Creating output directories...",
            "progress": 20
        })
        
        # Create output directories
        os.makedirs("generated_images", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        
        initialization_status.update({
            "status": "ready",
            "message": "API server ready. Model will load on first generation request.",
            "progress": 100,
            "model_loaded": False
        })
        
        print("✅ API Server initialized successfully")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        initialization_status.update({
            "status": "error",
            "message": f"Startup failed: {str(e)}",
            "progress": 0,
            "error": str(e)
        })

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": generator is not None and generator.pipe is not None
    }

# Status endpoint
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status with detailed initialization info"""
    global generator, current_generation, generation_queue, initialization_status
    
    memory_info = {}
    if torch.cuda.is_available():
        try:
            # Clear any stale memory before checking
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get fresh memory values
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            
            # Ensure values are sane (prevent corruption)
            if allocated_memory > total_memory:
                print(f"⚠️ Memory corruption detected: {allocated_memory} > {total_memory}")
                allocated_memory = min(allocated_memory, total_memory)
            
            free_memory = max(0, total_memory - allocated_memory)
            
            memory_info = {
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "cached_memory": cached_memory,
                "free_memory": free_memory,
                "device_name": torch.cuda.get_device_name(0)
            }
        except Exception as e:
            memory_info = {"error": f"Could not get memory info: {str(e)}"}
    
    # Enhanced status with initialization details
    status_data = {
        "model_loaded": generator is not None and generator.pipe is not None,
        "device": generator.device if generator else "unknown",
        "memory_info": memory_info,
        "current_generation": current_generation,
        "queue_length": len(generation_queue),
        "initialization": initialization_status.copy()
    }
    
    return StatusResponse(**status_data)

# Initialize model endpoint
@app.post("/initialize")
async def initialize_model():
    """Initialize the Qwen-Image model with detailed progress tracking"""
    global generator, initialization_status
    
    try:
        initialization_status.update({
            "status": "loading_model",
            "message": "Loading Qwen-Image model... This may take a few minutes.",
            "progress": 30,
            "model_loaded": False
        })
        
        if generator is None:
            generator = QwenImageGenerator()
        
        initialization_status.update({
            "status": "downloading_model",
            "message": "Downloading model files (if needed)...",
            "progress": 50
        })
        
        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        initialization_status.update({
            "status": "loading_to_gpu",
            "message": "Loading model to GPU...",
            "progress": 70
        })
        
        success = generator.load_model()
        
        if success:
            initialization_status.update({
                "status": "model_ready",
                "message": "Model loaded successfully and ready for generation!",
                "progress": 100,
                "model_loaded": True
            })
            return {"success": True, "message": "Model loaded successfully", "status": initialization_status}
        else:
            initialization_status.update({
                "status": "model_failed",
                "message": "Failed to load model. Check logs for details.",
                "progress": 0,
                "model_loaded": False,
                "error": "Model loading failed"
            })
            return {"success": False, "message": "Failed to load model", "status": initialization_status}
            
    except Exception as e:
        error_msg = str(e)
        initialization_status.update({
            "status": "error",
            "message": f"Model initialization failed: {error_msg}",
            "progress": 0,
            "model_loaded": False,
            "error": error_msg
        })
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {error_msg}")

# Text-to-image generation
@app.post("/generate/text-to-image", response_model=GenerationResponse)
async def generate_text_to_image(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate image from text prompt"""
    global generator, current_generation, generation_queue
    
    if generator is None or generator.pipe is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please initialize first.")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Add to queue
    generation_queue[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "created_at": datetime.now().isoformat(),
        "type": "text-to-image"
    }
    
    # Start generation in background
    background_tasks.add_task(process_text_to_image, job_id, request)
    
    return GenerationResponse(
        success=True,
        message="Generation started",
        job_id=job_id
    )

async def process_text_to_image(job_id: str, request: GenerationRequest):
    """Process text-to-image generation"""
    global generator, current_generation, generation_queue
    
    current_generation = job_id
    generation_queue[job_id]["status"] = "processing"
    generation_queue[job_id]["started_at"] = datetime.now().isoformat()
    
    try:
        start_time = time.time()
        
        # Generate image
        image, message = generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            language=request.language,
            enhance_prompt_flag=request.enhance_prompt
        )
        
        generation_time = time.time() - start_time
        
        if image is not None:
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_generated_{timestamp}_{job_id[:8]}.png"
            image_path = os.path.join("generated_images", filename)
            image.save(image_path)
            
            # Update queue
            generation_queue[job_id].update({
                "status": "completed",
                "image_path": image_path,
                "generation_time": generation_time,
                "completed_at": datetime.now().isoformat(),
                "message": message
            })
        else:
            generation_queue[job_id].update({
                "status": "failed",
                "error": message,
                "completed_at": datetime.now().isoformat()
            })
            
    except Exception as e:
        generation_queue[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
    finally:
        current_generation = None

# Image-to-image generation
@app.post("/generate/image-to-image", response_model=GenerationResponse)
async def generate_image_to_image(request: ImageToImageRequest, background_tasks: BackgroundTasks):
    """Generate image from image and text prompt"""
    global generator
    
    if generator is None or generator.pipe is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please initialize first.")
    
    # For now, return not implemented
    # This would need proper image upload handling
    raise HTTPException(status_code=501, detail="Image-to-image not yet implemented in API")

# Get aspect ratios
@app.get("/aspect-ratios")
async def get_aspect_ratios():
    """Get available aspect ratios"""
    return {"ratios": ASPECT_RATIOS}

# Queue management
@app.get("/queue")
async def get_queue():
    """Get current generation queue"""
    return {"queue": generation_queue, "current": current_generation}

@app.delete("/queue/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job"""
    if job_id in generation_queue:
        if generation_queue[job_id]["status"] == "queued":
            generation_queue[job_id]["status"] = "cancelled"
            return {"success": True, "message": "Job cancelled"}
        else:
            return {"success": False, "message": "Job cannot be cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

# Memory management
@app.get("/memory/clear")
async def clear_memory():
    """Clear GPU memory with enhanced cleanup"""
    try:
        if torch.cuda.is_available():
            # Get initial state
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_reserved()
            
            # Enhanced memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Second cleanup pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get final state
            final_allocated = torch.cuda.memory_allocated()
            final_cached = torch.cuda.memory_reserved()
            
            memory_info = {
                "total_memory": torch.cuda.get_device_properties(0).total_memory,
                "allocated_memory": final_allocated,
                "cached_memory": final_cached,
                "freed_allocated": initial_allocated - final_allocated,
                "freed_cached": initial_cached - final_cached,
                "device_name": torch.cuda.get_device_name(0)
            }
            
            return {
                "success": True,
                "message": f"GPU memory cleared. Freed {(initial_allocated - final_allocated) / 1e9:.2f}GB allocated, {(initial_cached - final_cached) / 1e9:.2f}GB cached",
                "memory_info": memory_info
            }
        else:
            return {"success": False, "message": "CUDA not available"}
    except Exception as e:
        return {"success": False, "message": f"Memory cleanup failed: {str(e)}"}

@app.get("/memory/status")
async def get_memory_status():
    """Get detailed memory status"""
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - allocated_memory
            
            return {
                "success": True,
                "memory_info": {
                    "device_name": torch.cuda.get_device_name(0),
                    "total_memory": total_memory,
                    "total_memory_gb": round(total_memory / 1e9, 2),
                    "allocated_memory": allocated_memory,
                    "allocated_memory_gb": round(allocated_memory / 1e9, 2),
                    "cached_memory": cached_memory,
                    "cached_memory_gb": round(cached_memory / 1e9, 2),
                    "free_memory": free_memory,
                    "free_memory_gb": round(free_memory / 1e9, 2),
                    "memory_usage_percent": round((allocated_memory / total_memory) * 100, 1)
                }
            }
        else:
            return {"success": False, "message": "CUDA not available"}
    except Exception as e:
        return {"success": False, "message": f"Failed to get memory status: {str(e)}"}

# Serve generated images
@app.get("/images/{filename}")
async def get_image(filename: str):
    """Serve generated images"""
    image_path = os.path.join("generated_images", filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

# Serve React frontend (if built)
frontend_build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")
frontend_static_path = os.path.join(frontend_build_path, "static")

if os.path.exists(frontend_build_path) and os.path.exists(frontend_static_path):
    app.mount("/static", StaticFiles(directory=frontend_static_path), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_build_path, "index.html"))
else:
    print("ℹ️ Frontend build not found. API-only mode enabled.")

if __name__ == "__main__":
    import uvicorn
    
    print("""
🎨 Qwen-Image API Server
========================

Features:
✅ Modern FastAPI backend
✅ React frontend support
✅ Real-time generation status
✅ Queue management
✅ Memory optimization
✅ CORS enabled

Starting server...
    """)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )