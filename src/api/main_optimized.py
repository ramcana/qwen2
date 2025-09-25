"""
High-Performance Qwen-Image FastAPI Server
Optimized for Threadripper PRO 5995WX + RTX 4080 + 128GB RAM
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.qwen_generator_optimized import OptimizedQwenImageGenerator
from src.qwen_image_config import ASPECT_RATIOS

# Initialize FastAPI app
app = FastAPI(
    title="Qwen-Image High-Performance API",
    description="Optimized for high-end hardware: Threadripper PRO + RTX 4080",
    version="2.1.0-optimized",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000",
        "http://localhost",  # Add Traefik frontend
        "http://qwen.localhost",  # Add alternative Traefik domain
        "http://127.0.0.1"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator: Optional[OptimizedQwenImageGenerator] = None
generation_queue: List[Dict[str, Any]] = []
is_generating = False

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1664
    height: int = 960
    num_inference_steps: int = 50
    cfg_scale: float = 4.0
    seed: int = -1
    language: str = "en"
    enhance_prompt: bool = True

class GenerationResponse(BaseModel):
    success: bool
    image_path: Optional[str] = None
    message: str
    generation_time: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None

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
    """Clear GPU memory aggressively"""
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"🧹 GPU memory cleared")

async def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    allocated = torch.cuda.memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    usage_percent = min(round(100 * allocated / total, 1), 100.0)
    
    return {
        "gpu_available": True,
        "allocated_gb": round(allocated / (1024**3), 2),
        "total_gb": round(total / (1024**3), 2),
        "usage_percent": usage_percent,
        "device_name": torch.cuda.get_device_name(0),
        "available_gb": round((total - allocated) / (1024**3), 2)
    }

# Generator management
async def get_generator() -> OptimizedQwenImageGenerator:
    """Get or initialize the optimized generator instance"""
    global generator
    if generator is None:
        generator = OptimizedQwenImageGenerator()
        success = generator.load_model()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load optimized model")
    return generator

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Qwen-Image High-Performance API",
        "version": "2.1.0-optimized",
        "description": "Optimized for Threadripper PRO 5995WX + RTX 4080 + 128GB RAM",
        "hardware": "High-end optimizations enabled",
        "status": "running",
        "endpoints": {
            "status": "/status",
            "generate": "/generate/text-to-image",
            "aspect-ratios": "/aspect-ratios",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API and model status"""
    memory_info = await get_memory_info()
    
    return StatusResponse(
        model_loaded=generator is not None,
        gpu_available=torch.cuda.is_available(),
        memory_info=memory_info,
        queue_size=len(generation_queue),
        is_generating=is_generating
    )

@app.get("/aspect-ratios", response_model=AspectRatioResponse)
async def get_aspect_ratios():
    """Get available aspect ratio presets"""
    return AspectRatioResponse(ratios=ASPECT_RATIOS)

@app.post("/generate/text-to-image", response_model=GenerationResponse)
async def generate_text_to_image(request: GenerationRequest):
    """Generate image from text with high-performance optimizations"""
    global is_generating
    
    if is_generating:
        return GenerationResponse(
            success=False,
            message="Another generation is in progress. Please wait."
        )
    
    try:
        is_generating = True
        gen = await get_generator()
        
        print(f"🚀 HIGH-PERFORMANCE GENERATION REQUEST:")
        print(f"   • Hardware: Threadripper PRO 5995WX + RTX 4080")
        print(f"   • Optimizations: Flash Attention, xFormers, torch.compile")
        print(f"   • Resolution: {request.width}×{request.height}")
        print(f"   • Steps: {request.num_inference_steps}")
        
        # Generate image with optimized generator
        image, result_path = gen.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            language=request.language,
            enhance_prompt=request.enhance_prompt
        )
        
        if image is None:
            return GenerationResponse(
                success=False,
                message=result_path  # Error message
            )
        
        # Extract generation time from the generator's last operation
        generation_time = getattr(gen, '_last_generation_time', None)
        
        return GenerationResponse(
            success=True,
            image_path=result_path,
            message="Image generated successfully with high-performance optimizations",
            generation_time=generation_time,
            parameters={
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.num_inference_steps,
                "cfg_scale": request.cfg_scale,
                "seed": request.seed,
                "optimizations": "Flash Attention + xFormers + torch.compile"
            }
        )
        
    except Exception as e:
        print(f"❌ High-performance generation failed: {e}")
        return GenerationResponse(
            success=False,
            message=f"Generation failed: {str(e)}"
        )
    finally:
        is_generating = False

@app.get("/memory/clear")
async def clear_memory():
    """Clear GPU memory"""
    try:
        await clear_gpu_memory()
        memory_info = await get_memory_info()
        return {
            "success": True,
            "message": "GPU memory cleared",
            "memory_info": memory_info
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clear memory: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    memory_info = await get_memory_info()
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "gpu_available": torch.cuda.is_available(),
        "memory_info": memory_info,
        "optimizations": "High-performance mode enabled",
        "hardware": "Threadripper PRO 5995WX + RTX 4080 + 128GB RAM"
    }

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Serve generated images"""
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup with high-performance settings"""
    print("🚀 Starting Qwen-Image HIGH-PERFORMANCE FastAPI server...")
    print("🔥 Hardware: Threadripper PRO 5995WX + RTX 4080 + 128GB RAM")
    print("⚡ Optimizations: Flash Attention + xFormers + torch.compile")
    print("🎯 Performance mode: MAXIMUM SPEED")
    print("🔗 CORS configured for React frontend")
    
    # Pre-warm the optimized generator
    try:
        await get_generator()
        print("✅ High-performance model pre-loaded successfully")
        print("🚀 Ready for BLAZING FAST image generation!")
    except Exception as e:
        print(f"⚠️ Model pre-loading failed: {e}")
        print("💡 Model will be loaded on first request")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await clear_gpu_memory()
    print("🛑 High-performance server shutdown complete")

if __name__ == "__main__":
    import uvicorn
    print("""
    🚀 Qwen-Image HIGH-PERFORMANCE API Server
    
    Hardware Optimized For:
    • AMD Ryzen Threadripper PRO 5995WX (64 cores)
    • NVIDIA RTX 4080 (16GB VRAM)
    • 128GB System RAM
    
    Performance Optimizations:
    • Attention slicing DISABLED (maximum speed)
    • Flash Attention 2.0 ENABLED
    • xFormers memory-efficient attention ENABLED
    • torch.compile UNet optimization ENABLED
    
    🌐 Server starting at: http://localhost:8000
    📚 API Docs: http://localhost:8000/docs
    """)
    
    uvicorn.run(
        "main_optimized:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for maximum performance
        log_level="info"
    )
