#!/usr/bin/env python3
"""
DiffSynth Enhanced API Server
Modern REST API backend focused on DiffSynth-Studio integration
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

# Import DiffSynth components
from diffsynth_service import DiffSynthService, DiffSynthConfig
from diffsynth_models import (
    ImageEditRequest, ImageEditResponse, InpaintRequest, 
    OutpaintRequest, StyleTransferRequest, EditOperation
)

# Import ControlNet components
from controlnet_service import (
    ControlNetService, ControlNetType,
    ControlNetDetectionResult, ControlMapResult
)

# Initialize FastAPI app
app = FastAPI(
    title="DiffSynth Enhanced API",
    description="Professional AI Image Generation API with DiffSynth-Studio",
    version="3.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
diffsynth_service: Optional[DiffSynthService] = None
controlnet_service: Optional[ControlNetService] = None

# Job tracking
active_jobs: Dict[str, dict] = {}

# Ensure directories exist
os.makedirs("generated_images", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

# API Models
class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "DiffSynth Enhanced API Server",
        "version": "3.0.0",
        "services": ["DiffSynth-Studio", "ControlNet"],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global diffsynth_service, controlnet_service
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "diffsynth": {
                "available": True,  # DiffSynth is available, just not initialized yet
                "status": str(diffsynth_service.status) if diffsynth_service else "not_initialized"
            },
            "controlnet": {
                "available": True,  # ControlNet is available, just not initialized yet
                "status": "ready" if controlnet_service else "not_initialized"
            }
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A",
            "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
        }
    }

@app.post("/api/generate/text-to-image")
async def generate_text_to_image(request: TextToImageRequest, background_tasks: BackgroundTasks):
    """Generate image from text using DiffSynth"""
    global diffsynth_service
    
    # Initialize DiffSynth service if needed
    if diffsynth_service is None:
        diffsynth_service = DiffSynthService()
        await asyncio.to_thread(diffsynth_service.initialize)
    
    # Create job ID for tracking
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Job queued for processing"
    }
    
    # Add background task
    background_tasks.add_task(
        process_text_to_image,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "queued"}

async def process_text_to_image(job_id: str, request: TextToImageRequest):
    """Process text-to-image generation in background"""
    global diffsynth_service
    
    try:
        # Update job status
        active_jobs[job_id].update({
            "status": "processing",
            "progress": 0.1,
            "message": "Starting image generation..."
        })
        
        # Create DiffSynth request
        edit_request = ImageEditRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            seed=request.seed if request.seed > 0 else None,
            operation=EditOperation.GENERATE
        )
        
        # Update progress
        active_jobs[job_id].update({
            "progress": 0.3,
            "message": "Generating image with DiffSynth..."
        })
        
        # Generate image with timeout and better error handling
        try:
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                asyncio.to_thread(diffsynth_service.edit_image, edit_request),
                timeout=300.0  # 5 minute timeout
            )
            
            # Update progress after generation
            active_jobs[job_id].update({
                "progress": 0.8,
                "message": "Processing completed, saving image..."
            })
            
        except asyncio.TimeoutError:
            active_jobs[job_id].update({
                "status": "failed",
                "progress": 0.0,
                "error": "Generation timed out after 5 minutes"
            })
            return
        except Exception as e:
            active_jobs[job_id].update({
                "status": "failed",
                "progress": 0.0,
                "error": f"Generation error: {str(e)}"
            })
            return
        
        if response.success:
            # Image is already saved by the service, get the path
            if response.image_path:
                filename = os.path.basename(response.image_path)
            else:
                # Fallback filename
                timestamp = int(time.time())
                filename = f"diffsynth_gen_{timestamp}_{job_id[:8]}.jpg"
            
            # Update job with success
            active_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "message": "Image generated successfully",
                "result": {
                    "image_url": f"/images/{filename}",
                    "filename": filename,
                    "prompt": request.prompt,
                    "seed": request.seed,
                    "steps": request.steps,
                    "cfg_scale": request.cfg_scale,
                    "processing_time": response.processing_time
                }
            })
        else:
            # Update job with error
            active_jobs[job_id].update({
                "status": "failed",
                "progress": 0.0,
                "error": response.error_details or response.message or "Unknown error occurred"
            })
            
    except Exception as e:
        # Update job with error
        active_jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "error": str(e)
        })

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = active_jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data.get("progress", 0.0),
        message=job_data.get("message", ""),
        result=job_data.get("result"),
        error=job_data.get("error")
    )

@app.post("/api/edit/image")
async def edit_image(request: ImageEditRequest, background_tasks: BackgroundTasks):
    """Edit image using DiffSynth"""
    global diffsynth_service
    
    # Initialize DiffSynth service if needed
    if diffsynth_service is None:
        diffsynth_service = DiffSynthService()
        await asyncio.to_thread(diffsynth_service.initialize)
    
    # Create job ID for tracking
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Image editing job queued"
    }
    
    # Add background task
    background_tasks.add_task(
        process_image_edit,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "queued"}

async def process_image_edit(job_id: str, request: ImageEditRequest):
    """Process image editing in background"""
    global diffsynth_service
    
    try:
        # Update job status
        active_jobs[job_id].update({
            "status": "processing",
            "progress": 0.1,
            "message": f"Starting {request.operation.value} operation..."
        })
        
        # Process with DiffSynth
        response = await asyncio.to_thread(diffsynth_service.edit_image, request)
        
        if response.success:
            # Save result image
            timestamp = int(time.time())
            filename = f"diffsynth_edit_{timestamp}_{job_id[:8]}.jpg"
            filepath = Path("generated_images") / filename
            
            response.result_image.save(filepath)
            
            # Update job with success
            active_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "message": f"{request.operation.value} completed successfully",
                "result": {
                    "image_url": f"/images/{filename}",
                    "filename": filename,
                    "operation": request.operation.value,
                    "prompt": request.prompt,
                    "seed": response.seed_used
                }
            })
        else:
            # Update job with error
            active_jobs[job_id].update({
                "status": "failed",
                "progress": 0.0,
                "error": response.error_details or response.message or "Image editing failed"
            })
            
    except Exception as e:
        # Update job with error
        active_jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "error": str(e)
        })

@app.get("/api/services/status")
async def get_services_status():
    """Get status of all services"""
    global diffsynth_service, controlnet_service
    
    return {
        "diffsynth": {
            "available": diffsynth_service is not None,
            "status": str(diffsynth_service.status) if diffsynth_service else "offline",
            "initialized": diffsynth_service.is_initialized() if diffsynth_service else False
        },
        "controlnet": {
            "available": controlnet_service is not None,
            "status": "ready" if controlnet_service else "offline"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting DiffSynth Enhanced API Server")
    print("=" * 50)
    print("📋 Server Configuration:")
    print("   • API Server: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Services: DiffSynth-Studio + ControlNet")
    print("   • GPU Optimized: RTX 4080 16GB")
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )