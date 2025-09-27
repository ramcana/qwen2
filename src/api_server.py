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
    title="Qwen-Image API",
    description="Professional AI Image Generation API",
    version="2.0.0"
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
generator: Optional[QwenImageGenerator] = None
diffsynth_service: Optional[DiffSynthService] = None
controlnet_service: Optional[ControlNetService] = None
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

# ControlNet API Models
class ControlNetRequest(BaseModel):
    """Request model for ControlNet-guided generation"""
    
    # Input data
    prompt: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    control_image_path: Optional[str] = None
    control_image_base64: Optional[str] = None
    
    # ControlNet configuration
    control_type: str = "auto"  # Will be converted to ControlNetType enum
    controlnet_conditioning_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    
    # Generation parameters
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 768
    height: int = 768
    seed: Optional[int] = None
    
    # Processing options
    use_tiled_processing: Optional[bool] = None
    additional_params: Optional[Dict[str, Any]] = None

class ControlDetectionRequest(BaseModel):
    """Request model for control type detection"""
    image_path: Optional[str] = None
    image_base64: Optional[str] = None

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

# DiffSynth API Endpoints

@app.post("/diffsynth/edit", response_model=ImageEditResponse)
async def diffsynth_edit_image(request: ImageEditRequest, background_tasks: BackgroundTasks):
    """General image editing using DiffSynth"""
    global diffsynth_service
    
    try:
        # Initialize DiffSynth service if needed
        if diffsynth_service is None:
            diffsynth_service = DiffSynthService()
        
        # Create job ID for tracking
        job_id = str(uuid.uuid4())
        
        # Add to queue
        generation_queue[job_id] = {
            "status": "queued",
            "request": request.dict(),
            "created_at": datetime.now().isoformat(),
            "type": "diffsynth-edit"
        }
        
        # Start processing in background
        background_tasks.add_task(process_diffsynth_edit, job_id, request)
        
        return ImageEditResponse(
            success=True,
            message="DiffSynth edit started",
            operation=EditOperation.EDIT,
            parameters={"job_id": job_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DiffSynth edit failed: {str(e)}")

@app.post("/diffsynth/inpaint", response_model=ImageEditResponse)
async def diffsynth_inpaint(request: InpaintRequest, background_tasks: BackgroundTasks):
    """Inpainting using DiffSynth"""
    global diffsynth_service
    
    try:
        # Initialize DiffSynth service if needed
        if diffsynth_service is None:
            diffsynth_service = DiffSynthService()
        
        # Create job ID for tracking
        job_id = str(uuid.uuid4())
        
        # Add to queue
        generation_queue[job_id] = {
            "status": "queued",
            "request": request.dict(),
            "created_at": datetime.now().isoformat(),
            "type": "diffsynth-inpaint"
        }
        
        # Start processing in background
        background_tasks.add_task(process_diffsynth_inpaint, job_id, request)
        
        return ImageEditResponse(
            success=True,
            message="DiffSynth inpaint started",
            operation=EditOperation.INPAINT,
            parameters={"job_id": job_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DiffSynth inpaint failed: {str(e)}")

@app.post("/diffsynth/outpaint", response_model=ImageEditResponse)
async def diffsynth_outpaint(request: OutpaintRequest, background_tasks: BackgroundTasks):
    """Image extension using DiffSynth"""
    global diffsynth_service
    
    try:
        # Initialize DiffSynth service if needed
        if diffsynth_service is None:
            diffsynth_service = DiffSynthService()
        
        # Create job ID for tracking
        job_id = str(uuid.uuid4())
        
        # Add to queue
        generation_queue[job_id] = {
            "status": "queued",
            "request": request.dict(),
            "created_at": datetime.now().isoformat(),
            "type": "diffsynth-outpaint"
        }
        
        # Start processing in background
        background_tasks.add_task(process_diffsynth_outpaint, job_id, request)
        
        return ImageEditResponse(
            success=True,
            message="DiffSynth outpaint started",
            operation=EditOperation.OUTPAINT,
            parameters={"job_id": job_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DiffSynth outpaint failed: {str(e)}")

@app.post("/diffsynth/style-transfer", response_model=ImageEditResponse)
async def diffsynth_style_transfer(request: StyleTransferRequest, background_tasks: BackgroundTasks):
    """Style transfer using DiffSynth"""
    global diffsynth_service
    
    try:
        # Initialize DiffSynth service if needed
        if diffsynth_service is None:
            diffsynth_service = DiffSynthService()
        
        # Create job ID for tracking
        job_id = str(uuid.uuid4())
        
        # Add to queue
        generation_queue[job_id] = {
            "status": "queued",
            "request": request.dict(),
            "created_at": datetime.now().isoformat(),
            "type": "diffsynth-style-transfer"
        }
        
        # Start processing in background
        background_tasks.add_task(process_diffsynth_style_transfer, job_id, request)
        
        return ImageEditResponse(
            success=True,
            message="DiffSynth style transfer started",
            operation=EditOperation.STYLE_TRANSFER,
            parameters={"job_id": job_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DiffSynth style transfer failed: {str(e)}")

# Background processing functions for DiffSynth operations

async def process_diffsynth_edit(job_id: str, request: ImageEditRequest):
    """Process DiffSynth edit operation"""
    global diffsynth_service, current_generation, generation_queue
    
    current_generation = job_id
    generation_queue[job_id]["status"] = "processing"
    generation_queue[job_id]["started_at"] = datetime.now().isoformat()
    
    try:
        # Process the edit request
        response = diffsynth_service.edit_image(request)
        
        # Update queue with results
        generation_queue[job_id].update({
            "status": "completed" if response.success else "failed",
            "response": response.dict(),
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

async def process_diffsynth_inpaint(job_id: str, request: InpaintRequest):
    """Process DiffSynth inpaint operation"""
    global diffsynth_service, current_generation, generation_queue
    
    current_generation = job_id
    generation_queue[job_id]["status"] = "processing"
    generation_queue[job_id]["started_at"] = datetime.now().isoformat()
    
    try:
        # Process the inpaint request
        response = diffsynth_service.inpaint(request)
        
        # Update queue with results
        generation_queue[job_id].update({
            "status": "completed" if response.success else "failed",
            "response": response.dict(),
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

async def process_diffsynth_outpaint(job_id: str, request: OutpaintRequest):
    """Process DiffSynth outpaint operation"""
    global diffsynth_service, current_generation, generation_queue
    
    current_generation = job_id
    generation_queue[job_id]["status"] = "processing"
    generation_queue[job_id]["started_at"] = datetime.now().isoformat()
    
    try:
        # Process the outpaint request
        response = diffsynth_service.outpaint(request)
        
        # Update queue with results
        generation_queue[job_id].update({
            "status": "completed" if response.success else "failed",
            "response": response.dict(),
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

async def process_diffsynth_style_transfer(job_id: str, request: StyleTransferRequest):
    """Process DiffSynth style transfer operation"""
    global diffsynth_service, current_generation, generation_queue
    
    current_generation = job_id
    generation_queue[job_id]["status"] = "processing"
    generation_queue[job_id]["started_at"] = datetime.now().isoformat()
    
    try:
        # Process the style transfer request
        response = diffsynth_service.style_transfer(request)
        
        # Update queue with results
        generation_queue[job_id].update({
            "status": "completed" if response.success else "failed",
            "response": response.dict(),
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

# ControlNet API Endpoints

@app.post("/controlnet/detect")
async def controlnet_detect_control_type(request: ControlDetectionRequest):
    """Detect appropriate ControlNet type for input image"""
    global controlnet_service
    
    try:
        # Validate input
        if not request.image_path and not request.image_base64:
            raise HTTPException(status_code=400, detail="Either image_path or image_base64 must be provided")
        
        # Initialize ControlNet service if needed
        if controlnet_service is None:
            controlnet_service = ControlNetService()
        
        # Determine input image
        input_image = request.image_path if request.image_path else request.image_base64
        
        # Detect control type
        detection_result = controlnet_service.detect_control_type(input_image)
        
        return {
            "success": True,
            "detected_type": detection_result.detected_type.value,
            "confidence": detection_result.confidence,
            "all_scores": {k.value: v for k, v in detection_result.all_scores.items()},
            "processing_time": detection_result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Control type detection failed: {str(e)}")

@app.post("/controlnet/generate")
async def controlnet_generate(request: ControlNetRequest, background_tasks: BackgroundTasks):
    """Generate image using ControlNet guidance"""
    global controlnet_service
    
    try:
        # Initialize ControlNet service if needed
        if controlnet_service is None:
            controlnet_service = ControlNetService()
        
        # Convert string control_type to enum
        try:
            control_type_enum = ControlNetType(request.control_type.lower())
        except ValueError:
            control_type_enum = ControlNetType.AUTO
        
        # Create job ID for tracking
        job_id = str(uuid.uuid4())
        
        # Add to queue
        generation_queue[job_id] = {
            "status": "queued",
            "request": request.dict(),
            "created_at": datetime.now().isoformat(),
            "type": "controlnet-generate"
        }
        
        # Start processing in background
        background_tasks.add_task(process_controlnet_generate, job_id, request, control_type_enum)
        
        return {
            "success": True,
            "message": "ControlNet generation started",
            "job_id": job_id,
            "control_type": control_type_enum.value
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ControlNet generation failed: {str(e)}")

@app.get("/controlnet/types")
async def get_controlnet_types():
    """Get available ControlNet types"""
    try:
        control_types = [
            {
                "type": control_type.value,
                "name": control_type.value.replace("_", " ").title(),
                "description": _get_control_type_description(control_type)
            }
            for control_type in ControlNetType
            if control_type != ControlNetType.AUTO
        ]
        
        return {
            "success": True,
            "control_types": control_types,
            "auto_detection_available": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ControlNet types: {str(e)}")

def _get_control_type_description(control_type: ControlNetType) -> str:
    """Get description for ControlNet type"""
    descriptions = {
        ControlNetType.CANNY: "Edge detection for structural control",
        ControlNetType.DEPTH: "Depth estimation for 3D-aware generation",
        ControlNetType.POSE: "Human pose detection for character control",
        ControlNetType.NORMAL: "Normal map generation for surface details",
        ControlNetType.SEGMENTATION: "Semantic segmentation for region control",
        ControlNetType.SCRIBBLE: "Scribble-based control for rough sketches",
        ControlNetType.LINEART: "Line art detection for clean outlines"
    }
    return descriptions.get(control_type, "Advanced control method")

# Background processing function for ControlNet

async def process_controlnet_generate(job_id: str, request: ControlNetRequest, control_type_enum: ControlNetType):
    """Process ControlNet generation operation"""
    global controlnet_service, current_generation, generation_queue
    
    current_generation = job_id
    generation_queue[job_id]["status"] = "processing"
    generation_queue[job_id]["started_at"] = datetime.now().isoformat()
    
    try:
        # Create ControlNet request object for the service
        # Import the service's ControlNetRequest class
        from controlnet_service import ControlNetRequest as ServiceControlNetRequest
        
        service_request = ServiceControlNetRequest(
            prompt=request.prompt,
            image_path=request.image_path,
            image_base64=request.image_base64,
            control_image_path=request.control_image_path,
            control_image_base64=request.control_image_base64,
            control_type=control_type_enum,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            control_guidance_start=request.control_guidance_start,
            control_guidance_end=request.control_guidance_end,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            use_tiled_processing=request.use_tiled_processing,
            additional_params=request.additional_params
        )
        
        # Process the ControlNet request
        response = controlnet_service.process_with_control(service_request)
        
        # Update queue with results
        generation_queue[job_id].update({
            "status": "completed" if response.get("success", False) else "failed",
            "response": response,
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

# Service Management Endpoints

@app.get("/services/status")
async def get_services_status():
    """Get status of all services"""
    global generator, diffsynth_service, controlnet_service
    
    try:
        # Check Qwen generator status
        qwen_status = {
            "name": "qwen-generator",
            "status": "ready" if (generator and generator.pipe) else "not_loaded",
            "model_loaded": generator is not None and generator.pipe is not None,
            "device": generator.device if generator else "unknown",
            "last_operation": None,
            "error_count": 0
        }
        
        # Check DiffSynth service status
        diffsynth_status = {
            "name": "diffsynth-service",
            "status": "not_initialized",
            "model_loaded": False,
            "device": "unknown",
            "last_operation": None,
            "error_count": 0
        }
        
        if diffsynth_service:
            diffsynth_status.update({
                "status": diffsynth_service.status.value,
                "model_loaded": diffsynth_service.pipeline is not None,
                "device": diffsynth_service.config.device,
                "last_operation": diffsynth_service.last_operation_time,
                "error_count": diffsynth_service.error_count,
                "operation_count": diffsynth_service.operation_count,
                "initialization_time": diffsynth_service.initialization_time
            })
        
        # Check ControlNet service status
        controlnet_status = {
            "name": "controlnet-service",
            "status": "ready" if controlnet_service else "not_initialized",
            "model_loaded": controlnet_service is not None,
            "device": controlnet_service.device if controlnet_service else "unknown",
            "available_types": [t.value for t in ControlNetType if t != ControlNetType.AUTO] if controlnet_service else []
        }
        
        # Get system resource information
        memory_info = {}
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                cached_memory = torch.cuda.memory_reserved()
                free_memory = max(0, total_memory - allocated_memory)
                
                memory_info = {
                    "total_memory": total_memory,
                    "allocated_memory": allocated_memory,
                    "cached_memory": cached_memory,
                    "free_memory": free_memory,
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_usage_percent": round((allocated_memory / total_memory) * 100, 1)
                }
            except Exception as e:
                memory_info = {"error": f"Could not get memory info: {str(e)}"}
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "qwen": qwen_status,
                "diffsynth": diffsynth_status,
                "controlnet": controlnet_status
            },
            "system": {
                "memory_info": memory_info,
                "current_generation": current_generation,
                "queue_length": len(generation_queue),
                "cuda_available": torch.cuda.is_available()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get services status: {str(e)}")

@app.post("/services/switch")
async def switch_service(service_name: str, action: str):
    """Switch service state (initialize, shutdown, restart)"""
    global generator, diffsynth_service, controlnet_service
    
    try:
        if service_name not in ["qwen", "diffsynth", "controlnet"]:
            raise HTTPException(status_code=400, detail="Invalid service name. Must be 'qwen', 'diffsynth', or 'controlnet'")
        
        if action not in ["initialize", "shutdown", "restart"]:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'initialize', 'shutdown', or 'restart'")
        
        result = {"service": service_name, "action": action, "success": False, "message": ""}
        
        if service_name == "qwen":
            if action == "initialize":
                if generator is None:
                    generator = QwenImageGenerator()
                success = generator.load_model()
                result["success"] = success
                result["message"] = "Qwen model loaded successfully" if success else "Failed to load Qwen model"
            
            elif action == "shutdown":
                if generator and generator.pipe:
                    # Clear GPU memory
                    generator.pipe = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                result["success"] = True
                result["message"] = "Qwen service shutdown"
            
            elif action == "restart":
                # Shutdown then initialize
                if generator and generator.pipe:
                    generator.pipe = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if generator is None:
                    generator = QwenImageGenerator()
                success = generator.load_model()
                result["success"] = success
                result["message"] = "Qwen service restarted successfully" if success else "Failed to restart Qwen service"
        
        elif service_name == "diffsynth":
            if action == "initialize":
                if diffsynth_service is None:
                    diffsynth_service = DiffSynthService()
                success = diffsynth_service.initialize()
                result["success"] = success
                result["message"] = "DiffSynth service initialized successfully" if success else "Failed to initialize DiffSynth service"
            
            elif action == "shutdown":
                if diffsynth_service:
                    diffsynth_service.pipeline = None
                    diffsynth_service.status = diffsynth_service.status.__class__.OFFLINE
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                result["success"] = True
                result["message"] = "DiffSynth service shutdown"
            
            elif action == "restart":
                # Shutdown then initialize
                if diffsynth_service:
                    diffsynth_service.pipeline = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if diffsynth_service is None:
                    diffsynth_service = DiffSynthService()
                success = diffsynth_service.initialize()
                result["success"] = success
                result["message"] = "DiffSynth service restarted successfully" if success else "Failed to restart DiffSynth service"
        
        elif service_name == "controlnet":
            if action == "initialize":
                if controlnet_service is None:
                    controlnet_service = ControlNetService()
                result["success"] = True
                result["message"] = "ControlNet service initialized successfully"
            
            elif action == "shutdown":
                if controlnet_service:
                    # Clear any loaded models
                    controlnet_service._controlnet_models = {}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                controlnet_service = None
                result["success"] = True
                result["message"] = "ControlNet service shutdown"
            
            elif action == "restart":
                # Shutdown then initialize
                if controlnet_service:
                    controlnet_service._controlnet_models = {}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                controlnet_service = ControlNetService()
                result["success"] = True
                result["message"] = "ControlNet service restarted successfully"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service switch failed: {str(e)}")

@app.get("/services/health")
async def check_services_health():
    """Perform health checks on all services"""
    global generator, diffsynth_service, controlnet_service
    
    try:
        health_results = {}
        overall_healthy = True
        
        # Check Qwen service health
        qwen_healthy = True
        qwen_issues = []
        
        try:
            if generator is None or generator.pipe is None:
                qwen_healthy = False
                qwen_issues.append("Model not loaded")
            else:
                # Try a simple operation to verify functionality
                # This is a basic health check - in production you might want a more thorough test
                pass
        except Exception as e:
            qwen_healthy = False
            qwen_issues.append(f"Health check failed: {str(e)}")
        
        health_results["qwen"] = {
            "healthy": qwen_healthy,
            "issues": qwen_issues,
            "status": "healthy" if qwen_healthy else "unhealthy"
        }
        
        # Check DiffSynth service health
        diffsynth_healthy = True
        diffsynth_issues = []
        
        try:
            if diffsynth_service is None:
                diffsynth_healthy = False
                diffsynth_issues.append("Service not initialized")
            elif diffsynth_service.status.value in ["error", "offline"]:
                diffsynth_healthy = False
                diffsynth_issues.append(f"Service in {diffsynth_service.status.value} state")
        except Exception as e:
            diffsynth_healthy = False
            diffsynth_issues.append(f"Health check failed: {str(e)}")
        
        health_results["diffsynth"] = {
            "healthy": diffsynth_healthy,
            "issues": diffsynth_issues,
            "status": "healthy" if diffsynth_healthy else "unhealthy"
        }
        
        # Check ControlNet service health
        controlnet_healthy = True
        controlnet_issues = []
        
        try:
            if controlnet_service is None:
                controlnet_healthy = False
                controlnet_issues.append("Service not initialized")
        except Exception as e:
            controlnet_healthy = False
            controlnet_issues.append(f"Health check failed: {str(e)}")
        
        health_results["controlnet"] = {
            "healthy": controlnet_healthy,
            "issues": controlnet_issues,
            "status": "healthy" if controlnet_healthy else "unhealthy"
        }
        
        # Overall health
        overall_healthy = all(service["healthy"] for service in health_results.values())
        
        return {
            "success": True,
            "overall_healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "services": health_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

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