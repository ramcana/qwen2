# DiffSynth Enhanced UI API Documentation

## Overview

This document provides comprehensive API documentation for the DiffSynth Enhanced UI system. The API extends the existing Qwen image generation endpoints with advanced image editing and ControlNet capabilities.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for local development. For production deployments, implement appropriate authentication mechanisms.

## Common Response Format

All API endpoints return responses in the following format:

```json
{
  "success": boolean,
  "message": string,
  "data": object,
  "error": string (optional),
  "timestamp": string (ISO 8601)
}
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Endpoint or resource not found
- `500`: Internal Server Error - Server-side error
- `503`: Service Unavailable - Service not initialized

### Error Response Format

```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error information",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Existing Endpoints (Qwen)

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "model_loaded": true
}
```

### System Status

```http
GET /status
```

**Response:**

```json
{
  "model_loaded": true,
  "device": "cuda:0",
  "memory_info": {
    "total_memory": 24000000000,
    "allocated_memory": 8000000000,
    "cached_memory": 2000000000,
    "free_memory": 16000000000,
    "device_name": "NVIDIA RTX 4090"
  },
  "current_generation": null,
  "queue_length": 0,
  "initialization": {
    "status": "ready",
    "message": "All services ready",
    "progress": 100,
    "model_loaded": true
  }
}
```

### Text-to-Image Generation

```http
POST /generate/text-to-image
```

**Request Body:**

```json
{
  "prompt": "A beautiful landscape with mountains and lakes",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 768,
  "num_inference_steps": 50,
  "cfg_scale": 7.5,
  "seed": -1,
  "language": "en",
  "enhance_prompt": true,
  "aspect_ratio": "4:3"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Generation started",
  "job_id": "uuid-string",
  "image_path": null,
  "generation_time": null,
  "parameters": {...}
}
```

## DiffSynth Endpoints

### General Image Editing

```http
POST /diffsynth/edit
```

Edit images using DiffSynth's general editing capabilities.

**Request Body:**

```json
{
  "image_path": "path/to/input/image.jpg",
  "image_base64": "base64-encoded-image-data",
  "prompt": "Transform this image into a painting",
  "negative_prompt": "blurry, distorted",
  "strength": 0.7,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": -1,
  "width": 1024,
  "height": 768,
  "use_tiled_processing": false,
  "additional_params": {
    "scheduler": "ddim",
    "eta": 0.0
  }
}
```

**Parameters:**

- `image_path` (string, optional): Path to input image file
- `image_base64` (string, optional): Base64-encoded image data
- `prompt` (string, required): Text description of desired edit
- `negative_prompt` (string, optional): What to avoid in the edit
- `strength` (float, 0.1-1.0): Edit strength (higher = more change)
- `num_inference_steps` (integer, 10-100): Quality vs speed tradeoff
- `guidance_scale` (float, 1.0-20.0): How closely to follow the prompt
- `seed` (integer, optional): Random seed for reproducibility
- `width` (integer, optional): Output width (will resize if different from input)
- `height` (integer, optional): Output height (will resize if different from input)
- `use_tiled_processing` (boolean, optional): Force tiled processing for large images

**Response:**

```json
{
  "success": true,
  "message": "DiffSynth edit started",
  "job_id": "uuid-string",
  "operation": "edit",
  "estimated_time": 30,
  "parameters": {
    "strength": 0.7,
    "steps": 50,
    "guidance_scale": 7.5
  }
}
```

### Inpainting

```http
POST /diffsynth/inpaint
```

Fill or modify specific areas of an image using a mask.

**Request Body:**

```json
{
  "image_path": "path/to/input/image.jpg",
  "image_base64": "base64-encoded-image-data",
  "mask_path": "path/to/mask/image.jpg",
  "mask_base64": "base64-encoded-mask-data",
  "prompt": "A beautiful flower garden",
  "negative_prompt": "wilted, dead plants",
  "strength": 0.8,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": -1,
  "mask_blur": 4,
  "inpaint_full_res": false,
  "inpaint_full_res_padding": 32
}
```

**Additional Parameters:**

- `mask_path` (string, optional): Path to mask image (white = inpaint, black = preserve)
- `mask_base64` (string, optional): Base64-encoded mask data
- `mask_blur` (integer, 0-64): Blur mask edges for smoother blending
- `inpaint_full_res` (boolean): Process at full resolution vs downscaled
- `inpaint_full_res_padding` (integer): Padding around inpaint area

**Response:**

```json
{
  "success": true,
  "message": "DiffSynth inpaint started",
  "job_id": "uuid-string",
  "operation": "inpaint",
  "mask_area_percentage": 15.5,
  "estimated_time": 25
}
```

### Outpainting

```http
POST /diffsynth/outpaint
```

Extend images beyond their original boundaries.

**Request Body:**

```json
{
  "image_path": "path/to/input/image.jpg",
  "image_base64": "base64-encoded-image-data",
  "prompt": "Continue the landscape with mountains",
  "negative_prompt": "cut off, incomplete",
  "direction": "right",
  "pixels": 256,
  "strength": 0.7,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": -1,
  "blend_width": 64,
  "multiple_directions": ["right", "up"]
}
```

**Additional Parameters:**

- `direction` (string): "up", "down", "left", "right", or "all"
- `pixels` (integer, 64-512): Number of pixels to extend
- `blend_width` (integer, 16-128): Width of blending area
- `multiple_directions` (array, optional): Extend in multiple directions

**Response:**

```json
{
  "success": true,
  "message": "DiffSynth outpaint started",
  "job_id": "uuid-string",
  "operation": "outpaint",
  "original_size": [1024, 768],
  "new_size": [1280, 768],
  "extension_area": 256
}
```

### Style Transfer

```http
POST /diffsynth/style-transfer
```

Apply the artistic style of one image to another.

**Request Body:**

```json
{
  "content_image_path": "path/to/content/image.jpg",
  "content_image_base64": "base64-encoded-content-data",
  "style_image_path": "path/to/style/image.jpg",
  "style_image_base64": "base64-encoded-style-data",
  "prompt": "Apply impressionist painting style",
  "negative_prompt": "photorealistic, sharp details",
  "style_strength": 0.8,
  "content_strength": 0.6,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": -1,
  "preserve_content_structure": true
}
```

**Additional Parameters:**

- `content_image_path` (string): Path to content image (what to stylize)
- `style_image_path` (string): Path to style reference image
- `style_strength` (float, 0.1-1.0): How strongly to apply style
- `content_strength` (float, 0.1-1.0): How much to preserve content
- `preserve_content_structure` (boolean): Maintain original composition

**Response:**

```json
{
  "success": true,
  "message": "DiffSynth style transfer started",
  "job_id": "uuid-string",
  "operation": "style_transfer",
  "style_analysis": {
    "dominant_colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
    "style_type": "impressionist",
    "complexity": "medium"
  }
}
```

## ControlNet Endpoints

### Control Type Detection

```http
POST /controlnet/detect
```

Automatically detect the best ControlNet type for an input image.

**Request Body:**

```json
{
  "image_path": "path/to/control/image.jpg",
  "image_base64": "base64-encoded-image-data"
}
```

**Response:**

```json
{
  "success": true,
  "detected_type": "canny",
  "confidence": 0.85,
  "all_scores": {
    "canny": 0.85,
    "depth": 0.72,
    "pose": 0.23,
    "normal": 0.45,
    "segmentation": 0.67
  },
  "processing_time": 1.2,
  "recommendations": [
    {
      "type": "canny",
      "reason": "Strong edge features detected",
      "confidence": 0.85
    },
    {
      "type": "depth",
      "reason": "Good depth variation present",
      "confidence": 0.72
    }
  ]
}
```

### ControlNet Generation

```http
POST /controlnet/generate
```

Generate images using ControlNet guidance.

**Request Body:**

```json
{
  "prompt": "A futuristic cityscape at sunset",
  "image_path": "path/to/input/image.jpg",
  "image_base64": "base64-encoded-image-data",
  "control_image_path": "path/to/control/image.jpg",
  "control_image_base64": "base64-encoded-control-data",
  "control_type": "canny",
  "controlnet_conditioning_scale": 1.0,
  "control_guidance_start": 0.0,
  "control_guidance_end": 1.0,
  "negative_prompt": "blurry, low quality",
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "width": 768,
  "height": 768,
  "seed": -1,
  "use_tiled_processing": false,
  "additional_params": {
    "scheduler": "ddim",
    "eta": 0.0
  }
}
```

**Parameters:**

- `control_type` (string): "auto", "canny", "depth", "pose", "normal", "segmentation"
- `controlnet_conditioning_scale` (float, 0.1-2.0): Strength of control guidance
- `control_guidance_start` (float, 0.0-1.0): When to start applying control
- `control_guidance_end` (float, 0.0-1.0): When to stop applying control

**Response:**

```json
{
  "success": true,
  "message": "ControlNet generation started",
  "job_id": "uuid-string",
  "control_type": "canny",
  "control_preview_path": "path/to/control/preview.jpg",
  "estimated_time": 35,
  "control_analysis": {
    "feature_density": "high",
    "complexity": "medium",
    "recommended_scale": 1.0
  }
}
```

### Available ControlNet Types

```http
GET /controlnet/types
```

Get list of available ControlNet types and their descriptions.

**Response:**

```json
{
  "success": true,
  "control_types": [
    {
      "type": "canny",
      "name": "Canny Edge Detection",
      "description": "Uses edge detection for structural guidance",
      "best_for": ["line art", "architectural drawings", "sketches"],
      "parameters": {
        "low_threshold": 100,
        "high_threshold": 200
      }
    },
    {
      "type": "depth",
      "name": "Depth Control",
      "description": "Uses depth information for 3D-aware generation",
      "best_for": ["landscapes", "interiors", "3D scenes"],
      "parameters": {
        "near_plane": 0.1,
        "far_plane": 1000.0
      }
    },
    {
      "type": "pose",
      "name": "Human Pose Detection",
      "description": "Uses human pose keypoints for character control",
      "best_for": ["human figures", "character art", "fashion"],
      "parameters": {
        "confidence_threshold": 0.5
      }
    }
  ]
}
```

## Service Management Endpoints

### Service Status

```http
GET /services/status
```

Get status of all services (Qwen, DiffSynth, ControlNet).

**Response:**

```json
{
  "success": true,
  "services": {
    "qwen": {
      "status": "running",
      "model_loaded": true,
      "memory_usage": "8.2GB",
      "last_activity": "2024-01-01T12:00:00Z"
    },
    "diffsynth": {
      "status": "running",
      "model_loaded": true,
      "memory_usage": "4.1GB",
      "last_activity": "2024-01-01T11:45:00Z"
    },
    "controlnet": {
      "status": "running",
      "available_types": ["canny", "depth", "pose", "normal", "segmentation"],
      "memory_usage": "2.8GB",
      "last_activity": "2024-01-01T11:30:00Z"
    }
  },
  "total_memory_usage": "15.1GB",
  "available_memory": "8.9GB"
}
```

### Service Switching

```http
POST /services/switch
```

Switch between services to optimize memory usage.

**Request Body:**

```json
{
  "primary_service": "diffsynth",
  "secondary_services": ["controlnet"],
  "unload_inactive": true,
  "memory_optimization": true
}
```

**Response:**

```json
{
  "success": true,
  "message": "Service switching completed",
  "active_services": ["diffsynth", "controlnet"],
  "inactive_services": ["qwen"],
  "memory_freed": "8.2GB",
  "switch_time": 2.5
}
```

## Job Management Endpoints

### Queue Status

```http
GET /queue
```

Get current processing queue status.

**Response:**

```json
{
  "success": true,
  "queue": {
    "job-uuid-1": {
      "status": "processing",
      "type": "diffsynth-inpaint",
      "created_at": "2024-01-01T12:00:00Z",
      "started_at": "2024-01-01T12:01:00Z",
      "estimated_completion": "2024-01-01T12:03:00Z",
      "progress": 65
    },
    "job-uuid-2": {
      "status": "queued",
      "type": "controlnet-generate",
      "created_at": "2024-01-01T12:02:00Z",
      "position": 1
    }
  },
  "current_generation": "job-uuid-1",
  "queue_length": 2,
  "estimated_wait_time": 120
}
```

### Job Status

```http
GET /queue/{job_id}
```

Get status of a specific job.

**Response:**

```json
{
  "success": true,
  "job": {
    "id": "job-uuid-1",
    "status": "completed",
    "type": "diffsynth-inpaint",
    "created_at": "2024-01-01T12:00:00Z",
    "started_at": "2024-01-01T12:01:00Z",
    "completed_at": "2024-01-01T12:03:00Z",
    "processing_time": 120,
    "result": {
      "image_path": "generated_images/inpaint_result_uuid.png",
      "thumbnail_path": "generated_images/thumbnails/inpaint_result_uuid_thumb.png",
      "metadata": {
        "original_size": [1024, 768],
        "final_size": [1024, 768],
        "parameters_used": {...}
      }
    }
  }
}
```

### Cancel Job

```http
DELETE /queue/{job_id}
```

Cancel a queued or processing job.

**Response:**

```json
{
  "success": true,
  "message": "Job cancelled successfully",
  "job_id": "job-uuid-1",
  "was_processing": false
}
```

## Memory Management Endpoints

### Memory Status

```http
GET /memory/status
```

Get detailed GPU memory information.

**Response:**

```json
{
  "success": true,
  "memory_info": {
    "device_name": "NVIDIA RTX 4090",
    "total_memory": 24000000000,
    "total_memory_gb": 24.0,
    "allocated_memory": 8000000000,
    "allocated_memory_gb": 8.0,
    "cached_memory": 2000000000,
    "cached_memory_gb": 2.0,
    "free_memory": 16000000000,
    "free_memory_gb": 16.0,
    "memory_usage_percent": 33.3,
    "fragmentation_percent": 8.3
  }
}
```

### Clear Memory

```http
GET /memory/clear
```

Clear GPU memory cache.

**Response:**

```json
{
  "success": true,
  "message": "GPU memory cleared. Freed 2.1GB allocated, 1.8GB cached",
  "memory_info": {
    "freed_allocated": 2100000000,
    "freed_cached": 1800000000,
    "total_freed": 3900000000,
    "remaining_allocated": 5900000000,
    "remaining_cached": 200000000
  }
}
```

## File Management Endpoints

### Upload Image

```http
POST /upload
```

Upload an image file for processing.

**Request:** Multipart form data with image file

**Response:**

```json
{
  "success": true,
  "message": "Image uploaded successfully",
  "file_path": "uploads/uploaded_image_uuid.jpg",
  "file_size": 2048576,
  "image_info": {
    "width": 1024,
    "height": 768,
    "format": "JPEG",
    "mode": "RGB"
  }
}
```

### Get Generated Image

```http
GET /images/{image_path}
```

Retrieve a generated image file.

**Response:** Image file (PNG/JPEG/WebP)

### List Generated Images

```http
GET /images
```

Get list of generated images with metadata.

**Response:**

```json
{
  "success": true,
  "images": [
    {
      "path": "generated_images/result_uuid.png",
      "created_at": "2024-01-01T12:00:00Z",
      "type": "diffsynth-inpaint",
      "size": 1048576,
      "dimensions": [1024, 768],
      "thumbnail": "generated_images/thumbnails/result_uuid_thumb.png",
      "metadata": {
        "prompt": "A beautiful garden",
        "parameters": {...}
      }
    }
  ],
  "total_count": 1,
  "total_size": 1048576
}
```

## WebSocket API (Real-time Updates)

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
```

### Message Types

**Job Progress Updates:**

```json
{
  "type": "job_progress",
  "job_id": "uuid-string",
  "progress": 65,
  "stage": "processing",
  "estimated_remaining": 30
}
```

**Job Completion:**

```json
{
  "type": "job_complete",
  "job_id": "uuid-string",
  "result": {
    "image_path": "path/to/result.png",
    "processing_time": 120
  }
}
```

**Service Status Updates:**

```json
{
  "type": "service_status",
  "service": "diffsynth",
  "status": "ready",
  "memory_usage": "4.1GB"
}
```

## Rate Limiting

- **Generation endpoints**: 10 requests per minute per IP
- **Status endpoints**: 60 requests per minute per IP
- **Upload endpoints**: 5 requests per minute per IP

## SDK Examples

### Python SDK

```python
import requests
import base64
from pathlib import Path

class DiffSynthAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def inpaint(self, image_path, mask_path, prompt, **kwargs):
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        with open(mask_path, 'rb') as f:
            mask_b64 = base64.b64encode(f.read()).decode()

        data = {
            "image_base64": image_b64,
            "mask_base64": mask_b64,
            "prompt": prompt,
            **kwargs
        }

        response = requests.post(f"{self.base_url}/diffsynth/inpaint", json=data)
        return response.json()

    def controlnet_generate(self, control_image_path, prompt, control_type="auto", **kwargs):
        with open(control_image_path, 'rb') as f:
            control_b64 = base64.b64encode(f.read()).decode()

        data = {
            "control_image_base64": control_b64,
            "prompt": prompt,
            "control_type": control_type,
            **kwargs
        }

        response = requests.post(f"{self.base_url}/controlnet/generate", json=data)
        return response.json()

# Usage
api = DiffSynthAPI()
result = api.inpaint("input.jpg", "mask.jpg", "A beautiful flower garden")
print(f"Job ID: {result['job_id']}")
```

### JavaScript SDK

```javascript
class DiffSynthAPI {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async uploadAndProcess(file, operation, params) {
    // Convert file to base64
    const base64 = await this.fileToBase64(file);

    const data = {
      image_base64: base64,
      ...params,
    };

    const response = await fetch(`${this.baseUrl}/diffsynth/${operation}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    return response.json();
  }

  async fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = (error) => reject(error);
    });
  }

  async getJobStatus(jobId) {
    const response = await fetch(`${this.baseUrl}/queue/${jobId}`);
    return response.json();
  }
}

// Usage
const api = new DiffSynthAPI();
const fileInput = document.getElementById("imageFile");
const result = await api.uploadAndProcess(fileInput.files[0], "edit", {
  prompt: "Transform into a painting",
  strength: 0.7,
});
console.log("Job ID:", result.job_id);
```

This comprehensive API documentation covers all the DiffSynth Enhanced UI endpoints with detailed examples and usage patterns. Use this as a reference for integrating with the system programmatically.
