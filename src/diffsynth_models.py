"""
DiffSynth Data Models
Pydantic models for DiffSynth image editing operations
"""

from typing import Optional, Dict, Any, Union, List, Tuple
from pydantic import BaseModel, Field, model_validator
from PIL import Image
import base64
import io
from enum import Enum


class EditOperation(str, Enum):
    """Supported edit operations"""
    GENERATE = "generate"
    EDIT = "edit"
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    STYLE_TRANSFER = "style_transfer"


class OutpaintDirection(str, Enum):
    """Outpainting directions"""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ALL = "all"


class ImageEditRequest(BaseModel):
    """Base request model for image editing operations"""
    
    prompt: str = Field(..., description="Text prompt for editing")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid")
    
    # Operation type
    operation: EditOperation = Field(EditOperation.EDIT, description="Type of edit operation")
    
    # Image input (can be path or base64)
    image_path: Optional[str] = Field(None, description="Path to input image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded input image")
    
    # Generation parameters
    num_inference_steps: Optional[int] = Field(20, ge=1, le=100, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    strength: Optional[float] = Field(0.8, ge=0.1, le=1.0, description="Edit strength")
    
    # Output parameters
    height: Optional[int] = Field(None, ge=256, le=2048, description="Output height")
    width: Optional[int] = Field(None, ge=256, le=2048, description="Output width")
    
    # Processing options
    use_tiled_processing: Optional[bool] = Field(None, description="Force tiled processing")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # EliGen options
    enable_eligen: Optional[bool] = Field(True, description="Enable EliGen enhancement")
    eligen_mode: Optional[str] = Field("balanced", description="EliGen quality mode: fast, balanced, quality, ultra")
    eligen_entity_detection: Optional[bool] = Field(True, description="Enable entity detection")
    eligen_quality_enhancement: Optional[bool] = Field(True, description="Enable quality enhancement")
    eligen_detail_enhancement: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Detail enhancement strength")
    eligen_color_enhancement: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Color enhancement strength")
    
    # Additional parameters
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional pipeline parameters")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    use_tiled_processing: Optional[bool] = Field(None, description="Use tiled processing for large images")
    
    # Additional parameters
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional processing parameters")
    
    @model_validator(mode='after')
    def validate_image_input(self):
        """Ensure image input is provided when needed"""
        # For GENERATE operation, no input image is needed
        if hasattr(self, 'operation') and self.operation == EditOperation.GENERATE:
            return self
        
        # For other operations, require input image
        if not self.image_path and not self.image_base64:
            raise ValueError("Either image_path or image_base64 must be provided for non-generation operations")
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Make this image more vibrant and colorful",
                "negative_prompt": "blurry, low quality",
                "image_path": "/path/to/image.jpg",
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "strength": 0.8,
                "seed": 42
            }
        }


class InpaintRequest(ImageEditRequest):
    """Request model for inpainting operations"""
    
    # Mask input (can be path or base64)
    mask_path: Optional[str] = Field(None, description="Path to mask image")
    mask_base64: Optional[str] = Field(None, description="Base64 encoded mask image")
    
    # Inpainting specific parameters
    invert_mask: Optional[bool] = Field(False, description="Invert the mask")
    mask_blur: Optional[int] = Field(4, ge=0, le=20, description="Mask blur radius")
    
    @model_validator(mode='after')
    def validate_mask_input(self):
        """Ensure at least one mask input is provided"""
        if not self.mask_path and not self.mask_base64:
            raise ValueError("Either mask_path or mask_base64 must be provided for inpainting")
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A beautiful flower in the garden",
                "image_path": "/path/to/image.jpg",
                "mask_path": "/path/to/mask.png",
                "num_inference_steps": 25,
                "guidance_scale": 8.0,
                "strength": 0.9,
                "mask_blur": 4
            }
        }


class OutpaintRequest(ImageEditRequest):
    """Request model for outpainting operations"""
    
    # Outpainting parameters
    direction: OutpaintDirection = Field(OutpaintDirection.ALL, description="Outpainting direction")
    pixels: int = Field(256, ge=64, le=1024, description="Number of pixels to extend")
    
    # Canvas expansion settings
    auto_expand_canvas: Optional[bool] = Field(True, description="Automatically expand canvas")
    fill_mode: Optional[str] = Field("edge", description="Fill mode for new areas (edge, constant, reflect)")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Extend this landscape with more mountains",
                "image_path": "/path/to/image.jpg",
                "direction": "all",
                "pixels": 256,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }


class StyleTransferRequest(ImageEditRequest):
    """Request model for style transfer operations"""
    
    # Style image input
    style_image_path: Optional[str] = Field(None, description="Path to style reference image")
    style_image_base64: Optional[str] = Field(None, description="Base64 encoded style image")
    
    # Style transfer parameters
    style_strength: Optional[float] = Field(0.7, ge=0.1, le=1.0, description="Style transfer strength")
    content_strength: Optional[float] = Field(0.3, ge=0.1, le=1.0, description="Content preservation strength")
    
    @model_validator(mode='after')
    def validate_style_input(self):
        """Ensure at least one style image input is provided"""
        if not self.style_image_path and not self.style_image_base64:
            raise ValueError("Either style_image_path or style_image_base64 must be provided for style transfer")
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Apply Van Gogh style to this photo",
                "image_path": "/path/to/content.jpg",
                "style_image_path": "/path/to/style.jpg",
                "style_strength": 0.7,
                "content_strength": 0.3,
                "num_inference_steps": 30
            }
        }


class ImageEditResponse(BaseModel):
    """Response model for image editing operations"""
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    
    # Output image
    image_path: Optional[str] = Field(None, description="Path to generated image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded generated image")
    
    # Operation metadata
    operation: Optional[EditOperation] = Field(None, description="Type of edit operation performed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    # Generation parameters used
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters used for generation")
    
    # Resource usage
    resource_usage: Optional[Dict[str, Any]] = Field(None, description="Resource usage information")
    
    # Error information (if applicable)
    error_details: Optional[str] = Field(None, description="Detailed error information")
    suggested_fixes: Optional[List[str]] = Field(None, description="Suggested fixes for errors")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Image edited successfully",
                "image_path": "/path/to/edited_image.jpg",
                "operation": "edit",
                "processing_time": 3.45,
                "parameters": {
                    "prompt": "Make this image more vibrant",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                }
            }
        }


class ProcessingMetrics(BaseModel):
    """Processing metrics for image operations"""
    
    operation_type: str = Field(..., description="Type of operation")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    # Resource usage
    gpu_memory_used: Optional[float] = Field(None, description="GPU memory used in GB")
    cpu_memory_used: Optional[float] = Field(None, description="CPU memory used in GB")
    
    # Image information
    input_resolution: Optional[Tuple[int, int]] = Field(None, description="Input image resolution (width, height)")
    output_resolution: Optional[Tuple[int, int]] = Field(None, description="Output image resolution (width, height)")
    
    # Processing details
    tiled_processing_used: Optional[bool] = Field(None, description="Whether tiled processing was used")
    num_tiles: Optional[int] = Field(None, description="Number of tiles processed")
    
    class Config:
        schema_extra = {
            "example": {
                "operation_type": "edit",
                "processing_time": 3.45,
                "gpu_memory_used": 2.1,
                "input_resolution": [768, 768],
                "output_resolution": [768, 768],
                "tiled_processing_used": False
            }
        }


# Utility functions for image handling
def encode_image_to_base64(image: Union[str, Image.Image]) -> str:
    """Convert image to base64 string"""
    if isinstance(image, str):
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def validate_image_format(image: Image.Image) -> bool:
    """Validate image format and properties"""
    try:
        # Make a copy to avoid modifying the original
        image_copy = image.copy()
        
        # Check dimensions
        width, height = image.size
        if width < 64 or height < 64:
            return False
        if width > 4096 or height > 4096:
            return False
            
        # Check mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False
            
        return True
    except Exception:
        return False