# Qwen-Image Configuration - Modern Architecture Support
# Optimized for MMDiT transformers and multimodal capabilities

from typing import Any, Dict, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import torch
import logging

logger = logging.getLogger(__name__)

# Architecture types for model detection
class ModelArchitecture(Enum):
    MMDIT = "mmdit"  # Modern MMDiT transformer architecture
    UNET = "unet"    # Legacy UNet architecture
    MULTIMODAL = "multimodal"  # Qwen2-VL multimodal architecture

# Optimization levels
class OptimizationLevel(Enum):
    ULTRA_FAST = "ultra_fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    MULTIMODAL = "multimodal"

@dataclass
class OptimizationConfig:
    """Configuration for architecture-specific optimizations"""
    
    # Model selection
    model_name: str = "Qwen/Qwen-Image"
    architecture: ModelArchitecture = ModelArchitecture.MMDIT
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Core settings
    torch_dtype: torch.dtype = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_safetensors: bool = True
    trust_remote_code: bool = True
    
    # Download settings
    resume_download: bool = True
    force_download: bool = False
    max_retries: int = 3
    timeout: int = 300
    
    # MMDiT-specific optimizations
    enable_scaled_dot_product_attention: bool = True
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    use_flash_attention_2: bool = False  # Compatibility dependent
    
    # Memory optimizations (disabled for performance)
    enable_attention_slicing: bool = False
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_model_cpu_offload: bool = False
    low_cpu_mem_usage: bool = False
    
    # Advanced optimizations
    use_torch_compile: bool = False  # Experimental for MMDiT
    enable_xformers: bool = False
    dynamic_batch_sizing: bool = True
    
    # Generation settings
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    true_cfg_scale: float = 4.0
    max_batch_size: int = 1
    output_type: str = "pil"
    
    # Qwen2-VL multimodal settings
    enable_qwen2vl_integration: bool = False
    qwen2vl_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    enable_prompt_enhancement: bool = False
    enable_image_analysis: bool = False
    
    # Performance targets
    target_generation_time: float = 5.0  # seconds per step
    target_total_time: float = 120.0     # seconds total
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Validate dimensions
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Width and height must be positive")
            
            # Validate steps and CFG scale
            if self.num_inference_steps <= 0:
                raise ValueError("Number of inference steps must be positive")
            
            if self.true_cfg_scale < 0:
                raise ValueError("CFG scale must be non-negative")
            
            # Architecture-specific validation
            if self.architecture == ModelArchitecture.MMDIT:
                # MMDiT works best with certain settings
                if self.enable_attention_slicing:
                    logger.warning("Attention slicing may reduce MMDiT performance")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def migrate_from_legacy(self, legacy_config: Dict[str, Any]) -> None:
        """Migrate from legacy configuration format"""
        try:
            # Map legacy model config
            if "model_name" in legacy_config:
                self.model_name = legacy_config["model_name"]
            
            if "torch_dtype" in legacy_config:
                self.torch_dtype = legacy_config["torch_dtype"]
            
            # Map legacy memory config
            memory_config = legacy_config.get("memory_config", {})
            self.enable_attention_slicing = memory_config.get("enable_attention_slicing", False)
            self.enable_cpu_offload = memory_config.get("enable_cpu_offload", False)
            self.enable_vae_slicing = memory_config.get("enable_vae_slicing", False)
            
            # Map legacy generation config
            gen_config = legacy_config.get("generation_config", {})
            self.width = gen_config.get("width", 1024)
            self.height = gen_config.get("height", 1024)
            self.num_inference_steps = gen_config.get("num_inference_steps", 25)
            self.true_cfg_scale = gen_config.get("true_cfg_scale", 4.0)
            
            logger.info("Successfully migrated legacy configuration")
            
        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
    
    def apply_optimization_level(self) -> None:
        """Apply settings based on optimization level"""
        if self.optimization_level == OptimizationLevel.ULTRA_FAST:
            self.num_inference_steps = 10
            self.true_cfg_scale = 2.5
            self.width = 768
            self.height = 768
            self.enable_all_optimizations()
            
        elif self.optimization_level == OptimizationLevel.BALANCED:
            self.num_inference_steps = 25
            self.true_cfg_scale = 4.0
            self.width = 1024
            self.height = 1024
            self.enable_performance_optimizations()
            
        elif self.optimization_level == OptimizationLevel.QUALITY:
            self.num_inference_steps = 40
            self.true_cfg_scale = 5.0
            self.width = 1280
            self.height = 1280
            self.enable_quality_optimizations()
            
        elif self.optimization_level == OptimizationLevel.MULTIMODAL:
            self.num_inference_steps = 30
            self.true_cfg_scale = 4.5
            self.enable_qwen2vl_integration = True
            self.enable_prompt_enhancement = True
            self.enable_performance_optimizations()
    
    def enable_all_optimizations(self) -> None:
        """Enable all performance optimizations for maximum speed"""
        self.enable_scaled_dot_product_attention = True
        self.enable_tf32 = True
        self.enable_cudnn_benchmark = True
        self.dynamic_batch_sizing = True
        
        # Disable memory-saving features
        self.enable_attention_slicing = False
        self.enable_cpu_offload = False
        self.enable_vae_slicing = False
        self.enable_vae_tiling = False
    
    def enable_performance_optimizations(self) -> None:
        """Enable balanced performance optimizations"""
        self.enable_scaled_dot_product_attention = True
        self.enable_tf32 = True
        self.enable_cudnn_benchmark = True
        
        # Keep some memory optimizations disabled
        self.enable_attention_slicing = False
        self.enable_cpu_offload = False
    
    def enable_quality_optimizations(self) -> None:
        """Enable optimizations focused on quality"""
        self.enable_scaled_dot_product_attention = True
        self.enable_tf32 = False  # Higher precision for quality
        
        # Allow some memory optimizations for higher resolution
        self.enable_vae_tiling = True
    
    def get_pipeline_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for pipeline initialization"""
        return {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "use_safetensors": self.use_safetensors,
            "trust_remote_code": self.trust_remote_code,
        }
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for image generation"""
        return {
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.true_cfg_scale,
            "output_type": self.output_type,
        }

# Configuration factory functions
def create_optimization_config(
    level: Union[OptimizationLevel, str] = OptimizationLevel.BALANCED,
    architecture: Union[ModelArchitecture, str] = ModelArchitecture.MMDIT,
    **kwargs
) -> OptimizationConfig:
    """Create an optimization configuration with specified level and architecture"""
    
    if isinstance(level, str):
        level = OptimizationLevel(level)
    if isinstance(architecture, str):
        architecture = ModelArchitecture(architecture)
    
    config = OptimizationConfig(
        optimization_level=level,
        architecture=architecture,
        **kwargs
    )
    
    # Store custom kwargs to preserve them after applying optimization level
    custom_kwargs = {k: v for k, v in kwargs.items() if hasattr(config, k)}
    
    config.apply_optimization_level()
    
    # Restore custom kwargs that were overridden
    for key, value in custom_kwargs.items():
        setattr(config, key, value)
    
    return config

def get_model_config_for_architecture(architecture: ModelArchitecture) -> Dict[str, Any]:
    """Get model-specific configuration based on architecture"""
    
    base_config = {
        "torch_dtype": torch.float16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_safetensors": True,
        "trust_remote_code": True,
        "resume_download": True,
        "force_download": False,
        "max_retries": 3,
        "timeout": 300,
    }
    
    if architecture == ModelArchitecture.MMDIT:
        base_config.update({
            "model_name": "Qwen/Qwen-Image",
            "enable_scaled_dot_product_attention": True,
            "enable_tf32": True,
            "enable_cudnn_benchmark": True,
        })
    elif architecture == ModelArchitecture.UNET:
        base_config.update({
            "model_name": "Qwen/Qwen-Image-Edit",
            "enable_xformers": True,
            "enable_attention_slicing": True,
        })
    elif architecture == ModelArchitecture.MULTIMODAL:
        base_config.update({
            "model_name": "Qwen/Qwen2-VL-7B-Instruct",
            "enable_qwen2vl_integration": True,
            "enable_prompt_enhancement": True,
        })
    
    return base_config

# Legacy configuration support (maintained for backward compatibility)
MODEL_CONFIG: Dict[str, Any] = get_model_config_for_architecture(ModelArchitecture.MMDIT)

MEMORY_CONFIG: Dict[str, bool] = {
    "enable_attention_slicing": False,     # DISABLED: Major performance killer for MMDiT
    "enable_cpu_offload": False,           # DISABLED: Keep everything on GPU
    "enable_sequential_cpu_offload": False, # DISABLED: Massive performance killer
    "enable_flash_attention": False,       # DISABLED: Compatibility issues with Qwen-Image
    "enable_xformers": False,              # DISABLED: Can cause slowdowns with MMDiT
    "use_torch_compile": False,            # DISABLED: Tensor unpacking issues with MMDiT
    "enable_vae_slicing": False,           # DISABLED: Quality and speed killer
    "enable_vae_tiling": False,            # DISABLED: Speed killer
    "enable_model_cpu_offload": False,     # DISABLED: Keep model on GPU
    "enable_sequential_cpu_offload": False, # DISABLED: Major bottleneck
    "low_cpu_mem_usage": False,            # DISABLED: Use available RAM for speed
}

GENERATION_CONFIG: Dict[str, Any] = {
    "width": 1024,                    # Optimized for MMDiT architecture
    "height": 1024,                   # Square is faster than rectangular
    "num_inference_steps": 25,        # Balanced for MMDiT performance
    "true_cfg_scale": 4.0,           # Optimal for MMDiT quality/speed
    "max_batch_size": 1,
    "output_type": "pil",
}

# Modern architecture-aware presets
QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "ultra_fast": {
        "num_inference_steps": 10,
        "true_cfg_scale": 2.5,
        "width": 768,
        "height": 768,
        "architecture": ModelArchitecture.MMDIT,
        "description": "Ultra fast MMDiT - 10 steps, optimized for speed"
    },
    "fast": {
        "num_inference_steps": 15,
        "true_cfg_scale": 3.0,
        "width": 1024,
        "height": 1024,
        "architecture": ModelArchitecture.MMDIT,
        "description": "Fast MMDiT - 15 steps, good for testing"
    },
    "balanced": {
        "num_inference_steps": 25,
        "true_cfg_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "architecture": ModelArchitecture.MMDIT,
        "description": "Balanced MMDiT - 25 steps, optimal quality/speed ratio"
    },
    "quality": {
        "num_inference_steps": 40,
        "true_cfg_scale": 5.0,
        "width": 1280,
        "height": 1280,
        "architecture": ModelArchitecture.MMDIT,
        "description": "Quality MMDiT - 40 steps, high quality generation"
    },
    "multimodal": {
        "num_inference_steps": 30,
        "true_cfg_scale": 4.5,
        "width": 1024,
        "height": 1024,
        "architecture": ModelArchitecture.MULTIMODAL,
        "enable_qwen2vl_integration": True,
        "enable_prompt_enhancement": True,
        "description": "Multimodal - Enhanced with Qwen2-VL integration"
    }
}

# MMDiT-optimized aspect ratios (square is fastest for transformers)
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),     # FASTEST - Square, optimal for MMDiT
    "1:1_hd": (1280, 1280),  # High-res square for quality mode
    "1:1_uhd": (1536, 1536), # Ultra high-res for maximum quality
    "16:9": (1344, 768),     # Landscape (MMDiT optimized)
    "9:16": (768, 1344),     # Portrait (MMDiT optimized)
    "4:3": (1152, 896),      # Photo ratio
    "3:4": (896, 1152),      # Portrait photo
    "3:2": (1152, 768),      # Classic photo ratio
    "2:3": (768, 1152),      # Portrait classic ratio
    "21:9": (1344, 576),     # Widescreen
    "2:1": (1024, 512),      # Panoramic
    "1:2": (512, 1024),      # Tall portrait
}

# Enhanced prompt enhancement with multimodal support
PROMPT_ENHANCEMENT: Dict[str, Dict[str, str]] = {
    "en": {
        "quality_keywords": "high quality, detailed, sharp",
        "artistic_keywords": "masterpiece, professional, artistic",
        "technical_keywords": "sharp focus, clear, well-composed",
        "mmdit_keywords": "transformer-generated, modern architecture",
        "multimodal_keywords": "contextually aware, semantically rich"
    },
    "zh": {
        "quality_keywords": "高质量，精细，清晰",
        "artistic_keywords": "杰作，专业，艺术",
        "technical_keywords": "清晰对焦，清楚，构图良好",
        "mmdit_keywords": "变换器生成，现代架构",
        "multimodal_keywords": "上下文感知，语义丰富"
    }
}

# Qwen2-VL specific configuration
QWEN2VL_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "trust_remote_code": True,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "enable_prompt_enhancement": True,
    "enable_image_analysis": True,
    "context_length": 2048,
}

# Configuration validation functions
def validate_architecture_compatibility(
    architecture: ModelArchitecture, 
    config: OptimizationConfig
) -> bool:
    """Validate that configuration is compatible with architecture"""
    
    if architecture == ModelArchitecture.MMDIT:
        # MMDiT works best without attention slicing
        if config.enable_attention_slicing:
            logger.warning("Attention slicing may reduce MMDiT performance")
            return False
        
        # MMDiT benefits from scaled dot-product attention
        if not config.enable_scaled_dot_product_attention:
            logger.info("Enabling scaled dot-product attention for MMDiT")
            config.enable_scaled_dot_product_attention = True
    
    elif architecture == ModelArchitecture.UNET:
        # UNet may benefit from xformers
        if not config.enable_xformers:
            logger.info("Consider enabling xformers for UNet architecture")
    
    elif architecture == ModelArchitecture.MULTIMODAL:
        # Multimodal requires Qwen2-VL integration
        if not config.enable_qwen2vl_integration:
            logger.info("Enabling Qwen2-VL integration for multimodal architecture")
            config.enable_qwen2vl_integration = True
    
    return True

def migrate_legacy_config(legacy_config: Dict[str, Any]) -> OptimizationConfig:
    """Migrate legacy configuration to modern OptimizationConfig"""
    
    config = OptimizationConfig()
    
    # Migrate model config
    model_config = legacy_config.get("MODEL_CONFIG", {})
    config.model_name = model_config.get("model_name", config.model_name)
    config.torch_dtype = model_config.get("torch_dtype", config.torch_dtype)
    config.device = model_config.get("device", config.device)
    
    # Migrate memory config
    memory_config = legacy_config.get("MEMORY_CONFIG", {})
    migrated_memory_settings = {
        "enable_attention_slicing": memory_config.get("enable_attention_slicing", False),
        "enable_cpu_offload": memory_config.get("enable_cpu_offload", False),
        "enable_vae_slicing": memory_config.get("enable_vae_slicing", False),
        "enable_xformers": memory_config.get("enable_xformers", False),
        "use_torch_compile": memory_config.get("use_torch_compile", False),
    }
    
    # Migrate generation config
    generation_config = legacy_config.get("GENERATION_CONFIG", {})
    migrated_generation_settings = {
        "width": generation_config.get("width", config.width),
        "height": generation_config.get("height", config.height),
        "num_inference_steps": generation_config.get("num_inference_steps", config.num_inference_steps),
        "true_cfg_scale": generation_config.get("true_cfg_scale", config.true_cfg_scale),
    }
    
    # Apply optimization level first (this may override some settings)
    config.apply_optimization_level()
    
    # Then apply migrated settings to preserve legacy configuration
    for key, value in migrated_memory_settings.items():
        setattr(config, key, value)
    
    for key, value in migrated_generation_settings.items():
        setattr(config, key, value)
    
    # Validate final configuration
    config.validate()
    
    logger.info("Successfully migrated legacy configuration to modern format")
    return config

# Default configurations for different use cases
DEFAULT_CONFIGS = {
    "ultra_fast": lambda: create_optimization_config(OptimizationLevel.ULTRA_FAST),
    "balanced": lambda: create_optimization_config(OptimizationLevel.BALANCED),
    "quality": lambda: create_optimization_config(OptimizationLevel.QUALITY),
    "multimodal": lambda: create_optimization_config(OptimizationLevel.MULTIMODAL),
}