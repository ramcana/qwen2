# Qwen-Image Configuration
# Optimized for RTX 4080 (16GB VRAM) + 128GB RAM

from typing import Any, Dict, Tuple

import torch

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen-Image",  # 20B parameter MMDiT model
    "torch_dtype": torch.bfloat16,  # Best for RTX 4080
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_safetensors": True,
    # Note: Qwen-Image doesn't use variants, load default model files
}

# Memory optimization settings
# NOTE: CPU offload disabled to prevent device mismatch issues
# If you encounter CUDA out-of-memory errors, try enabling sequential_cpu_offload
MEMORY_CONFIG: Dict[str, bool] = {
    "enable_attention_slicing": True,  # Reduce memory usage without device conflicts
    "enable_cpu_offload": False,  # DISABLED: Can cause device mismatch errors
    "enable_sequential_cpu_offload": False,  # Alternative if you need more VRAM (but may cause device issues)
}

# Default generation settings
GENERATION_CONFIG: Dict[str, Any] = {
    "width": 1664,  # Good for RTX 4080
    "height": 928,  # 16:9 aspect ratio
    "num_inference_steps": 50,  # Balanced quality/speed
    "true_cfg_scale": 4.0,  # Recommended CFG for Qwen-Image
    "max_batch_size": 1,  # Single image generation
}

# Quality presets
QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "num_inference_steps": 20,
        "true_cfg_scale": 3.0,
        "description": "Fast preview - lower quality",
    },
    "balanced": {
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "description": "Recommended - good balance",
    },
    "high": {
        "num_inference_steps": 80,
        "true_cfg_scale": 7.0,
        "description": "High quality - slower generation",
    },
}

# Aspect ratio presets (Updated to match official Qwen-Image documentation)
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (1328, 1328),  # Square
    "16:9": (1664, 928),  # Landscape
    "9:16": (928, 1664),  # Portrait
    "4:3": (1472, 1140),  # Photo
    "3:4": (1140, 1472),  # Portrait photo
    "3:2": (1584, 1056),  # Classic photo ratio
    "2:3": (1056, 1584),  # Portrait classic
    "21:9": (1792, 768),  # Ultra-wide
}

# Prompt enhancement templates (Updated with official "positive magic" strings)
PROMPT_ENHANCEMENT: Dict[str, Dict[str, str]] = {
    "en": {
        "quality_keywords": "Ultra HD, 4K, cinematic composition",  # Official positive magic
        "artistic_keywords": "masterpiece, trending on artstation, award winning, professional lighting",
        "technical_keywords": "photorealistic, high resolution, intricate details, perfect composition",
    },
    "zh": {
        "quality_keywords": "超清，4K，电影级构图",  # Official positive magic for Chinese
        "artistic_keywords": "杰作，艺术站热门，获奖作品，专业灯光",
        "technical_keywords": "逼真，高分辨率，精细细节，完美构图",
    },
}
