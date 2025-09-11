"""
Device Policy Helper for Qwen-Image Generator
Provides canonical device configuration and management functions.
"""

import gc
import logging
import os
import shutil
import time
import subprocess
from typing import Any, Dict, Optional, Tuple, Type

import torch
import yaml
from transformers import AutoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device_config() -> Dict[str, Any]:
    """
    Get canonical device configuration based on hardware capabilities.

    Returns:
        Dict containing device configuration parameters:
        - device_map: Device mapping strategy
        - torch_dtype: Optimal torch dtype (bfloat16 for 40-series, fp16 for 30-series)
        - attn_implementation: Attention implementation (sdpa by default)
    """
    config = {
        "device_map": "auto",
        "attn_implementation": get_attention_implementation(),
    }

    if torch.cuda.is_available():
        # Check GPU capability for bfloat16 support
        device_name = torch.cuda.get_device_name()
        if "RTX 40" in device_name or "A100" in device_name or "H100" in device_name:
            # 40-series and enterprise GPUs support bfloat16
            config["torch_dtype"] = torch.bfloat16
        else:
            # Older GPUs use fp16
            config["torch_dtype"] = torch.float16
    else:
        # CPU fallback
        config["torch_dtype"] = torch.float32

    return config


def get_hf_device_map() -> str:
    """
    Get HuggingFace device map strategy.

    Returns:
        str: Device map strategy ("auto", "balanced", etc.)
    """
    return "auto"


def get_optimal_torch_dtype() -> torch.dtype:
    """
    Get optimal torch dtype based on hardware.

    Returns:
        torch.dtype: Optimal dtype for current hardware
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        if "RTX 40" in device_name or "A100" in device_name or "H100" in device_name:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32


def get_attention_implementation() -> str:
    """
    Get optimal attention implementation with runtime probe.

    Returns:
        str: Attention implementation ("sdpa", "flash_attention_2", etc.)
    """
    # Default to SDPA for stability
    attn_impl = "sdpa"

    # Check if we can use Flash Attention
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        compute_capability = torch.cuda.get_device_capability()

        # Check if bfloat16 is supported (required for Flash Attention)
        has_bf16 = (
            "RTX 40" in device_name or "A100" in device_name or "H100" in device_name
        )
        has_high_compute = compute_capability[0] >= 8  # Compute capability 8.0+

        if has_bf16 and has_high_compute:
            try:
                # Try to import flash attention
                import flash_attn

                # Check if it's properly installed
                from flash_attn import __version__

                attn_impl = "flash_attention_2"
                logger.info(f"✅ Flash Attention 2 available (version {__version__})")
            except ImportError:
                # Flash attention not available
                logger.warning("⚠️ Flash Attention not available, using SDPA")
            except Exception as e:
                # Flash attention available but not working properly
                logger.warning(f"⚠️ Flash Attention available but not working: {e}")
                logger.info("💡 Falling back to SDPA")

    return attn_impl


def clear_gpu_memory() -> None:
    """
    Clear GPU memory and run garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def safe_model_switch_context() -> None:
    """
    Context for safe model switching.
    Clears memory and waits before loading new model.
    """
    # Clear existing model from memory
    clear_gpu_memory()

    # Short sleep to allow memory cleanup
    time.sleep(0.2)  # 200ms


def create_processor_eager(model_name: str, **kwargs) -> AutoProcessor:
    """
    Create processor immediately (eager loading).

    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments for processor creation

    Returns:
        AutoProcessor: Created processor
    """
    logger.info(f"🔧 Creating processor for {model_name} (eager loading)...")
    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    logger.info("✅ Processor created successfully")
    return processor


def load_model_with_retry(
    model_class: Type, model_name: str, max_retries: int = 2, **kwargs
) -> Any:
    """
    Load model with retry and backoff mechanism.

    Args:
        model_class: Class to instantiate (e.g., DiffusionPipeline)
        model_name (str): Name of the model
        max_retries (int): Maximum number of retry attempts
        **kwargs: Additional arguments for model loading

    Returns:
        Any: Loaded model instance

    Raises:
        Exception: If all retry attempts fail
    """
    # Apply canonical device policy
    device_config = get_device_config()

    # Merge device config with provided kwargs
    load_kwargs = {**device_config, **kwargs}

    # Log loading attempt
    logger.info(
        f"🔧 Loading model {model_name} with retry mechanism (max {max_retries} retries)..."
    )
    logger.info(f"   Device config: {device_config}")

    # Try to load model with retries
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"🔄 Retry attempt {attempt} of {max_retries}")
                # Clear memory before retry
                clear_gpu_memory()
                # Exponential backoff
                time.sleep(2**attempt)

            # Load model with optimal configuration
            model = model_class.from_pretrained(model_name, **load_kwargs)
            logger.info("✅ Model loaded successfully")
            return model

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's an OOM error
            is_oom = (
                "out of memory" in error_msg
                or "oom" in error_msg
                or "cuda out of memory" in error_msg
            )

            # Check if it's an HTTP error
            is_http = (
                "http" in error_msg
                or "connection" in error_msg
                or "timeout" in error_msg
            )

            if is_oom:
                logger.warning(f"⚠️ Out of memory error on attempt {attempt}: {e}")
                if attempt < max_retries:
                    # Try to lower max_pixels and clear cache before retry
                    logger.info(
                        "💡 Lowering max_pixels and clearing cache for retry..."
                    )
                    clear_gpu_memory()
                    # Reduce memory usage for next attempt
                    if "max_memory" in load_kwargs:
                        # Reduce max memory by 20%
                        for key in load_kwargs["max_memory"]:
                            if (
                                isinstance(load_kwargs["max_memory"][key], str)
                                and "GiB" in load_kwargs["max_memory"][key]
                            ):
                                # Parse and reduce GiB value
                                current = float(
                                    load_kwargs["max_memory"][key].replace("GiB", "")
                                )
                                load_kwargs["max_memory"][
                                    key
                                ] = f"{current * 0.8:.1f}GiB"
                    elif (
                        "torch_dtype" in load_kwargs
                        and load_kwargs["torch_dtype"] == torch.bfloat16
                    ):
                        # Fallback to float16
                        logger.info(
                            "💡 Falling back to float16 for reduced memory usage"
                        )
                        load_kwargs["torch_dtype"] = torch.float16
                continue

            elif is_http:
                logger.warning(f"⚠️ Network error on attempt {attempt}: {e}")
                if attempt < max_retries:
                    logger.info("💡 Retrying after network error...")
                continue

            else:
                # Other error - log and re-raise if final attempt
                logger.error(f"❌ Unexpected error on attempt {attempt}: {e}")
                if attempt >= max_retries:
                    # Log to reports/last_session.log
                    try:
                        log_path = os.path.join(
                            os.path.dirname(__file__),
                            "..",
                            "..",
                            "reports",
                            "last_session.log",
                        )
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "a") as f:
                            f.write(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loading failed: {e}\n"
                            )
                        logger.info(f"📝 Error logged to {log_path}")
                    except Exception as log_error:
                        logger.warning(f"⚠️ Could not log error to file: {log_error}")
                    raise e

    # This should never be reached due to the raise in the loop
    raise Exception("Model loading failed after all retry attempts")


def load_model_lazy(model_class: Type, model_name: str, **kwargs) -> Any:
    """
    Load model with lazy loading approach (wrapper for retry mechanism).

    Args:
        model_class: Class to instantiate (e.g., DiffusionPipeline)
        model_name (str): Name of the model
        **kwargs: Additional arguments for model loading

    Returns:
        Any: Loaded model instance
    """
    return load_model_with_retry(model_class, model_name, **kwargs)


def check_vram_availability() -> Dict[str, float]:
    """
    Check VRAM availability.

    Returns:
        Dict with VRAM information:
        - available_gb: Available VRAM in GB
        - total_gb: Total VRAM in GB
        - used_gb: Used VRAM in GB
        - usage_percent: VRAM usage percentage
    """
    if not torch.cuda.is_available():
        return {
            "available_gb": 0.0,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "usage_percent": 0.0,
        }

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - reserved_memory

    return {
        "available_gb": available_memory / 1e9,
        "total_gb": total_memory / 1e9,
        "used_gb": allocated_memory / 1e9,
        "usage_percent": (allocated_memory / total_memory) * 100,
    }


def check_disk_space(path: str = ".") -> Dict[str, float]:
    """
    Check available disk space.

    Args:
        path (str): Path to check disk space for (default: current directory)

    Returns:
        Dict with disk space information:
        - free_gb: Free disk space in GB
        - total_gb: Total disk space in GB
        - used_gb: Used disk space in GB
        - usage_percent: Disk usage percentage
    """
    try:
        total, used, free = shutil.disk_usage(path)
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        usage_percent = (used / total) * 100 if total > 0 else 0

        return {
            "free_gb": free_gb,
            "total_gb": total_gb,
            "used_gb": used_gb,
            "usage_percent": usage_percent,
        }
    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")
        return {"free_gb": 0.0, "total_gb": 0.0, "used_gb": 0.0, "usage_percent": 0.0}


def check_cuda_version() -> str:
    """
    Check CUDA version.

    Returns:
        str: CUDA version string or "Not available"
    """
    if torch.cuda.is_available():
        return torch.version.cuda or "Unknown"
    return "Not available"


def check_driver_version() -> str:
    """
    Check GPU driver version.

    Returns:
        str: Driver version string or "Not available"
    """
    if not torch.cuda.is_available():
        return "Not available"
    try:
        # Prefer querying through PyTorch when available
        return str(torch.cuda.driver_version())
    except Exception:
        try:
            import subprocess

            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                encoding="utf-8",
            )
            return output.strip().split("\n")[0]
        except Exception:
            return "Unknown"


def perform_preflight_checks(min_disk_space_gb: float = 25.0) -> Dict[str, Any]:
    """
    Perform comprehensive pre-flight checks.

    Args:
        min_disk_space_gb (float): Minimum required disk space in GB (default: 25GB)

    Returns:
        Dict with check results and recommendations
    """
    logger.info("🔍 Performing pre-flight checks...")

    results = {
        "vram": check_vram_availability(),
        "disk": check_disk_space(),
        "cuda_version": check_cuda_version(),
        "driver_version": check_driver_version(),
        "recommendations": [],
    }

    # Check VRAM
    vram_gb = results["vram"]["total_gb"]
    vram_percent = results["vram"]["usage_percent"]

    logger.info(
        f"📊 VRAM: {results['vram']['used_gb']:.1f}GB/{vram_gb:.1f}GB ({vram_percent:.1f}%)"
    )

    if vram_gb < 8.0:
        results["recommendations"].append(
            "Low VRAM detected. Consider reducing image size."
        )
        logger.warning("⚠️ Low VRAM system detected")
    elif vram_gb < 12.0:
        results["recommendations"].append(
            "Moderate VRAM. Image size may be automatically adjusted."
        )
        logger.info("⚠️ Moderate VRAM system")
    else:
        logger.info("✅ Sufficient VRAM available")

    # Check disk space
    free_disk_gb = results["disk"]["free_gb"]
    disk_percent = results["disk"]["usage_percent"]

    logger.info(f"💾 Disk space: {free_disk_gb:.1f}GB free ({disk_percent:.1f}% used)")

    if free_disk_gb < min_disk_space_gb:
        results["recommendations"].append(
            f"Low disk space. Ensure at least {min_disk_space_gb}GB free."
        )
        logger.warning(
            f"⚠️ Low disk space: {free_disk_gb:.1f}GB free (minimum {min_disk_space_gb}GB recommended)"
        )
    else:
        logger.info("✅ Sufficient disk space available")

    # Check CUDA version
    cuda_version = results["cuda_version"]
    logger.info(f"🔍 CUDA version: {cuda_version}")

    # Check driver version
    driver_version = results["driver_version"]
    logger.info(f"🔍 Driver version: {driver_version}")

    # Auto-adjust max_pixels based on available VRAM
    available_vram_gb = results["vram"]["available_gb"]
    if available_vram_gb < 8.0:
        results["recommendations"].append(
            "Safe Mode: Reduced pixel window will be used automatically."
        )
        logger.info("💡 Safe Mode: Reduced pixel window will be used automatically")
    elif available_vram_gb < 12.0:
        results["recommendations"].append(
            "Image size may be automatically adjusted for optimal performance."
        )
        logger.info("💡 Image size may be automatically adjusted")

    if results["recommendations"]:
        logger.info("📋 Recommendations:")
        for rec in results["recommendations"]:
            logger.info(f"   • {rec}")

    return results


def get_cache_dir() -> str:
    """
    Get model cache directory.

    Returns:
        str: Path to cache directory
    """
    # Check QWEN_HOME environment variable first
    qwen_home = os.environ.get("QWEN_HOME")
    if qwen_home:
        cache_dir = os.path.join(qwen_home, "cache", "huggingface", "hub")
    else:
        # Default to HuggingFace cache
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )

    # Ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_quality_presets() -> Dict[str, Any]:
    """
    Load quality presets from configuration file.

    Returns:
        Dict: Quality presets configuration
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "quality_presets.yaml"
    )
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"⚠️ Could not load quality presets: {e}")
        # Return default presets
        return {
            "vram_presets": {
                "medium": {"min_pixels": 518400, "max_pixels": 1327104},
                "high": {"min_pixels": 518400, "max_pixels": 2032128},
                "ultra": {"min_pixels": 518400, "max_pixels": 4194304},
            }
        }


def get_pixel_window_settings() -> Tuple[int, int]:
    """
    Get appropriate min/max pixel window settings based on available VRAM.

    Returns:
        Tuple[int, int]: (min_pixels, max_pixels)
    """
    # Load presets
    presets = load_quality_presets()

    # Check VRAM
    vram_info = check_vram_availability()
    available_vram_gb = vram_info["available_gb"]

    # Select appropriate preset based on VRAM
    if available_vram_gb >= 24:
        preset = presets["vram_presets"]["ultra"]
    elif available_vram_gb >= 16:
        preset = presets["vram_presets"]["high"]
    elif available_vram_gb >= 12:
        preset = presets["vram_presets"]["medium"]
    else:
        # Very low VRAM - use conservative settings
        preset = {"min_pixels": 259200, "max_pixels": 518400}  # 512x512 max

    return preset["min_pixels"], preset["max_pixels"]


def clamp_image_size(width: int, height: int) -> Tuple[int, int]:
    """
    Clamp image size to appropriate pixel window based on VRAM.

    Args:
        width (int): Requested width
        height (int): Requested height

    Returns:
        Tuple[int, int]: Clamped (width, height)
    """
    min_pixels, max_pixels = get_pixel_window_settings()
    requested_pixels = width * height

    # If requested size is within bounds, return as-is
    if min_pixels <= requested_pixels <= max_pixels:
        return width, height

    # If too large, scale down proportionally
    if requested_pixels > max_pixels:
        scale_factor = (max_pixels / requested_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return new_width, new_height

    # If too small, scale up proportionally (but not beyond max)
    if requested_pixels < min_pixels:
        scale_factor = (min_pixels / requested_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Check if scaled up version exceeds max
        scaled_pixels = new_width * new_height
        if scaled_pixels > max_pixels:
            scale_factor = (max_pixels / requested_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

        return new_width, new_height

    return width, height
