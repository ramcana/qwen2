"""
Environment Setup Manager for Quantized Qwen-Image Pipeline

This module provides comprehensive environment setup and GPU capability detection
for running quantized Qwen-Image models with optimal hardware configuration.
"""

import os
import subprocess
import logging
import torch
import warnings
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. GPU detection will be limited.")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    warnings.warn("bitsandbytes not available. Quantization will not work.")


class PrecisionType(Enum):
    """Supported precision types for model inference."""
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"


@dataclass
class GPUCapabilities:
    """GPU hardware capabilities and optimization settings."""
    compute_capability: Tuple[int, int]
    supports_bf16: bool
    supports_fp16: bool
    supports_tf32: bool
    memory_gb: float
    name: str
    driver_version: str
    cuda_version: str
    is_ada_architecture: bool


@dataclass
class EnvironmentConfig:
    """Environment configuration for quantized inference."""
    precision_type: PrecisionType
    cuda_alloc_conf: str
    tf32_enabled: bool
    cudnn_benchmark: bool
    torch_compile_enabled: bool
    memory_fraction: float


class EnvironmentSetupManager:
    """
    Manages environment setup and GPU capability detection for quantized inference.
    
    This class handles CUDA environment configuration, GPU capability detection,
    automatic precision selection, and fallback mechanisms for optimal performance.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the environment setup manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._gpu_capabilities: Optional[GPUCapabilities] = None
        self._environment_config: Optional[EnvironmentConfig] = None
        self._cuda_available = torch.cuda.is_available()
        
    def setup_cuda_environment(self) -> EnvironmentConfig:
        """
        Configure CUDA environment for optimal quantized inference.
        
        Returns:
            EnvironmentConfig: The applied environment configuration
            
        Raises:
            RuntimeError: If CUDA is not available or setup fails
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA is not available. Quantized inference requires CUDA.")
            
        self.logger.info("Setting up CUDA environment for quantized inference")
        
        # Configure PyTorch CUDA allocator for expandable segments
        cuda_alloc_conf = "expandable_segments:True,max_split_size_mb:512"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
        self.logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")
        
        # Detect GPU capabilities
        gpu_caps = self.detect_gpu_capabilities()
        
        # Choose optimal precision based on hardware
        precision_type = self.choose_optimal_precision(gpu_caps)
        
        # Configure TF32 for Ada architecture
        tf32_enabled = gpu_caps.is_ada_architecture and gpu_caps.supports_tf32
        if tf32_enabled:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("Enabled TF32 for Ada architecture")
        
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        self.logger.info("Enabled cuDNN benchmark")
        
        # Configure memory fraction (leave headroom for system)
        memory_fraction = 0.9  # Use 90% of available VRAM
        
        config = EnvironmentConfig(
            precision_type=precision_type,
            cuda_alloc_conf=cuda_alloc_conf,
            tf32_enabled=tf32_enabled,
            cudnn_benchmark=True,
            torch_compile_enabled=False,  # Disabled for quantized models initially
            memory_fraction=memory_fraction
        )
        
        self._environment_config = config
        self.logger.info(f"CUDA environment configured: {config}")
        return config
    
    def detect_gpu_capabilities(self) -> GPUCapabilities:
        """
        Detect GPU compute capabilities and hardware features.
        
        Returns:
            GPUCapabilities: Detected GPU capabilities
            
        Raises:
            RuntimeError: If GPU detection fails
        """
        if self._gpu_capabilities is not None:
            return self._gpu_capabilities
            
        if not self._cuda_available:
            raise RuntimeError("CUDA is not available for GPU detection")
            
        try:
            # Get basic PyTorch GPU info
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            compute_capability = torch.cuda.get_device_capability(device)
            
            # Get memory info
            memory_bytes = torch.cuda.get_device_properties(device).total_memory
            memory_gb = memory_bytes / (1024**3)
            
            # Detect precision support
            supports_bf16 = self._test_bf16_support()
            supports_fp16 = True  # All modern CUDA GPUs support FP16
            supports_tf32 = compute_capability >= (8, 0)  # Ampere and newer
            
            # Detect Ada architecture (RTX 40xx series)
            is_ada_architecture = compute_capability == (8, 9)
            
            # Get driver and CUDA version info
            driver_version, cuda_version = self._get_cuda_versions()
            
            capabilities = GPUCapabilities(
                compute_capability=compute_capability,
                supports_bf16=supports_bf16,
                supports_fp16=supports_fp16,
                supports_tf32=supports_tf32,
                memory_gb=memory_gb,
                name=gpu_name,
                driver_version=driver_version,
                cuda_version=cuda_version,
                is_ada_architecture=is_ada_architecture
            )
            
            self._gpu_capabilities = capabilities
            self.logger.info(f"Detected GPU capabilities: {capabilities}")
            return capabilities
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect GPU capabilities: {e}")
    
    def choose_optimal_precision(self, gpu_caps: Optional[GPUCapabilities] = None) -> PrecisionType:
        """
        Choose optimal precision type based on hardware capabilities.
        
        Args:
            gpu_caps: GPU capabilities (will detect if not provided)
            
        Returns:
            PrecisionType: Recommended precision type
        """
        if gpu_caps is None:
            gpu_caps = self.detect_gpu_capabilities()
            
        # Prefer bf16 for Ada architecture and newer
        if gpu_caps.supports_bf16 and gpu_caps.is_ada_architecture:
            self.logger.info("Selected bf16 precision for Ada architecture")
            return PrecisionType.BF16
        elif gpu_caps.supports_bf16 and gpu_caps.compute_capability >= (8, 0):
            self.logger.info("Selected bf16 precision for Ampere+ architecture")
            return PrecisionType.BF16
        elif gpu_caps.supports_fp16:
            self.logger.info("Selected fp16 precision (bf16 not optimal)")
            return PrecisionType.FP16
        else:
            self.logger.warning("Using fp32 precision (no mixed precision support)")
            return PrecisionType.FP32
    
    def test_precision_stability(self, precision_type: PrecisionType) -> bool:
        """
        Test if the selected precision type produces stable results.
        
        Args:
            precision_type: Precision type to test
            
        Returns:
            bool: True if precision is stable, False if NaN issues detected
        """
        self.logger.info(f"Testing {precision_type.value} precision stability")
        
        try:
            # Create test tensors
            device = torch.cuda.current_device()
            dtype = getattr(torch, precision_type.value)
            
            # Test basic operations that might produce NaN
            x = torch.randn(1000, 1000, dtype=dtype, device=device)
            y = torch.randn(1000, 1000, dtype=dtype, device=device)
            
            # Matrix multiplication (common source of NaN in mixed precision)
            result = torch.matmul(x, y)
            
            # Check for NaN or Inf
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            
            if has_nan or has_inf:
                self.logger.warning(f"{precision_type.value} precision produced NaN/Inf values")
                return False
                
            # Test gradient computation (important for fine-tuning)
            x.requires_grad_(True)
            loss = result.sum()
            loss.backward()
            
            grad_has_nan = torch.isnan(x.grad).any().item() if x.grad is not None else False
            
            if grad_has_nan:
                self.logger.warning(f"{precision_type.value} precision produced NaN gradients")
                return False
                
            self.logger.info(f"{precision_type.value} precision is stable")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing {precision_type.value} precision: {e}")
            return False
    
    def get_fallback_precision(self, failed_precision: PrecisionType) -> PrecisionType:
        """
        Get fallback precision when the preferred precision fails.
        
        Args:
            failed_precision: The precision type that failed
            
        Returns:
            PrecisionType: Fallback precision type
        """
        if failed_precision == PrecisionType.BF16:
            self.logger.info("Falling back from bf16 to fp16")
            return PrecisionType.FP16
        elif failed_precision == PrecisionType.FP16:
            self.logger.info("Falling back from fp16 to fp32")
            return PrecisionType.FP32
        else:
            self.logger.warning("Already using fp32, no further fallback available")
            return PrecisionType.FP32
    
    def validate_quantization_dependencies(self) -> Dict[str, bool]:
        """
        Validate that required dependencies for quantization are available.
        
        Returns:
            Dict[str, bool]: Validation results for each dependency
        """
        results = {}
        
        # Check bitsandbytes
        results['bitsandbytes'] = BITSANDBYTES_AVAILABLE
        if BITSANDBYTES_AVAILABLE:
            try:
                # Test basic bitsandbytes functionality
                bnb.nn.Linear4bit(10, 10)
                results['bitsandbytes_functional'] = True
            except Exception as e:
                self.logger.error(f"bitsandbytes not functional: {e}")
                results['bitsandbytes_functional'] = False
        else:
            results['bitsandbytes_functional'] = False
            
        # Check CUDA version compatibility
        cuda_version = torch.version.cuda
        results['cuda_compatible'] = cuda_version is not None and cuda_version >= "11.0"
        
        # Check PyTorch version
        torch_version = torch.__version__
        results['torch_compatible'] = torch_version >= "2.0.0"
        
        # Check GPU memory
        if self._cuda_available:
            gpu_caps = self.detect_gpu_capabilities()
            results['sufficient_vram'] = gpu_caps.memory_gb >= 8.0  # Minimum for quantized models
        else:
            results['sufficient_vram'] = False
            
        self.logger.info(f"Dependency validation results: {results}")
        return results
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """
        Get recommended settings based on detected hardware.
        
        Returns:
            Dict[str, Any]: Recommended configuration settings
        """
        if not self._cuda_available:
            return {"error": "CUDA not available"}
            
        gpu_caps = self.detect_gpu_capabilities()
        precision = self.choose_optimal_precision(gpu_caps)
        
        # Test precision stability and fallback if needed
        if not self.test_precision_stability(precision):
            precision = self.get_fallback_precision(precision)
            
        settings = {
            "precision": precision.value,
            "max_memory_gb": min(gpu_caps.memory_gb * 0.85, 14.0),  # Conservative limit
            "batch_size": 1 if gpu_caps.memory_gb < 16 else 2,
            "enable_tf32": gpu_caps.supports_tf32 and gpu_caps.is_ada_architecture,
            "enable_flash_attention": gpu_caps.compute_capability >= (8, 0),
            "quantization_recommended": gpu_caps.memory_gb < 24,
            "offload_recommended": gpu_caps.memory_gb < 20,
        }
        
        self.logger.info(f"Recommended settings: {settings}")
        return settings
    
    def _test_bf16_support(self) -> bool:
        """Test if bf16 is actually supported and functional."""
        try:
            if not hasattr(torch, 'bfloat16'):
                return False
                
            device = torch.cuda.current_device()
            x = torch.randn(10, 10, dtype=torch.bfloat16, device=device)
            y = torch.randn(10, 10, dtype=torch.bfloat16, device=device)
            result = torch.matmul(x, y)
            
            # Check if result is valid
            return not (torch.isnan(result).any() or torch.isinf(result).any())
        except Exception:
            return False
    
    def _get_cuda_versions(self) -> Tuple[str, str]:
        """Get CUDA driver and runtime versions."""
        driver_version = "unknown"
        cuda_version = "unknown"
        
        try:
            if PYNVML_AVAILABLE:
                pynvml.nvmlInit()
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode()
        except Exception:
            pass
            
        try:
            cuda_version = torch.version.cuda or "unknown"
        except Exception:
            pass
            
        return driver_version, cuda_version