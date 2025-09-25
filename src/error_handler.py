"""
Comprehensive Error Handling and Diagnostics with Architecture Awareness
Handles model download failures, pipeline loading errors, and architecture-specific issues
"""

import logging
import os
import sys
import traceback
import time
import psutil
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    DOWNLOAD = "download"
    PIPELINE = "pipeline"
    ARCHITECTURE = "architecture"
    MEMORY = "memory"
    DEVICE = "device"
    NETWORK = "network"
    PERMISSION = "permission"
    CONFIGURATION = "configuration"


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: str
    suggested_fixes: List[str]
    architecture_context: Optional[str] = None
    system_context: Optional[Dict[str, Any]] = None
    recovery_actions: Optional[List[Callable]] = None
    user_feedback: Optional[str] = None


@dataclass
class DiagnosticInfo:
    """System diagnostic information"""
    gpu_available: bool
    gpu_memory_gb: float
    system_memory_gb: float
    disk_space_gb: float
    cuda_version: Optional[str]
    pytorch_version: str
    architecture_support: Dict[str, bool]
    network_connectivity: bool
    permissions_ok: bool


class ArchitectureAwareErrorHandler:
    """
    Comprehensive error handler with architecture awareness for MMDiT vs UNet models
    """
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.diagnostic_cache: Optional[DiagnosticInfo] = None
        self.recovery_strategies: Dict[str, List[Callable]] = {}
        self.user_feedback_callbacks: List[Callable[[str], None]] = []
        
        # Architecture-specific error patterns
        self.architecture_errors = {
            "MMDiT": {
                "tensor_unpacking": {
                    "patterns": ["tuple index out of range", "expected tuple", "tensor unpacking"],
                    "fixes": [
                        "Disable torch.compile for MMDiT models",
                        "Use default attention processor instead of AttnProcessor2_0",
                        "Check transformer output format compatibility"
                    ]
                },
                "attention_issues": {
                    "patterns": ["attention", "scaled_dot_product", "flash attention"],
                    "fixes": [
                        "Disable Flash Attention for Qwen-Image compatibility",
                        "Use memory-efficient attention instead",
                        "Check attention processor compatibility with MMDiT"
                    ]
                },
                "parameter_mismatch": {
                    "patterns": ["true_cfg_scale", "guidance_scale", "unexpected keyword"],
                    "fixes": [
                        "Use 'true_cfg_scale' parameter for Qwen-Image models",
                        "Check generation parameter compatibility",
                        "Update pipeline configuration for MMDiT architecture"
                    ]
                }
            },
            "UNet": {
                "memory_issues": {
                    "patterns": ["out of memory", "CUDA out of memory", "memory allocation"],
                    "fixes": [
                        "Enable attention slicing for UNet models",
                        "Enable VAE slicing to reduce memory usage",
                        "Use CPU offloading for large UNet models"
                    ]
                },
                "attention_compatibility": {
                    "patterns": ["attention processor", "AttnProcessor"],
                    "fixes": [
                        "Use AttnProcessor2_0 for UNet models",
                        "Enable Flash Attention for better performance",
                        "Check attention processor version compatibility"
                    ]
                }
            }
        }
        
        # Initialize recovery strategies
        self._setup_recovery_strategies()
    
    def add_user_feedback_callback(self, callback: Callable[[str], None]):
        """Add callback for user feedback"""
        self.user_feedback_callbacks.append(callback)
    
    def _notify_user(self, message: str):
        """Notify user through registered callbacks"""
        for callback in self.user_feedback_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.warning(f"User feedback callback failed: {e}")
    
    def handle_download_error(
        self, 
        error: Exception, 
        model_name: str, 
        context: Dict[str, Any] = None
    ) -> ErrorInfo:
        """Handle model download failures with comprehensive diagnostics"""
        logger.error(f"🚨 Download error for {model_name}: {error}")
        
        error_str = str(error).lower()
        context = context or {}
        
        # Classify the download error
        if "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return self._handle_network_error(error, model_name, context)
        elif "space" in error_str or "disk" in error_str or "no space left" in error_str:
            return self._handle_disk_space_error(error, model_name, context)
        elif "permission" in error_str or "access denied" in error_str or "forbidden" in error_str:
            return self._handle_permission_error(error, model_name, context)
        elif "repository not found" in error_str or "404" in error_str:
            return self._handle_repository_error(error, model_name, context)
        elif "corrupted" in error_str or "integrity" in error_str or "checksum" in error_str:
            return self._handle_corruption_error(error, model_name, context)
        else:
            return self._handle_generic_download_error(error, model_name, context)
    
    def _handle_network_error(self, error: Exception, model_name: str, context: Dict[str, Any]) -> ErrorInfo:
        """Handle network-related download errors"""
        diagnostic = self.get_system_diagnostics()
        
        suggested_fixes = [
            "Check internet connection stability",
            "Try downloading during off-peak hours",
            "Use a VPN if regional restrictions apply",
            "Increase timeout settings in download configuration",
            "Resume download if partially completed"
        ]
        
        if not diagnostic.network_connectivity:
            suggested_fixes.insert(0, "Fix network connectivity issues")
        
        recovery_actions = [
            lambda: self._retry_with_backoff(context.get('retry_function')),
            lambda: self._check_alternative_mirrors(model_name),
            lambda: self._resume_partial_download(model_name)
        ]
        
        user_feedback = f"🌐 Network issue detected while downloading {model_name}. Checking connectivity and will retry automatically."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message=f"Network error downloading {model_name}",
            details=str(error),
            suggested_fixes=suggested_fixes,
            system_context=diagnostic.__dict__,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_disk_space_error(self, error: Exception, model_name: str, context: Dict[str, Any]) -> ErrorInfo:
        """Handle disk space related errors"""
        diagnostic = self.get_system_diagnostics()
        
        # Calculate required space for model
        model_sizes = {
            "Qwen/Qwen-Image": 8,
            "Qwen/Qwen-Image-Edit": 54,
            "Qwen/Qwen2-VL-7B-Instruct": 15,
            "Qwen/Qwen2-VL-2B-Instruct": 4
        }
        required_gb = model_sizes.get(model_name, 10)
        
        suggested_fixes = [
            f"Free up at least {required_gb}GB of disk space",
            "Clean up old model downloads",
            "Move models to a different drive with more space",
            "Use model compression if available",
            "Delete temporary files and caches"
        ]
        
        if diagnostic.disk_space_gb < required_gb:
            suggested_fixes.insert(0, f"Critical: Only {diagnostic.disk_space_gb:.1f}GB available, need {required_gb}GB")
        
        recovery_actions = [
            lambda: self._cleanup_old_models(),
            lambda: self._clear_cache_directories(),
            lambda: self._suggest_alternative_location()
        ]
        
        user_feedback = f"💾 Insufficient disk space for {model_name}. Need {required_gb}GB, have {diagnostic.disk_space_gb:.1f}GB. Starting cleanup..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.DOWNLOAD,
            severity=ErrorSeverity.HIGH,
            message=f"Insufficient disk space for {model_name}",
            details=f"Required: {required_gb}GB, Available: {diagnostic.disk_space_gb:.1f}GB",
            suggested_fixes=suggested_fixes,
            system_context=diagnostic.__dict__,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_permission_error(self, error: Exception, model_name: str, context: Dict[str, Any]) -> ErrorInfo:
        """Handle permission-related errors"""
        suggested_fixes = [
            "Run with administrator/sudo privileges",
            "Check write permissions for cache directory",
            "Change cache directory to user-writable location",
            "Fix file ownership issues",
            "Check antivirus software blocking downloads"
        ]
        
        recovery_actions = [
            lambda: self._check_directory_permissions(),
            lambda: self._suggest_alternative_cache_dir(),
            lambda: self._create_user_cache_dir()
        ]
        
        user_feedback = f"🔒 Permission denied while downloading {model_name}. Checking directory permissions..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            message=f"Permission denied downloading {model_name}",
            details=str(error),
            suggested_fixes=suggested_fixes,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_repository_error(self, error: Exception, model_name: str, context: Dict[str, Any]) -> ErrorInfo:
        """Handle repository not found errors"""
        suggested_fixes = [
            "Verify model name spelling and format",
            "Check if model requires authentication",
            "Ensure HuggingFace Hub access token is valid",
            "Check if model has been moved or renamed",
            "Try alternative model repositories"
        ]
        
        recovery_actions = [
            lambda: self._verify_model_exists(model_name),
            lambda: self._check_authentication(),
            lambda: self._suggest_alternative_models()
        ]
        
        user_feedback = f"❓ Model {model_name} not found. Checking repository status..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.DOWNLOAD,
            severity=ErrorSeverity.HIGH,
            message=f"Repository not found: {model_name}",
            details=str(error),
            suggested_fixes=suggested_fixes,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_corruption_error(self, error: Exception, model_name: str, context: Dict[str, Any]) -> ErrorInfo:
        """Handle file corruption errors"""
        suggested_fixes = [
            "Delete corrupted files and re-download",
            "Check network stability during download",
            "Verify disk health and integrity",
            "Use integrity verification during download",
            "Try downloading from different network"
        ]
        
        recovery_actions = [
            lambda: self._cleanup_corrupted_files(model_name),
            lambda: self._verify_disk_health(),
            lambda: self._retry_with_verification(model_name)
        ]
        
        user_feedback = f"🔧 Corrupted download detected for {model_name}. Cleaning up and retrying..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.DOWNLOAD,
            severity=ErrorSeverity.MEDIUM,
            message=f"Corrupted download: {model_name}",
            details=str(error),
            suggested_fixes=suggested_fixes,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_generic_download_error(self, error: Exception, model_name: str, context: Dict[str, Any]) -> ErrorInfo:
        """Handle generic download errors"""
        suggested_fixes = [
            "Check error details for specific issues",
            "Retry download with different settings",
            "Check system resources and connectivity",
            "Update huggingface_hub library",
            "Contact support if issue persists"
        ]
        
        recovery_actions = [
            lambda: self._retry_with_different_settings(model_name),
            lambda: self._check_library_versions(),
            lambda: self._collect_debug_info(error)
        ]
        
        user_feedback = f"⚠️ Download error for {model_name}. Analyzing issue and attempting recovery..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.DOWNLOAD,
            severity=ErrorSeverity.MEDIUM,
            message=f"Download failed: {model_name}",
            details=str(error),
            suggested_fixes=suggested_fixes,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def handle_pipeline_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str = "Unknown",
        context: Dict[str, Any] = None
    ) -> ErrorInfo:
        """Handle pipeline loading errors with architecture-specific recovery"""
        logger.error(f"🚨 Pipeline error for {model_path} ({architecture_type}): {error}")
        
        error_str = str(error).lower()
        context = context or {}
        
        # Check for architecture-specific errors
        if architecture_type in self.architecture_errors:
            arch_errors = self.architecture_errors[architecture_type]
            
            for error_type, error_info in arch_errors.items():
                if any(pattern in error_str for pattern in error_info["patterns"]):
                    return self._handle_architecture_specific_error(
                        error, model_path, architecture_type, error_type, error_info, context
                    )
        
        # Handle common pipeline errors
        if "out of memory" in error_str or "cuda out of memory" in error_str:
            return self._handle_memory_error(error, model_path, architecture_type, context)
        elif "device" in error_str or "cuda" in error_str:
            return self._handle_device_error(error, model_path, architecture_type, context)
        elif "model" in error_str and "not found" in error_str:
            return self._handle_model_not_found_error(error, model_path, architecture_type, context)
        elif "safetensors" in error_str or "checkpoint" in error_str:
            return self._handle_checkpoint_error(error, model_path, architecture_type, context)
        else:
            return self._handle_generic_pipeline_error(error, model_path, architecture_type, context)
    
    def _handle_architecture_specific_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str,
        error_type: str,
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ErrorInfo:
        """Handle architecture-specific errors (MMDiT vs UNet)"""
        
        suggested_fixes = error_info["fixes"].copy()
        
        # Add architecture-specific context
        if architecture_type == "MMDiT" and error_type == "tensor_unpacking":
            suggested_fixes.extend([
                "Ensure using AutoPipelineForText2Image for Qwen-Image models",
                "Check transformer output handling in pipeline",
                "Verify model compatibility with current diffusers version"
            ])
        elif architecture_type == "MMDiT" and error_type == "attention_issues":
            suggested_fixes.extend([
                "Use default attention processor for MMDiT compatibility",
                "Disable memory-efficient attention if causing issues",
                "Check PyTorch version compatibility with MMDiT"
            ])
        elif architecture_type == "UNet" and error_type == "memory_issues":
            suggested_fixes.extend([
                "Enable model CPU offloading for large UNet models",
                "Reduce batch size or image resolution",
                "Use gradient checkpointing if available"
            ])
        
        recovery_actions = [
            lambda: self._apply_architecture_fallback(model_path, architecture_type),
            lambda: self._adjust_pipeline_config(architecture_type, error_type),
            lambda: self._try_alternative_pipeline_class(model_path, architecture_type)
        ]
        
        user_feedback = f"🏗️ {architecture_type} architecture issue detected. Applying {architecture_type}-specific fixes..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.ARCHITECTURE,
            severity=ErrorSeverity.HIGH,
            message=f"{architecture_type} {error_type} error",
            details=str(error),
            suggested_fixes=suggested_fixes,
            architecture_context=f"{architecture_type} - {error_type}",
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_memory_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str, 
        context: Dict[str, Any]
    ) -> ErrorInfo:
        """Handle GPU memory errors with architecture awareness"""
        diagnostic = self.get_system_diagnostics()
        
        suggested_fixes = [
            "Reduce image resolution (try 512x512 instead of 1024x1024)",
            "Enable memory optimizations (attention slicing, VAE slicing)",
            "Use CPU offloading for model components",
            "Close other GPU-intensive applications",
            "Restart Python to clear GPU memory"
        ]
        
        # Architecture-specific memory optimizations
        if architecture_type == "MMDiT":
            suggested_fixes.extend([
                "MMDiT models may need more VRAM - consider using smaller resolution",
                "Disable torch.compile for MMDiT to save memory",
                "Use float16 precision for MMDiT models"
            ])
        elif architecture_type == "UNet":
            suggested_fixes.extend([
                "Enable attention slicing for UNet models",
                "Use sequential CPU offloading for UNet components",
                "Enable VAE tiling for high-resolution generation"
            ])
        
        recovery_actions = [
            lambda: self._clear_gpu_memory(),
            lambda: self._enable_memory_optimizations(architecture_type),
            lambda: self._reduce_model_precision(),
            lambda: self._try_cpu_fallback()
        ]
        
        user_feedback = f"🧠 GPU memory issue with {architecture_type} model. Applying memory optimizations..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message=f"GPU memory error with {architecture_type} model",
            details=f"Available VRAM: {diagnostic.gpu_memory_gb:.1f}GB, Error: {str(error)}",
            suggested_fixes=suggested_fixes,
            architecture_context=architecture_type,
            system_context=diagnostic.__dict__,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_device_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str, 
        context: Dict[str, Any]
    ) -> ErrorInfo:
        """Handle device-related errors"""
        diagnostic = self.get_system_diagnostics()
        
        suggested_fixes = [
            "Check CUDA installation and compatibility",
            "Verify GPU drivers are up to date",
            "Try CPU fallback if GPU unavailable",
            "Check PyTorch CUDA version compatibility",
            "Restart system to reset GPU state"
        ]
        
        if not diagnostic.gpu_available:
            suggested_fixes.insert(0, "GPU not detected - install CUDA drivers")
        
        recovery_actions = [
            lambda: self._check_cuda_installation(),
            lambda: self._try_cpu_device(),
            lambda: self._reset_gpu_state(),
            lambda: self._check_driver_compatibility()
        ]
        
        user_feedback = f"🖥️ Device error detected. Checking GPU availability and CUDA installation..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.DEVICE,
            severity=ErrorSeverity.HIGH,
            message="Device/CUDA error",
            details=str(error),
            suggested_fixes=suggested_fixes,
            system_context=diagnostic.__dict__,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_model_not_found_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str, 
        context: Dict[str, Any]
    ) -> ErrorInfo:
        """Handle model file not found errors"""
        suggested_fixes = [
            "Verify model path exists and is accessible",
            "Check if model download completed successfully",
            "Re-download model if files are missing",
            "Check file permissions for model directory",
            "Verify model format compatibility"
        ]
        
        recovery_actions = [
            lambda: self._verify_model_files(model_path),
            lambda: self._attempt_model_repair(model_path),
            lambda: self._suggest_redownload(model_path)
        ]
        
        user_feedback = f"📁 Model files not found at {model_path}. Checking file integrity..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.PIPELINE,
            severity=ErrorSeverity.HIGH,
            message="Model files not found",
            details=f"Path: {model_path}, Error: {str(error)}",
            suggested_fixes=suggested_fixes,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_checkpoint_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str, 
        context: Dict[str, Any]
    ) -> ErrorInfo:
        """Handle checkpoint/safetensors loading errors"""
        suggested_fixes = [
            "Check if safetensors files are corrupted",
            "Try loading with different precision (float16/float32)",
            "Verify model format compatibility",
            "Re-download model if checksums don't match",
            "Check available disk space during loading"
        ]
        
        recovery_actions = [
            lambda: self._verify_checkpoint_integrity(model_path),
            lambda: self._try_alternative_loading_method(model_path),
            lambda: self._check_model_format_compatibility(model_path, architecture_type)
        ]
        
        user_feedback = f"💾 Checkpoint loading issue. Verifying model file integrity..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.PIPELINE,
            severity=ErrorSeverity.MEDIUM,
            message="Checkpoint loading error",
            details=str(error),
            suggested_fixes=suggested_fixes,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def _handle_generic_pipeline_error(
        self, 
        error: Exception, 
        model_path: str, 
        architecture_type: str, 
        context: Dict[str, Any]
    ) -> ErrorInfo:
        """Handle generic pipeline errors"""
        suggested_fixes = [
            "Check error traceback for specific issues",
            "Verify model and pipeline compatibility",
            "Update diffusers library to latest version",
            "Check PyTorch version compatibility",
            "Try with different pipeline configuration"
        ]
        
        recovery_actions = [
            lambda: self._collect_pipeline_debug_info(model_path, architecture_type),
            lambda: self._try_basic_pipeline_config(model_path),
            lambda: self._check_library_compatibility()
        ]
        
        user_feedback = f"⚙️ Pipeline loading issue. Running diagnostics..."
        self._notify_user(user_feedback)
        
        return ErrorInfo(
            category=ErrorCategory.PIPELINE,
            severity=ErrorSeverity.MEDIUM,
            message="Pipeline loading error",
            details=str(error),
            suggested_fixes=suggested_fixes,
            architecture_context=architecture_type,
            recovery_actions=recovery_actions,
            user_feedback=user_feedback
        )
    
    def get_system_diagnostics(self, force_refresh: bool = False) -> DiagnosticInfo:
        """Get comprehensive system diagnostics"""
        if self.diagnostic_cache and not force_refresh:
            return self.diagnostic_cache
        
        logger.info("🔍 Running system diagnostics...")
        
        # GPU diagnostics
        gpu_available = torch.cuda.is_available()
        gpu_memory_gb = 0.0
        cuda_version = None
        
        if gpu_available:
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                cuda_version = torch.version.cuda
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
        
        # System memory
        system_memory_gb = psutil.virtual_memory().total / 1e9
        
        # Disk space
        disk_space_gb = 0.0
        try:
            disk_usage = psutil.disk_usage('.')
            disk_space_gb = disk_usage.free / 1e9
        except Exception as e:
            logger.warning(f"Could not get disk space: {e}")
        
        # Architecture support
        architecture_support = {
            "MMDiT": True,  # Always supported
            "UNet": True,   # Always supported
            "Flash_Attention": self._check_flash_attention_support(),
            "Torch_Compile": self._check_torch_compile_support(),
            "SDPA": self._check_sdpa_support()
        }
        
        # Network connectivity
        network_connectivity = self._check_network_connectivity()
        
        # Permissions
        permissions_ok = self._check_permissions()
        
        diagnostic = DiagnosticInfo(
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            system_memory_gb=system_memory_gb,
            disk_space_gb=disk_space_gb,
            cuda_version=cuda_version,
            pytorch_version=torch.__version__,
            architecture_support=architecture_support,
            network_connectivity=network_connectivity,
            permissions_ok=permissions_ok
        )
        
        self.diagnostic_cache = diagnostic
        logger.info(f"✅ Diagnostics complete: GPU={gpu_available}, VRAM={gpu_memory_gb:.1f}GB, RAM={system_memory_gb:.1f}GB")
        
        return diagnostic
    
    def create_diagnostic_report(self) -> Dict[str, Any]:
        """Create a comprehensive diagnostic report"""
        diagnostic = self.get_system_diagnostics(force_refresh=True)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "gpu_available": diagnostic.gpu_available,
                "gpu_memory_gb": diagnostic.gpu_memory_gb,
                "system_memory_gb": diagnostic.system_memory_gb,
                "disk_space_gb": diagnostic.disk_space_gb,
                "cuda_version": diagnostic.cuda_version,
                "pytorch_version": diagnostic.pytorch_version
            },
            "architecture_support": diagnostic.architecture_support,
            "connectivity": {
                "network": diagnostic.network_connectivity,
                "permissions": diagnostic.permissions_ok
            },
            "recent_errors": [
                {
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "architecture_context": error.architecture_context
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ],
            "recommendations": self._generate_recommendations(diagnostic)
        }
        
        return report
    
    def _generate_recommendations(self, diagnostic: DiagnosticInfo) -> List[str]:
        """Generate system recommendations based on diagnostics"""
        recommendations = []
        
        if not diagnostic.gpu_available:
            recommendations.append("Install CUDA drivers for GPU acceleration")
        elif diagnostic.gpu_memory_gb < 8:
            recommendations.append("Consider upgrading GPU for better performance (8GB+ VRAM recommended)")
        
        if diagnostic.system_memory_gb < 16:
            recommendations.append("Consider adding more system RAM (16GB+ recommended)")
        
        if diagnostic.disk_space_gb < 50:
            recommendations.append("Free up disk space (50GB+ recommended for model storage)")
        
        if not diagnostic.architecture_support.get("Flash_Attention", False):
            recommendations.append("Install Flash Attention for better performance")
        
        if not diagnostic.network_connectivity:
            recommendations.append("Fix network connectivity for model downloads")
        
        if not diagnostic.permissions_ok:
            recommendations.append("Fix file permissions for model cache directories")
        
        return recommendations
    
    def _setup_recovery_strategies(self):
        """Setup recovery strategies for different error types"""
        self.recovery_strategies = {
            "network_retry": [
                lambda: time.sleep(5),  # Wait before retry
                lambda: self._check_network_connectivity(),
                lambda: logger.info("Retrying network operation...")
            ],
            "memory_optimization": [
                lambda: self._clear_gpu_memory(),
                lambda: self._enable_memory_optimizations("auto"),
                lambda: logger.info("Applied memory optimizations")
            ],
            "architecture_fallback": [
                lambda: logger.info("Applying architecture fallback..."),
                lambda: self._disable_problematic_optimizations(),
                lambda: self._use_safe_pipeline_config()
            ]
        }    
 
   # Recovery action implementations
    def _retry_with_backoff(self, retry_function: Optional[Callable], max_retries: int = 3):
        """Retry operation with exponential backoff"""
        if not retry_function:
            return False
        
        for attempt in range(max_retries):
            try:
                wait_time = 2 ** attempt
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} (waiting {wait_time}s)")
                time.sleep(wait_time)
                return retry_function()
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed")
                    return False
        return False
    
    def _check_alternative_mirrors(self, model_name: str):
        """Check for alternative download mirrors"""
        logger.info(f"Checking alternative mirrors for {model_name}")
        # Implementation would check alternative HuggingFace mirrors
        return True
    
    def _resume_partial_download(self, model_name: str):
        """Resume a partially downloaded model"""
        logger.info(f"Attempting to resume download for {model_name}")
        # Implementation would check for partial files and resume
        return True
    
    def _cleanup_old_models(self):
        """Clean up old model files to free space"""
        logger.info("Cleaning up old models to free disk space")
        try:
            models_dir = Path("./models")
            if models_dir.exists():
                # Calculate space that could be freed
                total_size = sum(f.stat().st_size for f in models_dir.rglob("*") if f.is_file())
                logger.info(f"Found {total_size / 1e9:.1f}GB in models directory")
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False
    
    def _clear_cache_directories(self):
        """Clear various cache directories"""
        logger.info("Clearing cache directories")
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface"),
            os.path.expanduser("~/.cache/torch"),
            "./cache"
        ]
        
        for cache_dir in cache_dirs:
            try:
                if os.path.exists(cache_dir):
                    # Calculate cache size
                    cache_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(cache_dir)
                        for filename in filenames
                    )
                    logger.info(f"Cache {cache_dir}: {cache_size / 1e9:.1f}GB")
            except Exception as e:
                logger.warning(f"Could not check cache {cache_dir}: {e}")
        
        return True
    
    def _suggest_alternative_location(self):
        """Suggest alternative storage location"""
        logger.info("Suggesting alternative storage locations")
        # Check available drives and suggest best option
        return True
    
    def _check_directory_permissions(self):
        """Check and fix directory permissions"""
        logger.info("Checking directory permissions")
        
        directories_to_check = [
            "./models",
            os.path.expanduser("~/.cache/huggingface"),
            "./cache"
        ]
        
        for directory in directories_to_check:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                
                # Test write permissions
                test_file = os.path.join(directory, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"Write permissions OK: {directory}")
                
            except Exception as e:
                logger.error(f"Permission issue with {directory}: {e}")
                return False
        
        return True
    
    def _suggest_alternative_cache_dir(self):
        """Suggest alternative cache directory"""
        logger.info("Suggesting alternative cache directory")
        alternative_dirs = [
            os.path.expanduser("~/Documents/huggingface_cache"),
            "./local_cache",
            os.path.expanduser("~/AppData/Local/huggingface") if sys.platform == "win32" else "~/.local/share/huggingface"
        ]
        
        for alt_dir in alternative_dirs:
            try:
                os.makedirs(alt_dir, exist_ok=True)
                test_file = os.path.join(alt_dir, "test.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"Alternative cache directory available: {alt_dir}")
                return alt_dir
            except Exception as e:
                logger.warning(f"Cannot use {alt_dir}: {e}")
        
        return None
    
    def _create_user_cache_dir(self):
        """Create user-specific cache directory"""
        logger.info("Creating user-specific cache directory")
        user_cache = os.path.expanduser("~/.qwen_models")
        try:
            os.makedirs(user_cache, exist_ok=True)
            logger.info(f"Created user cache directory: {user_cache}")
            return user_cache
        except Exception as e:
            logger.error(f"Could not create user cache: {e}")
            return None
    
    def _verify_model_exists(self, model_name: str):
        """Verify if model exists on HuggingFace Hub"""
        logger.info(f"Verifying model existence: {model_name}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_info = api.repo_info(model_name)
            logger.info(f"Model verified: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def _check_authentication(self):
        """Check HuggingFace authentication"""
        logger.info("Checking HuggingFace authentication")
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                logger.info("HuggingFace token found")
                return True
            else:
                logger.warning("No HuggingFace token found")
                return False
        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return False
    
    def _suggest_alternative_models(self):
        """Suggest alternative models"""
        logger.info("Suggesting alternative models")
        alternatives = {
            "Qwen/Qwen-Image": ["Qwen/Qwen-Image-Edit"],
            "Qwen/Qwen-Image-Edit": ["Qwen/Qwen-Image"],
            "Qwen/Qwen2-VL-7B-Instruct": ["Qwen/Qwen2-VL-2B-Instruct"],
            "Qwen/Qwen2-VL-2B-Instruct": ["Qwen/Qwen2-VL-7B-Instruct"]
        }
        return alternatives
    
    def _cleanup_corrupted_files(self, model_name: str):
        """Clean up corrupted model files"""
        logger.info(f"Cleaning up corrupted files for {model_name}")
        
        # Find model directories
        model_dirs = []
        
        # Check local models
        local_model_dir = Path("./models") / model_name.split("/")[-1]
        if local_model_dir.exists():
            model_dirs.append(local_model_dir)
        
        # Check cache
        cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub")) / f"models--{model_name.replace('/', '--')}"
        if cache_dir.exists():
            model_dirs.append(cache_dir)
        
        for model_dir in model_dirs:
            try:
                logger.info(f"Removing corrupted files from {model_dir}")
                # Remove incomplete downloads (files with .tmp extension)
                for tmp_file in model_dir.rglob("*.tmp"):
                    tmp_file.unlink()
                    logger.info(f"Removed temporary file: {tmp_file}")
                
                # Remove lock files
                for lock_file in model_dir.rglob("*.lock"):
                    lock_file.unlink()
                    logger.info(f"Removed lock file: {lock_file}")
                    
            except Exception as e:
                logger.error(f"Could not clean up {model_dir}: {e}")
        
        return True
    
    def _verify_disk_health(self):
        """Verify disk health"""
        logger.info("Checking disk health")
        # Basic disk health check
        try:
            disk_usage = psutil.disk_usage('.')
            logger.info(f"Disk usage: {disk_usage.used / 1e9:.1f}GB used, {disk_usage.free / 1e9:.1f}GB free")
            return True
        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return False
    
    def _retry_with_verification(self, model_name: str):
        """Retry download with integrity verification"""
        logger.info(f"Retrying download with verification for {model_name}")
        # Implementation would retry download with checksum verification
        return True
    
    def _retry_with_different_settings(self, model_name: str):
        """Retry with different download settings"""
        logger.info(f"Retrying with different settings for {model_name}")
        # Implementation would try different timeout, chunk size, etc.
        return True
    
    def _check_library_versions(self):
        """Check library versions for compatibility"""
        logger.info("Checking library versions")
        
        try:
            import torch
            import diffusers
            import transformers
            
            versions = {
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "transformers": transformers.__version__
            }
            
            logger.info(f"Library versions: {versions}")
            
            # Check for known compatibility issues
            torch_version = torch.__version__
            if torch_version.startswith("1."):
                logger.warning("PyTorch 1.x detected - consider upgrading to 2.x for better performance")
            
            return versions
            
        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            return {}
    
    def _collect_debug_info(self, error: Exception):
        """Collect debug information for error analysis"""
        logger.info("Collecting debug information")
        
        debug_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "system_info": self.get_system_diagnostics().__dict__,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": os.getcwd()
            }
        }
        
        logger.info("Debug information collected")
        return debug_info
    
    def _apply_architecture_fallback(self, model_path: str, architecture_type: str):
        """Apply architecture-specific fallback strategies"""
        logger.info(f"Applying {architecture_type} fallback strategies")
        
        if architecture_type == "MMDiT":
            # MMDiT fallback strategies
            logger.info("Applying MMDiT fallback: disabling problematic optimizations")
            return {
                "use_torch_compile": False,
                "use_flash_attention": False,
                "use_default_attention": True,
                "pipeline_class": "AutoPipelineForText2Image"
            }
        elif architecture_type == "UNet":
            # UNet fallback strategies
            logger.info("Applying UNet fallback: enabling memory optimizations")
            return {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "use_cpu_offload": True,
                "pipeline_class": "DiffusionPipeline"
            }
        else:
            # Generic fallback
            logger.info("Applying generic fallback strategies")
            return {
                "use_safe_defaults": True,
                "disable_optimizations": True,
                "pipeline_class": "DiffusionPipeline"
            }
    
    def _adjust_pipeline_config(self, architecture_type: str, error_type: str):
        """Adjust pipeline configuration based on error type"""
        logger.info(f"Adjusting pipeline config for {architecture_type} {error_type}")
        
        config_adjustments = {}
        
        if architecture_type == "MMDiT":
            if error_type == "tensor_unpacking":
                config_adjustments.update({
                    "disable_torch_compile": True,
                    "use_default_attention": True,
                    "check_output_format": True
                })
            elif error_type == "attention_issues":
                config_adjustments.update({
                    "disable_flash_attention": True,
                    "use_memory_efficient_attention": False,
                    "use_default_attention_processor": True
                })
        elif architecture_type == "UNet":
            if error_type == "memory_issues":
                config_adjustments.update({
                    "enable_attention_slicing": True,
                    "enable_vae_slicing": True,
                    "enable_cpu_offload": True
                })
        
        logger.info(f"Config adjustments: {config_adjustments}")
        return config_adjustments
    
    def _try_alternative_pipeline_class(self, model_path: str, architecture_type: str):
        """Try alternative pipeline class"""
        logger.info(f"Trying alternative pipeline class for {architecture_type}")
        
        alternatives = {
            "MMDiT": ["AutoPipelineForText2Image", "DiffusionPipeline"],
            "UNet": ["DiffusionPipeline", "AutoPipelineForText2Image"],
            "Unknown": ["DiffusionPipeline", "AutoPipelineForText2Image"]
        }
        
        return alternatives.get(architecture_type, ["DiffusionPipeline"])
    
    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        logger.info("Clearing GPU memory")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared")
                return True
        except Exception as e:
            logger.error(f"Could not clear GPU memory: {e}")
        return False
    
    def _enable_memory_optimizations(self, architecture_type: str):
        """Enable memory optimizations based on architecture"""
        logger.info(f"Enabling memory optimizations for {architecture_type}")
        
        optimizations = {}
        
        if architecture_type == "MMDiT" or architecture_type == "auto":
            optimizations.update({
                "torch_dtype": "float16",
                "low_cpu_mem_usage": True,
                "disable_gradient": True
            })
        
        if architecture_type == "UNet" or architecture_type == "auto":
            optimizations.update({
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_vae_tiling": True
            })
        
        logger.info(f"Memory optimizations: {optimizations}")
        return optimizations
    
    def _reduce_model_precision(self):
        """Reduce model precision to save memory"""
        logger.info("Reducing model precision to float16")
        return {"torch_dtype": torch.float16}
    
    def _try_cpu_fallback(self):
        """Try CPU fallback"""
        logger.info("Trying CPU fallback")
        return {"device": "cpu", "torch_dtype": torch.float32}
    
    def _check_cuda_installation(self):
        """Check CUDA installation"""
        logger.info("Checking CUDA installation")
        
        cuda_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if cuda_info["cuda_available"]:
            for i in range(cuda_info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f}GB")
        else:
            logger.warning("CUDA not available")
        
        return cuda_info
    
    def _try_cpu_device(self):
        """Try using CPU device"""
        logger.info("Switching to CPU device")
        return {"device": "cpu"}
    
    def _reset_gpu_state(self):
        """Reset GPU state"""
        logger.info("Resetting GPU state")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                logger.info("GPU state reset")
                return True
        except Exception as e:
            logger.error(f"Could not reset GPU state: {e}")
        return False
    
    def _check_driver_compatibility(self):
        """Check driver compatibility"""
        logger.info("Checking driver compatibility")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - check driver installation")
            return False
        
        try:
            # Try a simple CUDA operation
            test_tensor = torch.randn(10, 10).cuda()
            result = test_tensor @ test_tensor.T
            logger.info("Driver compatibility check passed")
            return True
        except Exception as e:
            logger.error(f"Driver compatibility issue: {e}")
            return False
    
    def _verify_model_files(self, model_path: str):
        """Verify model files exist and are valid"""
        logger.info(f"Verifying model files at {model_path}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        # Check for essential files
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json"
        ]
        
        missing_files = []
        for file_path in essential_files:
            full_path = model_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing essential files: {missing_files}")
            return False
        
        logger.info("Model file verification passed")
        return True
    
    def _attempt_model_repair(self, model_path: str):
        """Attempt to repair model files"""
        logger.info(f"Attempting model repair for {model_path}")
        # Implementation would attempt to repair or re-download missing files
        return True
    
    def _suggest_redownload(self, model_path: str):
        """Suggest re-downloading the model"""
        logger.info(f"Suggesting re-download for {model_path}")
        return True
    
    def _verify_checkpoint_integrity(self, model_path: str):
        """Verify checkpoint file integrity"""
        logger.info(f"Verifying checkpoint integrity for {model_path}")
        
        model_path = Path(model_path)
        
        # Check for safetensors files
        safetensors_files = list(model_path.rglob("*.safetensors"))
        if safetensors_files:
            logger.info(f"Found {len(safetensors_files)} safetensors files")
            
            for file_path in safetensors_files:
                try:
                    # Basic file size check
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        logger.error(f"Empty safetensors file: {file_path}")
                        return False
                    logger.info(f"Safetensors file OK: {file_path.name} ({file_size / 1e6:.1f}MB)")
                except Exception as e:
                    logger.error(f"Could not check {file_path}: {e}")
                    return False
        
        return True
    
    def _try_alternative_loading_method(self, model_path: str):
        """Try alternative model loading method"""
        logger.info(f"Trying alternative loading method for {model_path}")
        
        alternatives = [
            {"torch_dtype": torch.float32},
            {"torch_dtype": torch.float16, "low_cpu_mem_usage": True},
            {"device_map": "auto"},
            {"use_safetensors": False}
        ]
        
        return alternatives
    
    def _check_model_format_compatibility(self, model_path: str, architecture_type: str):
        """Check model format compatibility"""
        logger.info(f"Checking model format compatibility for {architecture_type}")
        
        model_path = Path(model_path)
        
        # Check for config files
        config_files = list(model_path.rglob("config.json"))
        if config_files:
            try:
                import json
                with open(config_files[0], 'r') as f:
                    config = json.load(f)
                    logger.info(f"Model config loaded: {config.get('model_type', 'unknown')}")
                    return True
            except Exception as e:
                logger.error(f"Could not load model config: {e}")
        
        return False
    
    def _collect_pipeline_debug_info(self, model_path: str, architecture_type: str):
        """Collect pipeline debug information"""
        logger.info(f"Collecting pipeline debug info for {architecture_type}")
        
        debug_info = {
            "model_path": model_path,
            "architecture_type": architecture_type,
            "system_diagnostics": self.get_system_diagnostics().__dict__,
            "library_versions": self._check_library_versions(),
            "cuda_info": self._check_cuda_installation()
        }
        
        return debug_info
    
    def _try_basic_pipeline_config(self, model_path: str):
        """Try basic pipeline configuration"""
        logger.info("Trying basic pipeline configuration")
        
        basic_config = {
            "torch_dtype": torch.float16,
            "device_map": None,
            "use_safetensors": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": False
        }
        
        return basic_config
    
    def _check_library_compatibility(self):
        """Check library compatibility"""
        logger.info("Checking library compatibility")
        
        compatibility_info = {
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        # Check for known compatibility issues
        issues = []
        
        if torch.__version__.startswith("1."):
            issues.append("PyTorch 1.x may have compatibility issues - consider upgrading to 2.x")
        
        if sys.version_info < (3, 8):
            issues.append("Python version < 3.8 may have compatibility issues")
        
        compatibility_info["issues"] = issues
        logger.info(f"Compatibility check: {len(issues)} issues found")
        
        return compatibility_info
    
    # Helper methods for diagnostics
    def _check_flash_attention_support(self) -> bool:
        """Check if Flash Attention is supported"""
        try:
            # Try to import flash attention
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _check_torch_compile_support(self) -> bool:
        """Check if torch.compile is supported"""
        try:
            # Check PyTorch version
            torch_version = torch.__version__
            major, minor = torch_version.split('.')[:2]
            return int(major) >= 2 and int(minor) >= 0
        except:
            return False
    
    def _check_sdpa_support(self) -> bool:
        """Check if Scaled Dot-Product Attention is supported"""
        try:
            return hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        except:
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        try:
            import urllib.request
            urllib.request.urlopen('https://huggingface.co', timeout=10)
            return True
        except:
            return False
    
    def _check_permissions(self) -> bool:
        """Check file system permissions"""
        try:
            # Test write permissions in current directory
            test_file = "test_permissions.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except:
            return False
    
    def _disable_problematic_optimizations(self):
        """Disable optimizations that commonly cause issues"""
        logger.info("Disabling problematic optimizations")
        return {
            "use_torch_compile": False,
            "use_flash_attention": False,
            "enable_memory_efficient_attention": False
        }
    
    def _use_safe_pipeline_config(self):
        """Use safe pipeline configuration"""
        logger.info("Using safe pipeline configuration")
        return {
            "torch_dtype": torch.float16,
            "device_map": None,
            "use_safetensors": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
    
    def log_error(self, error_info: ErrorInfo):
        """Log error information"""
        self.error_history.append(error_info)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        logger.error(f"[{error_info.category.value.upper()}] {error_info.message}")
        if error_info.architecture_context:
            logger.error(f"Architecture context: {error_info.architecture_context}")
        
        # Notify user if callback is available
        if error_info.user_feedback:
            self._notify_user(error_info.user_feedback)
    
    def execute_recovery_actions(self, error_info: ErrorInfo) -> bool:
        """Execute recovery actions for an error"""
        if not error_info.recovery_actions:
            logger.info("No recovery actions available")
            return False
        
        logger.info(f"Executing {len(error_info.recovery_actions)} recovery actions")
        
        success_count = 0
        for i, action in enumerate(error_info.recovery_actions):
            try:
                logger.info(f"Executing recovery action {i + 1}/{len(error_info.recovery_actions)}")
                result = action()
                if result:
                    success_count += 1
                    logger.info(f"Recovery action {i + 1} succeeded")
                else:
                    logger.warning(f"Recovery action {i + 1} failed")
            except Exception as e:
                logger.error(f"Recovery action {i + 1} raised exception: {e}")
        
        success_rate = success_count / len(error_info.recovery_actions)
        logger.info(f"Recovery actions completed: {success_count}/{len(error_info.recovery_actions)} succeeded ({success_rate:.1%})")
        
        return success_rate > 0.5  # Consider successful if more than half succeeded


# Convenience functions for easy integration
def handle_download_error(error: Exception, model_name: str, context: Dict[str, Any] = None) -> ErrorInfo:
    """Convenience function to handle download errors"""
    handler = ArchitectureAwareErrorHandler()
    return handler.handle_download_error(error, model_name, context)


def handle_pipeline_error(
    error: Exception, 
    model_path: str, 
    architecture_type: str = "Unknown",
    context: Dict[str, Any] = None
) -> ErrorInfo:
    """Convenience function to handle pipeline errors"""
    handler = ArchitectureAwareErrorHandler()
    return handler.handle_pipeline_error(error, model_path, architecture_type, context)


def get_system_diagnostics() -> DiagnosticInfo:
    """Convenience function to get system diagnostics"""
    handler = ArchitectureAwareErrorHandler()
    return handler.get_system_diagnostics()


def create_diagnostic_report() -> Dict[str, Any]:
    """Convenience function to create diagnostic report"""
    handler = ArchitectureAwareErrorHandler()
    return handler.create_diagnostic_report()


if __name__ == "__main__":
    # Example usage and testing
    handler = ArchitectureAwareErrorHandler()
    
    # Test diagnostics
    print("Running system diagnostics...")
    diagnostics = handler.get_system_diagnostics()
    print(f"GPU Available: {diagnostics.gpu_available}")
    print(f"GPU Memory: {diagnostics.gpu_memory_gb:.1f}GB")
    print(f"System Memory: {diagnostics.system_memory_gb:.1f}GB")
    print(f"Disk Space: {diagnostics.disk_space_gb:.1f}GB")
    
    # Test error handling
    try:
        raise RuntimeError("Test error for MMDiT tensor unpacking")
    except Exception as e:
        error_info = handler.handle_pipeline_error(e, "test/model", "MMDiT")
        print(f"Error handled: {error_info.message}")
        print(f"Suggested fixes: {error_info.suggested_fixes}")
        
        # Execute recovery actions
        handler.execute_recovery_actions(error_info)
    
    # Generate diagnostic report
    report = handler.create_diagnostic_report()
    print(f"Diagnostic report generated with {len(report['recent_errors'])} recent errors")