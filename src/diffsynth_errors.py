"""
DiffSynth Error Handling System
Comprehensive error hierarchy and recovery mechanisms for DiffSynth operations
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.error_handler import ErrorInfo, ErrorCategory, ErrorSeverity, ArchitectureAwareErrorHandler

logger = logging.getLogger(__name__)


class DiffSynthErrorType(Enum):
    """Specific DiffSynth error types"""
    SERVICE_INITIALIZATION = "service_initialization"
    PIPELINE_LOADING = "pipeline_loading"
    IMAGE_PROCESSING = "image_processing"
    MEMORY_ALLOCATION = "memory_allocation"
    MODEL_COMPATIBILITY = "model_compatibility"
    CONTROLNET_PROCESSING = "controlnet_processing"
    INPAINTING_ERROR = "inpainting_error"
    OUTPAINTING_ERROR = "outpainting_error"
    STYLE_TRANSFER_ERROR = "style_transfer_error"
    TILED_PROCESSING_ERROR = "tiled_processing_error"
    ELIGEN_INTEGRATION_ERROR = "eligen_integration_error"
    RESOURCE_MANAGEMENT_ERROR = "resource_management_error"
    FALLBACK_FAILURE = "fallback_failure"


class DiffSynthErrorSeverity(Enum):
    """DiffSynth-specific error severity levels"""
    RECOVERABLE = "recoverable"  # Can continue with fallback
    DEGRADED = "degraded"        # Service works but with reduced functionality
    CRITICAL = "critical"        # Service unavailable, requires intervention
    FATAL = "fatal"             # Complete system failure


@dataclass
class DiffSynthErrorContext:
    """Context information for DiffSynth errors"""
    operation_type: str
    model_name: Optional[str] = None
    image_dimensions: Optional[tuple] = None
    memory_usage_gb: Optional[float] = None
    processing_time: Optional[float] = None
    pipeline_config: Optional[Dict[str, Any]] = None
    eligen_enabled: bool = False
    tiled_processing: bool = False
    controlnet_type: Optional[str] = None
    fallback_attempted: bool = False


class DiffSynthError(Exception):
    """Base exception for all DiffSynth-related errors"""
    
    def __init__(
        self,
        message: str,
        error_type: DiffSynthErrorType,
        severity: DiffSynthErrorSeverity = DiffSynthErrorSeverity.RECOVERABLE,
        context: Optional[DiffSynthErrorContext] = None,
        original_error: Optional[Exception] = None,
        suggested_fixes: Optional[List[str]] = None,
        recovery_actions: Optional[List[Callable]] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or DiffSynthErrorContext(operation_type="unknown")
        self.original_error = original_error
        self.suggested_fixes = suggested_fixes or []
        self.recovery_actions = recovery_actions or []
        self.timestamp = time.time()
        self.error_id = f"diffsynth_{int(self.timestamp)}_{id(self)}"


class DiffSynthServiceError(DiffSynthError):
    """Errors related to DiffSynth service initialization and management"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.SERVICE_INITIALIZATION,
            severity=DiffSynthErrorSeverity.CRITICAL,
            **kwargs
        )


class DiffSynthPipelineError(DiffSynthError):
    """Errors related to DiffSynth pipeline loading and configuration"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.PIPELINE_LOADING,
            severity=DiffSynthErrorSeverity.CRITICAL,
            **kwargs
        )


class DiffSynthProcessingError(DiffSynthError):
    """Errors during image processing operations"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.IMAGE_PROCESSING,
            severity=DiffSynthErrorSeverity.RECOVERABLE,
            **kwargs
        )


class DiffSynthMemoryError(DiffSynthError):
    """Memory-related errors in DiffSynth operations"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.MEMORY_ALLOCATION,
            severity=DiffSynthErrorSeverity.DEGRADED,
            **kwargs
        )


class DiffSynthControlNetError(DiffSynthError):
    """ControlNet-specific processing errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.CONTROLNET_PROCESSING,
            severity=DiffSynthErrorSeverity.RECOVERABLE,
            **kwargs
        )


class DiffSynthInpaintingError(DiffSynthError):
    """Inpainting operation errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.INPAINTING_ERROR,
            severity=DiffSynthErrorSeverity.RECOVERABLE,
            **kwargs
        )


class DiffSynthOutpaintingError(DiffSynthError):
    """Outpainting operation errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.OUTPAINTING_ERROR,
            severity=DiffSynthErrorSeverity.RECOVERABLE,
            **kwargs
        )


class DiffSynthStyleTransferError(DiffSynthError):
    """Style transfer operation errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.STYLE_TRANSFER_ERROR,
            severity=DiffSynthErrorSeverity.RECOVERABLE,
            **kwargs
        )


class DiffSynthTiledProcessingError(DiffSynthError):
    """Tiled processing errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.TILED_PROCESSING_ERROR,
            severity=DiffSynthErrorSeverity.DEGRADED,
            **kwargs
        )


class DiffSynthEliGenError(DiffSynthError):
    """EliGen integration errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.ELIGEN_INTEGRATION_ERROR,
            severity=DiffSynthErrorSeverity.DEGRADED,
            **kwargs
        )


class DiffSynthResourceError(DiffSynthError):
    """Resource management errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.RESOURCE_MANAGEMENT_ERROR,
            severity=DiffSynthErrorSeverity.DEGRADED,
            **kwargs
        )


class DiffSynthFallbackError(DiffSynthError):
    """Fallback mechanism failures"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_type=DiffSynthErrorType.FALLBACK_FAILURE,
            severity=DiffSynthErrorSeverity.FATAL,
            **kwargs
        )


@dataclass
class RecoveryStrategy:
    """Recovery strategy for DiffSynth errors"""
    name: str
    description: str
    action: Callable
    success_probability: float  # 0.0 to 1.0
    resource_cost: str  # "low", "medium", "high"
    user_impact: str   # "none", "minimal", "moderate", "high"


class DiffSynthErrorHandler:
    """
    Comprehensive error handler for DiffSynth operations with recovery mechanisms
    """
    
    def __init__(self):
        self.base_handler = ArchitectureAwareErrorHandler()
        self.error_history: List[DiffSynthError] = []
        self.recovery_strategies: Dict[DiffSynthErrorType, List[RecoveryStrategy]] = {}
        self.fallback_callbacks: List[Callable] = []
        self.user_notification_callbacks: List[Callable[[str, str], None]] = []
        
        # Initialize recovery strategies
        self._setup_recovery_strategies()
        
        # Error pattern tracking
        self.error_patterns: Dict[str, int] = {}
        self.consecutive_failures: Dict[DiffSynthErrorType, int] = {}
        
        logger.info("DiffSynth error handler initialized")
    
    def add_fallback_callback(self, callback: Callable) -> None:
        """Add fallback callback for when DiffSynth fails"""
        self.fallback_callbacks.append(callback)
    
    def add_user_notification_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for user notifications (message, severity)"""
        self.user_notification_callbacks.append(callback)
    
    def _notify_user(self, message: str, severity: str = "info") -> None:
        """Notify user through registered callbacks"""
        for callback in self.user_notification_callbacks:
            try:
                callback(message, severity)
            except Exception as e:
                logger.warning(f"User notification callback failed: {e}")
    
    def _setup_recovery_strategies(self) -> None:
        """Setup recovery strategies for different error types"""
        
        # Service initialization recovery strategies
        self.recovery_strategies[DiffSynthErrorType.SERVICE_INITIALIZATION] = [
            RecoveryStrategy(
                name="retry_initialization",
                description="Retry service initialization with clean state",
                action=self._retry_service_initialization,
                success_probability=0.7,
                resource_cost="medium",
                user_impact="minimal"
            ),
            RecoveryStrategy(
                name="fallback_to_cpu",
                description="Initialize service on CPU instead of GPU",
                action=self._fallback_to_cpu_initialization,
                success_probability=0.9,
                resource_cost="low",
                user_impact="moderate"
            ),
            RecoveryStrategy(
                name="reduce_memory_usage",
                description="Initialize with reduced memory configuration",
                action=self._reduce_memory_initialization,
                success_probability=0.8,
                resource_cost="low",
                user_impact="minimal"
            )
        ]
        
        # Memory error recovery strategies
        self.recovery_strategies[DiffSynthErrorType.MEMORY_ALLOCATION] = [
            RecoveryStrategy(
                name="clear_gpu_cache",
                description="Clear GPU memory cache and retry",
                action=self._clear_gpu_cache,
                success_probability=0.6,
                resource_cost="low",
                user_impact="none"
            ),
            RecoveryStrategy(
                name="enable_cpu_offload",
                description="Enable CPU offloading for model components",
                action=self._enable_cpu_offload,
                success_probability=0.8,
                resource_cost="medium",
                user_impact="minimal"
            ),
            RecoveryStrategy(
                name="reduce_image_resolution",
                description="Process with reduced image resolution",
                action=self._reduce_image_resolution,
                success_probability=0.9,
                resource_cost="low",
                user_impact="moderate"
            ),
            RecoveryStrategy(
                name="enable_tiled_processing",
                description="Use tiled processing for large images",
                action=self._enable_tiled_processing,
                success_probability=0.85,
                resource_cost="medium",
                user_impact="minimal"
            )
        ]
        
        # Image processing error recovery strategies
        self.recovery_strategies[DiffSynthErrorType.IMAGE_PROCESSING] = [
            RecoveryStrategy(
                name="retry_with_different_seed",
                description="Retry processing with different random seed",
                action=self._retry_with_different_seed,
                success_probability=0.5,
                resource_cost="low",
                user_impact="none"
            ),
            RecoveryStrategy(
                name="adjust_generation_parameters",
                description="Adjust inference steps and guidance scale",
                action=self._adjust_generation_parameters,
                success_probability=0.7,
                resource_cost="low",
                user_impact="minimal"
            ),
            RecoveryStrategy(
                name="disable_eligen",
                description="Disable EliGen enhancement and retry",
                action=self._disable_eligen_retry,
                success_probability=0.8,
                resource_cost="low",
                user_impact="moderate"
            ),
            RecoveryStrategy(
                name="fallback_to_basic_processing",
                description="Use basic processing without advanced features",
                action=self._fallback_to_basic_processing,
                success_probability=0.9,
                resource_cost="low",
                user_impact="high"
            )
        ]
        
        # ControlNet error recovery strategies
        self.recovery_strategies[DiffSynthErrorType.CONTROLNET_PROCESSING] = [
            RecoveryStrategy(
                name="retry_controlnet_detection",
                description="Retry ControlNet feature detection",
                action=self._retry_controlnet_detection,
                success_probability=0.6,
                resource_cost="medium",
                user_impact="minimal"
            ),
            RecoveryStrategy(
                name="use_alternative_controlnet",
                description="Try alternative ControlNet model",
                action=self._use_alternative_controlnet,
                success_probability=0.7,
                resource_cost="medium",
                user_impact="minimal"
            ),
            RecoveryStrategy(
                name="disable_controlnet",
                description="Process without ControlNet guidance",
                action=self._disable_controlnet_processing,
                success_probability=0.95,
                resource_cost="low",
                user_impact="high"
            )
        ]
        
        # Tiled processing error recovery strategies
        self.recovery_strategies[DiffSynthErrorType.TILED_PROCESSING_ERROR] = [
            RecoveryStrategy(
                name="adjust_tile_size",
                description="Reduce tile size for processing",
                action=self._adjust_tile_size,
                success_probability=0.8,
                resource_cost="low",
                user_impact="minimal"
            ),
            RecoveryStrategy(
                name="disable_tiled_processing",
                description="Process image as single tile",
                action=self._disable_tiled_processing,
                success_probability=0.6,
                resource_cost="high",
                user_impact="moderate"
            )
        ]
    
    def handle_error(
        self,
        error: Exception,
        context: DiffSynthErrorContext,
        auto_recover: bool = True
    ) -> DiffSynthError:
        """
        Handle DiffSynth error with automatic recovery attempts
        
        Args:
            error: Original exception
            context: Error context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            DiffSynthError with recovery information
        """
        # Convert to DiffSynth error
        diffsynth_error = self._convert_to_diffsynth_error(error, context)
        
        # Add to error history
        self.error_history.append(diffsynth_error)
        
        # Track error patterns
        self._track_error_pattern(diffsynth_error)
        
        # Log error
        self._log_diffsynth_error(diffsynth_error)
        
        # Notify user
        self._notify_user_of_error(diffsynth_error)
        
        # Attempt recovery if enabled
        if auto_recover:
            recovery_success = self._attempt_recovery(diffsynth_error)
            if recovery_success:
                self._notify_user(
                    f"✅ Recovered from {diffsynth_error.error_type.value} error",
                    "success"
                )
            else:
                self._notify_user(
                    f"❌ Failed to recover from {diffsynth_error.error_type.value} error",
                    "error"
                )
        
        return diffsynth_error
    
    def _convert_to_diffsynth_error(
        self,
        error: Exception,
        context: DiffSynthErrorContext
    ) -> DiffSynthError:
        """Convert generic exception to DiffSynth-specific error"""
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Determine error type based on context and error content
        if context.operation_type == "initialization":
            if "import" in error_str or "module" in error_str:
                return DiffSynthServiceError(
                    f"DiffSynth import failed: {error}",
                    context=context,
                    original_error=error,
                    suggested_fixes=[
                        "Install DiffSynth-Studio: pip install diffsynth-studio",
                        "Check Python environment compatibility",
                        "Verify PyTorch installation"
                    ]
                )
            else:
                return DiffSynthServiceError(
                    f"Service initialization failed: {error}",
                    context=context,
                    original_error=error
                )
        
        elif "memory" in error_str or "cuda out of memory" in error_str:
            return DiffSynthMemoryError(
                f"Memory allocation failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Reduce image resolution",
                    "Enable CPU offloading",
                    "Use tiled processing",
                    "Close other GPU applications"
                ]
            )
        
        elif context.controlnet_type:
            return DiffSynthControlNetError(
                f"ControlNet processing failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Check ControlNet model compatibility",
                    "Verify control image format",
                    "Try alternative ControlNet type"
                ]
            )
        
        elif context.operation_type == "inpainting":
            return DiffSynthInpaintingError(
                f"Inpainting failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Check mask image format and size",
                    "Verify mask-image alignment",
                    "Reduce inpainting strength"
                ]
            )
        
        elif context.operation_type == "outpainting":
            return DiffSynthOutpaintingError(
                f"Outpainting failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Reduce outpainting dimensions",
                    "Check available memory",
                    "Use tiled processing"
                ]
            )
        
        elif context.operation_type == "style_transfer":
            return DiffSynthStyleTransferError(
                f"Style transfer failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Check style image compatibility",
                    "Adjust style transfer strength",
                    "Verify image formats"
                ]
            )
        
        elif context.tiled_processing:
            return DiffSynthTiledProcessingError(
                f"Tiled processing failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Adjust tile size",
                    "Check tile overlap settings",
                    "Disable tiled processing"
                ]
            )
        
        elif context.eligen_enabled:
            return DiffSynthEliGenError(
                f"EliGen processing failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Disable EliGen enhancement",
                    "Check EliGen configuration",
                    "Reduce EliGen quality settings"
                ]
            )
        
        else:
            return DiffSynthProcessingError(
                f"Image processing failed: {error}",
                context=context,
                original_error=error,
                suggested_fixes=[
                    "Check input image format",
                    "Verify processing parameters",
                    "Try with different settings"
                ]
            )
    
    def _track_error_pattern(self, error: DiffSynthError) -> None:
        """Track error patterns for analysis"""
        pattern_key = f"{error.error_type.value}_{error.context.operation_type}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
        
        # Track consecutive failures
        self.consecutive_failures[error.error_type] = self.consecutive_failures.get(error.error_type, 0) + 1
        
        # Reset consecutive failures for other error types
        for error_type in DiffSynthErrorType:
            if error_type != error.error_type:
                self.consecutive_failures[error_type] = 0
    
    def _log_diffsynth_error(self, error: DiffSynthError) -> None:
        """Log DiffSynth error with context"""
        logger.error(
            f"🚨 DiffSynth Error [{error.error_id}]: {error.error_type.value} - {error}"
        )
        logger.error(f"   Context: {error.context.operation_type}")
        if error.context.model_name:
            logger.error(f"   Model: {error.context.model_name}")
        if error.context.image_dimensions:
            logger.error(f"   Image: {error.context.image_dimensions}")
        if error.suggested_fixes:
            logger.error(f"   Suggested fixes: {', '.join(error.suggested_fixes[:3])}")
    
    def _notify_user_of_error(self, error: DiffSynthError) -> None:
        """Notify user of error with appropriate message"""
        severity_map = {
            DiffSynthErrorSeverity.RECOVERABLE: "warning",
            DiffSynthErrorSeverity.DEGRADED: "warning", 
            DiffSynthErrorSeverity.CRITICAL: "error",
            DiffSynthErrorSeverity.FATAL: "error"
        }
        
        user_message = self._create_user_friendly_message(error)
        severity = severity_map.get(error.severity, "error")
        
        self._notify_user(user_message, severity)
    
    def _create_user_friendly_message(self, error: DiffSynthError) -> str:
        """Create user-friendly error message"""
        operation_names = {
            "edit": "image editing",
            "inpainting": "inpainting",
            "outpainting": "outpainting", 
            "style_transfer": "style transfer",
            "controlnet": "ControlNet processing",
            "initialization": "service startup"
        }
        
        operation = operation_names.get(error.context.operation_type, error.context.operation_type)
        
        if error.severity == DiffSynthErrorSeverity.RECOVERABLE:
            return f"⚠️ {operation.title()} encountered an issue. Attempting automatic recovery..."
        elif error.severity == DiffSynthErrorSeverity.DEGRADED:
            return f"🔧 {operation.title()} completed with reduced functionality. Some features may be unavailable."
        elif error.severity == DiffSynthErrorSeverity.CRITICAL:
            return f"❌ {operation.title()} failed. Please check your settings and try again."
        else:
            return f"💥 Critical error in {operation}. Service may need to be restarted."
    
    def _attempt_recovery(self, error: DiffSynthError) -> bool:
        """Attempt recovery using available strategies"""
        strategies = self.recovery_strategies.get(error.error_type, [])
        
        if not strategies:
            logger.warning(f"No recovery strategies available for {error.error_type.value}")
            return False
        
        # Sort strategies by success probability
        strategies.sort(key=lambda s: s.success_probability, reverse=True)
        
        for strategy in strategies:
            try:
                logger.info(f"🔄 Attempting recovery: {strategy.name}")
                success = strategy.action(error)
                
                if success:
                    logger.info(f"✅ Recovery successful: {strategy.name}")
                    return True
                else:
                    logger.warning(f"❌ Recovery failed: {strategy.name}")
                    
            except Exception as e:
                logger.error(f"❌ Recovery strategy {strategy.name} raised exception: {e}")
        
        logger.error(f"❌ All recovery strategies failed for {error.error_type.value}")
        return False
    
    # Recovery action implementations
    def _retry_service_initialization(self, error: DiffSynthError) -> bool:
        """Retry service initialization with clean state"""
        # This would be implemented by the service itself
        return False
    
    def _fallback_to_cpu_initialization(self, error: DiffSynthError) -> bool:
        """Fallback to CPU initialization"""
        # This would be implemented by the service itself
        return False
    
    def _reduce_memory_initialization(self, error: DiffSynthError) -> bool:
        """Initialize with reduced memory configuration"""
        # This would be implemented by the service itself
        return False
    
    def _clear_gpu_cache(self, error: DiffSynthError) -> bool:
        """Clear GPU memory cache"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return True
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")
        return False
    
    def _enable_cpu_offload(self, error: DiffSynthError) -> bool:
        """Enable CPU offloading"""
        # This would be implemented by the service itself
        return False
    
    def _reduce_image_resolution(self, error: DiffSynthError) -> bool:
        """Reduce image resolution for processing"""
        # This would be implemented by the service itself
        return False
    
    def _enable_tiled_processing(self, error: DiffSynthError) -> bool:
        """Enable tiled processing"""
        # This would be implemented by the service itself
        return False
    
    def _retry_with_different_seed(self, error: DiffSynthError) -> bool:
        """Retry with different random seed"""
        # This would be implemented by the service itself
        return False
    
    def _adjust_generation_parameters(self, error: DiffSynthError) -> bool:
        """Adjust generation parameters"""
        # This would be implemented by the service itself
        return False
    
    def _disable_eligen_retry(self, error: DiffSynthError) -> bool:
        """Disable EliGen and retry"""
        # This would be implemented by the service itself
        return False
    
    def _fallback_to_basic_processing(self, error: DiffSynthError) -> bool:
        """Fallback to basic processing"""
        # This would be implemented by the service itself
        return False
    
    def _retry_controlnet_detection(self, error: DiffSynthError) -> bool:
        """Retry ControlNet detection"""
        # This would be implemented by the ControlNet service
        return False
    
    def _use_alternative_controlnet(self, error: DiffSynthError) -> bool:
        """Use alternative ControlNet model"""
        # This would be implemented by the ControlNet service
        return False
    
    def _disable_controlnet_processing(self, error: DiffSynthError) -> bool:
        """Disable ControlNet processing"""
        # This would be implemented by the ControlNet service
        return False
    
    def _adjust_tile_size(self, error: DiffSynthError) -> bool:
        """Adjust tile size for processing"""
        # This would be implemented by the tiled processor
        return False
    
    def _disable_tiled_processing(self, error: DiffSynthError) -> bool:
        """Disable tiled processing"""
        # This would be implemented by the tiled processor
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {"total_errors": 0, "error_patterns": {}, "consecutive_failures": {}}
        
        # Error type distribution
        error_type_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_type_counts[error.error_type.value] = error_type_counts.get(error.error_type.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": total_errors,
            "error_type_distribution": error_type_counts,
            "severity_distribution": severity_counts,
            "error_patterns": dict(self.error_patterns),
            "consecutive_failures": {k.value: v for k, v in self.consecutive_failures.items()},
            "recent_errors": [
                {
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "operation": error.context.operation_type,
                    "timestamp": error.timestamp
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def create_error_report(self) -> Dict[str, Any]:
        """Create comprehensive error report"""
        stats = self.get_error_statistics()
        
        return {
            "report_timestamp": time.time(),
            "error_statistics": stats,
            "recovery_strategies_available": {
                error_type.value: len(strategies)
                for error_type, strategies in self.recovery_strategies.items()
            },
            "system_health": {
                "consecutive_failures": max(self.consecutive_failures.values()) if self.consecutive_failures else 0,
                "error_rate": len(self.error_history) / max(1, (time.time() - self.error_history[0].timestamp) / 3600) if self.error_history else 0,
                "critical_errors": sum(1 for e in self.error_history if e.severity in [DiffSynthErrorSeverity.CRITICAL, DiffSynthErrorSeverity.FATAL])
            }
        }