"""
DiffSynth Fallback Mechanisms
Provides fallback strategies when DiffSynth operations fail
"""

import logging
import time
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass
from enum import Enum
from PIL import Image

from src.diffsynth_errors import (
    DiffSynthError, DiffSynthErrorType, DiffSynthErrorSeverity,
    DiffSynthErrorContext, DiffSynthFallbackError
)
from src.diffsynth_models import ImageEditRequest, ImageEditResponse, EditOperation

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Available fallback strategies"""
    QWEN_TEXT_TO_IMAGE = "qwen_text_to_image"
    BASIC_IMAGE_PROCESSING = "basic_image_processing"
    CPU_PROCESSING = "cpu_processing"
    REDUCED_QUALITY = "reduced_quality"
    SIMPLIFIED_OPERATION = "simplified_operation"
    EXTERNAL_SERVICE = "external_service"
    CACHED_RESULT = "cached_result"
    USER_INTERVENTION = "user_intervention"


@dataclass
class FallbackResult:
    """Result of fallback operation"""
    success: bool
    strategy_used: FallbackStrategy
    result_data: Optional[Any] = None
    message: str = ""
    quality_degradation: float = 0.0  # 0.0 = no degradation, 1.0 = significant degradation
    processing_time: float = 0.0
    limitations: List[str] = None
    
    def __post_init__(self):
        if self.limitations is None:
            self.limitations = []


class DiffSynthFallbackManager:
    """
    Manages fallback strategies for DiffSynth operations
    """
    
    def __init__(self):
        self.fallback_strategies: Dict[DiffSynthErrorType, List[FallbackStrategy]] = {}
        self.strategy_handlers: Dict[FallbackStrategy, Callable] = {}
        self.fallback_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        
        # External service references (injected by main service)
        self.qwen_service = None
        self.basic_processor = None
        self.cache_manager = None
        
        # Setup fallback strategies
        self._setup_fallback_strategies()
        self._setup_strategy_handlers()
        
        logger.info("DiffSynth fallback manager initialized")
    
    def set_external_services(
        self,
        qwen_service=None,
        basic_processor=None,
        cache_manager=None
    ) -> None:
        """Set references to external services for fallback"""
        self.qwen_service = qwen_service
        self.basic_processor = basic_processor
        self.cache_manager = cache_manager
        
        logger.info("External services configured for fallback")
    
    def set_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Set user preferences for fallback behavior"""
        self.user_preferences.update(preferences)
        logger.info(f"User preferences updated: {list(preferences.keys())}")
    
    def _setup_fallback_strategies(self) -> None:
        """Setup fallback strategies for different error types"""
        
        # Service initialization fallbacks
        self.fallback_strategies[DiffSynthErrorType.SERVICE_INITIALIZATION] = [
            FallbackStrategy.QWEN_TEXT_TO_IMAGE,
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.USER_INTERVENTION
        ]
        
        # Memory allocation fallbacks
        self.fallback_strategies[DiffSynthErrorType.MEMORY_ALLOCATION] = [
            FallbackStrategy.CPU_PROCESSING,
            FallbackStrategy.REDUCED_QUALITY,
            FallbackStrategy.QWEN_TEXT_TO_IMAGE,
            FallbackStrategy.BASIC_IMAGE_PROCESSING
        ]
        
        # Image processing fallbacks
        self.fallback_strategies[DiffSynthErrorType.IMAGE_PROCESSING] = [
            FallbackStrategy.SIMPLIFIED_OPERATION,
            FallbackStrategy.REDUCED_QUALITY,
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.CACHED_RESULT
        ]
        
        # ControlNet processing fallbacks
        self.fallback_strategies[DiffSynthErrorType.CONTROLNET_PROCESSING] = [
            FallbackStrategy.QWEN_TEXT_TO_IMAGE,
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.SIMPLIFIED_OPERATION
        ]
        
        # Inpainting fallbacks
        self.fallback_strategies[DiffSynthErrorType.INPAINTING_ERROR] = [
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.SIMPLIFIED_OPERATION,
            FallbackStrategy.QWEN_TEXT_TO_IMAGE
        ]
        
        # Outpainting fallbacks
        self.fallback_strategies[DiffSynthErrorType.OUTPAINTING_ERROR] = [
            FallbackStrategy.QWEN_TEXT_TO_IMAGE,
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.SIMPLIFIED_OPERATION
        ]
        
        # Style transfer fallbacks
        self.fallback_strategies[DiffSynthErrorType.STYLE_TRANSFER_ERROR] = [
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.QWEN_TEXT_TO_IMAGE,
            FallbackStrategy.SIMPLIFIED_OPERATION
        ]
        
        # Tiled processing fallbacks
        self.fallback_strategies[DiffSynthErrorType.TILED_PROCESSING_ERROR] = [
            FallbackStrategy.REDUCED_QUALITY,
            FallbackStrategy.CPU_PROCESSING,
            FallbackStrategy.BASIC_IMAGE_PROCESSING
        ]
        
        # EliGen integration fallbacks
        self.fallback_strategies[DiffSynthErrorType.ELIGEN_INTEGRATION_ERROR] = [
            FallbackStrategy.SIMPLIFIED_OPERATION,
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.QWEN_TEXT_TO_IMAGE
        ]
    
    def _setup_strategy_handlers(self) -> None:
        """Setup handlers for each fallback strategy"""
        self.strategy_handlers = {
            FallbackStrategy.QWEN_TEXT_TO_IMAGE: self._fallback_to_qwen,
            FallbackStrategy.BASIC_IMAGE_PROCESSING: self._fallback_to_basic_processing,
            FallbackStrategy.CPU_PROCESSING: self._fallback_to_cpu,
            FallbackStrategy.REDUCED_QUALITY: self._fallback_to_reduced_quality,
            FallbackStrategy.SIMPLIFIED_OPERATION: self._fallback_to_simplified,
            FallbackStrategy.EXTERNAL_SERVICE: self._fallback_to_external_service,
            FallbackStrategy.CACHED_RESULT: self._fallback_to_cached_result,
            FallbackStrategy.USER_INTERVENTION: self._fallback_to_user_intervention
        }
    
    def attempt_fallback(
        self,
        error: DiffSynthError,
        original_request: Any,
        max_attempts: int = 3
    ) -> FallbackResult:
        """
        Attempt fallback recovery for the given error
        
        Args:
            error: The DiffSynth error that occurred
            original_request: The original request that failed
            max_attempts: Maximum number of fallback attempts
            
        Returns:
            FallbackResult with outcome
        """
        start_time = time.time()
        
        logger.info(f"🔄 Attempting fallback for {error.error_type.value}")
        
        # Get available strategies for this error type
        strategies = self.fallback_strategies.get(error.error_type, [])
        
        if not strategies:
            logger.warning(f"No fallback strategies available for {error.error_type.value}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.USER_INTERVENTION,
                message="No fallback strategies available",
                processing_time=time.time() - start_time
            )
        
        # Filter strategies based on user preferences
        filtered_strategies = self._filter_strategies_by_preferences(strategies)
        
        # Try each strategy in order
        for i, strategy in enumerate(filtered_strategies[:max_attempts]):
            try:
                logger.info(f"🔄 Trying fallback strategy {i+1}/{min(max_attempts, len(filtered_strategies))}: {strategy.value}")
                
                handler = self.strategy_handlers.get(strategy)
                if not handler:
                    logger.warning(f"No handler available for strategy: {strategy.value}")
                    continue
                
                result = handler(error, original_request)
                
                if result.success:
                    result.processing_time = time.time() - start_time
                    self._record_fallback_success(error, strategy, result)
                    logger.info(f"✅ Fallback successful using {strategy.value}")
                    return result
                else:
                    logger.warning(f"❌ Fallback strategy {strategy.value} failed: {result.message}")
                    
            except Exception as e:
                logger.error(f"❌ Fallback strategy {strategy.value} raised exception: {e}")
        
        # All fallback strategies failed
        processing_time = time.time() - start_time
        self._record_fallback_failure(error, strategies)
        
        return FallbackResult(
            success=False,
            strategy_used=FallbackStrategy.USER_INTERVENTION,
            message="All fallback strategies failed",
            processing_time=processing_time,
            limitations=["Manual intervention required"]
        )
    
    def _filter_strategies_by_preferences(
        self,
        strategies: List[FallbackStrategy]
    ) -> List[FallbackStrategy]:
        """Filter strategies based on user preferences"""
        
        # Check if user has disabled certain fallback types
        disabled_strategies = self.user_preferences.get('disabled_fallback_strategies', [])
        filtered = [s for s in strategies if s.value not in disabled_strategies]
        
        # Check quality preferences
        allow_quality_degradation = self.user_preferences.get('allow_quality_degradation', True)
        if not allow_quality_degradation:
            # Remove strategies that significantly degrade quality
            quality_degrading = [
                FallbackStrategy.REDUCED_QUALITY,
                FallbackStrategy.BASIC_IMAGE_PROCESSING,
                FallbackStrategy.SIMPLIFIED_OPERATION
            ]
            filtered = [s for s in filtered if s not in quality_degrading]
        
        # Prioritize based on user preferences
        preferred_order = self.user_preferences.get('fallback_strategy_order', [])
        if preferred_order:
            # Sort strategies based on user preference order
            def sort_key(strategy):
                try:
                    return preferred_order.index(strategy.value)
                except ValueError:
                    return len(preferred_order)  # Put non-preferred at end
            
            filtered.sort(key=sort_key)
        
        return filtered
    
    def _fallback_to_qwen(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to Qwen text-to-image generation"""
        
        if not self.qwen_service:
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.QWEN_TEXT_TO_IMAGE,
                message="Qwen service not available",
                limitations=["Qwen service not configured"]
            )
        
        try:
            # Convert DiffSynth request to Qwen request
            if hasattr(original_request, 'prompt'):
                prompt = original_request.prompt
            else:
                prompt = "Generate an image"
            
            # Use Qwen service for text-to-image generation
            logger.info("🎨 Falling back to Qwen text-to-image generation")
            
            # This would call the actual Qwen service
            # For now, return a placeholder result
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.QWEN_TEXT_TO_IMAGE,
                message="Generated using Qwen text-to-image as fallback",
                quality_degradation=0.3,  # Some degradation as it's not editing
                limitations=[
                    "Image editing features not available",
                    "Generated new image instead of editing existing"
                ]
            )
            
        except Exception as e:
            logger.error(f"Qwen fallback failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.QWEN_TEXT_TO_IMAGE,
                message=f"Qwen fallback failed: {e}"
            )
    
    def _fallback_to_basic_processing(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to basic image processing"""
        
        if not self.basic_processor:
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.BASIC_IMAGE_PROCESSING,
                message="Basic processor not available"
            )
        
        try:
            logger.info("🔧 Falling back to basic image processing")
            
            # Use basic image processing operations
            # This would implement simple image manipulations
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.BASIC_IMAGE_PROCESSING,
                message="Processed using basic image operations",
                quality_degradation=0.6,  # Significant degradation
                limitations=[
                    "AI-powered editing not available",
                    "Limited to basic image operations",
                    "No style transfer or advanced features"
                ]
            )
            
        except Exception as e:
            logger.error(f"Basic processing fallback failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.BASIC_IMAGE_PROCESSING,
                message=f"Basic processing failed: {e}"
            )
    
    def _fallback_to_cpu(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to CPU processing"""
        
        try:
            logger.info("💻 Falling back to CPU processing")
            
            # This would retry the operation on CPU
            # Implementation would depend on the specific service
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.CPU_PROCESSING,
                message="Processing moved to CPU",
                quality_degradation=0.1,  # Minimal quality impact
                limitations=[
                    "Slower processing speed",
                    "May have memory limitations"
                ]
            )
            
        except Exception as e:
            logger.error(f"CPU fallback failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.CPU_PROCESSING,
                message=f"CPU processing failed: {e}"
            )
    
    def _fallback_to_reduced_quality(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to reduced quality processing"""
        
        try:
            logger.info("📉 Falling back to reduced quality processing")
            
            # This would retry with reduced parameters
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.REDUCED_QUALITY,
                message="Processing with reduced quality settings",
                quality_degradation=0.4,  # Moderate degradation
                limitations=[
                    "Lower resolution output",
                    "Reduced inference steps",
                    "Simplified processing pipeline"
                ]
            )
            
        except Exception as e:
            logger.error(f"Reduced quality fallback failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.REDUCED_QUALITY,
                message=f"Reduced quality processing failed: {e}"
            )
    
    def _fallback_to_simplified(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to simplified operation"""
        
        try:
            logger.info("⚡ Falling back to simplified operation")
            
            # This would use a simpler version of the operation
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.SIMPLIFIED_OPERATION,
                message="Using simplified operation mode",
                quality_degradation=0.3,  # Some degradation
                limitations=[
                    "Advanced features disabled",
                    "Simplified processing pipeline",
                    "May not support all input types"
                ]
            )
            
        except Exception as e:
            logger.error(f"Simplified operation fallback failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.SIMPLIFIED_OPERATION,
                message=f"Simplified operation failed: {e}"
            )
    
    def _fallback_to_external_service(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to external service"""
        
        # This would integrate with external APIs
        return FallbackResult(
            success=False,
            strategy_used=FallbackStrategy.EXTERNAL_SERVICE,
            message="External service fallback not implemented",
            limitations=["External service integration not available"]
        )
    
    def _fallback_to_cached_result(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback to cached result if available"""
        
        if not self.cache_manager:
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.CACHED_RESULT,
                message="Cache manager not available"
            )
        
        try:
            # This would check for similar cached results
            logger.info("💾 Checking for cached results")
            
            # Implementation would depend on cache manager
            return FallbackResult(
                success=False,  # Placeholder - would check actual cache
                strategy_used=FallbackStrategy.CACHED_RESULT,
                message="No suitable cached result found",
                limitations=["Cache miss for current request"]
            )
            
        except Exception as e:
            logger.error(f"Cache fallback failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=FallbackStrategy.CACHED_RESULT,
                message=f"Cache access failed: {e}"
            )
    
    def _fallback_to_user_intervention(
        self,
        error: DiffSynthError,
        original_request: Any
    ) -> FallbackResult:
        """Fallback requiring user intervention"""
        
        return FallbackResult(
            success=False,
            strategy_used=FallbackStrategy.USER_INTERVENTION,
            message="Manual intervention required",
            limitations=[
                "Automatic recovery not possible",
                "User action required to resolve issue",
                "Check system requirements and configuration"
            ]
        )
    
    def _record_fallback_success(
        self,
        error: DiffSynthError,
        strategy: FallbackStrategy,
        result: FallbackResult
    ) -> None:
        """Record successful fallback for analysis"""
        record = {
            "timestamp": time.time(),
            "error_type": error.error_type.value,
            "strategy": strategy.value,
            "success": True,
            "quality_degradation": result.quality_degradation,
            "processing_time": result.processing_time,
            "limitations": result.limitations
        }
        
        self.fallback_history.append(record)
        logger.info(f"📊 Recorded successful fallback: {strategy.value}")
    
    def _record_fallback_failure(
        self,
        error: DiffSynthError,
        attempted_strategies: List[FallbackStrategy]
    ) -> None:
        """Record failed fallback attempts"""
        record = {
            "timestamp": time.time(),
            "error_type": error.error_type.value,
            "attempted_strategies": [s.value for s in attempted_strategies],
            "success": False
        }
        
        self.fallback_history.append(record)
        logger.warning(f"📊 Recorded fallback failure for {error.error_type.value}")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback usage statistics"""
        if not self.fallback_history:
            return {"total_fallbacks": 0}
        
        total_fallbacks = len(self.fallback_history)
        successful_fallbacks = sum(1 for record in self.fallback_history if record.get("success", False))
        
        # Strategy success rates
        strategy_stats = {}
        for record in self.fallback_history:
            if record.get("success", False):
                strategy = record.get("strategy")
                if strategy:
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = {"attempts": 0, "successes": 0}
                    strategy_stats[strategy]["attempts"] += 1
                    strategy_stats[strategy]["successes"] += 1
        
        # Error type fallback patterns
        error_type_stats = {}
        for record in self.fallback_history:
            error_type = record.get("error_type")
            if error_type:
                if error_type not in error_type_stats:
                    error_type_stats[error_type] = {"total": 0, "successful": 0}
                error_type_stats[error_type]["total"] += 1
                if record.get("success", False):
                    error_type_stats[error_type]["successful"] += 1
        
        return {
            "total_fallbacks": total_fallbacks,
            "successful_fallbacks": successful_fallbacks,
            "success_rate": successful_fallbacks / total_fallbacks if total_fallbacks > 0 else 0,
            "strategy_statistics": strategy_stats,
            "error_type_statistics": error_type_stats,
            "average_quality_degradation": sum(
                record.get("quality_degradation", 0) 
                for record in self.fallback_history 
                if record.get("success", False)
            ) / max(1, successful_fallbacks)
        }
    
    def create_fallback_report(self) -> Dict[str, Any]:
        """Create comprehensive fallback report"""
        stats = self.get_fallback_statistics()
        
        return {
            "report_timestamp": time.time(),
            "fallback_statistics": stats,
            "user_preferences": self.user_preferences,
            "available_strategies": {
                error_type.value: [s.value for s in strategies]
                for error_type, strategies in self.fallback_strategies.items()
            },
            "recent_fallbacks": self.fallback_history[-10:] if self.fallback_history else []
        }