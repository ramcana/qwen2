"""
Tests for DiffSynth Error Handling System
Comprehensive tests for error handling, recovery mechanisms, and fallback strategies
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.diffsynth_errors import (
    DiffSynthError, DiffSynthErrorType, DiffSynthErrorSeverity,
    DiffSynthErrorContext, DiffSynthErrorHandler,
    DiffSynthServiceError, DiffSynthMemoryError, DiffSynthProcessingError,
    DiffSynthControlNetError, DiffSynthInpaintingError, DiffSynthOutpaintingError,
    DiffSynthStyleTransferError, DiffSynthTiledProcessingError,
    DiffSynthEliGenError, DiffSynthResourceError, DiffSynthFallbackError
)
from src.diffsynth_fallback import (
    DiffSynthFallbackManager, FallbackStrategy, FallbackResult
)


class TestDiffSynthErrorHierarchy:
    """Test DiffSynth error class hierarchy"""
    
    def test_base_diffsynth_error_creation(self):
        """Test creating base DiffSynth error"""
        context = DiffSynthErrorContext(operation_type="test")
        error = DiffSynthError(
            message="Test error",
            error_type=DiffSynthErrorType.IMAGE_PROCESSING,
            severity=DiffSynthErrorSeverity.RECOVERABLE,
            context=context
        )
        
        assert str(error) == "Test error"
        assert error.error_type == DiffSynthErrorType.IMAGE_PROCESSING
        assert error.severity == DiffSynthErrorSeverity.RECOVERABLE
        assert error.context.operation_type == "test"
        assert error.error_id.startswith("diffsynth_")
        assert error.timestamp > 0
    
    def test_service_error_creation(self):
        """Test DiffSynth service error creation"""
        error = DiffSynthServiceError("Service failed to initialize")
        
        assert error.error_type == DiffSynthErrorType.SERVICE_INITIALIZATION
        assert error.severity == DiffSynthErrorSeverity.CRITICAL
        assert "Service failed to initialize" in str(error)
    
    def test_memory_error_creation(self):
        """Test DiffSynth memory error creation"""
        context = DiffSynthErrorContext(
            operation_type="edit",
            memory_usage_gb=8.5
        )
        error = DiffSynthMemoryError(
            "Out of GPU memory",
            context=context,
            suggested_fixes=["Reduce image size", "Enable CPU offload"]
        )
        
        assert error.error_type == DiffSynthErrorType.MEMORY_ALLOCATION
        assert error.severity == DiffSynthErrorSeverity.DEGRADED
        assert error.context.memory_usage_gb == 8.5
        assert len(error.suggested_fixes) == 2
    
    def test_controlnet_error_creation(self):
        """Test ControlNet error creation"""
        context = DiffSynthErrorContext(
            operation_type="controlnet",
            controlnet_type="canny"
        )
        error = DiffSynthControlNetError(
            "ControlNet detection failed",
            context=context
        )
        
        assert error.error_type == DiffSynthErrorType.CONTROLNET_PROCESSING
        assert error.context.controlnet_type == "canny"
    
    def test_error_context_creation(self):
        """Test error context creation with various parameters"""
        context = DiffSynthErrorContext(
            operation_type="inpainting",
            model_name="Qwen/Qwen-Image-Edit",
            image_dimensions=(1024, 1024),
            memory_usage_gb=4.2,
            processing_time=15.5,
            eligen_enabled=True,
            tiled_processing=True,
            controlnet_type="depth",
            fallback_attempted=False
        )
        
        assert context.operation_type == "inpainting"
        assert context.model_name == "Qwen/Qwen-Image-Edit"
        assert context.image_dimensions == (1024, 1024)
        assert context.memory_usage_gb == 4.2
        assert context.processing_time == 15.5
        assert context.eligen_enabled is True
        assert context.tiled_processing is True
        assert context.controlnet_type == "depth"
        assert context.fallback_attempted is False


class TestDiffSynthErrorHandler:
    """Test DiffSynth error handler functionality"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing"""
        return DiffSynthErrorHandler()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample error context"""
        return DiffSynthErrorContext(
            operation_type="edit",
            model_name="test-model",
            image_dimensions=(512, 512)
        )
    
    def test_error_handler_initialization(self, error_handler):
        """Test error handler initialization"""
        assert error_handler is not None
        assert len(error_handler.recovery_strategies) > 0
        assert len(error_handler.error_history) == 0
        assert len(error_handler.error_patterns) == 0
    
    def test_handle_service_initialization_error(self, error_handler, sample_context):
        """Test handling service initialization errors"""
        sample_context.operation_type = "initialization"
        original_error = ImportError("No module named 'diffsynth'")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthServiceError)
        assert diffsynth_error.error_type == DiffSynthErrorType.SERVICE_INITIALIZATION
        assert "DiffSynth import failed" in str(diffsynth_error)
        assert len(diffsynth_error.suggested_fixes) > 0
        assert "pip install diffsynth-studio" in diffsynth_error.suggested_fixes[0]
    
    def test_handle_memory_error(self, error_handler, sample_context):
        """Test handling memory errors"""
        original_error = RuntimeError("CUDA out of memory")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthMemoryError)
        assert diffsynth_error.error_type == DiffSynthErrorType.MEMORY_ALLOCATION
        assert "Memory allocation failed" in str(diffsynth_error)
        assert any("Reduce image resolution" in fix for fix in diffsynth_error.suggested_fixes)
    
    def test_handle_controlnet_error(self, error_handler, sample_context):
        """Test handling ControlNet errors"""
        sample_context.controlnet_type = "canny"
        original_error = ValueError("Invalid control image format")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthControlNetError)
        assert diffsynth_error.error_type == DiffSynthErrorType.CONTROLNET_PROCESSING
        assert diffsynth_error.context.controlnet_type == "canny"
    
    def test_handle_inpainting_error(self, error_handler, sample_context):
        """Test handling inpainting errors"""
        sample_context.operation_type = "inpainting"
        original_error = ValueError("Mask size mismatch")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthInpaintingError)
        assert diffsynth_error.error_type == DiffSynthErrorType.INPAINTING_ERROR
        assert any("mask" in fix.lower() for fix in diffsynth_error.suggested_fixes)
    
    def test_handle_outpainting_error(self, error_handler, sample_context):
        """Test handling outpainting errors"""
        sample_context.operation_type = "outpainting"
        original_error = RuntimeError("Canvas expansion failed")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthOutpaintingError)
        assert diffsynth_error.error_type == DiffSynthErrorType.OUTPAINTING_ERROR
    
    def test_handle_style_transfer_error(self, error_handler, sample_context):
        """Test handling style transfer errors"""
        sample_context.operation_type = "style_transfer"
        original_error = ValueError("Style image incompatible")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthStyleTransferError)
        assert diffsynth_error.error_type == DiffSynthErrorType.STYLE_TRANSFER_ERROR
    
    def test_handle_tiled_processing_error(self, error_handler, sample_context):
        """Test handling tiled processing errors"""
        sample_context.tiled_processing = True
        original_error = RuntimeError("Tile processing failed")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthTiledProcessingError)
        assert diffsynth_error.error_type == DiffSynthErrorType.TILED_PROCESSING_ERROR
    
    def test_handle_eligen_error(self, error_handler, sample_context):
        """Test handling EliGen errors"""
        sample_context.eligen_enabled = True
        original_error = RuntimeError("EliGen enhancement failed")
        
        diffsynth_error = error_handler.handle_error(
            original_error, sample_context, auto_recover=False
        )
        
        assert isinstance(diffsynth_error, DiffSynthEliGenError)
        assert diffsynth_error.error_type == DiffSynthErrorType.ELIGEN_INTEGRATION_ERROR
    
    def test_error_pattern_tracking(self, error_handler, sample_context):
        """Test error pattern tracking"""
        # Generate multiple similar errors
        for i in range(3):
            error = RuntimeError(f"Memory error {i}")
            error_handler.handle_error(error, sample_context, auto_recover=False)
        
        # Check pattern tracking
        pattern_key = f"{DiffSynthErrorType.MEMORY_ALLOCATION.value}_edit"
        assert error_handler.error_patterns.get(pattern_key, 0) == 3
        assert error_handler.consecutive_failures[DiffSynthErrorType.MEMORY_ALLOCATION] == 3
    
    def test_error_statistics(self, error_handler, sample_context):
        """Test error statistics generation"""
        # Generate various errors
        errors = [
            RuntimeError("CUDA out of memory"),
            ValueError("Invalid input"),
            ImportError("Module not found")
        ]
        
        contexts = [
            sample_context,
            DiffSynthErrorContext(operation_type="controlnet"),
            DiffSynthErrorContext(operation_type="initialization")
        ]
        
        for error, context in zip(errors, contexts):
            error_handler.handle_error(error, context, auto_recover=False)
        
        stats = error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert len(stats["error_type_distribution"]) > 0
        assert len(stats["severity_distribution"]) > 0
        assert len(stats["recent_errors"]) == 3
    
    def test_error_report_generation(self, error_handler, sample_context):
        """Test comprehensive error report generation"""
        # Generate some errors
        error_handler.handle_error(
            RuntimeError("Test error"), sample_context, auto_recover=False
        )
        
        report = error_handler.create_error_report()
        
        assert "report_timestamp" in report
        assert "error_statistics" in report
        assert "recovery_strategies_available" in report
        assert "system_health" in report
        assert report["error_statistics"]["total_errors"] == 1
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_clear_gpu_cache_recovery(self, mock_synchronize, mock_empty_cache, mock_is_available, error_handler):
        """Test GPU cache clearing recovery action"""
        mock_is_available.return_value = True
        
        # Test the recovery action
        success = error_handler._clear_gpu_cache(Mock())
        
        assert success is True
        mock_empty_cache.assert_called_once()
        mock_synchronize.assert_called_once()
    
    def test_user_notification_callbacks(self, error_handler):
        """Test user notification callbacks"""
        notifications = []
        
        def capture_notification(message, severity):
            notifications.append((message, severity))
        
        error_handler.add_user_notification_callback(capture_notification)
        
        # Generate an error
        context = DiffSynthErrorContext(operation_type="test")
        error_handler.handle_error(
            RuntimeError("Test error"), context, auto_recover=False
        )
        
        assert len(notifications) > 0
        message, severity = notifications[0]
        assert "test" in message.lower()
        assert severity in ["info", "warning", "error"]


class TestDiffSynthFallbackManager:
    """Test DiffSynth fallback manager functionality"""
    
    @pytest.fixture
    def fallback_manager(self):
        """Create fallback manager for testing"""
        return DiffSynthFallbackManager()
    
    @pytest.fixture
    def sample_error(self):
        """Create sample error for testing"""
        context = DiffSynthErrorContext(operation_type="edit")
        return DiffSynthMemoryError("Out of memory", context=context)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample request for testing"""
        return Mock(prompt="Test prompt", width=512, height=512)
    
    def test_fallback_manager_initialization(self, fallback_manager):
        """Test fallback manager initialization"""
        assert fallback_manager is not None
        assert len(fallback_manager.fallback_strategies) > 0
        assert len(fallback_manager.strategy_handlers) > 0
        assert len(fallback_manager.fallback_history) == 0
    
    def test_set_external_services(self, fallback_manager):
        """Test setting external services"""
        qwen_service = Mock()
        basic_processor = Mock()
        cache_manager = Mock()
        
        fallback_manager.set_external_services(
            qwen_service=qwen_service,
            basic_processor=basic_processor,
            cache_manager=cache_manager
        )
        
        assert fallback_manager.qwen_service == qwen_service
        assert fallback_manager.basic_processor == basic_processor
        assert fallback_manager.cache_manager == cache_manager
    
    def test_set_user_preferences(self, fallback_manager):
        """Test setting user preferences"""
        preferences = {
            "allow_quality_degradation": False,
            "disabled_fallback_strategies": ["basic_image_processing"]
        }
        
        fallback_manager.set_user_preferences(preferences)
        
        assert fallback_manager.user_preferences["allow_quality_degradation"] is False
        assert "basic_image_processing" in fallback_manager.user_preferences["disabled_fallback_strategies"]
    
    def test_attempt_fallback_no_strategies(self, fallback_manager):
        """Test fallback attempt when no strategies available"""
        # Create error type with no strategies
        context = DiffSynthErrorContext(operation_type="unknown")
        error = DiffSynthError(
            "Unknown error",
            error_type=DiffSynthErrorType.FALLBACK_FAILURE,  # No strategies for this
            context=context
        )
        
        result = fallback_manager.attempt_fallback(error, Mock())
        
        assert result.success is False
        assert result.strategy_used == FallbackStrategy.USER_INTERVENTION
        assert "No fallback strategies available" in result.message
    
    def test_attempt_fallback_with_strategies(self, fallback_manager, sample_error, sample_request):
        """Test fallback attempt with available strategies"""
        # Mock a successful fallback strategy
        def mock_successful_fallback(error, request):
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.CPU_PROCESSING,
                message="Fallback successful"
            )
        
        fallback_manager.strategy_handlers[FallbackStrategy.CPU_PROCESSING] = mock_successful_fallback
        
        result = fallback_manager.attempt_fallback(sample_error, sample_request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.CPU_PROCESSING
        assert "Fallback successful" in result.message
    
    def test_filter_strategies_by_preferences(self, fallback_manager):
        """Test strategy filtering based on user preferences"""
        strategies = [
            FallbackStrategy.QWEN_TEXT_TO_IMAGE,
            FallbackStrategy.BASIC_IMAGE_PROCESSING,
            FallbackStrategy.REDUCED_QUALITY
        ]
        
        # Set preferences to disable quality degradation
        fallback_manager.set_user_preferences({
            "allow_quality_degradation": False
        })
        
        filtered = fallback_manager._filter_strategies_by_preferences(strategies)
        
        # Should filter out quality-degrading strategies
        assert FallbackStrategy.QWEN_TEXT_TO_IMAGE in filtered
        assert FallbackStrategy.BASIC_IMAGE_PROCESSING not in filtered
        assert FallbackStrategy.REDUCED_QUALITY not in filtered
    
    def test_qwen_fallback_success(self, fallback_manager, sample_error, sample_request):
        """Test successful Qwen fallback"""
        # Mock Qwen service
        qwen_service = Mock()
        fallback_manager.set_external_services(qwen_service=qwen_service)
        
        result = fallback_manager._fallback_to_qwen(sample_error, sample_request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.QWEN_TEXT_TO_IMAGE
        assert result.quality_degradation > 0  # Some degradation expected
        assert len(result.limitations) > 0
    
    def test_qwen_fallback_no_service(self, fallback_manager, sample_error, sample_request):
        """Test Qwen fallback when service not available"""
        # Don't set Qwen service
        result = fallback_manager._fallback_to_qwen(sample_error, sample_request)
        
        assert result.success is False
        assert result.strategy_used == FallbackStrategy.QWEN_TEXT_TO_IMAGE
        assert "Qwen service not available" in result.message
    
    def test_basic_processing_fallback(self, fallback_manager, sample_error, sample_request):
        """Test basic processing fallback"""
        # Mock basic processor
        basic_processor = Mock()
        fallback_manager.set_external_services(basic_processor=basic_processor)
        
        result = fallback_manager._fallback_to_basic_processing(sample_error, sample_request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.BASIC_IMAGE_PROCESSING
        assert result.quality_degradation > 0.5  # Significant degradation
    
    def test_cpu_fallback(self, fallback_manager, sample_error, sample_request):
        """Test CPU processing fallback"""
        result = fallback_manager._fallback_to_cpu(sample_error, sample_request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.CPU_PROCESSING
        assert result.quality_degradation < 0.2  # Minimal degradation
        assert any("Slower processing" in limitation for limitation in result.limitations)
    
    def test_reduced_quality_fallback(self, fallback_manager, sample_error, sample_request):
        """Test reduced quality fallback"""
        result = fallback_manager._fallback_to_reduced_quality(sample_error, sample_request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.REDUCED_QUALITY
        assert 0.3 <= result.quality_degradation <= 0.5  # Moderate degradation
    
    def test_simplified_operation_fallback(self, fallback_manager, sample_error, sample_request):
        """Test simplified operation fallback"""
        result = fallback_manager._fallback_to_simplified(sample_error, sample_request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.SIMPLIFIED_OPERATION
        assert result.quality_degradation > 0
        assert any("Advanced features disabled" in limitation for limitation in result.limitations)
    
    def test_cached_result_fallback_no_cache(self, fallback_manager, sample_error, sample_request):
        """Test cached result fallback when cache not available"""
        result = fallback_manager._fallback_to_cached_result(sample_error, sample_request)
        
        assert result.success is False
        assert result.strategy_used == FallbackStrategy.CACHED_RESULT
        assert "Cache manager not available" in result.message
    
    def test_user_intervention_fallback(self, fallback_manager, sample_error, sample_request):
        """Test user intervention fallback"""
        result = fallback_manager._fallback_to_user_intervention(sample_error, sample_request)
        
        assert result.success is False
        assert result.strategy_used == FallbackStrategy.USER_INTERVENTION
        assert "Manual intervention required" in result.message
        assert any("User action required" in limitation for limitation in result.limitations)
    
    def test_fallback_statistics_empty(self, fallback_manager):
        """Test fallback statistics when no fallbacks attempted"""
        stats = fallback_manager.get_fallback_statistics()
        
        assert stats["total_fallbacks"] == 0
    
    def test_fallback_statistics_with_data(self, fallback_manager, sample_error, sample_request):
        """Test fallback statistics with fallback data"""
        # Record some fallback attempts
        fallback_manager._record_fallback_success(
            sample_error,
            FallbackStrategy.CPU_PROCESSING,
            FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.CPU_PROCESSING,
                quality_degradation=0.1,
                processing_time=5.0
            )
        )
        
        fallback_manager._record_fallback_failure(
            sample_error,
            [FallbackStrategy.QWEN_TEXT_TO_IMAGE]
        )
        
        stats = fallback_manager.get_fallback_statistics()
        
        assert stats["total_fallbacks"] == 2
        assert stats["successful_fallbacks"] == 1
        assert stats["success_rate"] == 0.5
        assert "cpu_processing" in stats["strategy_statistics"]
    
    def test_fallback_report_generation(self, fallback_manager):
        """Test comprehensive fallback report generation"""
        report = fallback_manager.create_fallback_report()
        
        assert "report_timestamp" in report
        assert "fallback_statistics" in report
        assert "user_preferences" in report
        assert "available_strategies" in report
        assert "recent_fallbacks" in report


class TestErrorHandlingIntegration:
    """Test integration between error handling and fallback systems"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated error handling and fallback system"""
        error_handler = DiffSynthErrorHandler()
        fallback_manager = DiffSynthFallbackManager()
        
        # Connect systems
        error_handler.add_fallback_callback(fallback_manager.attempt_fallback)
        
        return error_handler, fallback_manager
    
    def test_error_to_fallback_flow(self, integrated_system):
        """Test flow from error handling to fallback"""
        error_handler, fallback_manager = integrated_system
        
        # Mock successful fallback
        def mock_fallback(error, request):
            return FallbackResult(
                success=True,
                strategy_used=FallbackStrategy.CPU_PROCESSING,
                message="Fallback successful"
            )
        
        fallback_manager.attempt_fallback = Mock(return_value=mock_fallback(None, None))
        
        # Generate error
        context = DiffSynthErrorContext(operation_type="edit")
        original_error = RuntimeError("CUDA out of memory")
        
        diffsynth_error = error_handler.handle_error(
            original_error, context, auto_recover=True
        )
        
        assert isinstance(diffsynth_error, DiffSynthMemoryError)
        # Fallback should have been attempted (though mocked)
    
    def test_user_notification_integration(self, integrated_system):
        """Test user notification integration"""
        error_handler, fallback_manager = integrated_system
        
        notifications = []
        
        def capture_notification(message, severity):
            notifications.append((message, severity))
        
        error_handler.add_user_notification_callback(capture_notification)
        
        # Generate error that triggers fallback
        context = DiffSynthErrorContext(operation_type="edit")
        error_handler.handle_error(
            RuntimeError("Test error"), context, auto_recover=True
        )
        
        # Should have received notifications
        assert len(notifications) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])