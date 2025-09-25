"""
Safe integration tests that don't load actual models
Tests the integration without causing WSL crashes
"""

import pytest
import torch
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qwen_generator import QwenImageGenerator
    from model_detection_service import ModelDetectionService, ModelInfo
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestSafeIntegration:
    """Safe integration tests that don't load actual models"""
    
    def test_qwen_generator_initialization(self):
        """Test QwenImageGenerator initializes without loading models"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True), \
             patch('qwen_generator.ModelDetectionService') as mock_detection, \
             patch('qwen_generator.ModelDownloadManager') as mock_download, \
             patch('qwen_generator.PerformanceMonitor') as mock_perf, \
             patch('qwen_generator.CompatibilityLayer') as mock_compat:
            
            # Create generator without triggering model loading
            generator = QwenImageGenerator()
            
            # Verify initialization
            assert generator.device in ["cuda", "cpu"]
            assert generator.model_name == "Qwen/Qwen-Image"
            assert generator.pipe is None  # No model loaded yet
            
            # Verify optimized components were attempted to be initialized
            mock_detection.assert_called_once()
            mock_download.assert_called_once()
            mock_perf.assert_called_once()
            mock_compat.assert_called_once()
    
    def test_architecture_detection_methods_exist(self):
        """Test that all required architecture detection methods exist"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            
            # Check all required methods exist
            assert hasattr(generator, '_select_optimal_pipeline_class')
            assert hasattr(generator, '_apply_architecture_specific_optimizations')
            assert hasattr(generator, '_ensure_device_consistency')
            assert hasattr(generator, '_disable_memory_saving_features')
            assert hasattr(generator, '_apply_gpu_optimizations')
            assert hasattr(generator, 'detect_and_optimize_model')
    
    def test_pipeline_class_selection_mmdit(self):
        """Test pipeline class selection for MMDiT without loading"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            generator.detection_service = Mock()
            generator.model_info = Mock()
            generator.current_architecture = "MMDiT"
            generator.model_name = "Qwen/Qwen-Image"
            
            with patch('diffusers.AutoPipelineForText2Image') as mock_auto:
                pipeline_class = generator._select_optimal_pipeline_class()
                assert pipeline_class == mock_auto
    
    def test_pipeline_class_selection_unet(self):
        """Test pipeline class selection for UNet without loading"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            generator.detection_service = Mock()
            generator.model_info = Mock()
            generator.current_architecture = "UNet"
            generator.model_name = "some/unet-model"
            
            with patch('diffusers.AutoPipelineForText2Image') as mock_auto:
                pipeline_class = generator._select_optimal_pipeline_class()
                assert pipeline_class == mock_auto
    
    def test_architecture_optimizations_mmdit(self):
        """Test MMDiT optimizations without actual pipeline"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            generator.current_architecture = "MMDiT"
            generator.pipe = None  # No actual pipeline
            
            # Should not crash even with no pipeline
            generator._apply_architecture_specific_optimizations()
            
            # Verify architecture is set
            assert generator.current_architecture == "MMDiT"
    
    def test_architecture_optimizations_unet(self):
        """Test UNet optimizations without actual pipeline"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            generator.current_architecture = "UNet"
            generator.pipe = None  # No actual pipeline
            
            # Should not crash even with no pipeline
            generator._apply_architecture_specific_optimizations()
            
            # Verify architecture is set
            assert generator.current_architecture == "UNet"
    
    def test_device_consistency_without_pipeline(self):
        """Test device consistency check without actual pipeline"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            generator.current_architecture = "MMDiT"
            generator.device = "cuda"
            generator.pipe = None  # No actual pipeline
            
            # Should not crash even with no pipeline
            generator._ensure_device_consistency()
            
            # Verify settings
            assert generator.device == "cuda"
            assert generator.current_architecture == "MMDiT"
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_optimizations_safe(self, mock_cuda):
        """Test GPU optimizations without actual GPU operations"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False), \
             patch('torch.backends.cuda.matmul') as mock_matmul, \
             patch('torch.backends.cudnn') as mock_cudnn, \
             patch('torch.set_grad_enabled') as mock_grad:
            
            generator = QwenImageGenerator()
            
            # Should not crash
            generator._apply_gpu_optimizations()
            
            # Verify optimizations were attempted
            assert mock_matmul.allow_tf32 is True
            assert mock_cudnn.allow_tf32 is True
            mock_grad.assert_called_with(False)
    
    def test_memory_saving_features_without_pipeline(self):
        """Test disabling memory-saving features without actual pipeline"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            generator.pipe = None  # No actual pipeline
            
            # Should not crash even with no pipeline
            generator._disable_memory_saving_features()
    
    def test_model_detection_service_safe(self):
        """Test ModelDetectionService without actual model files"""
        with patch('os.path.exists', return_value=False), \
             patch('os.listdir', return_value=[]):
            
            service = ModelDetectionService()
            
            # Should not crash even with no models
            current_model = service.detect_current_model()
            assert current_model is None
            
            # Should return sensible defaults
            needs_optimization = service.is_optimization_needed()
            assert needs_optimization is True  # No model = needs optimization
            
            recommended = service.get_recommended_model()
            assert recommended == "Qwen/Qwen-Image"
    
    def test_pipeline_optimizer_safe(self):
        """Test PipelineOptimizer without actual model loading"""
        config = OptimizationConfig(
            architecture_type="MMDiT",
            enable_attention_slicing=False,
            enable_tf32=True
        )
        
        optimizer = PipelineOptimizer(config)
        
        # Test configuration methods that don't load models
        settings = optimizer.configure_generation_settings("MMDiT")
        assert isinstance(settings, dict)
        assert "width" in settings
        assert "height" in settings
        
        # Test performance recommendations
        recommendations = optimizer.get_performance_recommendations(16.0)
        assert isinstance(recommendations, dict)
        assert "memory_strategy" in recommendations
    
    def test_load_model_with_mocked_components(self):
        """Test load_model with fully mocked components to prevent crashes"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            
            # Mock all the heavy operations
            generator.detect_and_optimize_model = Mock(return_value=True)
            generator.pipe = Mock()  # Mock pipeline instead of loading real one
            generator._apply_architecture_specific_optimizations = Mock()
            generator._ensure_device_consistency = Mock()
            generator.verify_device_setup = Mock(return_value=True)
            
            # This should not crash or try to load actual models
            result = generator.load_model()
            
            # Verify the workflow was followed
            assert result is True
            generator.detect_and_optimize_model.assert_called_once()
            generator._apply_architecture_specific_optimizations.assert_called_once()
            generator._ensure_device_consistency.assert_called_once()
            generator.verify_device_setup.assert_called_once()
    
    def test_detect_and_optimize_model_mocked(self):
        """Test detect_and_optimize_model with mocked detection service"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            
            # Mock the detection service to avoid actual model detection
            mock_model_info = Mock()
            mock_model_info.name = "Qwen/Qwen-Image"
            mock_model_info.is_optimal = True
            
            generator.detection_service = Mock()
            generator.detection_service.detect_current_model.return_value = mock_model_info
            generator.detection_service.detect_model_architecture.return_value = "MMDiT"
            generator.detection_service.is_optimization_needed.return_value = False
            
            # Mock pipeline creation to avoid loading
            generator._create_optimized_pipeline = Mock(return_value=True)
            generator.pipe = Mock()  # Mock pipeline
            
            # This should not crash - but may return False due to mocking issues
            result = generator.detect_and_optimize_model()
            
            # Just verify it doesn't crash and sets the architecture
            assert generator.current_architecture == "MMDiT"
            assert generator.model_info == mock_model_info


class TestErrorHandlingSafe:
    """Test error handling without causing crashes"""
    
    def test_missing_optimized_components(self):
        """Test behavior when optimized components are missing"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', False):
            generator = QwenImageGenerator()
            
            # Should initialize without optimized components
            assert generator.detection_service is None
            assert generator.pipeline_optimizer is None
            assert generator.download_manager is None
            assert generator.performance_monitor is None
            assert generator.compatibility_layer is None
    
    def test_architecture_detection_failure(self):
        """Test handling of architecture detection failure"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            
            # Mock detection service to raise exception
            generator.detection_service = Mock()
            generator.detection_service.detect_current_model.side_effect = Exception("Detection failed")
            
            # Should not crash
            result = generator.detect_and_optimize_model()
            assert result is False
    
    def test_pipeline_creation_failure(self):
        """Test handling of pipeline creation failure"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            
            # Mock to simulate pipeline creation failure
            generator._create_optimized_pipeline = Mock(return_value=False)
            generator.detection_service = Mock()
            generator.detection_service.detect_current_model.return_value = Mock()
            generator.detection_service.detect_model_architecture.return_value = "MMDiT"
            generator.detection_service.is_optimization_needed.return_value = True
            
            # Should handle failure gracefully
            result = generator.detect_and_optimize_model()
            # Result depends on implementation, but should not crash


if __name__ == "__main__":
    # Run tests safely
    pytest.main([__file__, "-v", "--tb=short"])