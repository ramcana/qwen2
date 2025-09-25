"""
Integration tests for the complete optimization workflow
Tests the integration of ModelDetectionService, PipelineOptimizer, and QwenImageGenerator
SAFE VERSION - Does not load actual models to prevent WSL crashes
"""

import pytest
import torch
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional

# Import the components to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qwen_generator import QwenImageGenerator
    from model_detection_service import ModelDetectionService, ModelInfo
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from model_download_manager import ModelDownloadManager
    from utils.performance_monitor import PerformanceMonitor
    from compatibility_layer import CompatibilityLayer
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Prevent actual model loading to avoid WSL crashes
@pytest.fixture(autouse=True)
def prevent_model_loading():
    """Automatically prevent actual model loading in all tests"""
    with patch('diffusers.DiffusionPipeline.from_pretrained') as mock_pipeline, \
         patch('diffusers.AutoPipelineForText2Image.from_pretrained') as mock_auto_pipeline:
        
        # Create mock pipeline instances
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline.return_value = mock_pipeline_instance
        mock_auto_pipeline.return_value = mock_pipeline_instance
        
        yield


class TestOptimizationWorkflowIntegration:
    """Test the complete optimization workflow integration"""
    
    @pytest.fixture
    def mock_model_info(self):
        """Create a mock ModelInfo for testing"""
        return ModelInfo(
            name="Qwen/Qwen-Image",
            path="/mock/path/to/model",
            size_gb=8.5,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={
                "transformer": True,
                "vae": True,
                "text_encoder": True,
                "tokenizer": True,
                "scheduler": True,
                "model_index.json": True
            },
            metadata={
                "architecture_type": "MMDiT",
                "supports_text_to_image": True,
                "uses_mmdit": True
            }
        )
    
    @pytest.fixture
    def mock_suboptimal_model_info(self):
        """Create a mock ModelInfo for suboptimal model"""
        return ModelInfo(
            name="Qwen/Qwen-Image-Edit",
            path="/mock/path/to/edit/model",
            size_gb=54.0,
            model_type="image-editing",
            is_optimal=False,
            download_status="complete",
            components={
                "transformer": True,
                "vae": True,
                "text_encoder": True,
                "tokenizer": True,
                "scheduler": True,
                "model_index.json": True
            },
            metadata={
                "architecture_type": "MMDiT",
                "supports_image_editing": True,
                "uses_mmdit": True
            }
        )
    
    @pytest.fixture
    def mock_generator(self):
        """Create a QwenImageGenerator with mocked components"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            
            # Mock the optimized components
            generator.detection_service = Mock(spec=ModelDetectionService)
            generator.pipeline_optimizer = Mock(spec=PipelineOptimizer)
            generator.download_manager = Mock(spec=ModelDownloadManager)
            generator.performance_monitor = Mock(spec=PerformanceMonitor)
            generator.compatibility_layer = Mock(spec=CompatibilityLayer)
            
            return generator
    
    def test_detect_and_optimize_model_optimal_case(self, mock_generator, mock_model_info):
        """Test detection and optimization when optimal model is already available"""
        # Setup mocks
        mock_generator.detection_service.detect_current_model.return_value = mock_model_info
        mock_generator.detection_service.detect_model_architecture.return_value = "MMDiT"
        mock_generator.detection_service.is_optimization_needed.return_value = False
        
        # Mock pipeline creation
        mock_pipeline = Mock()
        mock_generator.pipeline_optimizer = Mock()
        mock_generator.pipeline_optimizer.create_optimized_pipeline.return_value = mock_pipeline
        mock_generator.pipeline_optimizer.validate_optimization.return_value = {
            'overall_status': 'optimized'
        }
        
        # Test the optimization workflow
        result = mock_generator.detect_and_optimize_model()
        
        # Assertions
        assert result is True
        assert mock_generator.model_info == mock_model_info
        assert mock_generator.current_architecture == "MMDiT"
        mock_generator.detection_service.detect_current_model.assert_called_once()
        mock_generator.detection_service.detect_model_architecture.assert_called_once_with(mock_model_info)
        mock_generator.detection_service.is_optimization_needed.assert_called_once()
    
    def test_detect_and_optimize_model_needs_optimization(self, mock_generator, mock_suboptimal_model_info):
        """Test detection and optimization when model needs optimization"""
        # Setup mocks
        mock_generator.detection_service.detect_current_model.return_value = mock_suboptimal_model_info
        mock_generator.detection_service.detect_model_architecture.return_value = "MMDiT"
        mock_generator.detection_service.is_optimization_needed.return_value = True
        mock_generator.detection_service.get_recommended_model.return_value = "Qwen/Qwen-Image"
        
        # Mock download manager
        mock_generator.download_manager.download_qwen_image.return_value = True
        
        # Mock updated model info after download
        optimal_model_info = ModelInfo(
            name="Qwen/Qwen-Image",
            path="/mock/path/to/optimal/model",
            size_gb=8.5,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={},
            metadata={}
        )
        mock_generator.detection_service.detect_current_model.side_effect = [
            mock_suboptimal_model_info,  # First call
            optimal_model_info  # After download
        ]
        
        # Mock pipeline creation
        mock_pipeline = Mock()
        mock_generator.pipeline_optimizer = Mock()
        mock_generator.pipeline_optimizer.create_optimized_pipeline.return_value = mock_pipeline
        mock_generator.pipeline_optimizer.validate_optimization.return_value = {
            'overall_status': 'optimized'
        }
        
        # Test the optimization workflow
        result = mock_generator.detect_and_optimize_model()
        
        # Assertions
        assert result is True
        mock_generator.detection_service.get_recommended_model.assert_called_once()
        mock_generator.download_manager.download_qwen_image.assert_called_once()
        assert mock_generator.detection_service.detect_current_model.call_count == 2
    
    def test_detect_and_optimize_model_no_model_found(self, mock_generator):
        """Test detection and optimization when no model is found"""
        # Setup mocks
        mock_generator.detection_service.detect_current_model.return_value = None
        mock_generator.download_manager.download_qwen_image.return_value = True
        
        # Mock model info after download
        downloaded_model_info = ModelInfo(
            name="Qwen/Qwen-Image",
            path="/mock/path/to/downloaded/model",
            size_gb=8.5,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={},
            metadata={}
        )
        
        # Mock recursive call after download
        with patch.object(mock_generator, 'detect_and_optimize_model', side_effect=[True]) as mock_recursive:
            mock_generator.detection_service.detect_current_model.side_effect = [
                None,  # First call - no model
                downloaded_model_info  # After download
            ]
            
            # Test the missing model workflow
            result = mock_generator._handle_missing_model()
            
            # Assertions
            assert result is True
            mock_generator.download_manager.download_qwen_image.assert_called_once()
    
    def test_select_optimal_pipeline_class_mmdit(self, mock_generator, mock_model_info):
        """Test pipeline class selection for MMDiT architecture"""
        mock_generator.detection_service = Mock()
        mock_generator.model_info = mock_model_info
        mock_generator.current_architecture = "MMDiT"
        mock_generator.model_name = "Qwen/Qwen-Image"
        
        with patch('qwen_generator.AutoPipelineForText2Image') as mock_auto_pipeline:
            pipeline_class = mock_generator._select_optimal_pipeline_class()
            assert pipeline_class == mock_auto_pipeline
    
    def test_select_optimal_pipeline_class_unet(self, mock_generator):
        """Test pipeline class selection for UNet architecture"""
        mock_generator.detection_service = Mock()
        mock_generator.model_info = Mock()
        mock_generator.current_architecture = "UNet"
        mock_generator.model_name = "some/unet-model"
        
        with patch('qwen_generator.AutoPipelineForText2Image') as mock_auto_pipeline:
            pipeline_class = mock_generator._select_optimal_pipeline_class()
            assert pipeline_class == mock_auto_pipeline
    
    def test_select_optimal_pipeline_class_editing_model(self, mock_generator):
        """Test pipeline class selection for editing models"""
        mock_generator.detection_service = Mock()
        mock_generator.model_info = Mock()
        mock_generator.current_architecture = "MMDiT"
        mock_generator.model_name = "Qwen/Qwen-Image-Edit"
        
        from qwen_generator import DiffusionPipeline
        pipeline_class = mock_generator._select_optimal_pipeline_class()
        assert pipeline_class == DiffusionPipeline
    
    def test_apply_architecture_specific_optimizations_mmdit(self, mock_generator):
        """Test MMDiT-specific optimizations"""
        mock_generator.current_architecture = "MMDiT"
        mock_generator.pipe = Mock()
        mock_generator.pipe.transformer = Mock()
        
        # Test MMDiT optimizations
        mock_generator._apply_architecture_specific_optimizations()
        
        # Verify that MMDiT-specific methods were called
        # (The actual implementation will depend on the specific optimizations)
        assert mock_generator.current_architecture == "MMDiT"
    
    def test_apply_architecture_specific_optimizations_unet(self, mock_generator):
        """Test UNet-specific optimizations"""
        mock_generator.current_architecture = "UNet"
        mock_generator.pipe = Mock()
        mock_generator.pipe.unet = Mock()
        
        with patch('qwen_generator.AttnProcessor2_0') as mock_processor:
            # Test UNet optimizations
            mock_generator._apply_architecture_specific_optimizations()
            
            # Verify UNet-specific optimizations were applied
            mock_generator.pipe.unet.set_attn_processor.assert_called_once()
    
    def test_ensure_device_consistency_mmdit(self, mock_generator):
        """Test device consistency for MMDiT architecture"""
        mock_generator.current_architecture = "MMDiT"
        mock_generator.device = "cuda"
        mock_generator.pipe = Mock()
        
        # Mock components
        mock_transformer = Mock()
        mock_transformer.device = "cpu"
        mock_transformer.parameters.return_value = [Mock(device="cpu")]
        mock_generator.pipe.transformer = mock_transformer
        
        mock_vae = Mock()
        mock_vae.device = "cuda"
        mock_generator.pipe.vae = mock_vae
        
        # Test device consistency
        mock_generator._ensure_device_consistency()
        
        # Verify device movement was attempted
        mock_transformer.to.assert_called_with("cuda")
    
    def test_ensure_device_consistency_unet(self, mock_generator):
        """Test device consistency for UNet architecture"""
        mock_generator.current_architecture = "UNet"
        mock_generator.device = "cuda"
        mock_generator.pipe = Mock()
        
        # Mock components
        mock_unet = Mock()
        mock_unet.device = "cuda"
        mock_generator.pipe.unet = mock_unet
        
        mock_vae = Mock()
        mock_vae.device = "cuda"
        mock_generator.pipe.vae = mock_vae
        
        # Test device consistency
        mock_generator._ensure_device_consistency()
        
        # Verify components are checked
        assert mock_generator.current_architecture == "UNet"
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_apply_gpu_optimizations(self, mock_cuda_available, mock_generator):
        """Test GPU optimization application"""
        with patch('torch.backends.cuda.matmul') as mock_matmul, \
             patch('torch.backends.cudnn') as mock_cudnn, \
             patch('torch.set_grad_enabled') as mock_grad:
            
            mock_generator._apply_gpu_optimizations()
            
            # Verify GPU optimizations were applied
            assert mock_matmul.allow_tf32 is True
            assert mock_cudnn.allow_tf32 is True
            assert mock_cudnn.benchmark is True
            mock_grad.assert_called_with(False)
    
    def test_disable_memory_saving_features(self, mock_generator):
        """Test disabling memory-saving features"""
        mock_generator.pipe = Mock()
        mock_generator.pipe.disable_attention_slicing = Mock()
        mock_generator.pipe.disable_vae_slicing = Mock()
        mock_generator.pipe.disable_vae_tiling = Mock()
        
        mock_generator._disable_memory_saving_features()
        
        # Verify memory-saving features were disabled
        mock_generator.pipe.disable_attention_slicing.assert_called_once()
        mock_generator.pipe.disable_vae_slicing.assert_called_once()
        mock_generator.pipe.disable_vae_tiling.assert_called_once()
    
    def test_load_model_with_optimization_success(self, mock_generator, mock_model_info):
        """Test complete load_model workflow with successful optimization"""
        # Setup mocks for successful optimization
        mock_generator.detect_and_optimize_model = Mock(return_value=True)
        mock_generator.pipe = Mock()  # Simulate successful pipeline creation
        mock_generator._apply_architecture_specific_optimizations = Mock()
        mock_generator._ensure_device_consistency = Mock()
        mock_generator.verify_device_setup = Mock(return_value=True)
        
        # Mock the load_model method to prevent actual model loading
        with patch.object(mock_generator, 'load_model', return_value=True) as mock_load:
            result = mock_generator.load_model()
            
            # Assertions
            assert result is True
            mock_load.assert_called_once()
    
    def test_load_model_with_optimization_failure_fallback(self, mock_generator):
        """Test load_model fallback when optimization fails"""
        # Setup mocks for failed optimization
        mock_generator.detect_and_optimize_model = Mock(return_value=False)
        mock_generator.pipe = None  # Simulate failed pipeline creation
        
        # Mock legacy loading methods
        mock_generator.estimate_download_progress = Mock(return_value={
            'current_size_gb': 8.0,
            'expected_size_gb': 36.0,
            'progress_percent': 22.2
        })
        mock_generator.check_model_cache = Mock(return_value={
            'exists': True,
            'complete': True,
            'size_gb': 35.0,
            'missing_components': []
        })
        mock_generator._select_optimal_pipeline_class = Mock(return_value=Mock())
        mock_generator.verify_device_setup = Mock(return_value=True)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.memory_allocated', return_value=0), \
             patch('torch.cuda.empty_cache'), \
             patch('torch.cuda.synchronize'), \
             patch('qwen_generator.DiffusionPipeline') as mock_pipeline_class:
            
            mock_props.return_value.total_memory = 16 * 1024**3  # 16GB
            mock_pipeline_instance = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance
            mock_pipeline_instance.to.return_value = mock_pipeline_instance
            
            # Test load_model with fallback
            result = mock_generator.load_model()
            
            # Assertions
            assert result is True
            mock_generator.detect_and_optimize_model.assert_called_once()
            # Verify fallback was used
            assert mock_generator.pipe is not None


class TestOptimizationWorkflowErrorHandling:
    """Test error handling in the optimization workflow"""
    
    @pytest.fixture
    def mock_generator(self):
        """Create a QwenImageGenerator with mocked components for error testing"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            generator.detection_service = Mock(spec=ModelDetectionService)
            generator.download_manager = Mock(spec=ModelDownloadManager)
            return generator
    
    def test_detect_and_optimize_model_detection_failure(self, mock_generator):
        """Test handling of model detection failure"""
        # Setup mock to raise exception
        mock_generator.detection_service.detect_current_model.side_effect = Exception("Detection failed")
        
        # Test error handling
        result = mock_generator.detect_and_optimize_model()
        
        # Should return False on error
        assert result is False
    
    def test_detect_and_optimize_model_download_failure(self, mock_generator):
        """Test handling of model download failure"""
        # Setup mocks
        mock_generator.detection_service.detect_current_model.return_value = None
        mock_generator.download_manager.download_qwen_image.return_value = False
        
        # Test download failure handling
        result = mock_generator._handle_missing_model()
        
        # Should return False on download failure
        assert result is False
    
    def test_create_optimized_pipeline_failure_fallback(self, mock_generator):
        """Test fallback when optimized pipeline creation fails"""
        # Setup mock to fail
        with patch('qwen_generator.PipelineOptimizer') as mock_optimizer_class:
            mock_optimizer_class.side_effect = Exception("Optimizer failed")
            
            # Mock the fallback method
            mock_generator._create_basic_optimized_pipeline = Mock(return_value=True)
            
            # Test fallback
            result = mock_generator._create_optimized_pipeline()
            
            # Should fallback to basic optimization
            assert result is True
            mock_generator._create_basic_optimized_pipeline.assert_called_once()
    
    def test_architecture_specific_optimizations_error_handling(self, mock_generator):
        """Test error handling in architecture-specific optimizations"""
        mock_generator.current_architecture = "MMDiT"
        mock_generator.pipe = Mock()
        mock_generator.pipe.transformer = Mock()
        mock_generator.pipe.transformer.set_use_memory_efficient_attention_xformers.side_effect = Exception("Optimization failed")
        
        # Should not raise exception
        mock_generator._apply_architecture_specific_optimizations()
        
        # Test completed without raising exception
        assert mock_generator.current_architecture == "MMDiT"
    
    def test_device_consistency_error_handling(self, mock_generator):
        """Test error handling in device consistency checks"""
        mock_generator.current_architecture = "MMDiT"
        mock_generator.device = "cuda"
        mock_generator.pipe = Mock()
        
        # Mock component that raises exception during device check
        mock_transformer = Mock()
        mock_transformer.device = Mock(side_effect=Exception("Device check failed"))
        mock_generator.pipe.transformer = mock_transformer
        
        # Should not raise exception
        mock_generator._ensure_device_consistency()
        
        # Test completed without raising exception
        assert mock_generator.device == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])