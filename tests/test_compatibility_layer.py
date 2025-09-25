"""
Integration tests for backward compatibility layer
Verifies no breaking changes to existing functionality
"""

import pytest
import os
import json
import tempfile
import torch
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Import the compatibility layer components
from src.compatibility_layer import (
    CompatibilityLayer, 
    LegacyQwenImageGenerator, 
    LegacyConfig,
    Qwen2VLIntegration
)
from src.model_detection_service import ModelInfo


class TestLegacyConfig:
    """Test legacy configuration handling"""
    
    def test_legacy_config_creation(self):
        """Test creating legacy configuration"""
        config = LegacyConfig(
            model_name="Qwen/Qwen-Image",
            device="cuda",
            torch_dtype=torch.float16,
            generation_settings={"width": 1024, "height": 1024},
            memory_settings={"enable_attention_slicing": False}
        )
        
        assert config.model_name == "Qwen/Qwen-Image"
        assert config.device == "cuda"
        assert config.torch_dtype == torch.float16
        assert config.generation_settings["width"] == 1024
        assert not config.memory_settings["enable_attention_slicing"]
    
    def test_legacy_config_serialization(self):
        """Test configuration serialization/deserialization"""
        original_config = LegacyConfig(
            model_name="Qwen/Qwen-Image",
            device="cuda",
            torch_dtype=torch.float16,
            generation_settings={"width": 1024, "height": 1024},
            memory_settings={"enable_attention_slicing": False}
        )
        
        # Serialize to dict
        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert "torch.float16" in config_dict["torch_dtype"]
        
        # Deserialize from dict
        restored_config = LegacyConfig.from_dict(config_dict)
        assert restored_config.model_name == original_config.model_name
        assert restored_config.device == original_config.device
        assert restored_config.torch_dtype == original_config.torch_dtype


class TestQwen2VLIntegration:
    """Test Qwen2-VL integration functionality"""
    
    def test_qwen2vl_initialization(self):
        """Test Qwen2-VL integration initialization"""
        with patch('src.compatibility_layer.ModelDetectionService') as mock_detection:
            # Mock no Qwen2-VL available
            mock_detection.return_value.detect_qwen2_vl_capabilities.return_value = {
                "integration_possible": False,
                "recommended_model": None
            }
            
            qwen2vl = Qwen2VLIntegration()
            assert not qwen2vl.available
            assert qwen2vl.model is None
    
    def test_qwen2vl_prompt_enhancement_fallback(self):
        """Test prompt enhancement fallback when Qwen2-VL not available"""
        with patch('src.compatibility_layer.ModelDetectionService') as mock_detection:
            mock_detection.return_value.detect_qwen2_vl_capabilities.return_value = {
                "integration_possible": False,
                "recommended_model": None
            }
            
            qwen2vl = Qwen2VLIntegration()
            original_prompt = "a beautiful landscape"
            enhanced_prompt = qwen2vl.enhance_prompt(original_prompt)
            
            # Should return original prompt when not available
            assert enhanced_prompt == original_prompt
    
    def test_qwen2vl_image_analysis_fallback(self):
        """Test image analysis fallback when Qwen2-VL not available"""
        with patch('src.compatibility_layer.ModelDetectionService') as mock_detection:
            mock_detection.return_value.detect_qwen2_vl_capabilities.return_value = {
                "integration_possible": False,
                "recommended_model": None
            }
            
            qwen2vl = Qwen2VLIntegration()
            
            # Create a dummy image
            dummy_image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            analysis = qwen2vl.analyze_image_for_context(dummy_image)
            
            # Should return empty string when not available
            assert analysis == ""


class TestCompatibilityLayer:
    """Test main compatibility layer functionality"""
    
    @pytest.fixture
    def mock_model_info(self):
        """Create mock model info for testing"""
        return ModelInfo(
            name="Qwen/Qwen-Image",
            path="/mock/path",
            size_gb=8.0,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={"transformer": True, "vae": True, "text_encoder": True},
            metadata={"architecture_type": "MMDiT"}
        )
    
    @pytest.fixture
    def compatibility_layer(self, mock_model_info):
        """Create compatibility layer with mocked dependencies"""
        with patch('src.compatibility_layer.ModelDetectionService') as mock_detection, \
             patch('src.compatibility_layer.PipelineOptimizer') as mock_optimizer, \
             patch('src.compatibility_layer.Qwen2VLIntegration') as mock_qwen2vl:
            
            # Mock detection service
            mock_detection.return_value.detect_current_model.return_value = mock_model_info
            mock_detection.return_value.detect_model_architecture.return_value = "MMDiT"
            mock_detection.return_value.is_optimization_needed.return_value = False
            mock_detection.return_value.get_recommended_model.return_value = "Qwen/Qwen-Image"
            
            # Mock Qwen2-VL integration
            mock_qwen2vl.return_value.available = False
            
            layer = CompatibilityLayer()
            return layer
    
    def test_compatibility_layer_initialization(self, compatibility_layer):
        """Test compatibility layer initialization"""
        assert compatibility_layer.detection_service is not None
        assert compatibility_layer.qwen2_vl is not None
        assert compatibility_layer.current_architecture == "Unknown"
        assert not compatibility_layer.backend_switched
    
    def test_config_migration_with_existing_config(self, compatibility_layer):
        """Test configuration migration with existing settings"""
        existing_config = {
            "model_name": "Qwen/Qwen-Image-Edit",
            "device": "cpu",
            "torch_dtype": "torch.bfloat16",
            "width": 512,
            "height": 512,
            "num_inference_steps": 30,
            "generation_settings": {
                "true_cfg_scale": 4.5
            },
            "memory_settings": {
                "enable_attention_slicing": True
            }
        }
        
        migrated_config = compatibility_layer.migrate_existing_config(existing_config)
        
        assert migrated_config.model_name == "Qwen/Qwen-Image-Edit"
        assert migrated_config.device == "cpu"
        assert migrated_config.torch_dtype == torch.bfloat16
        assert migrated_config.generation_settings["width"] == 512
        assert migrated_config.generation_settings["height"] == 512
        assert migrated_config.generation_settings["num_inference_steps"] == 30
        assert migrated_config.generation_settings["true_cfg_scale"] == 4.5
        assert migrated_config.memory_settings["enable_attention_slicing"] == True
    
    def test_config_migration_with_defaults(self, compatibility_layer):
        """Test configuration migration with default values"""
        # Mock _load_existing_config to return empty config
        with patch.object(compatibility_layer, '_load_existing_config', return_value={}):
            migrated_config = compatibility_layer.migrate_existing_config(None)
        
        # The model name comes from MODEL_CONFIG which is "Qwen/Qwen-Image"
        assert migrated_config.model_name == "Qwen/Qwen-Image"
        assert migrated_config.device in ["cuda", "cpu"]
        assert migrated_config.torch_dtype == torch.float16
        assert "width" in migrated_config.generation_settings
        assert "height" in migrated_config.generation_settings
    
    def test_backend_detection_and_switching(self, compatibility_layer, mock_model_info):
        """Test backend detection and switching logic"""
        with patch.object(compatibility_layer.detection_service, 'detect_current_model', return_value=mock_model_info), \
             patch.object(compatibility_layer.detection_service, 'detect_model_architecture', return_value="MMDiT"), \
             patch.object(compatibility_layer.detection_service, 'is_optimization_needed', return_value=True), \
             patch.object(compatibility_layer.detection_service, 'get_recommended_model', return_value="Qwen/Qwen-Image"), \
             patch('src.compatibility_layer.PipelineOptimizer') as mock_optimizer_class:
            
            # Mock pipeline optimizer
            mock_optimizer = Mock()
            mock_pipeline = Mock()
            mock_optimizer.create_optimized_pipeline.return_value = mock_pipeline
            mock_optimizer_class.return_value = mock_optimizer
            
            # Migrate config first
            compatibility_layer.migrate_existing_config()
            
            # Test backend switching
            success = compatibility_layer.detect_and_switch_backend()
            
            assert success
            assert compatibility_layer.current_architecture == "MMDiT"
            assert compatibility_layer.backend_switched
            assert compatibility_layer.optimized_pipeline == mock_pipeline
    
    def test_validation_functionality(self, compatibility_layer):
        """Test compatibility validation"""
        # Set up some state
        compatibility_layer.legacy_config = LegacyConfig(
            model_name="Qwen/Qwen-Image",
            device="cuda",
            torch_dtype=torch.float16,
            generation_settings={},
            memory_settings={}
        )
        
        validation_results = compatibility_layer.validate_compatibility()
        
        assert isinstance(validation_results, dict)
        assert "config_migration" in validation_results
        assert "backend_detection" in validation_results
        assert "pipeline_optimization" in validation_results
        assert "qwen2_vl_integration" in validation_results
        assert "overall_compatibility" in validation_results
        assert "warnings" in validation_results
        assert "errors" in validation_results


class TestLegacyQwenImageGenerator:
    """Test legacy interface wrapper"""
    
    @pytest.fixture
    def mock_compatibility_layer(self):
        """Create mock compatibility layer"""
        mock_layer = Mock()
        mock_layer.legacy_config = LegacyConfig(
            model_name="Qwen/Qwen-Image",
            device="cuda",
            torch_dtype=torch.float16,
            generation_settings={"width": 1024, "height": 1024},
            memory_settings={"enable_attention_slicing": False}
        )
        mock_layer.optimized_pipeline = Mock()
        return mock_layer
    
    def test_legacy_generator_initialization(self, mock_compatibility_layer):
        """Test legacy generator initialization"""
        generator = LegacyQwenImageGenerator(mock_compatibility_layer)
        
        assert generator.compatibility_layer == mock_compatibility_layer
        assert generator.device == "cuda"
        assert generator.model_name == "Qwen/Qwen-Image"
        assert generator.pipe == mock_compatibility_layer.optimized_pipeline
        assert os.path.exists(generator.output_dir)
    
    def test_legacy_load_model_method(self, mock_compatibility_layer):
        """Test legacy load_model method compatibility"""
        mock_compatibility_layer.legacy_config = None  # Force migration
        mock_compatibility_layer.detect_and_switch_backend.return_value = True
        mock_compatibility_layer.optimized_pipeline = Mock()
        
        generator = LegacyQwenImageGenerator(mock_compatibility_layer)
        success = generator.load_model()
        
        assert success
        mock_compatibility_layer.migrate_existing_config.assert_called_once()
        mock_compatibility_layer.detect_and_switch_backend.assert_called_once()
    
    def test_legacy_generate_image_method(self, mock_compatibility_layer):
        """Test legacy generate_image method compatibility"""
        # Mock the generate_image_with_compatibility method
        mock_image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        mock_compatibility_layer.generate_image_with_compatibility.return_value = mock_image
        
        generator = LegacyQwenImageGenerator(mock_compatibility_layer)
        
        # Test with various parameter combinations
        result = generator.generate_image(
            "test prompt",
            width=512,
            height=512,
            num_inference_steps=20,
            true_cfg_scale=3.5
        )
        
        assert result == mock_image
        mock_compatibility_layer.generate_image_with_compatibility.assert_called_once()
        
        # Check that parameters were passed correctly
        call_args = mock_compatibility_layer.generate_image_with_compatibility.call_args
        assert call_args[0][0] == "test prompt"  # First positional arg is prompt
        assert call_args[1]["width"] == 512
        assert call_args[1]["height"] == 512
        assert call_args[1]["num_inference_steps"] == 20
        assert call_args[1]["true_cfg_scale"] == 3.5
    
    def test_legacy_verify_device_setup_method(self, mock_compatibility_layer):
        """Test legacy verify_device_setup method"""
        mock_compatibility_layer.pipeline_optimizer = Mock()
        mock_compatibility_layer.pipeline_optimizer.validate_optimization.return_value = {
            "overall_status": "optimized"
        }
        
        generator = LegacyQwenImageGenerator(mock_compatibility_layer)
        result = generator.verify_device_setup()
        
        assert result == True
    
    def test_legacy_check_model_cache_method(self, mock_compatibility_layer):
        """Test legacy check_model_cache method"""
        mock_model_info = ModelInfo(
            name="Qwen/Qwen-Image",
            path="/mock/path",
            size_gb=8.0,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={"transformer": True, "vae": True, "text_encoder": False},
            metadata={}
        )
        
        mock_compatibility_layer.detection_service.detect_current_model.return_value = mock_model_info
        
        generator = LegacyQwenImageGenerator(mock_compatibility_layer)
        cache_info = generator.check_model_cache()
        
        assert cache_info["exists"] == True
        assert cache_info["complete"] == True
        assert cache_info["size_gb"] == 8.0
        assert "text_encoder" in cache_info["missing_components"]
        assert cache_info["snapshot_path"] == "/mock/path"


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    def test_full_backward_compatibility_workflow(self):
        """Test complete backward compatibility workflow"""
        with patch('src.compatibility_layer.ModelDetectionService') as mock_detection, \
             patch('src.compatibility_layer.PipelineOptimizer') as mock_optimizer_class, \
             patch('src.compatibility_layer.Qwen2VLIntegration') as mock_qwen2vl_class:
            
            # Mock model detection
            mock_model_info = ModelInfo(
                name="Qwen/Qwen-Image-Edit",  # Wrong model initially
                path="/mock/path",
                size_gb=54.0,
                model_type="image-editing",
                is_optimal=False,
                download_status="complete",
                components={"transformer": True, "vae": True, "text_encoder": True},
                metadata={}
            )
            
            mock_detection_service = mock_detection.return_value
            mock_detection_service.detect_current_model.return_value = mock_model_info
            mock_detection_service.detect_model_architecture.return_value = "MMDiT"
            mock_detection_service.is_optimization_needed.return_value = True
            mock_detection_service.get_recommended_model.return_value = "Qwen/Qwen-Image"
            
            # Mock pipeline optimizer
            mock_optimizer = Mock()
            mock_pipeline = Mock()
            mock_optimizer.create_optimized_pipeline.return_value = mock_pipeline
            mock_optimizer.configure_generation_settings.return_value = {
                "width": 1024, "height": 1024, "num_inference_steps": 20
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            # Mock Qwen2-VL
            mock_qwen2vl = Mock()
            mock_qwen2vl.available = False
            mock_qwen2vl_class.return_value = mock_qwen2vl
            
            # Create compatibility layer
            compatibility_layer = CompatibilityLayer()
            
            # Migrate existing config
            existing_config = {
                "model_name": "Qwen/Qwen-Image-Edit",
                "width": 512,
                "height": 512
            }
            migrated_config = compatibility_layer.migrate_existing_config(existing_config)
            
            # Detect and switch backend
            success = compatibility_layer.detect_and_switch_backend()
            assert success
            assert compatibility_layer.backend_switched
            
            # Get legacy interface
            legacy_generator = compatibility_layer.get_legacy_interface()
            assert isinstance(legacy_generator, LegacyQwenImageGenerator)
            
            # Test legacy load_model method
            load_success = legacy_generator.load_model()
            assert load_success
            
            # Validate compatibility
            validation = compatibility_layer.validate_compatibility()
            assert validation["config_migration"]
            assert validation["backend_detection"]
    
    def test_no_breaking_changes_to_existing_api(self):
        """Test that existing API calls continue to work"""
        with patch('src.compatibility_layer.ModelDetectionService') as mock_detection, \
             patch('src.compatibility_layer.PipelineOptimizer') as mock_optimizer_class, \
             patch('src.compatibility_layer.Qwen2VLIntegration') as mock_qwen2vl_class:
            
            # Mock optimal model (no switching needed)
            mock_model_info = ModelInfo(
                name="Qwen/Qwen-Image",
                path="/mock/path",
                size_gb=8.0,
                model_type="text-to-image",
                is_optimal=True,
                download_status="complete",
                components={"transformer": True, "vae": True, "text_encoder": True},
                metadata={}
            )
            
            mock_detection_service = mock_detection.return_value
            mock_detection_service.detect_current_model.return_value = mock_model_info
            mock_detection_service.is_optimization_needed.return_value = False
            
            # Mock Qwen2-VL
            mock_qwen2vl = Mock()
            mock_qwen2vl.available = False
            mock_qwen2vl_class.return_value = mock_qwen2vl
            
            # Create compatibility layer and legacy interface
            compatibility_layer = CompatibilityLayer()
            compatibility_layer.migrate_existing_config()
            
            legacy_generator = compatibility_layer.get_legacy_interface()
            
            # Test that all expected methods exist and are callable
            assert hasattr(legacy_generator, 'load_model')
            assert callable(legacy_generator.load_model)
            
            assert hasattr(legacy_generator, 'generate_image')
            assert callable(legacy_generator.generate_image)
            
            assert hasattr(legacy_generator, 'verify_device_setup')
            assert callable(legacy_generator.verify_device_setup)
            
            assert hasattr(legacy_generator, 'check_model_cache')
            assert callable(legacy_generator.check_model_cache)
            
            # Test that expected attributes exist
            assert hasattr(legacy_generator, 'device')
            assert hasattr(legacy_generator, 'model_name')
            assert hasattr(legacy_generator, 'pipe')
            assert hasattr(legacy_generator, 'edit_pipe')
            assert hasattr(legacy_generator, 'output_dir')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])