"""
Unit tests for Qwen Image Configuration Management
Tests the modern architecture-aware configuration system with MMDiT support
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.qwen_image_config import (
    OptimizationConfig,
    ModelArchitecture,
    OptimizationLevel,
    create_optimization_config,
    get_model_config_for_architecture,
    validate_architecture_compatibility,
    migrate_legacy_config,
    DEFAULT_CONFIGS,
    QUALITY_PRESETS,
    QWEN2VL_CONFIG,
)


class TestOptimizationConfig:
    """Test the OptimizationConfig dataclass"""
    
    def test_default_initialization(self):
        """Test default configuration initialization"""
        config = OptimizationConfig()
        
        assert config.model_name == "Qwen/Qwen-Image"
        assert config.architecture == ModelArchitecture.MMDIT
        assert config.optimization_level == OptimizationLevel.BALANCED
        assert config.torch_dtype == torch.float16
        assert config.enable_scaled_dot_product_attention is True
        assert config.enable_attention_slicing is False
        assert config.width == 1024
        assert config.height == 1024
    
    def test_custom_initialization(self):
        """Test custom configuration initialization"""
        config = OptimizationConfig(
            model_name="custom/model",
            architecture=ModelArchitecture.UNET,
            optimization_level=OptimizationLevel.ULTRA_FAST,
            width=512,
            height=512
        )
        
        assert config.model_name == "custom/model"
        assert config.architecture == ModelArchitecture.UNET
        assert config.optimization_level == OptimizationLevel.ULTRA_FAST
        assert config.width == 512
        assert config.height == 512
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_validation_success(self, mock_cuda):
        """Test successful configuration validation"""
        config = OptimizationConfig()
        assert config.validate() is True
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_validation_cuda_fallback(self, mock_cuda):
        """Test CUDA fallback during validation"""
        config = OptimizationConfig(device="cuda")
        assert config.validate() is True
        assert config.device == "cpu"
    
    def test_validation_invalid_dimensions(self):
        """Test validation with invalid dimensions"""
        config = OptimizationConfig(width=-1, height=0)
        assert config.validate() is False
    
    def test_validation_invalid_steps(self):
        """Test validation with invalid inference steps"""
        config = OptimizationConfig(num_inference_steps=0)
        assert config.validate() is False
    
    def test_validation_invalid_cfg_scale(self):
        """Test validation with invalid CFG scale"""
        config = OptimizationConfig(true_cfg_scale=-1.0)
        assert config.validate() is False
    
    def test_migrate_from_legacy(self):
        """Test migration from legacy configuration"""
        legacy_config = {
            "model_name": "legacy/model",
            "torch_dtype": torch.float32,
            "memory_config": {
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "enable_vae_slicing": True
            },
            "generation_config": {
                "width": 768,
                "height": 768,
                "num_inference_steps": 20,
                "true_cfg_scale": 3.0
            }
        }
        
        config = OptimizationConfig()
        config.migrate_from_legacy(legacy_config)
        
        assert config.model_name == "legacy/model"
        assert config.torch_dtype == torch.float32
        assert config.enable_attention_slicing is True
        assert config.enable_cpu_offload is True
        assert config.enable_vae_slicing is True
        assert config.width == 768
        assert config.height == 768
        assert config.num_inference_steps == 20
        assert config.true_cfg_scale == 3.0
    
    def test_apply_ultra_fast_optimization(self):
        """Test ultra fast optimization level application"""
        config = OptimizationConfig(optimization_level=OptimizationLevel.ULTRA_FAST)
        config.apply_optimization_level()
        
        assert config.num_inference_steps == 10
        assert config.true_cfg_scale == 2.5
        assert config.width == 768
        assert config.height == 768
        assert config.enable_scaled_dot_product_attention is True
        assert config.enable_attention_slicing is False
    
    def test_apply_balanced_optimization(self):
        """Test balanced optimization level application"""
        config = OptimizationConfig(optimization_level=OptimizationLevel.BALANCED)
        config.apply_optimization_level()
        
        assert config.num_inference_steps == 25
        assert config.true_cfg_scale == 4.0
        assert config.width == 1024
        assert config.height == 1024
        assert config.enable_scaled_dot_product_attention is True
    
    def test_apply_quality_optimization(self):
        """Test quality optimization level application"""
        config = OptimizationConfig(optimization_level=OptimizationLevel.QUALITY)
        config.apply_optimization_level()
        
        assert config.num_inference_steps == 40
        assert config.true_cfg_scale == 5.0
        assert config.width == 1280
        assert config.height == 1280
        assert config.enable_tf32 is False  # Higher precision for quality
        assert config.enable_vae_tiling is True
    
    def test_apply_multimodal_optimization(self):
        """Test multimodal optimization level application"""
        config = OptimizationConfig(optimization_level=OptimizationLevel.MULTIMODAL)
        config.apply_optimization_level()
        
        assert config.num_inference_steps == 30
        assert config.true_cfg_scale == 4.5
        assert config.enable_qwen2vl_integration is True
        assert config.enable_prompt_enhancement is True
    
    def test_enable_all_optimizations(self):
        """Test enabling all performance optimizations"""
        config = OptimizationConfig()
        config.enable_all_optimizations()
        
        assert config.enable_scaled_dot_product_attention is True
        assert config.enable_tf32 is True
        assert config.enable_cudnn_benchmark is True
        assert config.dynamic_batch_sizing is True
        assert config.enable_attention_slicing is False
        assert config.enable_cpu_offload is False
        assert config.enable_vae_slicing is False
    
    def test_get_pipeline_kwargs(self):
        """Test pipeline kwargs generation"""
        config = OptimizationConfig(device="cuda")
        kwargs = config.get_pipeline_kwargs()
        
        expected_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        
        assert kwargs == expected_kwargs
    
    def test_get_generation_kwargs(self):
        """Test generation kwargs generation"""
        config = OptimizationConfig(
            width=512,
            height=768,
            num_inference_steps=20,
            true_cfg_scale=3.5
        )
        kwargs = config.get_generation_kwargs()
        
        expected_kwargs = {
            "width": 512,
            "height": 768,
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "output_type": "pil",
        }
        
        assert kwargs == expected_kwargs


class TestConfigurationFactory:
    """Test configuration factory functions"""
    
    def test_create_optimization_config_with_enum(self):
        """Test creating config with enum parameters"""
        config = create_optimization_config(
            OptimizationLevel.ULTRA_FAST,
            ModelArchitecture.MMDIT
        )
        
        assert config.optimization_level == OptimizationLevel.ULTRA_FAST
        assert config.architecture == ModelArchitecture.MMDIT
        assert config.num_inference_steps == 10  # Applied from level
    
    def test_create_optimization_config_with_string(self):
        """Test creating config with string parameters"""
        config = create_optimization_config("quality", "unet")
        
        assert config.optimization_level == OptimizationLevel.QUALITY
        assert config.architecture == ModelArchitecture.UNET
        assert config.num_inference_steps == 40  # Applied from level
    
    def test_create_optimization_config_with_kwargs(self):
        """Test creating config with additional kwargs"""
        config = create_optimization_config(
            OptimizationLevel.BALANCED,
            ModelArchitecture.MMDIT,
            width=2048,
            height=2048,
            model_name="custom/model"
        )
        
        assert config.width == 2048
        assert config.height == 2048
        assert config.model_name == "custom/model"
    
    def test_get_model_config_for_mmdit(self):
        """Test getting model config for MMDiT architecture"""
        config = get_model_config_for_architecture(ModelArchitecture.MMDIT)
        
        assert config["model_name"] == "Qwen/Qwen-Image"
        assert config["enable_scaled_dot_product_attention"] is True
        assert config["enable_tf32"] is True
        assert config["enable_cudnn_benchmark"] is True
    
    def test_get_model_config_for_unet(self):
        """Test getting model config for UNet architecture"""
        config = get_model_config_for_architecture(ModelArchitecture.UNET)
        
        assert config["model_name"] == "Qwen/Qwen-Image-Edit"
        assert config["enable_xformers"] is True
        assert config["enable_attention_slicing"] is True
    
    def test_get_model_config_for_multimodal(self):
        """Test getting model config for multimodal architecture"""
        config = get_model_config_for_architecture(ModelArchitecture.MULTIMODAL)
        
        assert config["model_name"] == "Qwen/Qwen2-VL-7B-Instruct"
        assert config["enable_qwen2vl_integration"] is True
        assert config["enable_prompt_enhancement"] is True


class TestConfigurationValidation:
    """Test configuration validation functions"""
    
    def test_validate_mmdit_compatibility_success(self):
        """Test successful MMDiT compatibility validation"""
        config = OptimizationConfig(
            architecture=ModelArchitecture.MMDIT,
            enable_attention_slicing=False,
            enable_scaled_dot_product_attention=True
        )
        
        assert validate_architecture_compatibility(ModelArchitecture.MMDIT, config) is True
    
    def test_validate_mmdit_compatibility_warning(self):
        """Test MMDiT compatibility validation with warning"""
        config = OptimizationConfig(
            architecture=ModelArchitecture.MMDIT,
            enable_attention_slicing=True
        )
        
        with patch('src.qwen_image_config.logger') as mock_logger:
            result = validate_architecture_compatibility(ModelArchitecture.MMDIT, config)
            mock_logger.warning.assert_called_once()
            assert result is False
    
    def test_validate_mmdit_compatibility_auto_enable(self):
        """Test MMDiT compatibility auto-enabling optimizations"""
        config = OptimizationConfig(
            architecture=ModelArchitecture.MMDIT,
            enable_scaled_dot_product_attention=False
        )
        
        with patch('src.qwen_image_config.logger') as mock_logger:
            validate_architecture_compatibility(ModelArchitecture.MMDIT, config)
            mock_logger.info.assert_called_once()
            assert config.enable_scaled_dot_product_attention is True
    
    def test_validate_unet_compatibility(self):
        """Test UNet compatibility validation"""
        config = OptimizationConfig(
            architecture=ModelArchitecture.UNET,
            enable_xformers=False
        )
        
        with patch('src.qwen_image_config.logger') as mock_logger:
            result = validate_architecture_compatibility(ModelArchitecture.UNET, config)
            mock_logger.info.assert_called_once()
            assert result is True
    
    def test_validate_multimodal_compatibility(self):
        """Test multimodal compatibility validation"""
        config = OptimizationConfig(
            architecture=ModelArchitecture.MULTIMODAL,
            enable_qwen2vl_integration=False
        )
        
        with patch('src.qwen_image_config.logger') as mock_logger:
            validate_architecture_compatibility(ModelArchitecture.MULTIMODAL, config)
            mock_logger.info.assert_called_once()
            assert config.enable_qwen2vl_integration is True


class TestLegacyMigration:
    """Test legacy configuration migration"""
    
    def test_migrate_legacy_config_complete(self):
        """Test complete legacy configuration migration"""
        legacy_config = {
            "MODEL_CONFIG": {
                "model_name": "Qwen/Legacy-Model",
                "torch_dtype": torch.float32,
                "device": "cpu"
            },
            "MEMORY_CONFIG": {
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "enable_vae_slicing": True,
                "enable_xformers": True,
                "use_torch_compile": True
            },
            "GENERATION_CONFIG": {
                "width": 512,
                "height": 768,
                "num_inference_steps": 15,
                "true_cfg_scale": 2.5
            }
        }
        
        with patch('src.qwen_image_config.logger') as mock_logger:
            config = migrate_legacy_config(legacy_config)
            mock_logger.info.assert_called_once()
        
        assert config.model_name == "Qwen/Legacy-Model"
        assert config.torch_dtype == torch.float32
        assert config.device == "cpu"
        assert config.enable_attention_slicing is True
        assert config.enable_cpu_offload is True
        assert config.enable_vae_slicing is True
        assert config.enable_xformers is True
        assert config.use_torch_compile is True
        assert config.width == 512
        assert config.height == 768
        assert config.num_inference_steps == 15
        assert config.true_cfg_scale == 2.5
    
    def test_migrate_legacy_config_partial(self):
        """Test partial legacy configuration migration"""
        legacy_config = {
            "MODEL_CONFIG": {
                "model_name": "Qwen/Partial-Model"
            }
        }
        
        config = migrate_legacy_config(legacy_config)
        
        # Should have migrated model name but kept defaults for others
        assert config.model_name == "Qwen/Partial-Model"
        assert config.torch_dtype == torch.float16  # Default
        assert config.width == 1024  # Default
    
    def test_migrate_legacy_config_empty(self):
        """Test migration with empty legacy config"""
        legacy_config = {}
        
        config = migrate_legacy_config(legacy_config)
        
        # Should be all defaults
        assert config.model_name == "Qwen/Qwen-Image"
        assert config.torch_dtype == torch.float16
        assert config.width == 1024


class TestDefaultConfigurations:
    """Test default configuration presets"""
    
    def test_default_configs_creation(self):
        """Test that all default configs can be created"""
        for name, factory in DEFAULT_CONFIGS.items():
            config = factory()
            assert isinstance(config, OptimizationConfig)
            assert config.validate() is True
    
    def test_ultra_fast_default(self):
        """Test ultra fast default configuration"""
        config = DEFAULT_CONFIGS["ultra_fast"]()
        
        assert config.optimization_level == OptimizationLevel.ULTRA_FAST
        assert config.num_inference_steps == 10
        assert config.width == 768
        assert config.height == 768
    
    def test_balanced_default(self):
        """Test balanced default configuration"""
        config = DEFAULT_CONFIGS["balanced"]()
        
        assert config.optimization_level == OptimizationLevel.BALANCED
        assert config.num_inference_steps == 25
        assert config.width == 1024
        assert config.height == 1024
    
    def test_quality_default(self):
        """Test quality default configuration"""
        config = DEFAULT_CONFIGS["quality"]()
        
        assert config.optimization_level == OptimizationLevel.QUALITY
        assert config.num_inference_steps == 40
        assert config.width == 1280
        assert config.height == 1280
    
    def test_multimodal_default(self):
        """Test multimodal default configuration"""
        config = DEFAULT_CONFIGS["multimodal"]()
        
        assert config.optimization_level == OptimizationLevel.MULTIMODAL
        assert config.enable_qwen2vl_integration is True
        assert config.enable_prompt_enhancement is True


class TestQualityPresets:
    """Test quality preset configurations"""
    
    def test_quality_presets_structure(self):
        """Test that quality presets have required structure"""
        required_keys = ["num_inference_steps", "true_cfg_scale", "description"]
        
        for preset_name, preset_config in QUALITY_PRESETS.items():
            for key in required_keys:
                assert key in preset_config, f"Missing {key} in {preset_name}"
            
            assert isinstance(preset_config["architecture"], ModelArchitecture)
    
    def test_multimodal_preset_features(self):
        """Test multimodal preset has required features"""
        multimodal_preset = QUALITY_PRESETS["multimodal"]
        
        assert multimodal_preset["architecture"] == ModelArchitecture.MULTIMODAL
        assert multimodal_preset["enable_qwen2vl_integration"] is True
        assert multimodal_preset["enable_prompt_enhancement"] is True


class TestQwen2VLConfig:
    """Test Qwen2-VL specific configuration"""
    
    def test_qwen2vl_config_structure(self):
        """Test Qwen2-VL configuration has required keys"""
        required_keys = [
            "model_name", "torch_dtype", "device_map", "trust_remote_code",
            "max_new_tokens", "temperature", "top_p", "do_sample",
            "enable_prompt_enhancement", "enable_image_analysis", "context_length"
        ]
        
        for key in required_keys:
            assert key in QWEN2VL_CONFIG, f"Missing {key} in QWEN2VL_CONFIG"
    
    def test_qwen2vl_config_values(self):
        """Test Qwen2-VL configuration values"""
        assert QWEN2VL_CONFIG["model_name"] == "Qwen/Qwen2-VL-7B-Instruct"
        assert QWEN2VL_CONFIG["torch_dtype"] == torch.float16
        assert QWEN2VL_CONFIG["device_map"] == "auto"
        assert QWEN2VL_CONFIG["trust_remote_code"] is True
        assert QWEN2VL_CONFIG["enable_prompt_enhancement"] is True
        assert QWEN2VL_CONFIG["enable_image_analysis"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])