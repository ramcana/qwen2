"""
Unit tests for EnvironmentSetupManager

Tests environment detection, GPU capability detection, precision selection,
and fallback mechanisms for quantized inference setup.
"""

import pytest
import torch
import os
import logging
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.environment_setup_manager import (
    EnvironmentSetupManager,
    GPUCapabilities,
    EnvironmentConfig,
    PrecisionType
)


class TestEnvironmentSetupManager:
    """Test suite for EnvironmentSetupManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a test manager instance."""
        logger = logging.getLogger("test")
        return EnvironmentSetupManager(logger=logger)
    
    @pytest.fixture
    def mock_gpu_capabilities(self):
        """Mock GPU capabilities for RTX 4080."""
        return GPUCapabilities(
            compute_capability=(8, 9),  # Ada architecture
            supports_bf16=True,
            supports_fp16=True,
            supports_tf32=True,
            memory_gb=16.0,
            name="NVIDIA GeForce RTX 4080",
            driver_version="535.98",
            cuda_version="12.1",
            is_ada_architecture=True
        )
    
    @pytest.fixture
    def mock_older_gpu_capabilities(self):
        """Mock GPU capabilities for older GPU."""
        return GPUCapabilities(
            compute_capability=(7, 5),  # Turing architecture
            supports_bf16=False,
            supports_fp16=True,
            supports_tf32=False,
            memory_gb=8.0,
            name="NVIDIA GeForce GTX 1080 Ti",
            driver_version="470.82",
            cuda_version="11.8",
            is_ada_architecture=False
        )

    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.logger is not None
        assert manager._gpu_capabilities is None
        assert manager._environment_config is None
        assert isinstance(manager._cuda_available, bool)

    @patch('torch.cuda.is_available')
    def test_init_no_cuda(self, mock_cuda_available):
        """Test initialization when CUDA is not available."""
        mock_cuda_available.return_value = False
        manager = EnvironmentSetupManager()
        assert not manager._cuda_available

    @patch('torch.cuda.is_available')
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_cuda_environment_no_cuda(self, mock_cuda_available, manager):
        """Test CUDA environment setup when CUDA is not available."""
        mock_cuda_available.return_value = False
        manager._cuda_available = False
        
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            manager.setup_cuda_environment()

    @patch('torch.cuda.is_available')
    @patch('torch.backends.cuda.matmul.allow_tf32', True)
    @patch('torch.backends.cudnn.allow_tf32', True)
    @patch('torch.backends.cudnn.benchmark', True)
    def test_setup_cuda_environment_success(self, manager, mock_gpu_capabilities):
        """Test successful CUDA environment setup."""
        manager._cuda_available = True
        
        with patch.object(manager, 'detect_gpu_capabilities', return_value=mock_gpu_capabilities):
            with patch.object(manager, 'choose_optimal_precision', return_value=PrecisionType.BF16):
                config = manager.setup_cuda_environment()
                
                assert isinstance(config, EnvironmentConfig)
                assert config.precision_type == PrecisionType.BF16
                assert config.tf32_enabled == True
                assert config.cudnn_benchmark == True
                assert "expandable_segments:True" in config.cuda_alloc_conf
                assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None

    @patch('torch.cuda.current_device')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_properties')
    def test_detect_gpu_capabilities_success(self, mock_props, mock_capability, 
                                           mock_name, mock_device, manager):
        """Test successful GPU capability detection."""
        manager._cuda_available = True
        
        # Mock return values
        mock_device.return_value = 0
        mock_name.return_value = "NVIDIA GeForce RTX 4080"
        mock_capability.return_value = (8, 9)
        
        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value = mock_device_props
        
        with patch.object(manager, '_test_bf16_support', return_value=True):
            with patch.object(manager, '_get_cuda_versions', return_value=("535.98", "12.1")):
                capabilities = manager.detect_gpu_capabilities()
                
                assert capabilities.compute_capability == (8, 9)
                assert capabilities.supports_bf16 == True
                assert capabilities.supports_fp16 == True
                assert capabilities.supports_tf32 == True
                assert capabilities.memory_gb == 16.0
                assert capabilities.name == "NVIDIA GeForce RTX 4080"
                assert capabilities.is_ada_architecture == True

    def test_detect_gpu_capabilities_no_cuda(self, manager):
        """Test GPU capability detection when CUDA is not available."""
        manager._cuda_available = False
        
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            manager.detect_gpu_capabilities()

    def test_detect_gpu_capabilities_cached(self, manager, mock_gpu_capabilities):
        """Test that GPU capabilities are cached after first detection."""
        manager._gpu_capabilities = mock_gpu_capabilities
        
        # Should return cached value without calling detection methods
        result = manager.detect_gpu_capabilities()
        assert result == mock_gpu_capabilities

    def test_choose_optimal_precision_ada_architecture(self, manager, mock_gpu_capabilities):
        """Test precision selection for Ada architecture."""
        precision = manager.choose_optimal_precision(mock_gpu_capabilities)
        assert precision == PrecisionType.BF16

    def test_choose_optimal_precision_ampere_architecture(self, manager):
        """Test precision selection for Ampere architecture."""
        ampere_caps = GPUCapabilities(
            compute_capability=(8, 0),
            supports_bf16=True,
            supports_fp16=True,
            supports_tf32=True,
            memory_gb=24.0,
            name="NVIDIA A100",
            driver_version="535.98",
            cuda_version="12.1",
            is_ada_architecture=False
        )
        
        precision = manager.choose_optimal_precision(ampere_caps)
        assert precision == PrecisionType.BF16

    def test_choose_optimal_precision_older_gpu(self, manager, mock_older_gpu_capabilities):
        """Test precision selection for older GPU without bf16."""
        precision = manager.choose_optimal_precision(mock_older_gpu_capabilities)
        assert precision == PrecisionType.FP16

    def test_choose_optimal_precision_no_mixed_precision(self, manager):
        """Test precision selection for GPU without mixed precision support."""
        old_caps = GPUCapabilities(
            compute_capability=(6, 1),
            supports_bf16=False,
            supports_fp16=False,
            supports_tf32=False,
            memory_gb=8.0,
            name="NVIDIA GTX 1070",
            driver_version="470.82",
            cuda_version="11.8",
            is_ada_architecture=False
        )
        
        precision = manager.choose_optimal_precision(old_caps)
        assert precision == PrecisionType.FP32

    @patch('torch.cuda.current_device')
    @patch('torch.randn')
    @patch('torch.matmul')
    def test_test_precision_stability_success(self, mock_matmul, mock_randn, 
                                            mock_device, manager):
        """Test precision stability testing with stable results."""
        mock_device.return_value = 0
        
        # Mock stable tensor operations
        mock_tensor = Mock()
        mock_tensor.requires_grad_ = Mock(return_value=mock_tensor)
        mock_randn.return_value = mock_tensor
        
        mock_result = Mock()
        mock_result.sum.return_value = Mock()
        mock_result.sum.return_value.backward = Mock()
        mock_matmul.return_value = mock_result
        
        # Mock no NaN/Inf detection
        with patch('torch.isnan') as mock_isnan:
            with patch('torch.isinf') as mock_isinf:
                mock_isnan.return_value.any.return_value.item.return_value = False
                mock_isinf.return_value.any.return_value.item.return_value = False
                mock_tensor.grad = Mock()
                
                result = manager.test_precision_stability(PrecisionType.BF16)
                assert result == True

    @patch('torch.cuda.current_device')
    @patch('torch.randn')
    @patch('torch.matmul')
    def test_test_precision_stability_nan_detected(self, mock_matmul, mock_randn, 
                                                 mock_device, manager):
        """Test precision stability testing with NaN detection."""
        mock_device.return_value = 0
        mock_randn.return_value = Mock()
        mock_matmul.return_value = Mock()
        
        # Mock NaN detection
        with patch('torch.isnan') as mock_isnan:
            with patch('torch.isinf') as mock_isinf:
                mock_isnan.return_value.any.return_value.item.return_value = True
                mock_isinf.return_value.any.return_value.item.return_value = False
                
                result = manager.test_precision_stability(PrecisionType.BF16)
                assert result == False

    def test_get_fallback_precision_bf16(self, manager):
        """Test fallback from bf16 to fp16."""
        fallback = manager.get_fallback_precision(PrecisionType.BF16)
        assert fallback == PrecisionType.FP16

    def test_get_fallback_precision_fp16(self, manager):
        """Test fallback from fp16 to fp32."""
        fallback = manager.get_fallback_precision(PrecisionType.FP16)
        assert fallback == PrecisionType.FP32

    def test_get_fallback_precision_fp32(self, manager):
        """Test fallback from fp32 (no further fallback)."""
        fallback = manager.get_fallback_precision(PrecisionType.FP32)
        assert fallback == PrecisionType.FP32

    @patch('src.environment_setup_manager.BITSANDBYTES_AVAILABLE', True)
    @patch('src.environment_setup_manager.bnb')
    def test_validate_quantization_dependencies_success(self, mock_bnb, manager, 
                                                       mock_gpu_capabilities):
        """Test successful quantization dependency validation."""
        manager._cuda_available = True
        
        # Mock successful bitsandbytes test
        mock_bnb.nn.Linear4bit.return_value = Mock()
        
        with patch('torch.version.cuda', "12.1"):
            with patch('torch.__version__', "2.1.0"):
                with patch.object(manager, 'detect_gpu_capabilities', 
                                return_value=mock_gpu_capabilities):
                    results = manager.validate_quantization_dependencies()
                    
                    assert results['bitsandbytes'] == True
                    assert results['bitsandbytes_functional'] == True
                    assert results['cuda_compatible'] == True
                    assert results['torch_compatible'] == True
                    assert results['sufficient_vram'] == True

    @patch('src.environment_setup_manager.BITSANDBYTES_AVAILABLE', False)
    def test_validate_quantization_dependencies_no_bitsandbytes(self, manager):
        """Test quantization dependency validation without bitsandbytes."""
        results = manager.validate_quantization_dependencies()
        
        assert results['bitsandbytes'] == False
        assert results['bitsandbytes_functional'] == False

    def test_get_recommended_settings_no_cuda(self, manager):
        """Test recommended settings when CUDA is not available."""
        manager._cuda_available = False
        
        settings = manager.get_recommended_settings()
        assert "error" in settings
        assert settings["error"] == "CUDA not available"

    def test_get_recommended_settings_success(self, manager, mock_gpu_capabilities):
        """Test successful recommended settings generation."""
        manager._cuda_available = True
        
        with patch.object(manager, 'detect_gpu_capabilities', 
                         return_value=mock_gpu_capabilities):
            with patch.object(manager, 'choose_optimal_precision', 
                            return_value=PrecisionType.BF16):
                with patch.object(manager, 'test_precision_stability', 
                                return_value=True):
                    settings = manager.get_recommended_settings()
                    
                    assert settings['precision'] == 'bfloat16'
                    assert settings['max_memory_gb'] == 13.6  # 16 * 0.85
                    assert settings['batch_size'] == 2  # 16GB GPU
                    assert settings['enable_tf32'] == True
                    assert settings['enable_flash_attention'] == True
                    assert settings['quantization_recommended'] == True
                    assert settings['offload_recommended'] == True

    def test_get_recommended_settings_with_fallback(self, manager, mock_gpu_capabilities):
        """Test recommended settings with precision fallback."""
        manager._cuda_available = True
        
        with patch.object(manager, 'detect_gpu_capabilities', 
                         return_value=mock_gpu_capabilities):
            with patch.object(manager, 'choose_optimal_precision', 
                            return_value=PrecisionType.BF16):
                with patch.object(manager, 'test_precision_stability', 
                                return_value=False):
                    with patch.object(manager, 'get_fallback_precision', 
                                    return_value=PrecisionType.FP16):
                        settings = manager.get_recommended_settings()
                        
                        assert settings['precision'] == 'float16'

    @patch('torch.randn')
    @patch('torch.matmul')
    @patch('torch.cuda.current_device')
    def test_test_bf16_support_success(self, mock_device, mock_matmul, mock_randn, manager):
        """Test successful bf16 support detection."""
        mock_device.return_value = 0
        mock_randn.return_value = Mock()
        mock_result = Mock()
        mock_matmul.return_value = mock_result
        
        with patch('torch.isnan') as mock_isnan:
            with patch('torch.isinf') as mock_isinf:
                mock_isnan.return_value.any.return_value = False
                mock_isinf.return_value.any.return_value = False
                
                result = manager._test_bf16_support()
                assert result == True

    def test_test_bf16_support_no_bfloat16_attr(self, manager):
        """Test bf16 support detection when bfloat16 attribute is missing."""
        with patch('torch.bfloat16', side_effect=AttributeError):
            result = manager._test_bf16_support()
            assert result == False

    @patch('src.environment_setup_manager.PYNVML_AVAILABLE', True)
    @patch('src.environment_setup_manager.pynvml')
    def test_get_cuda_versions_with_pynvml(self, mock_pynvml, manager):
        """Test CUDA version detection with pynvml available."""
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = b"535.98"
        
        with patch('torch.version.cuda', "12.1"):
            driver_version, cuda_version = manager._get_cuda_versions()
            
            assert driver_version == "535.98"
            assert cuda_version == "12.1"

    @patch('src.environment_setup_manager.PYNVML_AVAILABLE', False)
    def test_get_cuda_versions_without_pynvml(self, manager):
        """Test CUDA version detection without pynvml."""
        with patch('torch.version.cuda', "12.1"):
            driver_version, cuda_version = manager._get_cuda_versions()
            
            assert driver_version == "unknown"
            assert cuda_version == "12.1"

    def test_get_cuda_versions_exception_handling(self, manager):
        """Test CUDA version detection with exceptions."""
        with patch('torch.version.cuda', side_effect=Exception("Test error")):
            driver_version, cuda_version = manager._get_cuda_versions()
            
            assert driver_version == "unknown"
            assert cuda_version == "unknown"


if __name__ == "__main__":
    pytest.main([__file__])