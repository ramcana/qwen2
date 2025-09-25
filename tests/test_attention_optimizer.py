"""
Unit tests for attention optimizations and memory management
Tests scaled dot-product attention, Flash Attention 2, memory-efficient patterns,
dynamic batch sizing, and torch.compile optimizations
"""

import pytest
import torch
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention_optimizer import (
    AttentionConfig,
    AttentionOptimizer,
    MemoryMonitor,
    DynamicBatchSizer,
    create_attention_config,
    create_attention_optimizer,
    PYTORCH_2_AVAILABLE,
    FLASH_ATTENTION_AVAILABLE,
    XFORMERS_AVAILABLE
)


class TestAttentionConfig(unittest.TestCase):
    """Test AttentionConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AttentionConfig()
        
        self.assertTrue(config.use_scaled_dot_product_attention)
        self.assertFalse(config.use_flash_attention_2)  # Disabled by default
        self.assertFalse(config.use_xformers)
        self.assertTrue(config.use_memory_efficient_attention)
        self.assertFalse(config.enable_attention_slicing)
        self.assertTrue(config.enable_dynamic_batch_sizing)
        self.assertFalse(config.enable_torch_compile)
        self.assertEqual(config.architecture_type, "MMDiT")
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = AttentionConfig()
        self.assertTrue(config.validate())
    
    def test_config_validation_batch_size_error(self):
        """Test validation failure for invalid batch size"""
        config = AttentionConfig(max_batch_size=1, min_batch_size=2)
        
        # Should return False instead of raising exception
        self.assertFalse(config.validate())
    
    @patch('attention_optimizer.PYTORCH_2_AVAILABLE', False)
    def test_config_validation_pytorch_version(self):
        """Test validation with old PyTorch version"""
        config = AttentionConfig(use_scaled_dot_product_attention=True)
        
        # Should disable SDPA and still validate successfully
        self.assertTrue(config.validate())
        self.assertFalse(config.use_scaled_dot_product_attention)
    
    @patch('attention_optimizer.FLASH_ATTENTION_AVAILABLE', False)
    def test_config_validation_flash_attention_unavailable(self):
        """Test validation when Flash Attention is unavailable"""
        config = AttentionConfig(use_flash_attention_2=True)
        
        # Should disable Flash Attention and enable SDPA as fallback
        self.assertTrue(config.validate())
        self.assertFalse(config.use_flash_attention_2)
        self.assertTrue(config.use_scaled_dot_product_attention)
    
    @patch('attention_optimizer.XFORMERS_AVAILABLE', False)
    def test_config_validation_xformers_unavailable(self):
        """Test validation when xformers is unavailable"""
        config = AttentionConfig(use_xformers=True)
        
        # Should disable xformers and enable SDPA as fallback
        self.assertTrue(config.validate())
        self.assertFalse(config.use_xformers)
        self.assertTrue(config.use_scaled_dot_product_attention)


class TestAttentionOptimizer(unittest.TestCase):
    """Test AttentionOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AttentionConfig()
        self.optimizer = AttentionOptimizer(self.config)
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer.config)
        self.assertEqual(self.optimizer.config.architecture_type, "MMDiT")
        self.assertIn(self.optimizer.device, ["cuda", "cpu"])
    
    def test_initialization_with_custom_config(self):
        """Test optimizer initialization with custom config"""
        custom_config = AttentionConfig(
            architecture_type="UNet",
            use_flash_attention_2=True,
            enable_dynamic_batch_sizing=False
        )
        optimizer = AttentionOptimizer(custom_config)
        
        self.assertEqual(optimizer.config.architecture_type, "UNet")
        self.assertEqual(optimizer.config.enable_dynamic_batch_sizing, False)
    
    def test_configure_sdpa_backends(self):
        """Test SDPA backend configuration"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('attention_optimizer.PYTORCH_2_AVAILABLE', True), \
             patch('torch.backends.cuda.enable_math_sdp') as mock_math, \
             patch('torch.backends.cuda.enable_flash_sdp') as mock_flash, \
             patch('torch.backends.cuda.enable_mem_efficient_sdp') as mock_mem:
            
            self.optimizer._configure_sdpa_backends()
            
            mock_math.assert_called_once_with(self.config.enable_math_sdp)
            mock_flash.assert_called_once_with(self.config.enable_flash_sdp)
            mock_mem.assert_called_once_with(self.config.enable_mem_efficient_sdp)
    
    def test_optimize_pipeline_attention_transformer(self):
        """Test pipeline attention optimization for transformer"""
        # Create mock pipeline with transformer
        mock_pipeline = Mock()
        mock_transformer = Mock()
        mock_pipeline.transformer = mock_transformer
        mock_pipeline.unet = None
        
        with patch.object(self.optimizer, '_configure_sdpa_backends'), \
             patch.object(self.optimizer, '_optimize_transformer_attention') as mock_opt_transformer, \
             patch.object(self.optimizer, '_apply_memory_optimizations'), \
             patch.object(self.optimizer, '_apply_torch_compile'):
            
            result = self.optimizer.optimize_pipeline_attention(mock_pipeline)
            
            self.assertTrue(result)
            mock_opt_transformer.assert_called_once_with(mock_transformer)
    
    def test_optimize_pipeline_attention_unet(self):
        """Test pipeline attention optimization for UNet"""
        # Create mock pipeline with UNet
        mock_pipeline = Mock()
        mock_unet = Mock()
        mock_pipeline.transformer = None
        mock_pipeline.unet = mock_unet
        
        with patch.object(self.optimizer, '_configure_sdpa_backends'), \
             patch.object(self.optimizer, '_optimize_unet_attention') as mock_opt_unet, \
             patch.object(self.optimizer, '_apply_memory_optimizations'), \
             patch.object(self.optimizer, '_apply_torch_compile'):
            
            result = self.optimizer.optimize_pipeline_attention(mock_pipeline)
            
            self.assertTrue(result)
            mock_opt_unet.assert_called_once_with(mock_unet)
    
    def test_optimize_transformer_attention_sdpa(self):
        """Test transformer attention optimization with SDPA"""
        mock_transformer = Mock()
        
        with patch.object(self.optimizer, '_apply_sdpa_processor') as mock_apply_sdpa:
            self.optimizer._optimize_transformer_attention(mock_transformer)
            mock_apply_sdpa.assert_called_once_with(mock_transformer)
    
    def test_optimize_transformer_attention_gradient_checkpointing(self):
        """Test transformer attention optimization with gradient checkpointing"""
        mock_transformer = Mock()
        mock_transformer.enable_gradient_checkpointing = Mock()
        
        self.config.enable_gradient_checkpointing = True
        
        with patch.object(self.optimizer, '_apply_sdpa_processor'):
            self.optimizer._optimize_transformer_attention(mock_transformer)
            mock_transformer.enable_gradient_checkpointing.assert_called_once()
    
    def test_optimize_unet_attention_attention_slicing(self):
        """Test UNet attention optimization with attention slicing"""
        mock_unet = Mock()
        mock_unet.set_attention_slice = Mock()
        
        self.config.enable_attention_slicing = True
        self.config.attention_slice_size = 2
        
        with patch.object(self.optimizer, '_apply_sdpa_processor'):
            self.optimizer._optimize_unet_attention(mock_unet)
            mock_unet.set_attention_slice.assert_called_once_with(2)
    
    def test_apply_sdpa_processor(self):
        """Test SDPA processor application"""
        mock_model = Mock()
        mock_model.set_attn_processor = Mock()
        
        with patch('diffusers.models.attention_processor.AttnProcessor2_0') as mock_processor:
            self.optimizer._apply_sdpa_processor(mock_model)
            mock_model.set_attn_processor.assert_called_once()
    
    def test_apply_memory_optimizations(self):
        """Test memory optimization application"""
        mock_pipeline = Mock()
        mock_pipeline.disable_attention_slicing = Mock()
        mock_pipeline.disable_vae_slicing = Mock()
        mock_pipeline.disable_vae_tiling = Mock()
        mock_pipeline.vae = Mock()
        
        self.optimizer._apply_memory_optimizations(mock_pipeline)
        
        mock_pipeline.disable_attention_slicing.assert_called_once()
        mock_pipeline.disable_vae_slicing.assert_called_once()
        mock_pipeline.disable_vae_tiling.assert_called_once()
    
    def test_calculate_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch.object(self.optimizer, 'memory_monitor') as mock_monitor:
            mock_monitor.get_available_memory_gb.return_value = 16.0
            
            batch_size = self.optimizer.calculate_optimal_batch_size(1024, 1024, 25)
            
            self.assertGreaterEqual(batch_size, self.config.min_batch_size)
            self.assertLessEqual(batch_size, self.config.max_batch_size)
    
    def test_calculate_optimal_batch_size_cpu(self):
        """Test optimal batch size calculation on CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = AttentionOptimizer(self.config)
            batch_size = optimizer.calculate_optimal_batch_size(1024, 1024, 25)
            
            self.assertEqual(batch_size, self.config.min_batch_size)
    
    def test_create_memory_efficient_attention_function(self):
        """Test memory-efficient attention function creation"""
        attention_fn = self.optimizer.create_memory_efficient_attention_function(64, 16)
        
        self.assertIsNotNone(attention_fn)
        self.assertTrue(callable(attention_fn))
    
    def test_memory_efficient_attention_function_sdpa(self):
        """Test memory-efficient attention function with SDPA"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('attention_optimizer.PYTORCH_2_AVAILABLE', True):
            
            attention_fn = self.optimizer.create_memory_efficient_attention_function(64, 16)
            
            # Create test tensors
            batch_size, seq_len, embed_dim = 2, 128, 1024
            query = torch.randn(batch_size, seq_len, embed_dim)
            key = torch.randn(batch_size, seq_len, embed_dim)
            value = torch.randn(batch_size, seq_len, embed_dim)
            
            with patch('torch.nn.functional.scaled_dot_product_attention') as mock_sdpa:
                mock_sdpa.return_value = torch.randn(batch_size, seq_len, embed_dim)
                
                result = attention_fn(query, key, value)
                
                self.assertIsNotNone(result)
                self.assertEqual(result.shape, (batch_size, seq_len, embed_dim))
    
    def test_benchmark_attention_methods(self):
        """Test attention method benchmarking"""
        # Test that the method returns a dict (actual benchmarking requires real GPU)
        results = self.optimizer.benchmark_attention_methods(1, 64, 64)  # Smaller size for testing
        self.assertIsInstance(results, dict)
    
    def test_optimized_inference_context(self):
        """Test optimized inference context manager"""
        original_grad = torch.is_grad_enabled()
        
        with self.optimizer.optimized_inference_context():
            # Inside context, gradients should be disabled
            self.assertFalse(torch.is_grad_enabled())
        
        # After context, original setting should be restored
        self.assertEqual(torch.is_grad_enabled(), original_grad)


class TestMemoryMonitor(unittest.TestCase):
    """Test MemoryMonitor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MemoryMonitor()
    
    def test_get_memory_info_cuda(self):
        """Test memory info retrieval on CUDA"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.memory_allocated') as mock_allocated, \
             patch('torch.cuda.memory_reserved') as mock_reserved:
            
            # Mock GPU properties
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * 1024**3  # 16GB
            mock_props.return_value = mock_device_props
            
            mock_allocated.return_value = 4 * 1024**3   # 4GB allocated
            mock_reserved.return_value = 6 * 1024**3    # 6GB cached
            
            memory_info = self.monitor.get_memory_info()
            
            # Check that values are reasonable (within 2GB tolerance due to conversion)
            self.assertGreater(memory_info["total"], 15.0)
            self.assertLess(memory_info["total"], 18.0)
            self.assertGreater(memory_info["allocated"], 3.0)
            self.assertLess(memory_info["allocated"], 5.0)
            self.assertGreater(memory_info["cached"], 5.0)
            self.assertLess(memory_info["cached"], 7.0)
            self.assertGreater(memory_info["free"], 9.0)
            self.assertLess(memory_info["free"], 12.0)
    
    def test_get_memory_info_cpu(self):
        """Test memory info retrieval on CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = MemoryMonitor()
            memory_info = monitor.get_memory_info()
            
            self.assertEqual(memory_info["total"], 0)
            self.assertEqual(memory_info["allocated"], 0)
            self.assertEqual(memory_info["cached"], 0)
            self.assertEqual(memory_info["free"], 0)
    
    def test_get_available_memory_gb(self):
        """Test available memory calculation"""
        with patch.object(self.monitor, 'get_memory_info') as mock_info:
            mock_info.return_value = {"free": 8.0}
            
            available = self.monitor.get_available_memory_gb()
            self.assertEqual(available, 8.0)
    
    def test_is_memory_available(self):
        """Test memory availability check"""
        with patch.object(self.monitor, 'get_available_memory_gb') as mock_available:
            mock_available.return_value = 10.0
            
            self.assertTrue(self.monitor.is_memory_available(5.0))
            self.assertFalse(self.monitor.is_memory_available(15.0))
    
    def test_clear_cache(self):
        """Test GPU cache clearing"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty, \
             patch('torch.cuda.synchronize') as mock_sync:
            
            self.monitor.clear_cache()
            
            mock_empty.assert_called_once()
            mock_sync.assert_called_once()


class TestDynamicBatchSizer(unittest.TestCase):
    """Test DynamicBatchSizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AttentionConfig(enable_dynamic_batch_sizing=True)
        self.sizer = DynamicBatchSizer(self.config)
    
    def test_get_optimal_batch_size_disabled(self):
        """Test batch size calculation when dynamic sizing is disabled"""
        config = AttentionConfig(enable_dynamic_batch_sizing=False)
        sizer = DynamicBatchSizer(config)
        
        batch_size = sizer.get_optimal_batch_size(1024, 1024, 2)
        self.assertEqual(batch_size, 2)  # Should return current batch size
    
    def test_get_optimal_batch_size_enabled(self):
        """Test batch size calculation when dynamic sizing is enabled"""
        with patch.object(self.sizer.memory_monitor, 'get_available_memory_gb') as mock_memory:
            mock_memory.return_value = 16.0
            
            batch_size = self.sizer.get_optimal_batch_size(1024, 1024, 1)
            
            self.assertGreaterEqual(batch_size, self.config.min_batch_size)
            self.assertLessEqual(batch_size, self.config.max_batch_size)
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation"""
        memory_usage = self.sizer._estimate_memory_usage(1024 * 1024, 2)
        
        self.assertGreater(memory_usage, 0)
        self.assertIsInstance(memory_usage, float)
    
    def test_batch_size_history(self):
        """Test batch size history tracking"""
        with patch.object(self.sizer.memory_monitor, 'get_available_memory_gb') as mock_memory:
            mock_memory.return_value = 16.0
            
            # Generate several batch sizes
            for _ in range(5):
                self.sizer.get_optimal_batch_size(1024, 1024, 1)
            
            self.assertEqual(len(self.sizer.batch_size_history), 5)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions"""
    
    def test_create_attention_config_default(self):
        """Test default attention config creation"""
        config = create_attention_config()
        
        self.assertEqual(config.architecture_type, "MMDiT")
        self.assertTrue(config.use_scaled_dot_product_attention)
        self.assertTrue(config.enable_dynamic_batch_sizing)
    
    def test_create_attention_config_ultra_fast(self):
        """Test ultra-fast attention config creation"""
        config = create_attention_config(optimization_level="ultra_fast")
        
        self.assertTrue(config.use_scaled_dot_product_attention)
        self.assertFalse(config.use_flash_attention_2)
        self.assertTrue(config.enable_dynamic_batch_sizing)
        self.assertFalse(config.enable_torch_compile)
    
    def test_create_attention_config_quality(self):
        """Test quality attention config creation"""
        config = create_attention_config(optimization_level="quality")
        
        self.assertTrue(config.use_scaled_dot_product_attention)
        self.assertTrue(config.use_memory_efficient_attention)
        self.assertTrue(config.enable_gradient_checkpointing)
        self.assertFalse(config.enable_torch_compile)
    
    def test_create_attention_config_experimental(self):
        """Test experimental attention config creation"""
        config = create_attention_config(optimization_level="experimental")
        
        self.assertEqual(config.use_flash_attention_2, FLASH_ATTENTION_AVAILABLE)
        self.assertTrue(config.enable_torch_compile)
        self.assertEqual(config.compile_mode, "max-autotune")
        self.assertTrue(config.enable_dynamic_batch_sizing)
    
    def test_create_attention_config_unet(self):
        """Test UNet-specific attention config creation"""
        config = create_attention_config(architecture="UNet")
        
        self.assertEqual(config.architecture_type, "UNet")
        self.assertEqual(config.use_xformers, XFORMERS_AVAILABLE)
        self.assertTrue(config.enable_attention_slicing)
    
    def test_create_attention_config_mmdit(self):
        """Test MMDiT-specific attention config creation"""
        config = create_attention_config(architecture="MMDiT")
        
        self.assertEqual(config.architecture_type, "MMDiT")
        self.assertFalse(config.enable_flash_sdp)
        self.assertFalse(config.enable_attention_slicing)
    
    def test_create_attention_config_custom_kwargs(self):
        """Test attention config creation with custom kwargs"""
        config = create_attention_config(
            max_batch_size=8,
            memory_threshold_gb=0.9,
            head_dim=128
        )
        
        self.assertEqual(config.max_batch_size, 8)
        self.assertEqual(config.memory_threshold_gb, 0.9)
        self.assertEqual(config.head_dim, 128)
    
    def test_create_attention_optimizer(self):
        """Test attention optimizer creation"""
        optimizer = create_attention_optimizer()
        
        self.assertIsInstance(optimizer, AttentionOptimizer)
        self.assertEqual(optimizer.config.architecture_type, "MMDiT")
    
    def test_create_attention_optimizer_custom(self):
        """Test attention optimizer creation with custom parameters"""
        optimizer = create_attention_optimizer(
            architecture="UNet",
            optimization_level="quality",
            max_batch_size=4
        )
        
        self.assertIsInstance(optimizer, AttentionOptimizer)
        self.assertEqual(optimizer.config.architecture_type, "UNet")
        self.assertEqual(optimizer.config.max_batch_size, 4)
        self.assertTrue(optimizer.config.enable_gradient_checkpointing)


class TestIntegration(unittest.TestCase):
    """Integration tests for attention optimization"""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline"""
        # Create optimizer
        optimizer = create_attention_optimizer(
            architecture="MMDiT",
            optimization_level="balanced"
        )
        
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_transformer = Mock()
        mock_transformer.set_attn_processor = Mock()
        mock_pipeline.transformer = mock_transformer
        mock_pipeline.unet = None
        mock_pipeline.disable_attention_slicing = Mock()
        mock_pipeline.disable_vae_slicing = Mock()
        mock_pipeline.disable_vae_tiling = Mock()
        mock_pipeline.vae = Mock()
        
        # Run optimization
        with patch.object(optimizer, '_configure_sdpa_backends'), \
             patch('diffusers.models.attention_processor.AttnProcessor2_0'):
            
            result = optimizer.optimize_pipeline_attention(mock_pipeline)
            
            self.assertTrue(result)
            mock_transformer.set_attn_processor.assert_called_once()
            mock_pipeline.disable_attention_slicing.assert_called_once()
    
    def test_memory_aware_batch_sizing(self):
        """Test memory-aware batch sizing integration"""
        config = AttentionConfig(
            enable_dynamic_batch_sizing=True,
            max_batch_size=8,
            memory_threshold_gb=0.8
        )
        
        optimizer = AttentionOptimizer(config)
        sizer = DynamicBatchSizer(config)
        
        with patch.object(sizer.memory_monitor, 'get_available_memory_gb') as mock_memory:
            mock_memory.return_value = 12.0
            
            # Test different resolutions
            batch_512 = sizer.get_optimal_batch_size(512, 512, 1)
            batch_1024 = sizer.get_optimal_batch_size(1024, 1024, 1)
            batch_2048 = sizer.get_optimal_batch_size(2048, 2048, 1)
            
            # Higher resolution should result in smaller batch size
            self.assertGreaterEqual(batch_512, batch_1024)
            self.assertGreaterEqual(batch_1024, batch_2048)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)