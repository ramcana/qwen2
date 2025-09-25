"""
Unit tests for PipelineOptimizer class
Tests pipeline configuration and optimization settings for MMDiT architecture
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline_optimizer import PipelineOptimizer, OptimizationConfig


class TestOptimizationConfig(unittest.TestCase):
    """Test OptimizationConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = OptimizationConfig()
        
        # Test default values
        self.assertEqual(config.torch_dtype, torch.float16)
        self.assertEqual(config.device, "cuda" if torch.cuda.is_available() else "cpu")
        self.assertFalse(config.enable_attention_slicing)
        self.assertFalse(config.enable_vae_slicing)
        self.assertFalse(config.enable_cpu_offload)
        self.assertTrue(config.enable_tf32)
        self.assertTrue(config.enable_cudnn_benchmark)
        self.assertEqual(config.optimal_steps, 20)
        self.assertEqual(config.optimal_cfg_scale, 3.5)
        self.assertEqual(config.architecture_type, "MMDiT")
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = OptimizationConfig(
            torch_dtype=torch.float32,
            enable_attention_slicing=True,
            optimal_steps=30,
            architecture_type="UNet"
        )
        
        self.assertEqual(config.torch_dtype, torch.float32)
        self.assertTrue(config.enable_attention_slicing)
        self.assertEqual(config.optimal_steps, 30)
        self.assertEqual(config.architecture_type, "UNet")


class TestPipelineOptimizer(unittest.TestCase):
    """Test PipelineOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OptimizationConfig()
        self.optimizer = PipelineOptimizer(self.config)
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsInstance(self.optimizer.config, OptimizationConfig)
        self.assertEqual(self.optimizer.device, self.config.device)
    
    def test_initialization_with_cuda_unavailable(self):
        """Test initialization when CUDA is requested but unavailable"""
        with patch('torch.cuda.is_available', return_value=False):
            config = OptimizationConfig(device="cuda")
            optimizer = PipelineOptimizer(config)
            self.assertEqual(optimizer.device, "cpu")
            self.assertEqual(optimizer.config.device, "cpu")
    
    def test_select_pipeline_class_mmdit_text2image(self):
        """Test pipeline class selection for MMDiT text-to-image"""
        with patch('pipeline_optimizer.AutoPipelineForText2Image') as mock_auto:
            pipeline_class = self.optimizer._select_pipeline_class("Qwen/Qwen-Image", "MMDiT")
            self.assertEqual(pipeline_class, mock_auto)
    
    def test_select_pipeline_class_mmdit_editing(self):
        """Test pipeline class selection for MMDiT image editing"""
        with patch('pipeline_optimizer.DiffusionPipeline') as mock_diffusion:
            pipeline_class = self.optimizer._select_pipeline_class("Qwen/Qwen-Image-Edit", "MMDiT")
            self.assertEqual(pipeline_class, mock_diffusion)
    
    def test_select_pipeline_class_unet(self):
        """Test pipeline class selection for UNet architecture"""
        with patch('pipeline_optimizer.DiffusionPipeline') as mock_diffusion:
            pipeline_class = self.optimizer._select_pipeline_class("some-model", "UNet")
            self.assertEqual(pipeline_class, mock_diffusion)
    
    def test_prepare_loading_kwargs(self):
        """Test preparation of loading keyword arguments"""
        kwargs = self.optimizer._prepare_loading_kwargs()
        
        expected_keys = [
            'torch_dtype', 'use_safetensors', 'trust_remote_code',
            'low_cpu_mem_usage', 'resume_download', 'force_download', 'device_map'
        ]
        
        for key in expected_keys:
            self.assertIn(key, kwargs)
        
        self.assertEqual(kwargs['torch_dtype'], self.config.torch_dtype)
        self.assertTrue(kwargs['use_safetensors'])
        self.assertTrue(kwargs['trust_remote_code'])
        self.assertFalse(kwargs['force_download'])
    
    def test_prepare_loading_kwargs_with_cpu_offload(self):
        """Test loading kwargs with CPU offload enabled"""
        config = OptimizationConfig(enable_cpu_offload=True)
        optimizer = PipelineOptimizer(config)
        kwargs = optimizer._prepare_loading_kwargs()
        
        self.assertEqual(kwargs['device_map'], "balanced")
    
    @patch('torch.backends.cuda.matmul')
    @patch('torch.backends.cudnn')
    @patch('torch.backends.cuda')
    @patch('torch.set_grad_enabled')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_apply_gpu_optimizations_cuda(self, mock_sync, mock_cache, mock_grad, 
                                         mock_cuda_backends, mock_cudnn, mock_matmul):
        """Test GPU optimizations application with CUDA"""
        config = OptimizationConfig(device="cuda")
        optimizer = PipelineOptimizer(config)
        
        mock_pipeline = Mock()
        optimizer._apply_gpu_optimizations(mock_pipeline)
        
        # Verify TF32 settings
        self.assertTrue(mock_matmul.allow_tf32)
        self.assertTrue(mock_cudnn.allow_tf32)
        
        # Verify cuDNN settings
        self.assertTrue(mock_cudnn.benchmark)
        self.assertFalse(mock_cudnn.deterministic)
        
        # Verify gradient computation disabled
        mock_grad.assert_called_with(False)
        
        # Verify cache clearing
        mock_cache.assert_called_once()
        mock_sync.assert_called_once()
    
    def test_apply_gpu_optimizations_cpu(self):
        """Test GPU optimizations with CPU device"""
        config = OptimizationConfig(device="cpu")
        optimizer = PipelineOptimizer(config)
        
        mock_pipeline = Mock()
        # Should not raise any exceptions
        optimizer._apply_gpu_optimizations(mock_pipeline)
    
    def test_configure_memory_settings_disable_all(self):
        """Test memory settings configuration with all features disabled"""
        mock_pipeline = Mock()
        mock_pipeline.disable_attention_slicing = Mock()
        mock_pipeline.disable_vae_slicing = Mock()
        mock_pipeline.disable_vae_tiling = Mock()
        
        self.optimizer._configure_memory_settings(mock_pipeline)
        
        mock_pipeline.disable_attention_slicing.assert_called_once()
        mock_pipeline.disable_vae_slicing.assert_called_once()
        mock_pipeline.disable_vae_tiling.assert_called_once()
    
    def test_configure_memory_settings_enable_features(self):
        """Test memory settings with features enabled"""
        config = OptimizationConfig(
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            enable_cpu_offload=True
        )
        optimizer = PipelineOptimizer(config)
        
        mock_pipeline = Mock()
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline.enable_vae_slicing = Mock()
        mock_pipeline.enable_model_cpu_offload = Mock()
        
        optimizer._configure_memory_settings(mock_pipeline)
        
        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_vae_slicing.assert_called_once()
        mock_pipeline.enable_model_cpu_offload.assert_called_once()
    
    def test_setup_attention_processors_mmdit(self):
        """Test attention processor setup for MMDiT architecture"""
        mock_pipeline = Mock()
        
        # Should not set any attention processors for MMDiT (compatibility)
        self.optimizer._setup_attention_processors(mock_pipeline, "MMDiT")
        
        # Verify no attention processor methods were called
        self.assertFalse(hasattr(mock_pipeline, 'set_attn_processor') and 
                        mock_pipeline.set_attn_processor.called)
    
    @patch('diffusers.models.attention_processor.AttnProcessor2_0')
    def test_setup_attention_processors_unet(self, mock_attn_processor):
        """Test attention processor setup for UNet architecture"""
        mock_pipeline = Mock()
        mock_unet = Mock()
        mock_pipeline.unet = mock_unet
        
        self.optimizer._setup_attention_processors(mock_pipeline, "UNet")
        
        # Verify AttnProcessor2_0 was set for UNet
        mock_unet.set_attn_processor.assert_called_once()
    
    def test_configure_generation_settings_mmdit(self):
        """Test generation settings configuration for MMDiT"""
        settings = self.optimizer.configure_generation_settings("MMDiT")
        
        expected_keys = [
            'width', 'height', 'num_inference_steps', 'true_cfg_scale', 
            'output_type', 'return_dict'
        ]
        
        for key in expected_keys:
            self.assertIn(key, settings)
        
        self.assertEqual(settings['width'], self.config.optimal_width)
        self.assertEqual(settings['height'], self.config.optimal_height)
        self.assertEqual(settings['num_inference_steps'], self.config.optimal_steps)
        self.assertEqual(settings['true_cfg_scale'], self.config.optimal_cfg_scale)
        self.assertEqual(settings['output_type'], 'pil')
    
    def test_configure_generation_settings_unet(self):
        """Test generation settings configuration for UNet"""
        settings = self.optimizer.configure_generation_settings("UNet")
        
        expected_keys = [
            'width', 'height', 'num_inference_steps', 'guidance_scale', 
            'output_type', 'return_dict'
        ]
        
        for key in expected_keys:
            self.assertIn(key, settings)
        
        self.assertEqual(settings['guidance_scale'], self.config.optimal_cfg_scale)
        self.assertNotIn('true_cfg_scale', settings)
    
    def test_configure_generation_settings_unknown(self):
        """Test generation settings for unknown architecture"""
        settings = self.optimizer.configure_generation_settings("Unknown")
        
        # Should include both CFG scale parameter names
        self.assertIn('guidance_scale', settings)
        self.assertIn('true_cfg_scale', settings)
        self.assertEqual(settings['guidance_scale'], self.config.optimal_cfg_scale)
        self.assertEqual(settings['true_cfg_scale'], self.config.optimal_cfg_scale)
    
    def test_apply_torch_compile_disabled(self):
        """Test torch.compile when disabled in config"""
        result = self.optimizer.apply_torch_compile_optimization(Mock())
        self.assertFalse(result)
    
    def test_apply_torch_compile_mmdit_disabled(self):
        """Test torch.compile disabled for MMDiT architecture"""
        config = OptimizationConfig(use_torch_compile=True, architecture_type="MMDiT")
        optimizer = PipelineOptimizer(config)
        
        result = optimizer.apply_torch_compile_optimization(Mock())
        self.assertFalse(result)
    
    @patch('torch.compile')
    def test_apply_torch_compile_unet(self, mock_compile):
        """Test torch.compile for UNet architecture"""
        config = OptimizationConfig(use_torch_compile=True, architecture_type="UNet")
        optimizer = PipelineOptimizer(config)
        
        mock_pipeline = Mock()
        mock_unet = Mock()
        mock_pipeline.unet = mock_unet
        mock_compile.return_value = mock_unet
        
        result = optimizer.apply_torch_compile_optimization(mock_pipeline)
        
        self.assertTrue(result)
        mock_compile.assert_called_once_with(mock_unet, mode="max-autotune")
    
    def test_validate_optimization_basic(self):
        """Test optimization validation"""
        mock_pipeline = Mock()
        mock_pipeline.device = "cuda:0"
        
        results = self.optimizer.validate_optimization(mock_pipeline)
        
        self.assertIn('device_placement', results)
        self.assertIn('memory_optimizations', results)
        self.assertIn('attention_setup', results)
        self.assertIn('gpu_optimizations', results)
        self.assertIn('overall_status', results)
        
        self.assertIn("cuda:0", results['device_placement'])
    
    def test_validate_optimization_component_check(self):
        """Test optimization validation with component device checking"""
        mock_pipeline = Mock()
        del mock_pipeline.device  # Remove device attribute to trigger component checking
        
        # Mock transformer component
        mock_transformer = Mock()
        mock_param = Mock()
        mock_param.device = "cuda:0"
        mock_transformer.parameters.return_value = iter([mock_param])
        mock_pipeline.transformer = mock_transformer
        
        # Mock other components as None
        mock_pipeline.unet = None
        mock_pipeline.vae = None
        mock_pipeline.text_encoder = None
        
        config = OptimizationConfig(device="cuda")
        optimizer = PipelineOptimizer(config)
        
        results = optimizer.validate_optimization(mock_pipeline)
        
        self.assertIn("1/1", results['device_placement'])
    
    def test_get_performance_recommendations_high_vram(self):
        """Test performance recommendations for high-VRAM GPU"""
        recommendations = self.optimizer.get_performance_recommendations(16.0)
        
        self.assertEqual(recommendations['memory_strategy'], 'full_gpu')
        self.assertEqual(recommendations['expected_performance'], 'excellent (2-5 seconds per step)')
        self.assertFalse(recommendations['optimal_settings']['enable_attention_slicing'])
        self.assertFalse(recommendations['optimal_settings']['enable_vae_slicing'])
        self.assertEqual(recommendations['optimal_settings']['resolution'], '1024x1024')
    
    def test_get_performance_recommendations_medium_vram(self):
        """Test performance recommendations for medium-VRAM GPU"""
        recommendations = self.optimizer.get_performance_recommendations(12.0)
        
        self.assertEqual(recommendations['memory_strategy'], 'full_gpu')
        self.assertEqual(recommendations['expected_performance'], 'good (3-8 seconds per step)')
        self.assertFalse(recommendations['optimal_settings']['enable_attention_slicing'])
    
    def test_get_performance_recommendations_low_vram(self):
        """Test performance recommendations for low-VRAM GPU"""
        recommendations = self.optimizer.get_performance_recommendations(8.0)
        
        self.assertEqual(recommendations['memory_strategy'], 'memory_efficient')
        self.assertEqual(recommendations['expected_performance'], 'moderate (5-15 seconds per step)')
        self.assertTrue(recommendations['optimal_settings']['enable_attention_slicing'])
        self.assertTrue(recommendations['optimal_settings']['enable_vae_slicing'])
        self.assertEqual(recommendations['optimal_settings']['resolution'], '768x768')
    
    def test_get_performance_recommendations_very_low_vram(self):
        """Test performance recommendations for very low VRAM"""
        recommendations = self.optimizer.get_performance_recommendations(4.0)
        
        self.assertEqual(recommendations['memory_strategy'], 'cpu_offload')
        self.assertEqual(recommendations['expected_performance'], 'slow (15+ seconds per step)')
        self.assertTrue(recommendations['optimal_settings']['enable_cpu_offload'])
        self.assertEqual(recommendations['optimal_settings']['resolution'], '512x512')
        self.assertGreater(len(recommendations['warnings']), 0)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_performance_recommendations_no_gpu(self, mock_cuda):
        """Test performance recommendations with no GPU"""
        # Create optimizer with no CUDA available
        config = OptimizationConfig()
        optimizer = PipelineOptimizer(config)
        
        recommendations = optimizer.get_performance_recommendations(None)
        
        self.assertEqual(recommendations['memory_strategy'], 'cpu_only')
        self.assertEqual(recommendations['expected_performance'], 'slow (CPU inference)')
        self.assertGreater(len(recommendations['warnings']), 0)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_get_performance_recommendations_auto_detect(self, mock_props, mock_cuda):
        """Test automatic GPU memory detection"""
        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value = mock_device_props
        
        recommendations = self.optimizer.get_performance_recommendations()
        
        self.assertEqual(recommendations['memory_strategy'], 'full_gpu')
        mock_props.assert_called_once_with(0)


class TestPipelineOptimizerIntegration(unittest.TestCase):
    """Integration tests for PipelineOptimizer"""
    
    @patch('pipeline_optimizer.AutoPipelineForText2Image')
    @patch('torch.cuda.is_available', return_value=True)
    def test_create_optimized_pipeline_integration(self, mock_cuda, mock_pipeline_class):
        """Test complete pipeline creation and optimization flow"""
        # Setup mocks
        mock_pipeline = Mock()
        mock_pipeline.device = "cuda:0"
        mock_pipeline.disable_attention_slicing = Mock()
        mock_pipeline.disable_vae_slicing = Mock()
        mock_pipeline.disable_vae_tiling = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        
        # Add __name__ attribute to mock class
        mock_pipeline_class.__name__ = "AutoPipelineForText2Image"
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Create optimizer and pipeline
        config = OptimizationConfig(device="cuda")
        optimizer = PipelineOptimizer(config)
        
        result = optimizer.create_optimized_pipeline("Qwen/Qwen-Image", "MMDiT")
        
        # Verify pipeline creation
        mock_pipeline_class.from_pretrained.assert_called_once()
        mock_pipeline.to.assert_called_once_with("cuda")
        
        # Verify optimizations applied
        mock_pipeline.disable_attention_slicing.assert_called_once()
        mock_pipeline.disable_vae_slicing.assert_called_once()
        mock_pipeline.disable_vae_tiling.assert_called_once()
        
        self.assertEqual(result, mock_pipeline)
    
    def test_end_to_end_configuration_flow(self):
        """Test end-to-end configuration and optimization flow"""
        # Create optimizer with custom config
        config = OptimizationConfig(
            optimal_steps=25,
            optimal_cfg_scale=4.0,
            architecture_type="MMDiT"
        )
        optimizer = PipelineOptimizer(config)
        
        # Get generation settings
        settings = optimizer.configure_generation_settings()
        
        # Verify settings match config
        self.assertEqual(settings['num_inference_steps'], 25)
        self.assertEqual(settings['true_cfg_scale'], 4.0)
        
        # Get performance recommendations
        recommendations = optimizer.get_performance_recommendations(16.0)
        
        # Verify recommendations are appropriate
        self.assertEqual(recommendations['memory_strategy'], 'full_gpu')
        self.assertFalse(recommendations['optimal_settings']['enable_attention_slicing'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)