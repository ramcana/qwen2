#!/usr/bin/env python3
"""
Pipeline Optimizer Demo
Demonstrates how to use the PipelineOptimizer for modern GPU performance with MMDiT architecture
"""

import sys
import os
import time
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline_optimizer import PipelineOptimizer, OptimizationConfig


def demo_basic_usage():
    """Demonstrate basic PipelineOptimizer usage"""
    print("🚀 Pipeline Optimizer Demo - Basic Usage")
    print("=" * 50)
    
    # Create default configuration
    config = OptimizationConfig()
    print(f"Default configuration:")
    print(f"  Device: {config.device}")
    print(f"  Architecture: {config.architecture_type}")
    print(f"  Torch dtype: {config.torch_dtype}")
    print(f"  Optimal steps: {config.optimal_steps}")
    print(f"  CFG scale: {config.optimal_cfg_scale}")
    print()
    
    # Create optimizer
    optimizer = PipelineOptimizer(config)
    
    # Get performance recommendations
    print("📊 Performance Recommendations:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        recommendations = optimizer.get_performance_recommendations(gpu_memory)
        print(f"  GPU Memory: {gpu_memory:.1f}GB")
        print(f"  Strategy: {recommendations['memory_strategy']}")
        print(f"  Expected Performance: {recommendations['expected_performance']}")
        print(f"  Optimal Settings: {recommendations['optimal_settings']}")
        if recommendations['warnings']:
            print(f"  Warnings: {recommendations['warnings']}")
    else:
        recommendations = optimizer.get_performance_recommendations(None)
        print(f"  No GPU detected")
        print(f"  Strategy: {recommendations['memory_strategy']}")
        print(f"  Expected Performance: {recommendations['expected_performance']}")
    print()


def demo_mmdit_configuration():
    """Demonstrate MMDiT-specific configuration"""
    print("🎯 MMDiT Architecture Configuration")
    print("=" * 50)
    
    # Create MMDiT-optimized configuration
    config = OptimizationConfig(
        architecture_type="MMDiT",
        optimal_steps=20,
        optimal_cfg_scale=3.5,
        enable_attention_slicing=False,  # Disabled for high-VRAM GPUs
        enable_vae_slicing=False,        # Disabled for performance
        enable_tf32=True,                # Enabled for RTX 30/40 series
        enable_cudnn_benchmark=True      # Enabled for consistent input sizes
    )
    
    optimizer = PipelineOptimizer(config)
    
    # Get generation settings for MMDiT
    settings = optimizer.configure_generation_settings("MMDiT")
    print("MMDiT Generation Settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    print()
    
    # Compare with UNet settings
    unet_settings = optimizer.configure_generation_settings("UNet")
    print("UNet Generation Settings (for comparison):")
    for key, value in unet_settings.items():
        print(f"  {key}: {value}")
    print()


def demo_custom_optimization():
    """Demonstrate custom optimization configuration"""
    print("⚙️ Custom Optimization Configuration")
    print("=" * 50)
    
    # Create custom configuration for different scenarios
    scenarios = {
        "ultra_performance": OptimizationConfig(
            optimal_steps=15,
            optimal_cfg_scale=3.0,
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            enable_tf32=True,
            enable_cudnn_benchmark=True,
            optimal_width=1024,
            optimal_height=1024
        ),
        "balanced": OptimizationConfig(
            optimal_steps=25,
            optimal_cfg_scale=4.0,
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            optimal_width=1024,
            optimal_height=1024
        ),
        "memory_efficient": OptimizationConfig(
            optimal_steps=20,
            optimal_cfg_scale=3.5,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            optimal_width=768,
            optimal_height=768
        )
    }
    
    for scenario_name, config in scenarios.items():
        print(f"{scenario_name.upper()} Configuration:")
        optimizer = PipelineOptimizer(config)
        settings = optimizer.configure_generation_settings()
        
        print(f"  Steps: {settings['num_inference_steps']}")
        print(f"  CFG Scale: {settings.get('true_cfg_scale', settings.get('guidance_scale', 'N/A'))}")
        print(f"  Resolution: {settings['width']}x{settings['height']}")
        print(f"  Attention Slicing: {config.enable_attention_slicing}")
        print(f"  VAE Slicing: {config.enable_vae_slicing}")
        print()


def demo_validation():
    """Demonstrate optimization validation"""
    print("✅ Optimization Validation Demo")
    print("=" * 50)
    
    # Create a mock pipeline for validation demo
    class MockPipeline:
        def __init__(self):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._attention_slicing_enabled = False
            self._vae_slicing_enabled = False
    
    mock_pipeline = MockPipeline()
    
    config = OptimizationConfig()
    optimizer = PipelineOptimizer(config)
    
    # Validate the mock pipeline
    validation_results = optimizer.validate_optimization(mock_pipeline)
    
    print("Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    print()


def demo_architecture_comparison():
    """Demonstrate differences between MMDiT and UNet architectures"""
    print("🔄 Architecture Comparison: MMDiT vs UNet")
    print("=" * 50)
    
    optimizer = PipelineOptimizer()
    
    # MMDiT (Qwen-Image) characteristics
    print("MMDiT Architecture (Qwen-Image):")
    print("  ✅ Modern transformer-based architecture")
    print("  ✅ Optimized for text-to-image generation")
    print("  ✅ Built-in attention optimizations")
    print("  ⚠️ AttnProcessor2_0 not compatible (tensor unpacking issues)")
    print("  ⚠️ torch.compile disabled due to tensor format differences")
    
    mmdit_settings = optimizer.configure_generation_settings("MMDiT")
    print(f"  CFG Parameter: true_cfg_scale = {mmdit_settings['true_cfg_scale']}")
    print()
    
    # UNet characteristics
    print("UNet Architecture (Traditional):")
    print("  ✅ Well-established architecture")
    print("  ✅ Compatible with AttnProcessor2_0 (Flash Attention)")
    print("  ✅ torch.compile support available")
    print("  ⚠️ Older architecture, may be less optimized")
    
    unet_settings = optimizer.configure_generation_settings("UNet")
    print(f"  CFG Parameter: guidance_scale = {unet_settings['guidance_scale']}")
    print()


def main():
    """Run all demo functions"""
    print("🎨 Pipeline Optimizer Comprehensive Demo")
    print("=" * 60)
    print()
    
    try:
        demo_basic_usage()
        demo_mmdit_configuration()
        demo_custom_optimization()
        demo_validation()
        demo_architecture_comparison()
        
        print("✅ Demo completed successfully!")
        print()
        print("💡 Next Steps:")
        print("  1. Use PipelineOptimizer in your image generation code")
        print("  2. Customize OptimizationConfig for your hardware")
        print("  3. Monitor performance improvements with validation")
        print("  4. Adjust settings based on your quality/speed requirements")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()