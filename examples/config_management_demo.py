#!/usr/bin/env python3
"""
Configuration Management Demo
Demonstrates the new modern architecture-aware configuration system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
import torch


def demo_basic_configuration():
    """Demonstrate basic configuration creation and usage"""
    print("=== Basic Configuration Demo ===")
    
    # Create default configuration
    config = OptimizationConfig()
    print(f"Default config: {config.model_name}, {config.architecture.value}, {config.optimization_level.value}")
    print(f"Generation settings: {config.width}x{config.height}, {config.num_inference_steps} steps, CFG {config.true_cfg_scale}")
    print(f"Optimizations: SDPA={config.enable_scaled_dot_product_attention}, TF32={config.enable_tf32}")
    print()


def demo_optimization_levels():
    """Demonstrate different optimization levels"""
    print("=== Optimization Levels Demo ===")
    
    levels = [OptimizationLevel.ULTRA_FAST, OptimizationLevel.BALANCED, 
              OptimizationLevel.QUALITY, OptimizationLevel.MULTIMODAL]
    
    for level in levels:
        config = create_optimization_config(level)
        print(f"{level.value.upper()}:")
        print(f"  Steps: {config.num_inference_steps}, CFG: {config.true_cfg_scale}")
        print(f"  Resolution: {config.width}x{config.height}")
        print(f"  Qwen2-VL: {config.enable_qwen2vl_integration}")
        print()


def demo_architecture_specific_configs():
    """Demonstrate architecture-specific configurations"""
    print("=== Architecture-Specific Configs Demo ===")
    
    architectures = [ModelArchitecture.MMDIT, ModelArchitecture.UNET, ModelArchitecture.MULTIMODAL]
    
    for arch in architectures:
        config = get_model_config_for_architecture(arch)
        print(f"{arch.value.upper()} Architecture:")
        print(f"  Model: {config['model_name']}")
        print(f"  SDPA: {config.get('enable_scaled_dot_product_attention', 'N/A')}")
        print(f"  Xformers: {config.get('enable_xformers', 'N/A')}")
        print(f"  Qwen2-VL: {config.get('enable_qwen2vl_integration', 'N/A')}")
        print()


def demo_custom_configuration():
    """Demonstrate custom configuration with validation"""
    print("=== Custom Configuration Demo ===")
    
    # Create custom high-resolution configuration
    config = create_optimization_config(
        OptimizationLevel.QUALITY,
        ModelArchitecture.MMDIT,
        width=1536,
        height=1536,
        num_inference_steps=50,
        true_cfg_scale=6.0,
        enable_qwen2vl_integration=True
    )
    
    print("Custom High-Resolution Config:")
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  Steps: {config.num_inference_steps}, CFG: {config.true_cfg_scale}")
    print(f"  Architecture: {config.architecture.value}")
    print(f"  Qwen2-VL: {config.enable_qwen2vl_integration}")
    
    # Validate configuration
    is_valid = config.validate()
    print(f"  Valid: {is_valid}")
    
    # Check architecture compatibility
    is_compatible = validate_architecture_compatibility(config.architecture, config)
    print(f"  Compatible: {is_compatible}")
    print()


def demo_legacy_migration():
    """Demonstrate legacy configuration migration"""
    print("=== Legacy Migration Demo ===")
    
    # Simulate legacy configuration
    legacy_config = {
        "MODEL_CONFIG": {
            "model_name": "Qwen/Legacy-Model",
            "torch_dtype": torch.float32,
            "device": "cpu"
        },
        "MEMORY_CONFIG": {
            "enable_attention_slicing": True,
            "enable_cpu_offload": False,
            "enable_vae_slicing": True,
            "enable_xformers": True
        },
        "GENERATION_CONFIG": {
            "width": 768,
            "height": 768,
            "num_inference_steps": 30,
            "true_cfg_scale": 4.5
        }
    }
    
    print("Legacy config:")
    print(f"  Model: {legacy_config['MODEL_CONFIG']['model_name']}")
    print(f"  Resolution: {legacy_config['GENERATION_CONFIG']['width']}x{legacy_config['GENERATION_CONFIG']['height']}")
    print(f"  Attention slicing: {legacy_config['MEMORY_CONFIG']['enable_attention_slicing']}")
    
    # Migrate to modern configuration
    modern_config = migrate_legacy_config(legacy_config)
    
    print("Migrated modern config:")
    print(f"  Model: {modern_config.model_name}")
    print(f"  Resolution: {modern_config.width}x{modern_config.height}")
    print(f"  Attention slicing: {modern_config.enable_attention_slicing}")
    print(f"  Architecture: {modern_config.architecture.value}")
    print()


def demo_pipeline_and_generation_kwargs():
    """Demonstrate kwargs generation for pipeline and generation"""
    print("=== Pipeline & Generation Kwargs Demo ===")
    
    config = create_optimization_config(OptimizationLevel.BALANCED)
    
    pipeline_kwargs = config.get_pipeline_kwargs()
    generation_kwargs = config.get_generation_kwargs()
    
    print("Pipeline kwargs:")
    for key, value in pipeline_kwargs.items():
        print(f"  {key}: {value}")
    
    print("\nGeneration kwargs:")
    for key, value in generation_kwargs.items():
        print(f"  {key}: {value}")
    print()


def demo_quality_presets():
    """Demonstrate quality presets"""
    print("=== Quality Presets Demo ===")
    
    for preset_name, preset_config in QUALITY_PRESETS.items():
        print(f"{preset_name.upper()}:")
        print(f"  Description: {preset_config['description']}")
        print(f"  Steps: {preset_config['num_inference_steps']}")
        print(f"  CFG: {preset_config['true_cfg_scale']}")
        print(f"  Architecture: {preset_config['architecture'].value}")
        if 'enable_qwen2vl_integration' in preset_config:
            print(f"  Qwen2-VL: {preset_config['enable_qwen2vl_integration']}")
        print()


def demo_qwen2vl_config():
    """Demonstrate Qwen2-VL configuration"""
    print("=== Qwen2-VL Configuration Demo ===")
    
    print("Qwen2-VL Config:")
    for key, value in QWEN2VL_CONFIG.items():
        print(f"  {key}: {value}")
    print()


def main():
    """Run all configuration demos"""
    print("Qwen Image Configuration Management Demo")
    print("=" * 50)
    print()
    
    demo_basic_configuration()
    demo_optimization_levels()
    demo_architecture_specific_configs()
    demo_custom_configuration()
    demo_legacy_migration()
    demo_pipeline_and_generation_kwargs()
    demo_quality_presets()
    demo_qwen2vl_config()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()