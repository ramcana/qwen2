#!/usr/bin/env python3
"""
Create optimized configuration for 16GB GPU
"""
import json
from pathlib import Path

def create_optimized_config():
    """Create configuration optimized for 16GB GPU"""
    
    config = {
        "model_settings": {
            "model_name": "Qwen/Qwen-Image",
            "torch_dtype": "float16",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        },
        
        "memory_optimizations": {
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_vae_tiling": True,
            "enable_model_cpu_offload": True,
            "enable_sequential_cpu_offload": False  # Use only if needed
        },
        
        "generation_presets": {
            "fast": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 8,
                "guidance_scale": 2.5,
                "description": "Fast generation for testing"
            },
            "balanced": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 15,
                "guidance_scale": 3.5,
                "description": "Balanced speed/quality"
            },
            "quality": {
                "width": 768,
                "height": 768,
                "num_inference_steps": 20,
                "guidance_scale": 4.0,
                "description": "Higher quality (uses more memory)"
            },
            "max_quality": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 25,
                "guidance_scale": 4.5,
                "description": "Maximum quality (may exceed memory)"
            }
        },
        
        "hardware_limits": {
            "gpu_memory_gb": 16,
            "recommended_max_resolution": "768x768",
            "safe_max_resolution": "512x512",
            "max_batch_size": 1
        },
        
        "performance_tips": [
            "Use 512x512 resolution for best performance",
            "Enable all memory optimizations",
            "Use 8-15 inference steps for speed",
            "Clear GPU memory between generations",
            "Monitor memory usage to avoid overflow"
        ]
    }
    
    # Save configuration
    config_path = Path("config/optimized_settings.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Optimized configuration saved to: {config_path}")
    return config

def create_quick_start_script():
    """Create a quick start script with optimized settings"""
    
    script_content = '''#!/usr/bin/env python3
"""
Quick start script with optimized settings for 16GB GPU
"""
import torch
import gc
from diffusers import DiffusionPipeline
import json
from pathlib import Path

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def load_optimized_pipeline():
    """Load pipeline with memory optimizations"""
    print("🚀 Loading Qwen-Image with optimizations...")
    
    # Clear memory first
    clear_memory()
    
    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "./models/Qwen-Image",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply optimizations
    pipe.enable_attention_slicing()
    
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    if hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    
    print("✅ Pipeline loaded with optimizations")
    return pipe

def generate_optimized(pipe, prompt, preset="balanced"):
    """Generate image with optimized settings"""
    
    # Load presets
    presets = {
        "fast": {"width": 512, "height": 512, "steps": 8, "guidance": 2.5},
        "balanced": {"width": 512, "height": 512, "steps": 15, "guidance": 3.5},
        "quality": {"width": 768, "height": 768, "steps": 20, "guidance": 4.0}
    }
    
    settings = presets.get(preset, presets["balanced"])
    
    print(f"🎨 Generating with {preset} preset...")
    print(f"   Size: {settings['width']}x{settings['height']}")
    print(f"   Steps: {settings['steps']}")
    
    # Clear memory before generation
    clear_memory()
    
    # Generate
    with torch.inference_mode():
        result = pipe(
            prompt,
            width=settings["width"],
            height=settings["height"],
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance"]
        )
    
    return result.images[0]

def main():
    """Main function"""
    print("🚀 Qwen-Image Quick Start (Optimized)")
    print("=" * 40)
    
    # Load pipeline
    pipe = load_optimized_pipeline()
    
    # Test generation
    prompt = "A beautiful mountain landscape with clear blue sky"
    image = generate_optimized(pipe, prompt, "balanced")
    
    # Save result
    output_file = "quick_start_result.jpg"
    image.save(output_file)
    print(f"💾 Saved: {output_file}")
    
    print("\\n🎉 Quick start complete!")
    print("💡 Try different presets: 'fast', 'balanced', 'quality'")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("tools/quick_start_optimized.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"✅ Quick start script created: {script_path}")

def main():
    """Main function"""
    print("🔧 Creating Optimized Configuration")
    print("=" * 40)
    
    # Create configuration
    config = create_optimized_config()
    
    # Create quick start script
    create_quick_start_script()
    
    print("\\n📋 Configuration Summary:")
    print("- Memory optimizations enabled")
    print("- Safe resolution limits set (512x512)")
    print("- Multiple quality presets available")
    print("- Hardware limits documented")
    
    print("\\n🚀 Next Steps:")
    print("1. Test: python tools/quick_start_optimized.py")
    print("2. Use presets to avoid memory issues")
    print("3. Monitor GPU memory usage")

if __name__ == "__main__":
    main()