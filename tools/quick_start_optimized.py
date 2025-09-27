#!/usr/bin/env python3
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
    
    print("\n🎉 Quick start complete!")
    print("💡 Try different presets: 'fast', 'balanced', 'quality'")

if __name__ == "__main__":
    main()
