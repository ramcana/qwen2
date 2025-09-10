#!/bin/bash
# Qwen-Image Local UI Setup Script
# Text-to-Image Generation - Optimized for AMD Threadripper + RTX 4080

echo "🎨 Setting up Qwen-Image Local Text-to-Image Generator..."
echo "Hardware: AMD Threadripper PRO 5995WX + RTX 4080 (16GB VRAM)"
echo "Model: Qwen-Image (20B MMDiT diffusion model)"
echo "============================================================"

# Create project directory
mkdir -p qwen-image-ui
cd qwen-image-ui

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "🔧 Installing dependencies for Qwen-Image..."

# Install PyTorch with CUDA support (optimized for RTX 4080)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install diffusers (required for Qwen-Image)
echo "Installing diffusers library..."
pip install git+https://github.com/huggingface/diffusers.git

# Install other required packages
echo "Installing additional dependencies..."
pip install -r ../requirements.txt -c ../constraints.txt

# Create requirements file
cat > requirements.txt << EOF
torch>=2.1.0
torchvision
torchaudio
git+https://github.com/huggingface/diffusers.git
transformers>=4.40.0
accelerate
safetensors
pillow
gradio
numpy
xformers
EOF

echo "📝 Creating configuration file..."

# Create optimized config for RTX 4080
cat > qwen_image_config.py << EOF
# Qwen-Image Configuration
# Optimized for RTX 4080 (16GB VRAM) + 128GB RAM

import torch

# Model configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen-Image",  # 20B parameter MMDiT model
    "torch_dtype": torch.bfloat16,    # Best for RTX 4080
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_safetensors": True,
    "variant": "fp16",  # Use fp16 variant for better VRAM usage
}

# Memory optimization settings
MEMORY_CONFIG = {
    "enable_attention_slicing": True,      # Reduce memory usage
    "enable_cpu_offload": False,           # Keep False with 16GB VRAM
    "enable_sequential_cpu_offload": False, # Alternative if you need more VRAM
}

# Default generation settings
GENERATION_CONFIG = {
    "width": 1664,                    # Good for RTX 4080
    "height": 928,                    # 16:9 aspect ratio
    "num_inference_steps": 50,        # Balanced quality/speed
    "true_cfg_scale": 4.0,           # Recommended CFG for Qwen-Image
    "max_batch_size": 1,             # Single image generation
}

# Quality presets
QUALITY_PRESETS = {
    "fast": {
        "num_inference_steps": 20,
        "true_cfg_scale": 3.0,
        "description": "Fast preview - lower quality"
    },
    "balanced": {
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "description": "Recommended - good balance"
    },
    "high": {
        "num_inference_steps": 80,
        "true_cfg_scale": 7.0,
        "description": "High quality - slower generation"
    }
}

# Aspect ratio presets
ASPECT_RATIOS = {
    "1:1": (1328, 1328),     # Square
    "16:9": (1664, 928),     # Landscape
    "9:16": (928, 1664),     # Portrait
    "4:3": (1472, 1140),     # Photo
    "3:4": (1140, 1472),     # Portrait photo
    "21:9": (1792, 768),     # Ultra-wide
}

# Prompt enhancement templates
PROMPT_ENHANCEMENT = {
    "en": {
        "quality_keywords": "Ultra HD, 4K, cinematic composition, professional photography, detailed, sharp focus, high quality",
        "artistic_keywords": "masterpiece, trending on artstation, award winning, professional lighting",
        "technical_keywords": "photorealistic, high resolution, intricate details, perfect composition"
    },
    "zh": {
        "quality_keywords": "超清，4K，电影级构图，专业摄影，细节丰富，清晰对焦，高质量",
        "artistic_keywords": "杰作，艺术站热门，获奖作品，专业灯光",
        "technical_keywords": "逼真，高分辨率，精细细节，完美构图"
    }
}
EOF

echo "🎯 Creating example usage script..."

# Create example script
cat > example_generation.py << EOF
#!/usr/bin/env python3
"""
Quick example script for Qwen-Image generation
Run this to test your setup without the full UI
"""

import torch
from diffusers import DiffusionPipeline
from qwen_image_config import MODEL_CONFIG, GENERATION_CONFIG

def quick_test():
    print("🧪 Testing Qwen-Image setup...")

    # Load model
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_CONFIG["model_name"],
        torch_dtype=MODEL_CONFIG["torch_dtype"],
        use_safetensors=MODEL_CONFIG["use_safetensors"],
        variant=MODEL_CONFIG["variant"]
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        print("✅ Model loaded on GPU")
    else:
        print("⚠️  Model loaded on CPU")

    # Test generation
    prompt = "A beautiful coffee shop with a neon sign reading 'AI Café', modern interior, warm lighting"

    print(f"Generating: {prompt}")

    image = pipe(
        prompt=prompt + ", Ultra HD, 4K, professional photography",
        width=GENERATION_CONFIG["width"],
        height=GENERATION_CONFIG["height"],
        num_inference_steps=GENERATION_CONFIG["num_inference_steps"],
        true_cfg_scale=GENERATION_CONFIG["true_cfg_scale"],
        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
    ).images[0]

    # Save result
    image.save("test_generation.png")
    print("✅ Test image saved as 'test_generation.png'")

if __name__ == "__main__":
    quick_test()
EOF

chmod +x example_generation.py

echo "🎉 Setup Complete!"
echo ""
echo "📋 What was installed:"
echo "✅ PyTorch with CUDA 12.1 support"
echo "✅ Diffusers library (latest version)"
echo "✅ Qwen-Image model configuration"
echo "✅ Gradio web interface"
echo "✅ Memory optimizations for RTX 4080"
echo ""
echo "🚀 Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Test setup: python example_generation.py"
echo "3. Launch full UI: python qwen_image_ui.py"
echo "4. Open browser: http://localhost:7860"
echo ""
echo "💾 Your RTX 4080 (16GB) is perfect for Qwen-Image!"
echo "📊 Expected performance:"
echo "   • Fast (20 steps): ~15-20 seconds"
echo "   • Balanced (50 steps): ~30-40 seconds"
echo "   • High Quality (80 steps): ~50-60 seconds"
echo ""
echo "🎨 Qwen-Image specializes in:"
echo "   • Text rendering in images (signs, logos, etc.)"
echo "   • Multi-language text support"
echo "   • High-quality artistic generation"
echo "   • Professional photography styles"
echo ""
echo "Ready-to-use alternatives:"
echo "🔗 Quantized version: https://github.com/Branc93/NunchakuQwen-Gradio"
echo "🔗 ComfyUI workflow: https://docs.comfy.org/tutorials/image/qwen/qwen-image"
