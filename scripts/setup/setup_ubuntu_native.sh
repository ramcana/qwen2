#!/bin/bash
# Ubuntu-Native Qwen-Image Setup for RTX 4080 (16GB VRAM)
# Optimized for stability and performance

set -e

echo "🚀 Ubuntu-Native Qwen-Image Setup for RTX 4080"
echo "================================================"

# 1) Check GPU and CUDA
echo "🎮 Checking GPU and CUDA..."
nvidia-smi
echo ""

# 2) Create directories for models and offload
echo "📁 Setting up directories..."
mkdir -p ~/models ~/offload
mkdir -p ./models ./offload ./cache

# 3) Set environment variables for optimal performance
echo "⚙️ Setting environment variables..."

# Add to current session
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export HF_HOME="$HOME/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Add to .bashrc for persistence
echo "📝 Adding to ~/.bashrc..."
grep -q "PYTORCH_CUDA_ALLOC_CONF" ~/.bashrc || echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"' >> ~/.bashrc
grep -q "HF_HOME" ~/.bashrc || echo 'export HF_HOME="$HOME/.cache/huggingface"' >> ~/.bashrc
grep -q "ulimit -n" ~/.bashrc || echo 'ulimit -n 1048576' >> ~/.bashrc

# 4) Install optimized packages
echo "📦 Installing optimized packages..."

# Activate virtual environment
source .venv311/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 wheels
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install quantization and optimization packages
echo "⚡ Installing quantization packages..."
pip install bitsandbytes==0.43.3
pip install transformers accelerate safetensors sentencepiece
pip install diffusers

# Try to install flash-attention (optional)
echo "💫 Attempting to install flash-attention..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash-attention install failed (optional)"

# 5) Check swap (optional but recommended)
echo "💾 Checking swap configuration..."
if [ $(swapon --show | wc -l) -eq 0 ]; then
    echo "⚠️ No swap detected. Consider adding swap for large model offloading:"
    echo "   sudo fallocate -l 64G /swapfile"
    echo "   sudo chmod 600 /swapfile"
    echo "   sudo mkswap /swapfile"
    echo "   sudo swapon /swapfile"
else
    echo "✅ Swap is configured:"
    swapon --show
fi

echo ""
echo "✅ Ubuntu-native setup complete!"
echo "💡 Next step: Run the optimized loader script"