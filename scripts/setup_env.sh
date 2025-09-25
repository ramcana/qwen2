#!/usr/bin/env bash
# Setup Python 3.13 virtual environment and dependencies
set -euo pipefail

echo "🐍 Setting up Python 3.11 environment..."

# Deactivate any existing environment
deactivate 2>/dev/null || true

# Create Python 3.11 virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

echo "✅ Created and activated Python 3.11 virtual environment"
python -V

# Upgrade pip and install core tools
pip install -U pip setuptools wheel

echo "📦 Installing dependencies..."

# Uninstall any existing diffusers to avoid conflicts
pip uninstall -y diffusers 2>/dev/null || true

# Install requirements
pip install -r requirements.txt

echo "🔥 Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA 12.1 support
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "🧪 Testing installation..."
python -c "
import diffusers as d
from diffusers import QwenImageEditPipeline
import torch
print('✅ diffusers:', d.__version__, '- QwenImageEditPipeline available')
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA device:', torch.cuda.get_device_name(0))
"

echo "🎉 Environment setup complete!"
echo "💡 To activate: source .venv311/bin/activate"