#!/usr/bin/env bash
# Quick start script for daily development
set -e

echo "🚀 Starting Qwen Image Edit Development Environment"
echo "=" * 60

# Activate environment
echo "🐍 Activating Python 3.11 environment..."
source .venv311/bin/activate

# Quick health check
echo "🔍 Running health check..."
python -c "
import torch
from diffusers import QwenImageEditPipeline
print('✅ Python:', __import__('sys').version.split()[0])
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ QwenImageEditPipeline: Available')
"

echo ""
echo "🎨 Development Environment Ready!"
echo ""
echo "Available commands:"
echo "  make ui          - Start Gradio UI"
echo "  make run         - Start main application"
echo "  make smoke       - Quick pipeline test"
echo "  make format      - Format code"
echo "  make lint        - Run linting"
echo "  make help        - See all commands"
echo ""
echo "💡 To deactivate: run './dev-stop.sh' or 'deactivate'"
echo ""

# Keep shell active with environment
exec bash