#!/usr/bin/env bash
# Quick test script for daily development
set -e

echo "🧪 Running Qwen Image Edit Tests"
echo "=" * 40

# Activate environment
source .venv311/bin/activate

echo "🔍 System Check..."
python -c "
import torch
from diffusers import QwenImageEditPipeline
print('✅ Python 3.11 environment active')
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.cuda.is_available())
print('✅ QwenImageEditPipeline: Available')
"

echo ""
echo "🚀 Running smoke test..."
python examples/qwen_edit_smoke.py

echo ""
echo "✅ All tests passed!"
echo "💡 System ready for development"