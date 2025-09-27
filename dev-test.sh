#!/usr/bin/env bash
# Modern test script for DiffSynth integration
set -e

echo "🧪 Running DiffSynth Integration Tests"
echo "=" * 40

# Activate environment
source .venv311/bin/activate

echo "🔍 System Check..."
python -c "
import torch
import sys
sys.path.insert(0, 'src')
from diffsynth_service import DiffSynthService
print('✅ Python 3.11 environment active')
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.cuda.is_available())
print('✅ DiffSynth-Studio: Available')
"

echo ""
echo "🚀 Running DiffSynth model loading test..."
python test_diffsynth_model_loading.py

echo ""
echo "🧪 Running safe integration tests..."
python -m pytest tests/test_integration_safe.py -v

echo ""
echo "✅ All tests passed!"
echo "💡 DiffSynth system ready for development"