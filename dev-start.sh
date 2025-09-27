#!/usr/bin/env bash
# Modern development environment for DiffSynth-Studio integration
set -e

echo "🚀 Starting DiffSynth Enhanced Development Environment"
echo "=" * 60

# Activate environment
echo "🐍 Activating Python 3.11 environment..."
source .venv311/bin/activate

# Quick health check
echo "🔍 Running health check..."
python -c "
import torch
import sys
sys.path.insert(0, 'src')
from diffsynth_service import DiffSynthService
print('✅ Python:', sys.version.split()[0])
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('✅ DiffSynth-Studio: Available')
"

echo ""
echo "🎨 DiffSynth Development Environment Ready!"
echo ""
echo "Available commands:"
echo "  python start.py                    - Interactive launcher menu"
echo "  python start_backend.py           - Start API server only"
echo "  python start_frontend.py          - Start React frontend only"
echo "  python test_diffsynth_model_loading.py - Test DiffSynth integration"
echo "  python -m pytest tests/test_integration_safe.py - Run safe tests"
echo ""
echo "💡 To stop: run './dev-stop.sh' or press Ctrl+C"
echo ""

# Keep shell active with environment
exec bash