#!/usr/bin/env bash
# Modern UI launcher for DiffSynth Enhanced system
set -e

echo "🎨 Starting DiffSynth Enhanced UI"
echo "=" * 40

# Activate environment
source .venv311/bin/activate

# Quick check
echo "🔍 Quick system check..."
python -c "
import torch
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "🚀 Starting full-stack UI..."
echo "💡 This will start both backend and frontend"
echo ""

# Start the interactive launcher
python start.py