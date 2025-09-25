#!/usr/bin/env bash
# Quick UI launcher for daily use
set -e

echo "🎨 Starting Qwen Image Edit UI"
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
echo "🚀 Launching Gradio UI..."
echo "💡 Press Ctrl+C to stop"
echo ""

# Launch UI
python src/qwen_image_ui.py