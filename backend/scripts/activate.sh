#!/bin/bash
# Qwen2 Project Environment Activation Script
# This script activates the virtual environment and displays project info

echo "🎨 Activating Qwen2 Project Environment..."
echo "============================================"

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "🐍 Python version: $(python --version)"
echo "📦 Gradio version: $(python -c 'import gradio; print(gradio.__version__)')"
echo "📁 Current directory: $(pwd)"
echo ""
echo "Available commands:"
echo "  python src/qwen_image_ui.py    # Launch UI"
echo "  python examples/quick_test.py  # Quick test"
echo "  deactivate                     # Exit environment"
echo ""
echo "🚀 Ready to work with Qwen2 Image Generator!"

# Keep shell active with activated environment
exec "$SHELL"