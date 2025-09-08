#!/bin/bash
# Enhanced Qwen2 Restart Script - Direct Enhanced UI Launch
# Quick launcher for the enhanced features

echo "🔄 Enhanced Restart: Qwen2 Image Generator"
echo "=========================================="

# Kill any existing instances
echo "🛑 Stopping existing processes..."
pkill -f "python src/qwen_image_ui.py" 2>/dev/null || true
pkill -f "python src/qwen_image_enhanced_ui.py" 2>/dev/null || true
pkill -f "qwen_image_ui.py" 2>/dev/null || true
pkill -f "qwen_image_enhanced_ui.py" 2>/dev/null || true
sleep 3

# Activate environment
echo "🐍 Activating environment..."
source venv/bin/activate

# Light CUDA cleanup
echo "🧹 Light CUDA cleanup..."
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('✅ Light cleanup completed')
else:
    print('⚠️ CUDA not available')
" 2>/dev/null || true

echo ""
echo "🚀 Starting Enhanced Qwen2 UI..."
echo ""
echo "📋 Enhanced Features Available:"
echo "   ✅ Text-to-Image Generation (Qwen-Image)"
echo "   ✅ Image-to-Image Transformation"
echo "   ✅ Inpainting with Interactive Mask Editor"
echo "   ✅ Super-Resolution Enhancement (2x-4x)"
echo "   ✅ Dynamic UI Mode Switching"
echo "   ✅ Advanced Controls & Quality Presets"
echo "   ✅ Multi-language Support (EN/ZH)"
echo ""
echo "🌐 Server will be available at: http://localhost:7860"
echo "💡 Open that URL in your Windows browser"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Launch enhanced UI directly
python src/qwen_image_enhanced_ui.py