#!/bin/bash
# Safe Qwen2 Restart Script - Minimal Device Handling
# Avoids aggressive device manipulation that causes segfaults

echo "🔄 Safe Restart: Qwen2 Image Generator"
echo "======================================"

# Kill any existing instances
echo "🛑 Stopping existing processes..."
pkill -f "python src/qwen_image_ui.py" 2>/dev/null || true
pkill -f "qwen_image_ui.py" 2>/dev/null || true
sleep 3

# Activate environment
echo "🐍 Activating environment..."
source venv/bin/activate

# Light CUDA cleanup only
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
echo "🚀 Starting Qwen2 UI with SAFE device handling..."
echo ""
echo "📋 Safety features:"
echo "   ✅ Removed aggressive parameter manipulation"
echo "   ✅ Safe device verification only"
echo "   ✅ Minimal device moves during generation"
echo "   ✅ No recursive parameter scanning"
echo "   ✅ Emergency fallback systems removed"
echo ""
echo "🌐 Server will be available at: http://localhost:7860"
echo "💡 Open that URL in your Windows browser"
echo ""

# Launch with safe handling
python src/qwen_image_ui.py