#!/bin/bash
# Qwen2 Restart Script with Enhanced Device Handling
# Stops any running instances and starts fresh with comprehensive device fixes

echo "🔄 Restarting Qwen2 Image Generator with COMPREHENSIVE DEVICE FIXES..."
echo "====================================================================="

# Kill any existing instances
echo "🛑 Stopping existing processes..."
pkill -f "python src/qwen_image_ui.py" 2>/dev/null || true
pkill -f "qwen_image_ui.py" 2>/dev/null || true
sleep 3

# Activate environment
echo "🐍 Activating environment..."
source venv/bin/activate

# Enhanced CUDA cleanup
echo "🧹 Aggressive CUDA cleanup..."
python -c "
import torch
import gc
if torch.cuda.is_available():
    for i in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print('✅ CUDA cleanup completed')
else:
    print('⚠️ CUDA not available')
" 2>/dev/null || true

# Run comprehensive device diagnostics
echo "🔍 Running comprehensive device diagnostics..."
python tools/test_device.py

echo ""
echo "🚀 Starting Qwen2 UI with COMPREHENSIVE DEVICE SAFETY..."
echo ""
echo "📋 Device fixes applied:"
echo "   ✅ Removed invalid fp16 variant parameter"
echo "   ✅ Comprehensive parameter-level device consistency"
echo "   ✅ Pre-generation device verification and fixing"
echo "   ✅ Emergency CPU fallback for device errors"
echo "   ✅ Enhanced memory management and synchronization"
echo "   ✅ Progressive loading fallbacks with device safety"
echo "   ✅ CPU->GPU device reset capability"
echo ""
echo "🔧 New safety features:"
echo "   🛠️ Automatic device consistency enforcement"
echo "   🚨 Emergency CPU fallback on device errors"
echo "   🔄 Device reset and retry mechanisms"
echo "   📊 Comprehensive device state monitoring"
echo ""
echo "🌐 Server will be available at: http://localhost:7860"
echo "💡 Open that URL in your Windows browser"
echo ""
echo "⚠️ If device errors persist, run: python tools/emergency_device_fix.py"
echo ""

# Launch with enhanced device handling
python src/qwen_image_ui.py