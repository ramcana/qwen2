# 🔧 Qwen-Image-Edit Setup Guide

## Issue: "Qwen-Image-Edit pipeline not loaded"

This error occurs when the Qwen-Image-Edit model hasn't been downloaded yet. Here's how to fix it:

## 🚀 **Quick Solution**

### Method 1: Automatic Download Script
```bash
# Run the download helper
cd /home/ramji_t/projects/Qwen2
source venv/bin/activate
python download_qwen_edit.py
```

### Method 2: Manual Download
```bash
# In Python environment
cd /home/ramji_t/projects/Qwen2
source venv/bin/activate
python -c "
from diffusers import QwenImageEditPipeline
import torch

print('Downloading Qwen-Image-Edit...')
pipeline = QwenImageEditPipeline.from_pretrained(
    'Qwen/Qwen-Image-Edit',
    torch_dtype=torch.bfloat16,
    cache_dir='./models/qwen-image-edit'
)
print('Download complete!')
"
```

## 📋 **What's Happening**

1. **Qwen-Image** (for text-to-image) loads quickly (~5GB)
2. **Qwen-Image-Edit** (for img2img/inpainting) is much larger (~20GB)
3. First-time download can take 10-30 minutes depending on internet speed

## 🎯 **Available Features by Model**

| Feature | Qwen-Image | Qwen-Image-Edit |
|---------|------------|------------------|
| **Text-to-Image** | ✅ Available | ✅ Available |
| **Image-to-Image** | ❌ Not supported | ✅ Available |
| **Inpainting** | ❌ Not supported | ✅ Available |
| **Super-Resolution** | ✅ Basic upscaling | ✅ Enhanced |

## 🛠️ **Troubleshooting**

### If Download Fails:
1. **Check Internet**: Ensure stable connection to Hugging Face
2. **Check Space**: Need ~25GB free disk space
3. **Check Permissions**: Ensure write access to project directory

### If Still Having Issues:
```bash
# Check diffusers version
pip install git+https://github.com/huggingface/diffusers.git --upgrade

# Clear cache and retry
rm -rf ./models/qwen-image-edit
python download_qwen_edit.py
```

## 💡 **Pro Tips**

1. **Start Download Early**: Run the download script while doing other tasks
2. **Use Text-to-Image**: While waiting, use the standard text-to-image features
3. **Check Progress**: The download script shows progress and estimated time
4. **Background Download**: Download can run in background - just don't close terminal

## 🔄 **Current Workarounds**

Until Qwen-Image-Edit is downloaded, you can:
1. Use **Text-to-Image** mode with descriptive prompts
2. Use **Super-Resolution** for basic image enhancement
3. Create images with text like: "An image showing: [your description]"

## ✅ **Verification**

After download, you should see:
- "✅ Qwen-Image-Edit pipeline loaded successfully!"
- All UI modes become functional
- No more "pipeline not loaded" errors

---

**The enhanced features are worth the wait! Qwen-Image-Edit provides professional-grade image editing capabilities. 🎨**
