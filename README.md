# Qwen-Image Local Generator 🎨

A professional image editing system using the Qwen-Image-Edit model, optimized for high-end hardware with local deployment capabilities.

> **🚀 New Setup**: One-command installation with Python 3.11, resumable downloads, and comprehensive testing. Run `make setup` to get started!

## ✨ Features

- **🎯 Advanced Image Editing**: Specialized Qwen-Image-Edit pipeline for precise image modifications
- **🌍 Multi-language Support**: English and Chinese text generation
- **⚡ Hardware Optimized**: Specifically tuned for RTX 4080 + CUDA 12.1 setup
- **🎨 Professional UI**: Complete Gradio web interface with advanced controls
- **🔒 Local Deployment**: No cloud dependencies, complete privacy
- **📊 Metadata Management**: Automatic saving of generation parameters
- **🎛️ Multiple Presets**: Quality, aspect ratio, and style presets
- **🔄 Resumable Downloads**: Smart model downloading with automatic resume

## 🖥️ System Requirements

### Recommended Hardware

- **GPU**: RTX 4080 (16GB VRAM) or better
- **CPU**: AMD Threadripper or equivalent high-core-count processor
- **RAM**: 32GB minimum, 128GB recommended
- **Storage**: 60-70GB free space for models and generated images

### Software Requirements

- **OS**: Ubuntu 20.04+ or WSL2 with Ubuntu
- **Python**: 3.11 (standardized environment)
- **CUDA**: 12.1 or compatible
- **PyTorch**: 2.8.0+ with CUDA support

## 🚀 Quick Start

### **One-Time Setup**

```bash
# Complete setup: Python 3.11 venv + dependencies + PyTorch + models
make setup
```

### **Daily Use (After Setup)**

```bash
# Turn ON development environment
./dev-start.sh

# Launch UI for image editing
./dev-ui.sh

# Turn OFF when done
./dev-stop.sh
```

This will:

- ✅ Create Python 3.11 virtual environment (`.venv311/`)
- ✅ Install all dependencies with pinned versions
- ✅ Install PyTorch with CUDA 12.1 support
- ✅ Download Qwen-Image-Edit model (~20GB)

### **Manual Setup (Step by Step)**

```bash
# 1. Create environment
make venv

# 2. Install dependencies
make deps

# 3. Install PyTorch with CUDA
make torch

# 4. Download models
make models
```

### **Launch Application**

```bash
# Activate environment
source .venv311/bin/activate

# Start Gradio UI
make ui
# or: python src/qwen_image_ui.py

# Start main application
make run
# or: python start.py
```

### **Development Commands**

```bash
# Run smoke test
make smoke

# Format code
make format

# Run linting
make lint

# Clean up
make clean

# See all commands
make help
```

## 📁 Project Structure

```
qwen2/
├── .venv311/              # Python 3.11 virtual environment
├── src/                   # Main application code
│   ├── qwen_image_ui.py  # Gradio web interface
│   ├── qwen_generator.py # Core generation logic
│   ├── qwen_image_config.py # Configuration settings
│   ├── utils/            # Utility modules
│   └── presets/          # Preset configurations
├── scripts/              # Automation scripts
│   ├── setup_env.sh      # Environment setup
│   ├── setup_models.sh   # Model download
│   └── activate.sh       # Quick activation
├── tools/                # Development tools
│   ├── download_models.py # Resumable model downloader
│   ├── test_device.py    # Device diagnostics
│   └── emergency_device_fix.py # Emergency repairs
├── examples/             # Example scripts
│   └── qwen_edit_smoke.py # End-to-end test
├── models/               # Downloaded models
│   └── Qwen-Image-Edit/  # Main model (~20GB)
├── generated_images/     # Output directory
├── docs/                 # Documentation
├── tests/                # Test suite
├── Makefile             # Development shortcuts
├── SETUP.md             # Detailed setup guide
└── requirements.txt     # Pinned dependencies
```

## 💡 Usage Examples

### Quick Test

```bash
# Run end-to-end smoke test
make smoke
# This creates a test image with "Add a red kite flying in the sky"
```

### Basic Image Editing

```python
from diffusers import QwenImageEditPipeline
from PIL import Image
import torch

# Load pipeline
pipe = QwenImageEditPipeline.from_pretrained(
    "./models/Qwen-Image-Edit",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Edit image
image = Image.open("input.jpg")
result = pipe(
    prompt="Add a beautiful sunset in the background",
    image=image
).images[0]

result.save("edited_image.jpg")
```

### Advanced Settings

```python
# High-quality editing with custom parameters
result = pipe(
    prompt="Transform this into a cyberpunk scene with neon lights",
    image=image,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.8
).images[0]
```

## ⚙️ Configuration

### Hardware Optimization

The system automatically detects and optimizes for your hardware:

- **RTX 4080**: Uses float16 precision, memory optimization
- **CUDA 12.1**: Latest PyTorch with optimized CUDA kernels
- **High RAM**: Efficient model loading and caching

### Environment Features

- **🐍 Python 3.11**: Standardized environment for consistency
- **📦 Pinned Dependencies**: Reproducible builds with exact versions
- **🔄 Resumable Downloads**: Smart model downloading with auto-resume
- **🧪 Smoke Testing**: Quick validation of complete pipeline
- **🛠️ Development Tools**: Integrated linting, formatting, and testing

### Model Information

- **Model**: Qwen/Qwen-Image-Edit
- **Size**: ~20GB total
- **Type**: Image editing and enhancement
- **Precision**: float16 for RTX 40-series GPUs
- **VRAM Usage**: ~12-15GB during inference

## 📊 Performance

### Expected Performance (RTX 4080)

| Operation        | Time        | VRAM Usage | Notes                 |
| ---------------- | ----------- | ---------- | --------------------- |
| Model Loading    | 30-60s      | 12GB       | One-time startup      |
| Image Editing    | 10-30s      | 15GB       | Depends on complexity |
| Batch Processing | 5-15s/image | 15GB       | Multiple edits        |

### Memory Usage

- **VRAM**: 12-15GB during inference
- **System RAM**: 8-12GB active usage
- **Storage**: ~2-5MB per generated image
- **Model Cache**: ~20GB for Qwen-Image-Edit

### Optimization Tips

- Use `torch.float16` for RTX 40-series GPUs
- Enable attention slicing for memory efficiency
- Keep VRAM usage under 14GB for stability
- Use the smoke test to verify optimal performance

## 🔧 Troubleshooting

### Quick Diagnostics

```bash
# Test complete pipeline
make smoke

# Check system compatibility
python tools/test_device.py

# Emergency device fixes
python tools/emergency_device_fix.py
```

### Common Issues

1. **Model Download Stuck**:

   ```bash
   # Resume download
   make models
   ```

2. **CUDA Out of Memory**:

   ```bash
   # Clear GPU memory and restart
   make clean
   make smoke
   ```

3. **QwenImageEditPipeline not found**:

   ```bash
   # Reinstall latest diffusers
   source .venv311/bin/activate
   pip uninstall -y diffusers
   pip install git+https://github.com/huggingface/diffusers.git
   ```

4. **Environment Issues**:
   ```bash
   # Clean rebuild
   rm -rf .venv311
   make setup
   ```

### Resumable Downloads

If model download gets interrupted:

```bash
# Downloads automatically resume from where they left off
make models

# Check download progress
ls -la models/Qwen-Image-Edit/
du -sh models/Qwen-Image-Edit/
```

### Performance Issues

- **Slow generation**: Ensure CUDA is properly installed
- **Memory errors**: Use `torch.float16` precision
- **Crashes**: Run `make smoke` to validate setup

### Documentation

- 📖 **Setup Guide**: [`SETUP.md`](SETUP.md) - Comprehensive setup instructions
- 🔧 **Device Issues**: `docs/DEVICE_ERROR_FIX.md`
- 🌐 **UI Access**: `docs/UI_ACCESS_GUIDE.md`
- 🖥️ **WSL2 Setup**: `docs/WSL2_BROWSER_SETUP.md`

### Performance Tips

- **Use safe restart**: `./scripts/safe_restart.sh` prevents crashes
- **Diagnostic tools**: Run `python tools/test_device.py` for health checks
- Use bfloat16 precision for RTX 40-series GPUs
- Enable attention slicing for memory efficiency
- Keep VRAM usage under 14GB for stability

## 🚀 What's New

### ✨ Latest Updates

- **🐍 Python 3.11 Environment**: Standardized setup with `.venv311/`
- **📦 Pinned Dependencies**: Reproducible builds with exact versions
- **🔄 Resumable Downloads**: Smart model downloading with auto-resume
- **🛠️ Makefile Integration**: One-command setup and development shortcuts
- **🧪 Smoke Testing**: Quick end-to-end pipeline validation
- **📖 Comprehensive Docs**: Detailed setup guide in `SETUP.md`

### 🎯 Key Features

- **Latest Diffusers**: Direct from GitHub with QwenImageEditPipeline
- **CUDA 12.1 Optimized**: PyTorch 2.8.0+ with latest CUDA support
- **Memory Efficient**: Optimized for RTX 4080 with 16GB VRAM
- **Developer Friendly**: Integrated linting, formatting, and testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with `make smoke`
4. Run quality checks with `make lint` and `make format`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team**: For the amazing Qwen-Image-Edit model
- **Hugging Face**: For the diffusers library and model hosting
- **Gradio**: For the web interface framework
- **PyTorch Team**: For the excellent deep learning framework

---

**🎯 Hardware Optimized**: Specifically tuned for RTX 4080 + CUDA 12.1 setups  
**🔒 Local Privacy**: All processing happens on your hardware  
**⚡ Production Ready**: Professional-quality image editing pipeline  
**🛠️ Developer Friendly**: Complete development environment with one command
