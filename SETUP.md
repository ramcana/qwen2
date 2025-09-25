# Qwen Image Generator - Setup Guide

This guide will help you set up the Qwen Image Generator with a standardized Python 3.11 environment that ensures the QwenImageEditPipeline works reliably.

## Quick Setup (Recommended)

### 1. Complete Automated Setup

```bash
# One command to rule them all
make setup
```

This will:

- Create a Python 3.11 virtual environment (`.venv311/`)
- Install all dependencies with proper versions
- Install PyTorch with CUDA 12.1 support
- Download the Qwen-Image-Edit model

### 2. Activate Environment

```bash
source .venv311/bin/activate
```

### 3. Run Smoke Test

```bash
make smoke
```

## Manual Setup (Step by Step)

If you prefer to understand each step:

### 1. Create Virtual Environment

```bash
make venv
# or manually: python3.11 -m venv .venv311
```

### 2. Install Dependencies

```bash
make deps
# or manually: source .venv311/bin/activate && pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA

```bash
make torch
# or manually: pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 4. Download Models

```bash
make models
# or manually: python tools/download_models.py
```

## Verification

### Test Installation

```bash
# Quick smoke test
make smoke

# Full test suite
make test

# Check Python and packages
source .venv311/bin/activate
python -c "
import diffusers as d
from diffusers import QwenImageEditPipeline
import torch
print('✅ diffusers:', d.__version__)
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
"
```

## Running the Application

### Gradio UI

```bash
make ui
# or: source .venv311/bin/activate && python src/qwen_image_ui.py
```

### Main Application

```bash
make run
# or: source .venv311/bin/activate && python start.py
```

## Development Commands

```bash
# Format code
make format

# Run linting
make lint

# Clean up generated files
make clean
```

## Key Features of This Setup

### ✅ Standardized Environment

- Python 3.11 virtual environment in `.venv311/`
- Pinned dependency versions for reproducibility
- Latest diffusers from GitHub with QwenImageEditPipeline

### ✅ Resumable Model Downloads

- `tools/download_models.py` handles interrupted downloads
- Models stored in `./models/` directory
- Automatic retry and resume functionality

### ✅ VS Code Integration

- Automatic Python interpreter detection
- Configured linting and formatting
- Proper exclude patterns for generated files

### ✅ CUDA Optimization

- PyTorch with CUDA 12.1 support
- Automatic GPU detection and usage
- Memory-optimized settings

## Troubleshooting

### Common Issues

**"QwenImageEditPipeline not found"**

```bash
# Reinstall diffusers from GitHub
source .venv311/bin/activate
pip uninstall -y diffusers
pip install git+https://github.com/huggingface/diffusers.git
```

**"CUDA out of memory"**

- Reduce batch size in your scripts
- Use `torch.float16` instead of `torch.float32`
- Clear CUDA cache: `torch.cuda.empty_cache()`

**"Model download interrupted"**

```bash
# Resume download
make models
# The download will automatically resume from where it left off
```

### Frontend Issues (Optional)

If you need to fix frontend dependencies:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

## Environment Variables

Create a `.env` file in the project root for custom settings:

```bash
# GPU settings
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 4080

# Model settings
MODEL_CACHE_DIR=./models
HF_HOME=./models/.cache

# API settings
API_HOST=0.0.0.0
API_PORT=8000
```

## Next Steps

1. Run the smoke test: `make smoke`
2. Start the UI: `make ui`
3. Check out the examples in `examples/`
4. Read the main README.md for usage instructions

Happy generating! 🎨
