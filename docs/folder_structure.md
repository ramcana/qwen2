# Qwen-Image Local UI - Complete Folder Structure

# Qwen2 Project Structure

This document describes the complete folder structure of the Qwen2 Image Generator project after cleanup and organization.

## Main Project Structure

```
Qwen2/
├── README.md                     # Main project documentation
├── QUICK_START.md               # Quick reference guide
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── .gitignore                  # Git ignore rules
├── .pre-commit-config.yaml     # Code quality hooks
│
├── venv/                        # Virtual environment (created by setup)
│   ├── bin/
│   ├── lib/
│   └── ...
│
├── src/                         # Main application code
│   ├── __init__.py
│   ├── qwen_image_ui.py        # Main Gradio interface
│   ├── qwen_image_config.py    # Configuration settings
│   ├── qwen_generator.py       # Core generation logic (SEGFAULT FIXED)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── ...
│   └── presets/
│       ├── __init__.py
│       └── ...
│
├── scripts/                     # 🔧 ALL automation and utility scripts
│   ├── README.md               # Scripts documentation
│   ├── safe_restart.sh         # ⭐ RECOMMENDED launcher (no segfaults)
│   ├── restart_ui.sh           # Full diagnostic restart
│   ├── activate.sh             # Environment activation
│   ├── setup.sh                # Project setup
│   ├── launch_ui.sh            # Basic launcher
│   ├── lint.sh                 # Code quality checks
│   ├── quality_gate.sh         # CI/CD quality gate
│   ├── setup_precommit.sh      # Pre-commit setup
│   ├── error_detection.py      # Automated error detection
│   └── fix_common_issues.py    # Automated fixes
│
├── docs/                        # 📚 Complete documentation
│   ├── README.md               # Documentation index
│   ├── folder_structure.md     # This file
│   ├── DEVICE_ERROR_FIX.md     # GPU/CUDA troubleshooting
│   ├── UI_ACCESS_GUIDE.md      # UI setup and access
│   ├── WSL2_BROWSER_SETUP.md   # WSL2-specific setup
│   ├── CODE_QUALITY_SYSTEM.md  # Quality tools and standards
│   ├── MODEL_LOADING_FIX.md    # Model loading issues
│   ├── GRADIO_FIX_DOCUMENTATION.md # Gradio troubleshooting
│   ├── MVP_CHECKLIST.md        # Feature checklist
│   └── QUICK_FIX_SUMMARY.md    # Common solutions
│
├── tools/                       # 🛠️ Diagnostic and utility tools
│   ├── README.md               # Tools documentation
│   ├── test_device.py          # Device diagnostics
│   └── emergency_device_fix.py # Emergency system repair
│
├── reports/                     # 📊 Generated reports and logs
│   └── error_detection_report.json
│
├── generated_images/            # 🎨 Output directory for generated images
│   ├── qwen_image_YYYYMMDD_HHMMSS_SEED.png
│   ├── qwen_image_YYYYMMDD_HHMMSS_SEED_metadata.json
│   └── ...
│
├── configs/                     # ⚙️ Configuration files
│   └── default_config.yaml
│
├── examples/                    # 💡 Example usage and demos
│   ├── quick_test.py
│   └── ...
│
├── templates/                   # 🎨 UI templates and assets
│   └── {icons}/
│
├── tests/                       # 🧪 Test suite
│   └── ...
│
├── logs/                        # 📝 Application logs (empty by default)
│
└── launch.py                    # Alternative launcher
```

## Key Features of New Organization

### 🚀 **Launch Scripts (Recommended Order)**

1. **`scripts/safe_restart.sh`** ⭐ - Prevents segmentation faults
2. **`scripts/restart_ui.sh`** - Full diagnostic restart with comprehensive checks
3. **`scripts/launch_ui.sh`** - Basic launcher

### 🛠️ **Diagnostic Tools**

- **`tools/test_device.py`** - System health check
- **`tools/emergency_device_fix.py`** - Emergency repairs

### 📚 **Documentation**

- **`docs/README.md`** - Documentation index
- **`docs/DEVICE_ERROR_FIX.md`** - Resolves GPU/CUDA issues
- **`docs/UI_ACCESS_GUIDE.md`** - Browser setup for WSL2

### 🔧 **Development Tools**

- **`scripts/lint.sh`** - Code quality checks
- **`scripts/quality_gate.sh`** - CI/CD pipeline
- **`scripts/setup.sh`** - Complete project setup

### Configuration Files Structure

```
configs/
├── default_config.yaml           # Base configuration
├── rtx4080_optimized.yaml       # Your hardware-specific settings
├── ui_themes.yaml               # UI appearance settings
└── model_presets.yaml           # Pre-configured model settings
```

### Generated Images Organization

```
generated_images/
├── YYYYMMDD_HHMMSS_SEED.png     # Main image files
├── YYYYMMDD_HHMMSS_SEED_metadata.json  # Generation metadata
├── thumbnails/                   # Auto-generated previews
│   └── YYYYMMDD_HHMMSS_SEED_thumb.png
└── gallery/                      # Organized by date
    ├── 2025-09-07/
    │   ├── morning_session/
    │   ├── afternoon_session/
    │   └── evening_session/
    └── 2025-09-08/
```

### Scripts Directory

```
scripts/
├── install_dependencies.sh       # Automated dependency installation
├── download_models.py           # Pre-download models for offline use
├── test_setup.py               # Verify installation and performance
├── batch_generate.py           # Generate multiple images from prompts
├── cleanup_cache.py            # Clean model cache and temp files
└── benchmark_performance.py    # Performance testing on your hardware
```

### Model Cache Structure (Hugging Face)

```
models/Qwen--Qwen-Image/
├── config.json                  # Model configuration
├── model_index.json            # Pipeline index
├── scheduler/                   # Diffusion scheduler
│   ├── scheduler_config.json
│   └── ...
├── text_encoder/               # Text encoding components
├── tokenizer/                  # Text tokenization
├── transformer/                # Main MMDiT transformer
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ...
├── unet/ (if applicable)       # Legacy U-Net components
└── vae/                        # Variational AutoEncoder
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## Quick Setup Commands

### 1. Create the structure

```bash
mkdir -p qwen-image-ui/{src/{utils,presets},generated_images/{thumbnails,gallery},models,configs,scripts,examples/{sample_prompts},docs,templates/{icons},logs,tests/{benchmarks}}
```

### 2. Initialize the project

```bash
cd qwen-image-ui
python3 -m venv venv
source venv/bin/activate
```

### 3. Create essential files

```bash
touch src/__init__.py src/utils/__init__.py src/presets/__init__.py
touch tests/__init__.py
touch README.md .gitignore .env.example
```

## Storage Requirements

**Estimated Disk Usage:**

- **Base installation**: ~2-3 GB (PyTorch, dependencies)
- **Qwen-Image model**: ~40-50 GB (20B parameters)
- **Generated images**: Variable (each ~2-5 MB depending on resolution)
- **Cache and logs**: ~1-2 GB
- **Total recommended**: **60-70 GB free space**

## Memory Usage Patterns

**During Generation:**

- **VRAM**: 12-15 GB (perfect for RTX 4080's 16GB)
- **System RAM**: 8-12 GB (your 128GB is excellent)
- **Disk I/O**: Moderate (model loading, image saving)

This structure provides a professional, scalable foundation for your Qwen-Image local UI with room for future enhancements and easy maintenance.
