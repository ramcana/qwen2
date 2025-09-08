# Qwen-Image Local Generator 🎨

A professional text-to-image generation system using the Qwen-Image model, optimized for high-end hardware with local deployment capabilities.

## Features

- **Advanced Text Rendering**: Specialized in generating text within images with high accuracy
- **Multi-language Support**: English and Chinese text generation
- **Hardware Optimized**: Specifically tuned for RTX 4080 + AMD Threadripper setup
- **Professional UI**: Complete Gradio web interface with advanced controls
- **Local Deployment**: No cloud dependencies, complete privacy
- **Metadata Management**: Automatic saving of generation parameters
- **Multiple Presets**: Quality, aspect ratio, and style presets

## System Requirements

### Recommended Hardware

- **GPU**: RTX 4080 (16GB VRAM) or better
- **CPU**: AMD Threadripper or equivalent high-core-count processor
- **RAM**: 32GB minimum, 128GB recommended
- **Storage**: 60-70GB free space for models and generated images

### Software Requirements

- **OS**: Ubuntu 20.04+ or WSL2 with Ubuntu
- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 12.1 or compatible
- **PyTorch**: 2.1.0+

## Quick Start

### 🚀 **Easy Launch Options**

#### Interactive Launcher (Recommended)
```bash
# Choose your interface interactively
source venv/bin/activate
python launch.py
```

#### Direct Launch Commands
```bash
# Standard UI (Text-to-Image only)
python launch.py --mode standard

# Enhanced UI (All features)
python launch.py --mode enhanced

# Shell script with menu
./scripts/launch_ui.sh

# Direct enhanced launch
./scripts/restart_enhanced.sh
```

### 🛡️ **Safe Restart Options**
```bash
# Safe restart with UI choice
./scripts/safe_restart.sh

# Direct enhanced safe restart
./scripts/restart_enhanced.sh
```

## Project Structure

```
Qwen2/
├── src/                    # Main application code
│   ├── qwen_image_ui.py   # Gradio web interface
│   ├── qwen_generator.py  # Core generation logic
│   ├── qwen_image_config.py # Configuration settings
│   ├── utils/             # Utility modules
│   └── presets/           # Preset configurations
├── scripts/               # All automation scripts
│   ├── safe_restart.sh    # Recommended launcher (prevents segfaults)
│   ├── restart_ui.sh      # Full diagnostic restart
│   ├── activate.sh        # Environment activation
│   ├── setup.sh           # Project setup
│   └── lint.sh            # Code quality checks
├── docs/                  # Complete documentation
│   ├── README.md          # Documentation index
│   ├── DEVICE_ERROR_FIX.md # GPU troubleshooting
│   ├── UI_ACCESS_GUIDE.md  # UI setup guide
│   └── ...                # Additional guides
├── tools/                 # Diagnostic and utility tools
│   ├── test_device.py     # Device diagnostics
│   └── emergency_device_fix.py # Emergency repairs
├── reports/               # Generated reports and logs
├── configs/               # Configuration files
├── generated_images/      # Output directory
└── tests/                 # Test suite
```

## Usage Examples

### Basic Generation

```python
from src.qwen_generator import QwenImageGenerator

generator = QwenImageGenerator()
generator.load_model()

image, message = generator.generate_image(
    prompt="A coffee shop with a sign reading 'AI Café'",
    width=1664,
    height=928
)
```

### Advanced Settings

```python
image, message = generator.generate_image(
    prompt="Modern art gallery with text 'Innovation 2025'",
    negative_prompt="blurry, low quality",
    width=1472,
    height=1140,
    num_inference_steps=80,
    cfg_scale=7.0,
    seed=42,
    language="en"
)
```

## Configuration

### Hardware Optimization

The system automatically detects and optimizes for your hardware:

- **RTX 4080**: Uses bfloat16 precision, attention slicing
- **High RAM**: Enables larger batch processing
- **CUDA 12.1**: Optimized PyTorch installation

### Quality Presets

- **Fast**: 20 steps, CFG 3.0 (~15-20 seconds)
- **Balanced**: 50 steps, CFG 4.0 (~30-40 seconds)
- **High**: 80 steps, CFG 7.0 (~50-60 seconds)

### Aspect Ratios

- Square (1:1): 1328×1328
- Landscape (16:9): 1664×928
- Portrait (9:16): 928×1664
- Photo (4:3): 1472×1140
- Ultra-wide (21:9): 1792×768

## Performance

### Expected Generation Times (RTX 4080)

| Quality | Steps | Time | Use Case |
|---------|-------|------|----------|
| Fast | 20 | 15-20s | Quick previews |
| Balanced | 50 | 30-40s | General use |
| High | 80 | 50-60s | Final quality |

### Memory Usage

- **VRAM**: 12-15GB during generation
- **System RAM**: 8-12GB active usage
- **Storage**: ~2-5MB per generated image

## Troubleshooting

### Quick Diagnostics

```bash
# Test system compatibility
python tools/test_device.py

# Emergency device fixes
python tools/emergency_device_fix.py

# Safe restart (prevents segfaults)
./scripts/safe_restart.sh
```

### Common Issues

1. **Segmentation Fault**: Use `./scripts/safe_restart.sh` instead of direct launch
2. **CUDA Out of Memory**: Reduce image size or enable CPU offload
3. **Model Download Slow**: Use HuggingFace cache or local mirror
4. **Generation Fails**: Check PyTorch CUDA installation

### Documentation

- 📚 **Complete guides**: See `docs/` directory
- 🔧 **Device issues**: `docs/DEVICE_ERROR_FIX.md`
- 🌐 **UI access**: `docs/UI_ACCESS_GUIDE.md`
- 🖥️ **WSL2 setup**: `docs/WSL2_BROWSER_SETUP.md`

### Performance Tips

- **Use safe restart**: `./scripts/safe_restart.sh` prevents crashes
- **Diagnostic tools**: Run `python tools/test_device.py` for health checks
- Use bfloat16 precision for RTX 40-series GPUs
- Enable attention slicing for memory efficiency
- Keep VRAM usage under 14GB for stability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Qwen Team**: For the amazing Qwen-Image model
- **Hugging Face**: For the diffusers library
- **Gradio**: For the web interface framework

---

**Hardware Optimized**: Specifically tuned for RTX 4080 + AMD Threadripper setups
**Local Privacy**: All processing happens on your hardware
**Professional Quality**: Production-ready text-to-image generation
