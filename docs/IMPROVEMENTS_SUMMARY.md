# Qwen-Image Generator Improvements Summary

This document summarizes all the improvements implemented to enhance the Qwen-Image generator system.

## 1. Dependency Management

### requirements.txt Updates
- Pinned critical model-breaking libraries:
  - `transformers==4.46.*`
  - `diffusers==0.30.*`
  - `accelerate==0.34.*`
- Removed `torch` from requirements.txt
- Added clear instructions for CUDA-specific PyTorch installation

### README.md Updates
- Added detailed PyTorch installation instructions per CUDA version
- Documented RTX 4080 → CUDA 12.1 wheel recommendation
- Added Quick Start notes for Windows PowerShell scripts

## 2. Canonical Device Policy

### New Module: src/utils/devices.py
Created a centralized device policy helper with functions:
- `get_device_config()` - Canonical device configuration
- `get_attention_implementation()` - SDPA default with Flash-Attn runtime probe
- `load_model_lazy()` - Lazy model loading with retry mechanism
- `create_processor_eager()` - Eager processor creation
- `clamp_image_size()` - Pixel window governor based on VRAM
- `perform_preflight_checks()` - Comprehensive system checks
- `safe_model_switch_context()` - Graceful model switching

### Implementation in Generator Classes
- Updated `QwenImageGenerator` and `HighEndQwenImageGenerator` to use device helper
- Integrated eager processor, lazy weights approach
- Added pre-flight checks for VRAM, disk space, CUDA version

## 3. Model Cache Configuration

### QWEN_HOME Environment Variable
- Added support for `QWEN_HOME` environment variable
- Defaults to `~/.cache/huggingface/hub` if not set
- Configurable cache directory for fast NVMe storage
- Updated all configuration files to use `QWEN_HOME`

## 4. Graceful Model Switching

### Memory Management
- Implemented `safe_model_switch_context()` function
- Added `safe_model_switch()` context manager
- Automatic GPU memory clearing with `torch.cuda.empty_cache()`
- Garbage collection with `gc.collect()`
- Short sleep (200ms) to allow memory cleanup

### Switch Implementation
- Added `switch_to_model()` method for safe model transitions
- Proper cleanup of existing models before loading new ones

## 5. Windows PowerShell Support

### New PowerShell Scripts
- `scripts/launch_ui.ps1` - PowerShell version of launch_ui.sh
- `scripts/setup.ps1` - PowerShell version of setup.sh
- `scripts/safe_restart.ps1` - PowerShell version of safe_restart.sh

### Cross-Platform Compatibility
- All key scripts now have PowerShell equivalents
- Proper WSL integration for Windows users
- Consistent user experience across platforms

## 6. Pixel Window Governor

### configs/quality_presets.yaml
Created configuration file with:
- VRAM-based pixel window settings (medium, high, ultra)
- Aspect ratio presets with pixel counts
- Quality presets (fast, balanced, high)

### Dynamic Image Size Adjustment
- Automatic clamping based on available VRAM
- Proportional scaling when images exceed limits
- Prevents OOM errors before first image generation
- "Safe Mode: Reduced pixel window" notification

## 7. Model Loading Hardening

### Retry & Backoff Mechanism
- `load_model_with_retry()` function with exponential backoff
- Automatic retry on OOM and HTTP errors
- Intelligent fallback strategies:
  - Reduce max_memory on OOM
  - Fallback from bfloat16 to float16
  - Lower max_pixels automatically

### Error Logging
- Detailed error logging to `reports/last_session.log`
- Clear error messages with actionable solutions
- Session-based error tracking

### Eager Processor, Lazy Weights
- Processor creation happens immediately (fast)
- Model weights loading delayed until first generate
- UI remains responsive during initial loading
- Processor retained even if model loading fails

## 8. Stable Attention Backend

### Runtime Probe
- Automatic detection of Flash Attention availability
- Runtime check for bfloat16 support
- Compute capability verification
- Graceful fallback to SDPA when needed

### Implementation
- SDPA as default stable backend
- Flash Attention 2 as optional high-performance backend
- Automatic selection based on hardware capabilities

## 9. Pre-flight Checks

### Comprehensive System Validation
- VRAM availability checking
- Disk space verification (≥ model size + 2GB)
- CUDA version detection
- Driver version checking
- Automatic recommendations based on system status

### User Experience
- Clear status reporting
- Proactive issue detection
- Helpful recommendations for marginal systems

## 10. Download Reliability Improvements

### Fast Wins Implemented
1. **Rust Accelerator Support** - Up to 4x faster downloads with `hf_transfer`
2. **Environment Variable Integration** - Automatic Rust accelerator activation
3. **Resume Download Functionality** - Continue interrupted downloads
4. **Cache Directory Configuration** - Optimize for fast SSDs
5. **Cleanup Functionality** - Remove incomplete downloads
6. **Concurrency Optimization** - Better download performance
7. **Windows Compatibility** - Fixed symlink issues
8. **Retry Mechanism** - Automatic retry with backoff

### New Tools and Scripts
- `tools/robust_download.py` - New utility with all improvements
- `scripts/setup_hf_transfer.sh` - Automated setup script
- Enhanced existing download scripts with retry mechanisms
- Automatic cleanup of incomplete downloads

### Performance Improvements
- **4x faster downloads** with Rust accelerator
- **Automatic resume** capability for interrupted downloads
- **Reduced retry times** with exponential backoff
- **Better concurrency** with optimized worker counts
- **Automatic cleanup** of corrupted partial downloads

## 11. Integration Testing

### test_improvements.py
Created comprehensive test suite verifying:
- Device helper functions
- Model loading with retry
- Configuration files
- Pixel window governor
- All tests passing successfully

## Benefits

1. **Lower Maintenance**: Centralized device policy reduces code duplication
2. **Higher Reliability**: Retry mechanisms and graceful error handling
3. **Better User Experience**: Proactive checks and clear guidance
4. **Cross-Platform Support**: PowerShell scripts for Windows users
5. **Memory Safety**: Proper cleanup and memory management
6. **Performance Optimization**: VRAM-aware image sizing
7. **Flexible Configuration**: Customizable cache locations
8. **Stable Backends**: Reliable attention implementation selection
9. **Download Reliability**: Dramatically improved download success rates
10. **Faster Downloads**: Up to 4x speed improvement with Rust accelerator

## Usage Instructions

### Environment Setup
```bash
# Linux/macOS
export QWEN_HOME=/path/to/fast/storage

# Windows PowerShell
$env:QWEN_HOME="D:\fast-ssd\qwen-models"
```

### Model Loading
The system now automatically:
1. Checks system resources
2. Selects optimal device configuration
3. Loads processor eagerly
4. Loads weights lazily with retry protection
5. Adjusts image sizes based on VRAM
6. Uses stable attention backend

### Safe Model Switching
```python
generator.switch_to_model("new/model/name")
```

### Robust Downloads
```bash
# Enable Rust accelerator
./scripts/setup_hf_transfer.sh

# Fast, reliable downloads
python tools/robust_download.py Qwen/Qwen2.5-VL-7B-Instruct \
  --out-dir ./models/qwen-vl \
  --retries 3 \
  --workers 8
```

This implementation provides a robust, user-friendly, and maintainable foundation for the Qwen-Image generator system.