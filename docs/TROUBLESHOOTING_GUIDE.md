# Comprehensive Troubleshooting Guide

## Architecture-Aware Error Handling and Diagnostics

This guide covers common issues with Qwen image generation models and their solutions, with specific focus on MMDiT vs UNet architecture differences.

## Table of Contents

1. [Architecture-Specific Issues](#architecture-specific-issues)
2. [Download and Installation Issues](#download-and-installation-issues)
3. [Pipeline Loading Issues](#pipeline-loading-issues)
4. [Memory and Performance Issues](#memory-and-performance-issues)
5. [Device and CUDA Issues](#device-and-cuda-issues)
6. [Diagnostic Tools](#diagnostic-tools)
7. [Recovery Procedures](#recovery-procedures)

## Architecture-Specific Issues

### MMDiT (Qwen-Image) Architecture Issues

#### Tensor Unpacking Errors

**Symptoms:**

- `IndexError: tuple index out of range`
- `ValueError: expected tuple, got Tensor`
- `RuntimeError: tensor unpacking failed`

**Causes:**

- torch.compile incompatibility with MMDiT transformer output format
- Qwen-Image transformer returns single tensor instead of tuple
- AttnProcessor2_0 incompatibility with MMDiT attention mechanism

**Solutions:**

```python
# Disable torch.compile for MMDiT models
config = OptimizationConfig(use_torch_compile=False)

# Use default attention processor
pipeline.transformer.set_attn_processor({})  # Use default

# Use AutoPipelineForText2Image instead of generic DiffusionPipeline
from diffusers import AutoPipelineForText2Image
pipeline = AutoPipelineForText2Image.from_pretrained(model_path)
```

#### Attention Mechanism Issues

**Symptoms:**

- `RuntimeError: Flash attention not compatible`
- `CUDA error: invalid configuration argument`
- Slow generation despite high-end GPU

**Causes:**

- Flash Attention incompatibility with MMDiT architecture
- Incorrect attention processor configuration
- Memory-efficient attention conflicts

**Solutions:**

```python
# Disable Flash Attention for MMDiT
torch.backends.cuda.enable_flash_sdp(False)

# Use memory-efficient attention instead
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Configure MMDiT-specific settings
config = OptimizationConfig(
    enable_flash_attention=False,
    enable_memory_efficient_attention=True,
    architecture_type="MMDiT"
)
```

#### Parameter Mismatch Issues

**Symptoms:**

- `TypeError: unexpected keyword argument 'guidance_scale'`
- `TypeError: unexpected keyword argument 'true_cfg_scale'`

**Causes:**

- MMDiT models use `true_cfg_scale` instead of `guidance_scale`
- Parameter naming differences between architectures

**Solutions:**

```python
# Use correct parameter names for MMDiT
generation_kwargs = {
    "true_cfg_scale": 3.5,  # Not guidance_scale
    "num_inference_steps": 20,
    "width": 1024,
    "height": 1024
}

# Architecture-aware parameter selection
if architecture_type == "MMDiT":
    generation_kwargs["true_cfg_scale"] = cfg_scale
else:
    generation_kwargs["guidance_scale"] = cfg_scale
```

### UNet Architecture Issues

#### Memory Issues

**Symptoms:**

- `RuntimeError: CUDA out of memory`
- `torch.cuda.OutOfMemoryError`
- System freezing during generation

**Causes:**

- UNet models can be memory-intensive
- Insufficient VRAM for model size
- Memory fragmentation

**Solutions:**

```python
# Enable memory optimizations for UNet
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()

# Use CPU offloading
pipeline.enable_model_cpu_offload()

# Reduce precision
pipeline = pipeline.to(torch_dtype=torch.float16)
```

#### Attention Processor Compatibility

**Symptoms:**

- `AttributeError: 'UNet2DConditionModel' has no attribute 'set_attn_processor'`
- Performance degradation with default attention

**Causes:**

- Older UNet models may not support newer attention processors
- Version incompatibility

**Solutions:**

```python
# Use AttnProcessor2_0 for UNet models
from diffusers.models.attention_processor import AttnProcessor2_0

if hasattr(pipeline.unet, 'set_attn_processor'):
    pipeline.unet.set_attn_processor(AttnProcessor2_0())

# Enable Flash Attention for UNet (if compatible)
torch.backends.cuda.enable_flash_sdp(True)
```

## Download and Installation Issues

### Network Issues

**Symptoms:**

- `ConnectionError: Network connection failed`
- `TimeoutError: Request timed out`
- `HTTPError: 503 Service Unavailable`

**Automatic Recovery:**
The error handler automatically detects network issues and applies recovery strategies:

```python
# Network error recovery is automatic
from error_handler import handle_download_error

try:
    download_model("Qwen/Qwen-Image")
except Exception as e:
    error_info = handle_download_error(e, "Qwen/Qwen-Image")
    # Recovery actions executed automatically
```

**Manual Solutions:**

```bash
# Check network connectivity
ping huggingface.co

# Use VPN if regional restrictions
# Retry during off-peak hours
# Increase timeout settings
```

### Disk Space Issues

**Symptoms:**

- `OSError: No space left on device`
- `PermissionError: Disk quota exceeded`

**Automatic Recovery:**

```python
# Automatic cleanup and space management
error_handler = ArchitectureAwareErrorHandler()
error_handler._cleanup_old_models()
error_handler._clear_cache_directories()
```

**Manual Solutions:**

```bash
# Check disk space
df -h

# Clean up old models
rm -rf ~/.cache/huggingface/hub/models--*old-model*

# Move models to different drive
export HF_HOME=/path/to/larger/drive
```

### Permission Issues

**Symptoms:**

- `PermissionError: Access denied`
- `OSError: Permission denied`

**Automatic Recovery:**

```python
# Automatic permission checking and fixing
error_handler._check_directory_permissions()
error_handler._create_user_cache_dir()
```

**Manual Solutions:**

```bash
# Fix permissions
chmod -R 755 ~/.cache/huggingface
chown -R $USER ~/.cache/huggingface

# Use alternative cache directory
export HF_HOME=~/Documents/huggingface_cache
```

## Pipeline Loading Issues

### Model File Issues

**Symptoms:**

- `FileNotFoundError: Model file not found`
- `RuntimeError: Corrupted model file`
- `ValueError: Invalid model format`

**Automatic Recovery:**

```python
# Automatic model verification and repair
error_handler._verify_model_files(model_path)
error_handler._verify_checkpoint_integrity(model_path)
error_handler._cleanup_corrupted_files(model_name)
```

**Manual Solutions:**

```bash
# Re-download corrupted model
rm -rf models/Qwen-Image
python -c "from model_download_manager import download_qwen_image; download_qwen_image(force_redownload=True)"

# Verify model integrity
python -c "from model_download_manager import ModelDownloadManager; mgr = ModelDownloadManager(); mgr.verify_model_integrity('Qwen/Qwen-Image')"
```

### Configuration Issues

**Symptoms:**

- `ValueError: Invalid configuration`
- `TypeError: Incompatible pipeline configuration`

**Solutions:**

```python
# Use architecture-aware configuration
from pipeline_optimizer import PipelineOptimizer, OptimizationConfig

config = OptimizationConfig(
    architecture_type="MMDiT",  # or "UNet"
    enable_optimizations=True,
    use_torch_compile=False  # Disable for MMDiT
)

optimizer = PipelineOptimizer(config)
pipeline = optimizer.create_optimized_pipeline(model_path, "MMDiT")
```

## Memory and Performance Issues

### GPU Memory Issues

**Symptoms:**

- `RuntimeError: CUDA out of memory`
- Generation takes extremely long time
- System becomes unresponsive

**Automatic Recovery:**

```python
# Automatic memory optimization
error_handler._clear_gpu_memory()
error_handler._enable_memory_optimizations("MMDiT")  # or "UNet"
error_handler._reduce_model_precision()
```

**Manual Solutions:**

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Enable memory optimizations based on architecture
if architecture_type == "MMDiT":
    # MMDiT-specific optimizations
    config = OptimizationConfig(
        torch_dtype=torch.float16,
        enable_attention_slicing=False,  # May hurt MMDiT performance
        enable_vae_slicing=False,
        low_cpu_mem_usage=True
    )
elif architecture_type == "UNet":
    # UNet-specific optimizations
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
```

### Performance Issues

**Symptoms:**

- Generation takes 30+ seconds per step (should be 2-5 seconds)
- GPU utilization is low
- Using wrong model for task

**Solutions:**

```python
# Verify using correct model
from model_detection_service import ModelDetectionService

detector = ModelDetectionService()
current_model = detector.detect_current_model()

if current_model.model_type == "image-editing":
    print("⚠️ Using image-editing model for text-to-image task!")
    print("💡 Download Qwen-Image model for better performance")

    from model_download_manager import download_qwen_image
    download_qwen_image()

# Apply performance optimizations
optimizer = PipelineOptimizer()
recommendations = optimizer.get_performance_recommendations()
print(f"Expected performance: {recommendations['expected_performance']}")
```

## Device and CUDA Issues

### CUDA Installation Issues

**Symptoms:**

- `RuntimeError: CUDA not available`
- `ImportError: No module named 'torch.cuda'`

**Automatic Diagnostics:**

```python
# Automatic CUDA diagnostics
diagnostics = get_system_diagnostics()
print(f"CUDA Available: {diagnostics.gpu_available}")
print(f"CUDA Version: {diagnostics.cuda_version}")
print(f"GPU Memory: {diagnostics.gpu_memory_gb:.1f}GB")
```

**Manual Solutions:**

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Driver Issues

**Symptoms:**

- `CUDA error: invalid device ordinal`
- `RuntimeError: CUDA driver version is insufficient`

**Solutions:**

```bash
# Update NVIDIA drivers
# Ubuntu/Debian:
sudo apt update && sudo apt install nvidia-driver-535

# Check driver compatibility
nvidia-smi
```

## Diagnostic Tools

### System Diagnostics

```python
from error_handler import create_diagnostic_report

# Generate comprehensive diagnostic report
report = create_diagnostic_report()
print(f"GPU Available: {report['system_info']['gpu_available']}")
print(f"VRAM: {report['system_info']['gpu_memory_gb']:.1f}GB")
print(f"Architecture Support: {report['architecture_support']}")
print(f"Recent Errors: {len(report['recent_errors'])}")

# Get recommendations
for rec in report['recommendations']:
    print(f"💡 {rec}")
```

### Model Detection

```python
from model_detection_service import ModelDetectionService

detector = ModelDetectionService()

# Check current model
current_model = detector.detect_current_model()
if current_model:
    print(f"Current Model: {current_model.name}")
    print(f"Type: {current_model.model_type}")
    print(f"Architecture: {detector.detect_model_architecture(current_model)}")
    print(f"Optimal: {current_model.is_optimal}")

# Check optimization needs
if detector.is_optimization_needed():
    recommended = detector.get_recommended_model()
    print(f"💡 Recommended: {recommended}")
```

### Performance Analysis

```python
from model_detection_service import ModelDetectionService

detector = ModelDetectionService()
current_model = detector.detect_current_model()

if current_model:
    characteristics = detector.analyze_performance_characteristics(current_model)
    print(f"Expected Generation Time: {characteristics['expected_generation_time']}")
    print(f"Memory Usage: {characteristics['memory_usage']}")
    print(f"Optimization Level: {characteristics['optimization_level']}")

    if characteristics['bottlenecks']:
        print("⚠️ Bottlenecks:")
        for bottleneck in characteristics['bottlenecks']:
            print(f"  • {bottleneck}")
```

## Recovery Procedures

### Automatic Recovery

The error handling system provides automatic recovery for most issues:

```python
from error_handler import ArchitectureAwareErrorHandler

handler = ArchitectureAwareErrorHandler()

# Add user feedback callback
def user_feedback(message):
    print(f"🔧 {message}")

handler.add_user_feedback_callback(user_feedback)

# Errors are automatically handled and recovery attempted
try:
    # Your code here
    pass
except Exception as e:
    error_info = handler.handle_pipeline_error(e, model_path, architecture_type)
    handler.log_error(error_info)
    recovery_success = handler.execute_recovery_actions(error_info)

    if recovery_success:
        print("✅ Recovery successful, retrying operation...")
    else:
        print("❌ Recovery failed, manual intervention required")
```

### Manual Recovery Steps

#### Complete System Reset

```bash
# 1. Clear all caches
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
rm -rf ./models

# 2. Reinstall dependencies
pip uninstall torch torchvision diffusers transformers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate

# 3. Re-download models
python -c "from model_download_manager import download_qwen_image; download_qwen_image()"
```

#### Architecture-Specific Reset

```python
# Reset for MMDiT issues
config = OptimizationConfig(
    use_torch_compile=False,
    enable_flash_attention=False,
    architecture_type="MMDiT"
)

# Reset for UNet issues
config = OptimizationConfig(
    enable_attention_slicing=True,
    enable_vae_slicing=True,
    enable_cpu_offload=True,
    architecture_type="UNet"
)
```

## Getting Help

### Debug Information Collection

```python
# Collect comprehensive debug information
from error_handler import ArchitectureAwareErrorHandler

handler = ArchitectureAwareErrorHandler()
debug_info = handler._collect_debug_info(exception)

print("Debug Information:")
print(f"Error Type: {debug_info['error_type']}")
print(f"System Info: {debug_info['system_info']}")
print(f"Environment: {debug_info['environment']}")
```

### Reporting Issues

When reporting issues, please include:

1. **System Diagnostics:**

   ```python
   from error_handler import create_diagnostic_report
   report = create_diagnostic_report()
   # Include the full report
   ```

2. **Model Information:**

   ```python
   from model_detection_service import ModelDetectionService
   detector = ModelDetectionService()
   current_model = detector.detect_current_model()
   # Include model details
   ```

3. **Error Details:**

   - Full error traceback
   - Architecture type (MMDiT/UNet)
   - Model being used
   - System specifications

4. **Recovery Attempts:**
   - What recovery actions were tried
   - Results of recovery attempts
   - Any manual fixes attempted

This comprehensive troubleshooting guide should help users diagnose and resolve most issues with the Qwen image generation system, with specific attention to architecture-aware error handling and recovery procedures.
