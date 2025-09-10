# Qwen-Image-Edit Device Mapping Fix

## Problem Analysis

The error you encountered was:
```
⚠️ Could not download/load Qwen-Image-Edit: auto not supported. Supported strategies are: balanced, cuda
```

This error occurred because I added `device_map="auto"` to the model loading configuration, but the Qwen-Image-Edit pipeline doesn't support this parameter.

## Root Cause

The issue was in the [`_load_qwen_edit_pipeline`](\\wsl.localhost\Ubuntu\home\ramji_t\projects\Qwen2\src\qwen_generator.py) method where I incorrectly added:

```python
# PROBLEMATIC CODE (now fixed)
self.edit_pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=MODEL_CONFIG["torch_dtype"],
    low_cpu_mem_usage=False,
    resume_download=True,
    use_safetensors=True,
    device_map="auto"  # ❌ This caused the error
)
```

The `device_map` parameter is used in some HuggingFace models for automatic device placement, but Qwen-Image-Edit pipeline only supports specific strategies: `"balanced"` and `"cuda"`, not `"auto"`.

## Solution Implemented

### 1. **Removed Problematic Parameter**
```python
# FIXED CODE
self.edit_pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=MODEL_CONFIG["torch_dtype"],
    low_cpu_mem_usage=False,      # Optimized for 128GB RAM
    resume_download=True,         # HuggingFace Hub API benefit
    use_safetensors=True         # Faster loading
    # ✅ Removed device_map="auto"
)
```

### 2. **Improved Device Management**
```python
if torch.cuda.is_available():
    # Move to device and apply optimizations
    self.edit_pipe = self.edit_pipe.to(self.device)

    if MEMORY_CONFIG["enable_attention_slicing"]:
        try:
            self.edit_pipe.enable_attention_slicing()
            print("✅ Attention slicing enabled for Qwen-Image-Edit")
        except Exception as opt_error:
            print(f"⚠️ Could not enable attention slicing: {opt_error}")

    # Verify device consistency for edit pipeline
    self._verify_edit_pipeline_devices()
```

### 3. **Added Device Verification Method**
Created `_verify_edit_pipeline_devices()` method that:
- Checks each component (unet, vae, text_encoder) device
- Automatically moves components to correct device if needed
- Provides detailed logging of device status
- Handles errors gracefully

## Benefits of the Fix

1. **Eliminates the Error**: No more "auto not supported" error
2. **Better Device Management**: Explicit device placement with verification
3. **Improved Debugging**: Detailed device status logging
4. **Automatic Correction**: Components moved to correct device if misplaced
5. **Maintains HF Hub Benefits**: Still uses resume_download and other optimizations

## What You Should See Now

After the fix, when loading Qwen-Image-Edit you should see:
```
🔄 Loading Qwen-Image-Edit pipeline for enhanced features...
   Using HuggingFace Hub API for better download reliability
📊 Model size: 20.1 GB (~20GB)
💡 Download will resume automatically if interrupted
✅ Qwen-Image-Edit pipeline loaded successfully!
   • Image-to-Image editing available
   • Inpainting capabilities available
   • Text editing in images available
✅ Attention slicing enabled for Qwen-Image-Edit
🔍 Verifying Qwen-Image-Edit pipeline devices for cuda:
   UNET: cuda:0
   VAE: cuda:0
   TEXT_ENCODER: cuda:0
   SCHEDULER: present (FlowMatchEulerDiscreteScheduler)
✅ Qwen-Image-Edit pipeline verified on cuda
```

## Testing the Fix

You can test the fix using:

```bash
# Test the specific fix
python test_qwen_edit_fix.py

# Or test the full application
python launch.py --mode enhanced
```

## Next Steps

1. **If Model Not Downloaded**: Run `python tools/download_qwen_edit_hub.py`
2. **If Still Issues**: The enhanced downloader has better error handling
3. **For Full Testing**: Launch the enhanced UI to verify all features work

The fix maintains all the HuggingFace Hub API improvements while removing the incompatible device mapping parameter that caused the error.
