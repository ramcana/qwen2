# DiffSynth Enhanced UI Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues when using the DiffSynth Enhanced UI system. Issues are organized by category with step-by-step solutions.

## System Initialization Issues

### Service Fails to Start

**Symptoms**:

- Error messages during startup
- Services show as "not loaded" in status
- UI shows initialization errors

**Causes & Solutions**:

1. **Insufficient GPU Memory**

   ```bash
   # Check GPU memory
   nvidia-smi

   # Clear GPU memory
   curl -X GET http://localhost:8000/memory/clear
   ```

2. **Missing Dependencies**

   ```bash
   # Install missing packages
   pip install -r requirements.txt

   # For DiffSynth specific dependencies
   pip install diffsynth-api controlnet-aux
   ```

3. **Model Download Issues**

   ```bash
   # Check model download status
   python tools/download_models.py --check

   # Force re-download if corrupted
   python tools/download_models.py --force
   ```

### DiffSynth Service Won't Initialize

**Symptoms**:

- Qwen works but DiffSynth features unavailable
- "DiffSynth service not available" errors
- Edit mode shows fallback UI

**Solutions**:

1. **Check DiffSynth Installation**

   ```python
   # Test DiffSynth import
   try:
       from diffsynth import ModelManager
       print("DiffSynth available")
   except ImportError as e:
       print(f"DiffSynth not available: {e}")
   ```

2. **Verify Model Files**

   ```bash
   # Check if DiffSynth models are downloaded
   ls -la models/diffsynth/

   # Download missing models
   python tools/download_diffsynth_models.py
   ```

3. **Memory Allocation Issues**

   ```bash
   # Check available memory
   curl -X GET http://localhost:8000/memory/status

   # Restart with memory optimization
   python start.py --optimize-memory
   ```

## Image Processing Issues

### Upload Failures

**Symptoms**:

- Images won't upload
- "File format not supported" errors
- Upload progress stalls

**Solutions**:

1. **File Format Issues**

   - **Supported formats**: PNG, JPEG, WebP, TIFF
   - **Convert unsupported formats**:
     ```python
     from PIL import Image
     img = Image.open('image.bmp')
     img.save('image.png')
     ```

2. **File Size Limits**

   - **Maximum size**: 50MB per file
   - **Reduce file size**:
     ```python
     from PIL import Image
     img = Image.open('large_image.jpg')
     img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
     img.save('resized_image.jpg', quality=85)
     ```

3. **Network Issues**
   - Check browser console for network errors
   - Verify server is running: `curl http://localhost:8000/health`
   - Try uploading smaller test image first

### Processing Failures

**Symptoms**:

- Processing starts but fails
- "Out of memory" errors
- Processing hangs indefinitely

**Solutions**:

1. **Memory Issues**

   ```bash
   # Check GPU memory usage
   nvidia-smi

   # Clear memory and retry
   curl -X GET http://localhost:8000/memory/clear

   # Enable tiled processing for large images
   # (automatically enabled for images >2048px)
   ```

2. **Image Size Issues**

   - **Recommended sizes**: 512x512 to 2048x2048
   - **For larger images**: System uses automatic tiling
   - **Resize if needed**:
     ```python
     from PIL import Image
     img = Image.open('huge_image.jpg')
     img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
     img.save('processed_image.jpg')
     ```

3. **Parameter Issues**
   - **Reduce inference steps**: Try 20-30 instead of 50+
   - **Lower CFG scale**: Try 4.0-7.0 range
   - **Simplify prompts**: Avoid overly complex descriptions

## ControlNet Issues

### Control Detection Fails

**Symptoms**:

- "No control features detected" message
- Auto-detection returns low confidence
- Control preview shows poor quality

**Solutions**:

1. **Image Quality Issues**

   ```python
   # Enhance image contrast
   from PIL import Image, ImageEnhance
   img = Image.open('control_image.jpg')
   enhancer = ImageEnhance.Contrast(img)
   enhanced = enhancer.enhance(1.5)
   enhanced.save('enhanced_control.jpg')
   ```

2. **Wrong Control Type**

   - **For architectural images**: Use Canny edge detection
   - **For human figures**: Use Pose detection
   - **For landscapes**: Use Depth or Segmentation
   - **Try manual selection** instead of auto-detection

3. **Preprocessing Issues**
   ```python
   # Clean up control image
   from PIL import Image
   img = Image.open('noisy_control.jpg')
   # Convert to grayscale for edge detection
   gray = img.convert('L')
   gray.save('clean_control.jpg')
   ```

### Poor ControlNet Results

**Symptoms**:

- Generated image doesn't follow control
- Control is too strong or too weak
- Inconsistent results

**Solutions**:

1. **Adjust Conditioning Scale**

   - **Too weak control**: Increase scale to 1.2-1.5
   - **Too strong control**: Decrease scale to 0.6-0.8
   - **Balanced control**: Use 0.8-1.0

2. **Guidance Scheduling**

   ```json
   {
     "control_guidance_start": 0.0,
     "control_guidance_end": 0.8,
     "controlnet_conditioning_scale": 1.0
   }
   ```

3. **Control Image Quality**
   - Use high-resolution control images
   - Ensure clear, unambiguous features
   - Remove background noise or distractions

## Performance Issues

### Slow Processing

**Symptoms**:

- Generation takes much longer than expected
- UI becomes unresponsive
- High memory usage

**Solutions**:

1. **Optimize Parameters**

   ```json
   {
     "num_inference_steps": 25,
     "guidance_scale": 7.0,
     "use_tiled_processing": true
   }
   ```

2. **System Optimization**

   ```bash
   # Close unnecessary applications
   # Monitor GPU usage
   nvidia-smi -l 1

   # Use performance mode
   python start.py --performance-mode
   ```

3. **Batch Processing**
   - Process multiple similar images together
   - Use consistent parameters to avoid reloading
   - Queue multiple jobs instead of processing individually

### Memory Errors

**Symptoms**:

- "CUDA out of memory" errors
- System becomes unresponsive
- Services crash unexpectedly

**Solutions**:

1. **Immediate Memory Cleanup**

   ```bash
   # Clear GPU memory
   curl -X GET http://localhost:8000/memory/clear

   # Restart services if needed
   curl -X POST http://localhost:8000/services/restart
   ```

2. **Prevent Memory Issues**

   ```python
   # Use smaller batch sizes
   # Enable gradient checkpointing
   # Use mixed precision training
   ```

3. **System Configuration**
   ```bash
   # Increase swap space (Linux)
   sudo swapon --show
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

## UI and Frontend Issues

### Interface Not Loading

**Symptoms**:

- Blank page or loading screen
- JavaScript errors in browser console
- Features missing or not working

**Solutions**:

1. **Browser Issues**

   - **Clear browser cache**: Ctrl+Shift+R (Chrome/Firefox)
   - **Try different browser**: Chrome, Firefox, Safari
   - **Disable browser extensions**: Test in incognito/private mode

2. **Frontend Service Issues**

   ```bash
   # Check if frontend is running
   curl http://localhost:3000

   # Restart frontend service
   cd frontend
   npm start
   ```

3. **CORS Issues**
   - Check browser console for CORS errors
   - Verify API server CORS configuration
   - Use correct URLs (localhost vs 127.0.0.1)

### Mode Switching Problems

**Symptoms**:

- Can't switch between Generate/Edit/ControlNet modes
- Mode buttons not responding
- State not preserved between modes

**Solutions**:

1. **Clear Browser State**

   ```javascript
   // In browser console
   localStorage.clear();
   sessionStorage.clear();
   location.reload();
   ```

2. **Check Service Status**

   ```bash
   # Verify all services are running
   curl http://localhost:8000/services/status
   ```

3. **Reset UI State**
   - Refresh the page
   - Clear any stuck processing jobs
   - Check for JavaScript errors in console

## API and Integration Issues

### API Endpoints Not Responding

**Symptoms**:

- 404 errors for DiffSynth endpoints
- Timeout errors
- Connection refused errors

**Solutions**:

1. **Verify Service Status**

   ```bash
   # Check API server
   curl http://localhost:8000/health

   # Check specific endpoints
   curl http://localhost:8000/diffsynth/status
   curl http://localhost:8000/controlnet/types
   ```

2. **Check Service Configuration**

   ```python
   # Verify services are properly initialized
   import requests
   response = requests.get('http://localhost:8000/services/status')
   print(response.json())
   ```

3. **Restart Services**

   ```bash
   # Restart API server
   python src/api_server.py

   # Or use restart endpoint
   curl -X POST http://localhost:8000/services/restart
   ```

### Authentication/Permission Issues

**Symptoms**:

- 403 Forbidden errors
- Permission denied messages
- File access errors

**Solutions**:

1. **File Permissions**

   ```bash
   # Fix file permissions (Linux/Mac)
   chmod -R 755 generated_images/
   chmod -R 755 uploads/

   # Windows - run as administrator if needed
   ```

2. **Directory Access**

   ```bash
   # Ensure directories exist
   mkdir -p generated_images uploads cache

   # Check disk space
   df -h
   ```

## Diagnostic Tools

### System Health Check

```bash
# Comprehensive system check
curl http://localhost:8000/health
curl http://localhost:8000/status
curl http://localhost:8000/memory/status
curl http://localhost:8000/services/status
```

### Log Analysis

```bash
# Check application logs
tail -f logs/application.log

# Check error logs
grep -i error logs/application.log

# Check GPU logs
nvidia-smi -q -d MEMORY,UTILIZATION
```

### Performance Monitoring

```python
# Monitor system performance
import psutil
import GPUtil

# CPU usage
print(f"CPU: {psutil.cpu_percent()}%")

# Memory usage
memory = psutil.virtual_memory()
print(f"RAM: {memory.percent}%")

# GPU usage
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUtil*100:.1f}% memory")
```

## Getting Additional Help

### Log Collection

When reporting issues, collect these logs:

```bash
# System information
python --version
pip list | grep -E "(torch|diffsynth|controlnet)"
nvidia-smi

# Application logs
tail -100 logs/application.log > debug_logs.txt

# System status
curl http://localhost:8000/status > system_status.json
```

### Common Error Codes

- **500**: Internal server error - check logs
- **404**: Endpoint not found - verify URL and service status
- **400**: Bad request - check request parameters
- **403**: Permission denied - check file permissions
- **503**: Service unavailable - restart services

### Support Resources

- **Documentation**: Check other guides in the `docs/` folder
- **GitHub Issues**: Report bugs and feature requests
- **Community Forums**: Ask questions and share solutions
- **API Documentation**: `/docs` endpoint for interactive API docs

## Prevention Tips

### Regular Maintenance

```bash
# Weekly cleanup
curl -X GET http://localhost:8000/memory/clear
python tools/cleanup_old_files.py

# Monthly updates
pip install --upgrade -r requirements.txt
python tools/update_models.py
```

### Best Practices

1. **Monitor Resources**: Keep an eye on GPU memory and disk space
2. **Regular Backups**: Backup important generated images and presets
3. **Update Regularly**: Keep dependencies and models up to date
4. **Test Changes**: Test new configurations with small images first
5. **Document Issues**: Keep notes on successful parameter combinations

This troubleshooting guide covers the most common issues you might encounter. For specific problems not covered here, check the application logs and system status for more detailed error information.
