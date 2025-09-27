# DiffSynth Enhanced Image Generation System 🎨

A comprehensive AI image generation and editing system combining Qwen-Image models with DiffSynth-Studio capabilities, featuring multiple frontends and optimized for high-end hardware.

> **🚀 Latest**: Enhanced with DiffSynth-Studio integration, multiple frontend options, and organized project structure. Full CORS support for seamless web interface experience.

## ✨ Features

### Core Capabilities

- **🎯 Advanced Image Generation**: Text-to-image, image-to-image, inpainting, outpainting
- **🎨 DiffSynth Integration**: Professional-grade image editing with DiffSynth-Studio
- **🎛️ ControlNet Support**: Precise control over generation with various control types
- **🌍 Multi-language Support**: English and Chinese text generation
- **⚡ Hardware Optimized**: Specifically tuned for RTX 4080 + CUDA 12.1 setup

### User Interfaces

- **🌐 Multiple Frontend Options**: Clean, Enhanced, Simple, and Docker-optimized HTML interfaces
- **⚙️ FastAPI Backend**: RESTful API with comprehensive endpoints
- **🔄 Real-time Progress**: Job tracking and progress monitoring
- **📱 Responsive Design**: Works on desktop and mobile devices

### Technical Features

- **🔒 Local Deployment**: No cloud dependencies, complete privacy
- **📊 Metadata Management**: Automatic saving of generation parameters
- **🔄 Resumable Downloads**: Smart model downloading with automatic resume
- **🛡️ CORS Support**: Seamless cross-origin requests for web interfaces
- **🧪 Comprehensive Testing**: Automated testing suite with integration tests

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

### **Initial Setup**

```bash
# 1. Clone the repository
git clone https://github.com/ramcana/qwen2.git
cd qwen2

# 2. Clone DiffSynth-Studio (required dependency)
git clone https://github.com/modelscope/DiffSynth-Studio.git

# 3. Create Python 3.11 virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install DiffSynth-Studio
cd DiffSynth-Studio
pip install -e .
cd ..
```

### **Launch the System**

```bash
# Start the enhanced API server
python src/api_server_diffsynth.py

# In another terminal, start the frontend server
python serve_frontend.py

# Access the web interface at:
# http://localhost:3001/frontend/html/clean_frontend.html
```

### **Alternative: Full System Startup**

```bash
# Start everything with one command
./start-full-system.sh

# Stop everything
./stop-full-system.sh
```

### **Available Frontends**

Choose the interface that best fits your needs:

1. **Clean Frontend** (Recommended):

   ```
   http://localhost:3001/frontend/html/clean_frontend.html
   ```

   - Modern, clean design
   - All features supported
   - Real-time progress tracking

2. **Enhanced Frontend**:

   ```
   http://localhost:3001/frontend/html/enhanced_frontend.html
   ```

   - Feature-rich interface
   - Advanced controls
   - React-based components

3. **Simple Frontend**:

   ```
   http://localhost:3001/frontend/html/simple_frontend.html
   ```

   - Minimal, fast interface
   - Basic functionality
   - Quick testing

4. **Docker Frontend**:
   ```
   http://localhost:3001/frontend/html/docker_frontend.html
   ```
   - Optimized for containerized deployment
   - Production-ready

### **Development Commands**

```bash
# Test the system
python tools/quick_test.py

# Run frontend tests
./scripts/run-frontend-tests.sh

# Debug performance issues
python tools/debug/debug_performance_issues.py

# Check model information
python tools/debug/check_model_info.py

# Start/stop full system
./start-full-system.sh
./stop-full-system.sh
```

## 📁 Project Structure

```
qwen2/
├── .venv311/                    # Python 3.11 virtual environment
├── src/                         # Main application code
│   ├── api_server_diffsynth.py  # Enhanced DiffSynth API server
│   ├── api_server.py            # Original Qwen API server
│   ├── diffsynth_service.py     # DiffSynth integration service
│   ├── controlnet_service.py    # ControlNet integration
│   ├── qwen_generator.py        # Core Qwen generation logic
│   └── utils/                   # Utility modules
├── frontend/                    # Web interfaces
│   ├── html/                    # HTML frontend files
│   │   ├── clean_frontend.html  # Clean, modern interface
│   │   ├── enhanced_frontend.html # Feature-rich interface
│   │   ├── simple_frontend.html # Basic interface
│   │   └── docker_frontend.html # Docker-optimized interface
│   └── src/                     # React frontend (if using)
├── tests/                       # Test suite
│   ├── frontend/                # Frontend tests
│   │   ├── simple_test.html     # Basic API tests
│   │   ├── test_connection.html # Connection tests
│   │   └── test_ui.html         # UI tests
│   └── *.py                     # Python test files
├── tools/                       # Development tools
│   ├── debug/                   # Debug utilities
│   │   ├── debug_performance_issues.py
│   │   ├── check_model_info.py
│   │   └── check_model_size.py
│   └── quick_test.py            # Quick system test
├── scripts/                     # Automation scripts
│   ├── setup/                   # Setup scripts
│   │   ├── setup_ubuntu_native.sh
│   │   └── diffsynth_qwen_setup.py
│   ├── fix-frontend.sh          # Frontend fixes
│   └── run-frontend-tests.sh    # Frontend testing
├── docs/                        # Documentation
│   ├── troubleshooting/         # Troubleshooting guides
│   │   ├── WSL_CRASH_ANALYSIS.md
│   │   ├── CACHE_FIX_SUMMARY.md
│   │   └── SOLUTION.md
│   ├── DAILY_WORKFLOW.md        # Daily usage guide
│   └── integration_plan.md      # Integration documentation
├── config/                      # Configuration files
│   ├── docker/                  # Docker configurations
│   │   ├── docker-compose.prod.yml
│   │   ├── traefik.yml
│   │   └── nginx.conf
│   └── *.json                   # JSON config files
├── DiffSynth-Studio/            # External dependency (git clone)
├── models/                      # Downloaded models
├── generated_images/            # Output directory
├── serve_frontend.py            # Frontend server utility
├── start-full-system.sh         # System startup script
├── stop-full-system.sh          # System shutdown script
└── requirements.txt             # Pinned dependencies
```

## 💡 Usage Examples

### Web Interface Usage

1. **Access the Clean Frontend**:

   ```
   http://localhost:3001/frontend/html/clean_frontend.html
   ```

2. **Generate Images**:

   - Enter your prompt: "A beautiful sunset over mountains"
   - Adjust parameters (width, height, steps, CFG scale)
   - Click "Generate Image"
   - Monitor progress in real-time

3. **Image Editing**:
   - Upload an input image
   - Select edit mode (Edit, Inpaint, Outpaint, Style Transfer)
   - Enter editing prompt
   - Generate enhanced image

### API Usage

```python
import requests
import base64
from PIL import Image
import io

# Text-to-Image Generation
response = requests.post("http://localhost:8000/api/generate/text-to-image", json={
    "prompt": "A beautiful sunset over mountains, digital art",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "cfg_scale": 7.0
})

job_id = response.json()["job_id"]

# Monitor job progress
while True:
    status = requests.get(f"http://localhost:8000/api/jobs/{job_id}")
    job = status.json()
    if job["status"] == "completed":
        image_url = job["result"]["image_url"]
        break
```

### Image Editing with API

```python
# Convert image to base64
with open("input.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Edit image
response = requests.post("http://localhost:8000/api/edit/image", json={
    "prompt": "Add a rainbow in the sky",
    "operation": "edit",
    "image_base64": image_base64,
    "strength": 0.8,
    "steps": 20
})
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
- 🔧 **Troubleshooting**: `docs/troubleshooting/` - Common issues and solutions
- 🌐 **Daily Workflow**: `docs/DAILY_WORKFLOW.md` - Usage patterns
- 📋 **Integration Plan**: `docs/integration_plan.md` - System architecture

### Performance Tips

- **Frontend Server**: Use `python serve_frontend.py` for optimal CORS handling
- **API Health**: Check `http://localhost:8000/health` for system status
- **Memory Management**: Monitor GPU usage via API endpoints
- Use float16 precision for RTX 40-series GPUs
- Enable attention slicing for memory efficiency
- Keep VRAM usage under 14GB for stability

## 🚀 What's New

### ✨ Latest Updates (v3.0)

- **🎨 DiffSynth Integration**: Full DiffSynth-Studio integration with professional editing capabilities
- **🌐 Multiple Frontends**: Clean, Enhanced, Simple, and Docker-optimized web interfaces
- **⚙️ Enhanced API**: Comprehensive FastAPI backend with job tracking and progress monitoring
- **🛡️ CORS Support**: Seamless cross-origin requests for web interfaces
- **📁 Organized Structure**: Clean project organization with dedicated directories
- **🧪 Comprehensive Testing**: Frontend and backend testing suites

### 🎯 Key Features

- **🔄 Real-time Progress**: Live job monitoring with progress updates
- **🎛️ ControlNet Support**: Advanced control over image generation
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🔒 Local Privacy**: All processing happens on your hardware
- **⚡ Hardware Optimized**: Tuned for RTX 4080 + CUDA 12.1
- **🛠️ Developer Friendly**: Organized codebase with comprehensive documentation

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

## 🔗 Important Links

- **API Documentation**: `http://localhost:8000/docs` (when server is running)
- **Health Check**: `http://localhost:8000/health`
- **DiffSynth-Studio**: [GitHub Repository](https://github.com/modelscope/DiffSynth-Studio)
- **Frontend Server**: `http://localhost:3001`

## 📋 Prerequisites

Before starting, ensure you have:

- **Git**: For cloning repositories
- **Python 3.11**: Required for the virtual environment
- **CUDA 12.1+**: For GPU acceleration
- **16GB+ VRAM**: RTX 4080 or equivalent
- **50GB+ Storage**: For models and generated images

---

**🎯 Hardware Optimized**: Specifically tuned for RTX 4080 + CUDA 12.1 setups  
**🔒 Local Privacy**: All processing happens on your hardware  
**⚡ Production Ready**: Professional-quality image generation and editing  
**🌐 Web-First**: Modern web interfaces with real-time progress tracking  
**🛠️ Developer Friendly**: Organized codebase with comprehensive testing
