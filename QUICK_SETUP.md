# Quick Setup Guide

## 🚀 Get Started in 5 Minutes

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/qwen2.git
cd qwen2
```

### 2. Clone DiffSynth-Studio (Required)

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
```

### 3. Setup Environment

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install DiffSynth-Studio
cd DiffSynth-Studio
pip install -e .
cd ..
```

### 4. Start the System

```bash
# Option 1: Full system startup (recommended)
./start-full-system.sh

# Option 2: Manual startup
# Terminal 1: Start API server
python src/api_server_diffsynth.py

# Terminal 2: Start frontend server
python serve_frontend.py
```

### 5. Access the Web Interface

Open your browser to:

- **Clean Frontend**: http://localhost:3001/frontend/html/clean_frontend.html
- **Enhanced Frontend**: http://localhost:3001/frontend/html/enhanced_frontend.html
- **API Documentation**: http://localhost:8000/docs

## 🎯 Quick Test

```bash
# Test the system
python tools/quick_test.py

# Check API health
curl http://localhost:8000/health
```

## 🛑 Stop the System

```bash
./stop-full-system.sh
```

## 📁 Key Directories

- `frontend/html/` - Web interfaces
- `src/` - API servers and core logic
- `tests/` - Test files
- `tools/` - Utilities and debug tools
- `docs/` - Documentation

## 🔧 Troubleshooting

- **Port conflicts**: System uses ports 8000 (API) and 3001 (frontend)
- **CORS issues**: Use `python serve_frontend.py` instead of opening HTML files directly
- **Model loading**: First generation takes 2-5 minutes to download models

## 📖 More Information

- See `README.md` for detailed documentation
- Check `docs/` for troubleshooting guides
- Run `python tools/quick_test.py` for system diagnostics
