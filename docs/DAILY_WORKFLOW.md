# Daily Development Workflow 🚀

Quick reference for everyday use of the Qwen Image Edit development environment.

## ⚡ Quick Start (Recommended)

### Turn On Development Environment

```bash
# One command to start everything
./dev-start.sh
# or
make dev-start
```

This will:

- ✅ Activate Python 3.11 environment
- ✅ Run health checks (CUDA, GPU, dependencies)
- ✅ Show available commands
- ✅ Keep you in an active development shell

### Turn Off Development Environment

```bash
# Clean shutdown
./dev-stop.sh
# or
make dev-stop
```

This will:

- 🛑 Stop all running processes (Gradio, FastAPI, etc.)
- 🧹 Clear GPU memory
- 🐍 Deactivate virtual environment
- ✅ Clean exit

## 🎨 Launch UI Quickly

```bash
# Quick UI launcher
./dev-ui.sh
# or
make dev-ui
```

## 🧪 Quick Testing

```bash
# Run smoke test
./dev-test.sh
# or
make dev-test
```

## 📋 Daily Commands

### Morning Startup

```bash
cd ~/projects/qwen2
./dev-start.sh
```

### Launch UI for Work

```bash
./dev-ui.sh
# UI will be available at http://localhost:7860
```

### Test Changes

```bash
./dev-test.sh
```

### End of Day Shutdown

```bash
./dev-stop.sh
```

## 🛠️ Development Tasks

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Both together
make format lint
```

### Full Development Cycle

```bash
# 1. Start environment
./dev-start.sh

# 2. Make changes to code
# ... edit files ...

# 3. Test changes
./dev-test.sh

# 4. Format and lint
make format lint

# 5. Launch UI to test
./dev-ui.sh

# 6. When done
./dev-stop.sh
```

## 🔧 Troubleshooting

### Environment Issues

```bash
# Clean restart
./dev-stop.sh
./dev-start.sh
```

### GPU Memory Issues

```bash
# Clear GPU memory
./dev-stop.sh
# Wait a moment, then restart
./dev-start.sh
```

### Download Issues

```bash
# Clear locks and retry
make clear-locks
make models
```

### Complete Reset

```bash
# Nuclear option - clean everything
./dev-stop.sh
make clean
rm -rf .venv311
make setup
./dev-start.sh
```

## 📊 Status Checks

### Quick Health Check

```bash
# From activated environment
python -c "
import torch
from diffusers import QwenImageEditPipeline
print('✅ Environment ready')
print('✅ CUDA:', torch.cuda.is_available())
print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

### Model Status

```bash
ls -la models/Qwen-Image-Edit/
du -sh models/Qwen-Image-Edit/
```

### Process Status

```bash
# Check running processes
ps aux | grep -E "(gradio|python.*qwen)"
```

## 💡 Pro Tips

1. **Always use `./dev-start.sh`** - It sets up everything correctly
2. **Use `./dev-stop.sh` before shutdown** - Prevents GPU memory leaks
3. **Run `./dev-test.sh` after changes** - Quick validation
4. **Keep one terminal for development** - Use the activated shell from dev-start.sh
5. **Use `make help`** - See all available commands

## 🎯 Typical Day

```bash
# Morning
cd ~/projects/qwen2
./dev-start.sh

# Work session
./dev-ui.sh          # Launch UI
# ... do work ...
./dev-test.sh        # Test changes
make format lint     # Clean up code

# End of day
./dev-stop.sh
```

---

**🚀 Happy Coding!** The development environment is optimized for your RTX 4080 + Python 3.11 setup.
