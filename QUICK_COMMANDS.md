# Quick Commands Reference 🚀

## 🎯 Daily Use (Most Important)

```bash
# Turn ON development environment
./dev-start.sh

# Turn OFF development environment
./dev-stop.sh

# Launch UI quickly
./dev-ui.sh

# Quick test
./dev-test.sh
```

## 📋 Complete Daily Workflow

### Morning Startup

```bash
cd ~/projects/qwen2
./dev-start.sh          # Activates environment + health check
```

### Work Session

```bash
./dev-ui.sh             # Launch Gradio UI (http://localhost:7860)
# ... do your image editing work ...
./dev-test.sh           # Test after changes
make format lint        # Clean up code
```

### End of Day

```bash
./dev-stop.sh           # Clean shutdown + GPU cleanup
```

## 🛠️ Alternative Commands (Makefile)

```bash
make dev-start          # Same as ./dev-start.sh
make dev-stop           # Same as ./dev-stop.sh
make dev-ui             # Same as ./dev-ui.sh
make dev-test           # Same as ./dev-test.sh
make help               # See all commands
```

## 🔧 Troubleshooting

```bash
# If something is stuck
./dev-stop.sh && ./dev-start.sh

# If downloads fail
make clear-locks && make models

# Nuclear reset
./dev-stop.sh && rm -rf .venv311 && make setup
```

---

**💡 Pro Tip**: Always use `./dev-start.sh` first - it sets up everything correctly for your RTX 4080!
