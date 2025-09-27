#!/bin/bash
# Complete DiffSynth Enhanced System Shutdown

echo "🛑 Stopping DiffSynth Enhanced Image Generation System"
echo "=" * 60

# Function to gracefully stop process
stop_process() {
    local pid=$1
    local name=$2
    
    if kill -0 $pid 2>/dev/null; then
        echo "🔄 Stopping $name (PID: $pid)..."
        kill -TERM $pid
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $pid 2>/dev/null; then
                echo "✅ $name stopped gracefully"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "⚠️ Force killing $name..."
        kill -KILL $pid 2>/dev/null || true
    fi
}

# Stop processes by PID if available
if [ -f .system_pids ]; then
    source .system_pids
    
    if [ ! -z "$BACKEND_PID" ]; then
        stop_process $BACKEND_PID "Backend API"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        stop_process $FRONTEND_PID "Frontend Server"
    fi
    
    rm -f .system_pids
fi

# Stop by process name (fallback)
echo "🧹 Cleaning up any remaining processes..."

# Stop backend
BACKEND_PIDS=$(pgrep -f "api_server_diffsynth" 2>/dev/null || true)
if [ ! -z "$BACKEND_PIDS" ]; then
    echo "🔄 Stopping remaining backend processes..."
    echo "$BACKEND_PIDS" | xargs kill -TERM 2>/dev/null || true
    sleep 2
    echo "$BACKEND_PIDS" | xargs kill -KILL 2>/dev/null || true
fi

# Stop frontend
FRONTEND_PIDS=$(pgrep -f "serve_frontend.py" 2>/dev/null || true)
if [ ! -z "$FRONTEND_PIDS" ]; then
    echo "🔄 Stopping remaining frontend processes..."
    echo "$FRONTEND_PIDS" | xargs kill -TERM 2>/dev/null || true
    sleep 2
    echo "$FRONTEND_PIDS" | xargs kill -KILL 2>/dev/null || true
fi

# Kill by port (final cleanup)
echo "🧹 Final port cleanup..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3001/tcp 2>/dev/null || true

# Clear GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Clearing GPU memory..."
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
" 2>/dev/null || true
fi

# Verify shutdown
echo ""
echo "🔍 Verifying shutdown..."

# Check processes
REMAINING=$(ps aux | grep -E "(api_server_diffsynth|serve_frontend.py)" | grep -v grep | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo "✅ All processes stopped"
else
    echo "⚠️ Some processes may still be running:"
    ps aux | grep -E "(api_server_diffsynth|serve_frontend.py)" | grep -v grep
fi

# Check ports
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️ Port 8000 still in use:"
    lsof -i :8000
else
    echo "✅ Port 8000 is free"
fi

if lsof -i :3001 >/dev/null 2>&1; then
    echo "⚠️ Port 3001 still in use:"
    lsof -i :3001
else
    echo "✅ Port 3000 is free"
fi

echo ""
echo "🎉 DiffSynth Enhanced System shutdown complete!"
echo ""
echo "💡 To restart: ./start-full-system.sh"