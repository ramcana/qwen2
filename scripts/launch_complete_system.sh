#!/bin/bash
# Qwen-Image Complete System Launcher
# Starts both backend API and React frontend

# Remove set -e to prevent abrupt exits on errors
# set -e  # Exit on any error

echo "🎨 QWEN-IMAGE COMPLETE SYSTEM LAUNCHER"
echo "======================================"
echo "High-Performance Configuration for RTX 4080"
echo "Expected generation time: 15-60 seconds"
echo "======================================"

# Function to clean up background processes
cleanup() {
    echo -e "\n🛑 Shutting down services..."
    if [[ -n $BACKEND_PID ]]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "✅ Backend stopped"
    fi
    if [[ -n $FRONTEND_PID ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "✅ Frontend stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Set QWEN_HOME to prevent model redownloading
export QWEN_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "🏠 Setting QWEN_HOME to: $QWEN_HOME"

# Activate virtual environment
echo "✅ Activating virtual environment..."
cd "$QWEN_HOME" || { echo "❌ Failed to change directory"; exit 1; }
source venv/bin/activate

# Start backend API server
echo "🔄 Starting backend API server..."
python src/api/robust_server.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to initialize
sleep 5

# Check if backend is still running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start properly"
    echo "💡 Check the logs above for error messages"
    echo "💡 Try running: python src/api/robust_server.py directly to see detailed errors"
    exit 1
fi

# Start frontend
echo "🔄 Starting React frontend..."
cd frontend || { echo "❌ Failed to change to frontend directory"; exit 1; }
npm start &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

# Display access information
echo ""
echo "🎉 SYSTEM READY!"
echo "================"
echo "Backend API:    http://localhost:8000"
echo "Frontend UI:    http://localhost:3000"
echo "API Docs:       http://localhost:8000/docs"
echo ""
echo "🔧 TROUBLESHOOTING TIPS:"
echo "   • First generation may take 2-5 minutes (model loading)"
echo "   • Check backend logs if generation takes >5 minutes"
echo "   • Run 'nvidia-smi' to monitor GPU usage"
echo "   • Run 'python tools/performance_optimizer.py' for diagnostics"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes with error handling
while true; do
    # Check if processes are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null && ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Both processes have exited"
        break
    fi
    
    # Check backend process
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process has exited"
        # Try to restart backend
        echo "🔄 Attempting to restart backend..."
        cd "$QWEN_HOME" || exit
        python src/api/robust_server.py &
        BACKEND_PID=$!
        echo "✅ Backend restarted (PID: $BACKEND_PID)"
    fi
    
    # Check frontend process
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend process has exited"
        break
    fi
    
    sleep 5
done

# Cleanup on exit
cleanup