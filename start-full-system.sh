#!/bin/bash
# Complete DiffSynth Enhanced System Startup

echo "🚀 Starting DiffSynth Enhanced Image Generation System"
echo "=" * 60

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🐍 Activating Python 3.11 environment..."
    source .venv311/bin/activate
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Kill existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "api_server_diffsynth" 2>/dev/null || true
pkill -f "http.server.*3001" 2>/dev/null || true
sleep 2

# Start backend
echo "🔧 Starting DiffSynth API Server..."
if check_port 8000; then
    echo "⚠️ Port 8000 is already in use"
    lsof -i :8000
else
    nohup python src/api_server_diffsynth.py > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "✅ Backend started (PID: $BACKEND_PID)"
fi

# Wait for backend to initialize
echo "⏳ Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Start frontend
echo "🎨 Starting Frontend Server..."
if check_port 3001; then
    echo "⚠️ Port 3001 is already in use"
    lsof -i :3001
else
    nohup python serve_frontend.py > frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
fi

# Wait for frontend to be ready
echo "⏳ Waiting for frontend to be ready..."
sleep 3

# Check system status
echo ""
echo "📊 System Status:"
echo "=" * 30

# Backend status
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Backend API: http://localhost:8000"
    echo "✅ API Docs: http://localhost:8000/docs"
else
    echo "❌ Backend API: Not responding"
fi

# Frontend status
if curl -s http://localhost:3001 >/dev/null 2>&1; then
    echo "✅ Clean Frontend: http://localhost:3001/frontend/html/clean_frontend.html"
    echo "✅ Enhanced Frontend: http://localhost:3001/frontend/html/enhanced_frontend.html"
else
    echo "❌ Frontend: Not responding"
fi

# GPU status
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | head -1
fi

echo ""
echo "🎉 DiffSynth Enhanced System is Ready!"
echo ""
echo "🌐 Access URLs:"
echo "   • Clean Frontend: http://localhost:3001/frontend/html/clean_frontend.html"
echo "   • Enhanced Frontend: http://localhost:3001/frontend/html/enhanced_frontend.html"
echo "   • API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Available Features:"
echo "   • 🎨 Text-to-Image Generation"
echo "   • ✏️ Image Editing"
echo "   • 🖌️ Inpainting"
echo "   • 🔍 Outpainting"
echo "   • 🎭 Style Transfer"
echo ""
echo "🛑 To stop the system: ./stop-full-system.sh"
echo ""

# Save PIDs for cleanup
echo "BACKEND_PID=$BACKEND_PID" > .system_pids
echo "FRONTEND_PID=$FRONTEND_PID" >> .system_pids

echo "✨ System startup complete! Open your browser and start creating!"