#!/bin/bash
set -e

# Docker entrypoint script for Qwen-Image API Server
# Handles initialization, health checks, graceful startup, and failure recovery

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling function
handle_error() {
    local exit_code=$?
    log "ERROR: Command failed with exit code $exit_code"
    log "ERROR: Line $1"
    
    # Cleanup on error
    cleanup_on_error
    exit $exit_code
}

# Cleanup function for error scenarios
cleanup_on_error() {
    log "Performing cleanup due to error..."
    
    # Clear GPU memory if available
    if command -v nvidia-smi &> /dev/null; then
        log "Clearing GPU memory..."
        python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('GPU memory cleared')
" 2>/dev/null || true
    fi
    
    # Kill any hanging processes
    pkill -f "python.*api_server.py" 2>/dev/null || true
}

# Set error trap
trap 'handle_error $LINENO' ERR

# Graceful shutdown handler
shutdown_handler() {
    log "Received shutdown signal, performing graceful shutdown..."
    
    # Send SIGTERM to main process
    if [ ! -z "$MAIN_PID" ]; then
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local count=0
        while kill -0 "$MAIN_PID" 2>/dev/null && [ $count -lt 30 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            log "Force killing main process..."
            kill -KILL "$MAIN_PID" 2>/dev/null || true
        fi
    fi
    
    cleanup_on_error
    exit 0
}

# Set shutdown trap
trap shutdown_handler SIGTERM SIGINT

log "Starting Qwen-Image API Server..."
log "Python version: $(python --version)"
log "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check GPU availability with retry logic
check_gpu() {
    local retries=3
    local count=0
    
    while [ $count -lt $retries ]; do
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi &> /dev/null; then
                log "GPU Information:"
                nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
                return 0
            else
                log "WARNING: nvidia-smi command failed, attempt $((count + 1))/$retries"
            fi
        else
            log "No GPU detected, running in CPU mode"
            return 1
        fi
        
        count=$((count + 1))
        sleep 2
    done
    
    log "WARNING: GPU check failed after $retries attempts, continuing without GPU"
    return 1
}

check_gpu

# Create necessary directories if they don't exist
mkdir -p /app/cache/huggingface
mkdir -p /app/cache/torch
mkdir -p /app/cache/diffsynth
mkdir -p /app/cache/controlnet
mkdir -p /app/generated_images
mkdir -p /app/uploads
mkdir -p /app/offload
mkdir -p /app/logs

# Set proper permissions
chmod -R 755 /app/cache
chmod -R 755 /app/generated_images
chmod -R 755 /app/uploads
chmod -R 755 /app/offload

# Check if DiffSynth-Studio is properly installed
if [ -d "/app/DiffSynth-Studio" ]; then
    echo "DiffSynth-Studio found, checking installation..."
    cd /app/DiffSynth-Studio
    if python -c "import diffsynth" 2>/dev/null; then
        echo "DiffSynth-Studio successfully imported"
    else
        echo "Warning: DiffSynth-Studio import failed, attempting reinstall..."
        pip install -e . --no-deps || echo "DiffSynth reinstall failed, continuing without it"
    fi
    cd /app
else
    echo "Warning: DiffSynth-Studio directory not found"
fi

# Verify critical dependencies
echo "Verifying critical dependencies..."
python -c "
import sys
try:
    import torch
    print(f'✓ PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')
    sys.exit(1)

try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except ImportError as e:
    print(f'✗ Transformers import failed: {e}')
    sys.exit(1)

try:
    import fastapi
    print(f'✓ FastAPI {fastapi.__version__}')
except ImportError as e:
    print(f'✗ FastAPI import failed: {e}')
    sys.exit(1)

try:
    import diffusers
    print(f'✓ Diffusers {diffusers.__version__}')
except ImportError as e:
    print(f'✗ Diffusers import failed: {e}')
    sys.exit(1)
"

# Set memory optimization flags if enabled
if [ "$MEMORY_OPTIMIZATION" = "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_LAUNCH_BLOCKING=0
    echo "Memory optimization enabled"
fi

# Pre-flight checks
preflight_checks() {
    log "Performing pre-flight checks..."
    
    # Check disk space
    local available_space=$(df /app | awk 'NR==2 {print $4}')
    local required_space=1048576  # 1GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log "ERROR: Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
        return 1
    fi
    
    # Check memory
    local available_memory=$(free | awk 'NR==2{printf "%.0f", $7/1024}')
    local required_memory=2048  # 2GB in MB
    
    if [ "$available_memory" -lt "$required_memory" ]; then
        log "WARNING: Low available memory. Available: ${available_memory}MB, Recommended: ${required_memory}MB"
    fi
    
    # Check critical files
    local critical_files=(
        "/app/src/api_server.py"
        "/app/src/monitoring_config.py"
        "/app/src/qwen_generator.py"
    )
    
    for file in "${critical_files[@]}"; do
        if [ ! -f "$file" ]; then
            log "ERROR: Critical file missing: $file"
            return 1
        fi
    done
    
    log "Pre-flight checks completed successfully"
    return 0
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=0
    
    log "Waiting for API server to become healthy..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log "API server is healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
    done
    
    log "ERROR: API server failed to become healthy after $max_attempts attempts"
    return 1
}

# Start the application with monitoring
start_application() {
    log "Starting API server on port 8000..."
    
    # Start the main application in background
    "$@" &
    MAIN_PID=$!
    
    # Wait a bit for startup
    sleep 5
    
    # Check if process is still running
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
        log "ERROR: Main process died during startup"
        return 1
    fi
    
    # Perform health check
    if ! health_check; then
        log "ERROR: Health check failed"
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        return 1
    fi
    
    log "API server started successfully with PID $MAIN_PID"
    
    # Wait for main process
    wait "$MAIN_PID"
}

# Main execution
main() {
    # Perform pre-flight checks
    if ! preflight_checks; then
        log "ERROR: Pre-flight checks failed"
        exit 1
    fi
    
    # Start application with retry logic
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        log "Starting application (attempt $((retry + 1))/$max_retries)..."
        
        if start_application "$@"; then
            log "Application started successfully"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                log "Application start failed, retrying in 30 seconds..."
                cleanup_on_error
                sleep 30
            fi
        fi
    done
    
    log "ERROR: Failed to start application after $max_retries attempts"
    exit 1
}

# Execute main function
main "$@"