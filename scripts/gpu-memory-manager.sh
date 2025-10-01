#!/bin/bash
# =============================================================================
# GPU Memory Management Script for Qwen2 Docker Environment
# =============================================================================
# This script provides GPU memory monitoring, optimization, and cleanup
# for better resource utilization in containerized environments.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/gpu-memory.log"
VERBOSE=${VERBOSE:-false}

# GPU memory thresholds (in MB)
GPU_MEMORY_WARNING_THRESHOLD=${GPU_MEMORY_WARNING_THRESHOLD:-12288}  # 12GB
GPU_MEMORY_CRITICAL_THRESHOLD=${GPU_MEMORY_CRITICAL_THRESHOLD:-14336} # 14GB
GPU_MEMORY_CLEANUP_THRESHOLD=${GPU_MEMORY_CLEANUP_THRESHOLD:-15360}   # 15GB

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        case $level in
            "ERROR") echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
            "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
            "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
            "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" ;;
        esac
    fi
}

# Check if NVIDIA GPU is available
check_gpu_availability() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR" "nvidia-smi not found. NVIDIA GPU support not available."
        return 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log "ERROR" "nvidia-smi failed to run. Check NVIDIA driver installation."
        return 1
    fi
    
    return 0
}

# Get GPU memory information
get_gpu_memory_info() {
    local gpu_id=${1:-0}
    
    if ! check_gpu_availability; then
        return 1
    fi
    
    # Get memory information in MB
    local memory_info=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits)
    
    if [[ -z "$memory_info" ]]; then
        log "ERROR" "Failed to get GPU memory information"
        return 1
    fi
    
    echo "$memory_info"
}

# Display GPU status
show_gpu_status() {
    log "INFO" "GPU Status Report"
    log "INFO" "=================="
    
    if ! check_gpu_availability; then
        return 1
    fi
    
    # Get number of GPUs
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    log "INFO" "Number of GPUs: $gpu_count"
    
    for ((i=0; i<gpu_count; i++)); do
        log "INFO" "GPU $i Status:"
        
        local memory_info=$(get_gpu_memory_info "$i")
        if [[ $? -eq 0 ]]; then
            local total_memory=$(echo "$memory_info" | cut -d',' -f1 | tr -d ' ')
            local used_memory=$(echo "$memory_info" | cut -d',' -f2 | tr -d ' ')
            local free_memory=$(echo "$memory_info" | cut -d',' -f3 | tr -d ' ')
            
            local usage_percent=$((used_memory * 100 / total_memory))
            
            log "INFO" "  Total Memory: ${total_memory}MB"
            log "INFO" "  Used Memory:  ${used_memory}MB (${usage_percent}%)"
            log "INFO" "  Free Memory:  ${free_memory}MB"
            
            # Check thresholds
            if [[ $used_memory -gt $GPU_MEMORY_CRITICAL_THRESHOLD ]]; then
                log "ERROR" "  Status: CRITICAL - Memory usage above ${GPU_MEMORY_CRITICAL_THRESHOLD}MB"
            elif [[ $used_memory -gt $GPU_MEMORY_WARNING_THRESHOLD ]]; then
                log "WARN" "  Status: WARNING - Memory usage above ${GPU_MEMORY_WARNING_THRESHOLD}MB"
            else
                log "INFO" "  Status: OK"
            fi
        fi
        
        # Get GPU processes
        local processes=$(nvidia-smi --id="$i" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
        if [[ -n "$processes" ]]; then
            log "INFO" "  Active Processes:"
            while IFS=',' read -r pid process_name used_mem; do
                pid=$(echo "$pid" | tr -d ' ')
                process_name=$(echo "$process_name" | tr -d ' ')
                used_mem=$(echo "$used_mem" | tr -d ' ')
                log "INFO" "    PID: $pid, Process: $process_name, Memory: ${used_mem}MB"
            done <<< "$processes"
        else
            log "INFO" "  No active processes"
        fi
        
        echo ""
    done
}

# Clear GPU memory cache
clear_gpu_cache() {
    local container_name=${1:-"qwen-api"}
    
    log "INFO" "Clearing GPU memory cache in container: $container_name"
    
    # Check if container is running
    if ! docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        log "WARN" "Container $container_name is not running"
        return 1
    fi
    
    # Clear PyTorch GPU cache
    local clear_script='
import torch
import gc
import os

def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                # Clear cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Get memory info
                allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                cached = torch.cuda.memory_reserved(i) / 1024**2      # MB
                
                print(f"GPU {i} - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
        
        print("GPU memory cache cleared successfully")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    clear_gpu_memory()
'
    
    # Execute the script in the container
    if docker exec "$container_name" python -c "$clear_script" 2>/dev/null; then
        log "INFO" "GPU memory cache cleared successfully"
        return 0
    else
        log "ERROR" "Failed to clear GPU memory cache"
        return 1
    fi
}

# Optimize GPU memory settings
optimize_gpu_memory() {
    local container_name=${1:-"qwen-api"}
    
    log "INFO" "Optimizing GPU memory settings in container: $container_name"
    
    # Check if container is running
    if ! docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        log "WARN" "Container $container_name is not running"
        return 1
    fi
    
    # Set GPU memory optimization environment variables
    local optimization_script='
import os
import torch

def optimize_gpu_settings():
    """Set optimal GPU memory settings."""
    if torch.cuda.is_available():
        # Enable memory fraction allocation
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Enable memory growth (similar to TensorFlow)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Set memory pool settings
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        
        print("GPU memory optimization settings applied")
        
        # Display current settings
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**2:.0f}MB")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    optimize_gpu_settings()
'
    
    # Execute the optimization script
    if docker exec "$container_name" python -c "$optimization_script" 2>/dev/null; then
        log "INFO" "GPU memory optimization completed"
        return 0
    else
        log "ERROR" "Failed to optimize GPU memory settings"
        return 1
    fi
}

# Monitor GPU memory usage
monitor_gpu_memory() {
    local duration=${1:-60}  # Monitor for 60 seconds by default
    local interval=${2:-5}   # Check every 5 seconds
    
    log "INFO" "Monitoring GPU memory for ${duration} seconds (interval: ${interval}s)"
    
    if ! check_gpu_availability; then
        return 1
    fi
    
    local end_time=$(($(date +%s) + duration))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local memory_info=$(get_gpu_memory_info 0)
        if [[ $? -eq 0 ]]; then
            local used_memory=$(echo "$memory_info" | cut -d',' -f2 | tr -d ' ')
            local free_memory=$(echo "$memory_info" | cut -d',' -f3 | tr -d ' ')
            
            log "INFO" "GPU Memory - Used: ${used_memory}MB, Free: ${free_memory}MB"
            
            # Check if cleanup is needed
            if [[ $used_memory -gt $GPU_MEMORY_CLEANUP_THRESHOLD ]]; then
                log "WARN" "GPU memory usage critical, attempting cleanup..."
                clear_gpu_cache
            fi
        fi
        
        sleep "$interval"
    done
}

# Kill GPU processes if memory is critical
emergency_gpu_cleanup() {
    log "WARN" "Performing emergency GPU cleanup..."
    
    if ! check_gpu_availability; then
        return 1
    fi
    
    # Get all GPU processes
    local processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    
    if [[ -n "$processes" ]]; then
        log "INFO" "Found GPU processes, attempting graceful shutdown..."
        
        # Try graceful shutdown first
        while IFS= read -r pid; do
            pid=$(echo "$pid" | tr -d ' ')
            if [[ -n "$pid" && "$pid" != "pid" ]]; then
                log "INFO" "Sending SIGTERM to process $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done <<< "$processes"
        
        # Wait a bit for graceful shutdown
        sleep 5
        
        # Force kill if still running
        processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
        if [[ -n "$processes" ]]; then
            log "WARN" "Processes still running, forcing shutdown..."
            while IFS= read -r pid; do
                pid=$(echo "$pid" | tr -d ' ')
                if [[ -n "$pid" && "$pid" != "pid" ]]; then
                    log "WARN" "Sending SIGKILL to process $pid"
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            done <<< "$processes"
        fi
    fi
    
    # Clear cache after killing processes
    clear_gpu_cache
}

# Main function
main() {
    local action=${1:-"status"}
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "INFO" "GPU Memory Manager - Action: $action"
    
    case $action in
        "status"|"show")
            show_gpu_status
            ;;
        "clear"|"cleanup")
            clear_gpu_cache "${2:-qwen-api}"
            ;;
        "optimize")
            optimize_gpu_memory "${2:-qwen-api}"
            ;;
        "monitor")
            monitor_gpu_memory "${2:-60}" "${3:-5}"
            ;;
        "emergency")
            emergency_gpu_cleanup
            ;;
        *)
            log "ERROR" "Unknown action: $action"
            show_help
            exit 1
            ;;
    esac
}

# Help function
show_help() {
    cat << EOF
GPU Memory Manager for Qwen2 Docker Environment

Usage: $0 [ACTION] [OPTIONS]

Actions:
    status                          Show GPU status and memory usage (default)
    clear [container_name]          Clear GPU memory cache in container
    optimize [container_name]       Optimize GPU memory settings
    monitor [duration] [interval]   Monitor GPU memory usage
    emergency                       Emergency GPU cleanup (kill processes)
    
Options:
    -v, --verbose                   Enable verbose output
    -h, --help                      Show this help message
    
Environment Variables:
    GPU_MEMORY_WARNING_THRESHOLD    Warning threshold in MB (default: 12288)
    GPU_MEMORY_CRITICAL_THRESHOLD   Critical threshold in MB (default: 14336)
    GPU_MEMORY_CLEANUP_THRESHOLD    Cleanup threshold in MB (default: 15360)
    
Examples:
    $0                              Show GPU status
    $0 clear                        Clear GPU cache in qwen-api container
    $0 optimize qwen-api            Optimize GPU settings for qwen-api
    $0 monitor 120 10               Monitor for 2 minutes, check every 10 seconds
    $0 emergency                    Emergency cleanup of GPU processes

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # Pass remaining arguments to main function
            break
            ;;
    esac
done

# Run main function with remaining arguments
main "$@"