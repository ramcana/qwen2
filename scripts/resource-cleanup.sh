#!/bin/bash
# =============================================================================
# Resource Cleanup Script for Qwen2 Docker Environment
# =============================================================================
# This script provides automatic cleanup of temporary files, cache, and logs
# to manage disk space and optimize container performance.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/cleanup.log"
DRY_RUN=${DRY_RUN:-false}
VERBOSE=${VERBOSE:-false}

# Cleanup thresholds (in MB)
CACHE_SIZE_THRESHOLD=${CACHE_SIZE_THRESHOLD:-10240}  # 10GB
LOG_SIZE_THRESHOLD=${LOG_SIZE_THRESHOLD:-1024}       # 1GB
TEMP_AGE_DAYS=${TEMP_AGE_DAYS:-7}                   # 7 days
GENERATED_IMAGES_AGE_DAYS=${GENERATED_IMAGES_AGE_DAYS:-30}  # 30 days

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

# Check if directory exists and get size
get_directory_size() {
    local dir=$1
    if [[ -d "$dir" ]]; then
        du -sm "$dir" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Clean Docker system resources
cleanup_docker_system() {
    log "INFO" "Starting Docker system cleanup..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would clean Docker system resources"
        docker system df
        return
    fi
    
    # Remove unused containers, networks, images, and build cache
    log "INFO" "Removing unused Docker resources..."
    docker system prune -f --volumes
    
    # Remove dangling images
    log "INFO" "Removing dangling images..."
    docker image prune -f
    
    # Remove unused volumes (be careful with this)
    log "INFO" "Removing unused volumes..."
    docker volume prune -f
    
    log "INFO" "Docker system cleanup completed"
}

# Clean HuggingFace cache
cleanup_huggingface_cache() {
    local cache_dir="${PROJECT_ROOT}/cache/huggingface"
    local cache_size=$(get_directory_size "$cache_dir")
    
    log "INFO" "HuggingFace cache size: ${cache_size}MB"
    
    if [[ $cache_size -gt $CACHE_SIZE_THRESHOLD ]]; then
        log "WARN" "HuggingFace cache exceeds threshold (${CACHE_SIZE_THRESHOLD}MB)"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Would clean HuggingFace cache"
            return
        fi
        
        # Remove old cached models (older than 30 days)
        log "INFO" "Cleaning old HuggingFace cache files..."
        find "$cache_dir" -type f -mtime +30 -delete 2>/dev/null || true
        
        # Clean temporary files
        find "$cache_dir" -name "*.tmp" -delete 2>/dev/null || true
        find "$cache_dir" -name "*.lock" -delete 2>/dev/null || true
        
        local new_size=$(get_directory_size "$cache_dir")
        log "INFO" "HuggingFace cache cleaned: ${cache_size}MB -> ${new_size}MB"
    else
        log "INFO" "HuggingFace cache size is within threshold"
    fi
}

# Clean PyTorch cache
cleanup_torch_cache() {
    local cache_dir="${PROJECT_ROOT}/cache/torch"
    local cache_size=$(get_directory_size "$cache_dir")
    
    log "INFO" "PyTorch cache size: ${cache_size}MB"
    
    if [[ $cache_size -gt $CACHE_SIZE_THRESHOLD ]]; then
        log "WARN" "PyTorch cache exceeds threshold (${CACHE_SIZE_THRESHOLD}MB)"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Would clean PyTorch cache"
            return
        fi
        
        # Remove old cached models
        log "INFO" "Cleaning old PyTorch cache files..."
        find "$cache_dir" -type f -mtime +30 -delete 2>/dev/null || true
        
        local new_size=$(get_directory_size "$cache_dir")
        log "INFO" "PyTorch cache cleaned: ${cache_size}MB -> ${new_size}MB"
    else
        log "INFO" "PyTorch cache size is within threshold"
    fi
}

# Clean temporary files
cleanup_temp_files() {
    log "INFO" "Cleaning temporary files..."
    
    local temp_dirs=(
        "${PROJECT_ROOT}/tmp"
        "${PROJECT_ROOT}/temp"
        "${PROJECT_ROOT}/.tmp"
        "/tmp/qwen*"
    )
    
    for temp_dir in "${temp_dirs[@]}"; do
        if [[ -d "$temp_dir" ]]; then
            local temp_size=$(get_directory_size "$temp_dir")
            log "INFO" "Temporary directory $temp_dir size: ${temp_size}MB"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log "INFO" "[DRY RUN] Would clean temporary files in $temp_dir"
                continue
            fi
            
            # Remove files older than specified days
            find "$temp_dir" -type f -mtime +$TEMP_AGE_DAYS -delete 2>/dev/null || true
            
            # Remove empty directories
            find "$temp_dir" -type d -empty -delete 2>/dev/null || true
        fi
    done
}

# Clean old generated images
cleanup_generated_images() {
    local images_dir="${PROJECT_ROOT}/generated_images"
    local images_size=$(get_directory_size "$images_dir")
    
    log "INFO" "Generated images directory size: ${images_size}MB"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would clean old generated images (older than ${GENERATED_IMAGES_AGE_DAYS} days)"
        return
    fi
    
    # Remove old generated images
    log "INFO" "Cleaning old generated images (older than ${GENERATED_IMAGES_AGE_DAYS} days)..."
    find "$images_dir" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | \
        xargs -I {} find {} -mtime +$GENERATED_IMAGES_AGE_DAYS -delete 2>/dev/null || true
    
    # Remove associated metadata files
    find "$images_dir" -name "*_metadata.json" -mtime +$GENERATED_IMAGES_AGE_DAYS -delete 2>/dev/null || true
    
    local new_size=$(get_directory_size "$images_dir")
    log "INFO" "Generated images cleaned: ${images_size}MB -> ${new_size}MB"
}

# Clean application logs
cleanup_logs() {
    local logs_dir="${PROJECT_ROOT}/logs"
    local logs_size=$(get_directory_size "$logs_dir")
    
    log "INFO" "Logs directory size: ${logs_size}MB"
    
    if [[ $logs_size -gt $LOG_SIZE_THRESHOLD ]]; then
        log "WARN" "Logs exceed threshold (${LOG_SIZE_THRESHOLD}MB)"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Would clean old log files"
            return
        fi
        
        # Compress old log files
        log "INFO" "Compressing old log files..."
        find "$logs_dir" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
        
        # Remove very old compressed logs
        find "$logs_dir" -name "*.log.gz" -mtime +30 -delete 2>/dev/null || true
        
        local new_size=$(get_directory_size "$logs_dir")
        log "INFO" "Logs cleaned: ${logs_size}MB -> ${new_size}MB"
    else
        log "INFO" "Logs size is within threshold"
    fi
}

# Clean Docker logs
cleanup_docker_logs() {
    log "INFO" "Cleaning Docker container logs..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would clean Docker container logs"
        return
    fi
    
    # Get all container IDs for the project
    local containers=$(docker ps -a --filter "label=com.qwen.project=qwen2-image-generation" --format "{{.ID}}" 2>/dev/null || true)
    
    if [[ -n "$containers" ]]; then
        for container in $containers; do
            local log_file=$(docker inspect --format='{{.LogPath}}' "$container" 2>/dev/null || true)
            if [[ -n "$log_file" && -f "$log_file" ]]; then
                local log_size=$(du -sm "$log_file" 2>/dev/null | cut -f1 || echo "0")
                if [[ $log_size -gt 100 ]]; then  # If log file > 100MB
                    log "INFO" "Truncating large log file for container $container (${log_size}MB)"
                    truncate -s 50M "$log_file" 2>/dev/null || true
                fi
            fi
        done
    fi
}

# GPU memory cleanup
cleanup_gpu_memory() {
    log "INFO" "Cleaning GPU memory..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log "INFO" "NVIDIA GPU not available, skipping GPU cleanup"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would clean GPU memory"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
        return
    fi
    
    # Clear GPU memory cache (this requires the containers to be running)
    local api_container=$(docker ps --filter "name=qwen-api" --format "{{.ID}}" 2>/dev/null || true)
    if [[ -n "$api_container" ]]; then
        log "INFO" "Clearing GPU memory cache in API container..."
        docker exec "$api_container" python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
else:
    print('CUDA not available')
" 2>/dev/null || log "WARN" "Failed to clear GPU cache"
    fi
}

# Main cleanup function
main() {
    log "INFO" "Starting resource cleanup process..."
    log "INFO" "Project root: $PROJECT_ROOT"
    log "INFO" "Dry run: $DRY_RUN"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run cleanup functions
    cleanup_docker_system
    cleanup_huggingface_cache
    cleanup_torch_cache
    cleanup_temp_files
    cleanup_generated_images
    cleanup_logs
    cleanup_docker_logs
    cleanup_gpu_memory
    
    log "INFO" "Resource cleanup completed successfully"
    
    # Display disk usage summary
    log "INFO" "Current disk usage:"
    df -h "$PROJECT_ROOT" | tail -1
}

# Help function
show_help() {
    cat << EOF
Resource Cleanup Script for Qwen2 Docker Environment

Usage: $0 [OPTIONS]

Options:
    -d, --dry-run                   Show what would be cleaned without making changes
    -v, --verbose                   Enable verbose output
    -h, --help                      Show this help message
    
Environment Variables:
    CACHE_SIZE_THRESHOLD           Cache size threshold in MB (default: 10240)
    LOG_SIZE_THRESHOLD             Log size threshold in MB (default: 1024)
    TEMP_AGE_DAYS                  Age threshold for temp files in days (default: 7)
    GENERATED_IMAGES_AGE_DAYS      Age threshold for generated images in days (default: 30)
    
Examples:
    $0                             Run cleanup with default settings
    $0 --dry-run                   Show what would be cleaned
    $0 --verbose                   Run with verbose output
    CACHE_SIZE_THRESHOLD=5120 $0   Run with custom cache threshold (5GB)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main "$@"