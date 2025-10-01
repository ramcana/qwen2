#!/bin/bash
# =============================================================================
# Enhanced Deployment Script with Resource Management for Qwen2
# =============================================================================
# This script deploys the Qwen2 application with comprehensive resource
# management, monitoring, and optimization features.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/deployment.log"

# Deployment options
ENVIRONMENT=${ENVIRONMENT:-"production"}
ENABLE_MONITORING=${ENABLE_MONITORING:-true}
ENABLE_CLEANUP=${ENABLE_CLEANUP:-true}
ENABLE_RESOURCE_LIMITS=${ENABLE_RESOURCE_LIMITS:-true}
DRY_RUN=${DRY_RUN:-false}

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
    
    case $level in
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log "ERROR" "Docker Compose is not installed"
        exit 1
    fi
    
    # Check available disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=20971520  # 20GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        log "ERROR" "Insufficient disk space. Required: 20GB, Available: $((available_space / 1024 / 1024))GB"
        exit 1
    fi
    
    # Check GPU availability (optional)
    if command -v nvidia-smi &> /dev/null; then
        log "INFO" "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        log "WARN" "No NVIDIA GPU detected - running in CPU mode"
    fi
    
    log "INFO" "Prerequisites check completed"
}

# Setup directories and permissions
setup_directories() {
    log "INFO" "Setting up directories and permissions..."
    
    local directories=(
        "logs"
        "logs/api"
        "logs/traefik"
        "cache"
        "cache/huggingface"
        "cache/torch"
        "cache/diffsynth"
        "cache/controlnet"
        "models"
        "models/diffsynth"
        "models/controlnet"
        "generated_images"
        "uploads"
        "offload"
        "data"
        "data/redis"
        "data/prometheus"
        "data/grafana"
    )
    
    for dir in "${directories[@]}"; do
        local full_path="${PROJECT_ROOT}/${dir}"
        if [[ ! -d "$full_path" ]]; then
            log "INFO" "Creating directory: $dir"
            mkdir -p "$full_path"
        fi
        
        # Set appropriate permissions
        chmod 755 "$full_path"
    done
    
    # Make scripts executable
    chmod +x "${PROJECT_ROOT}/scripts"/*.sh 2>/dev/null || true
    
    log "INFO" "Directory setup completed"
}

# Configure resource limits based on system
configure_resource_limits() {
    log "INFO" "Configuring resource limits based on system capabilities..."
    
    # Get system information
    local total_memory=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    local cpu_cores=$(nproc)
    
    log "INFO" "System resources - Memory: ${total_memory}MB, CPU cores: ${cpu_cores}"
    
    # Create environment-specific configuration
    local env_file="${PROJECT_ROOT}/.env.resource-limits"
    
    cat > "$env_file" << EOF
# Auto-generated resource limits based on system capabilities
# Generated: $(date)

# System information
TOTAL_SYSTEM_MEMORY=${total_memory}
TOTAL_CPU_CORES=${cpu_cores}

# API server limits (60% of system memory, 75% of CPU cores)
API_MEMORY_LIMIT=$((total_memory * 60 / 100))m
API_CPU_LIMIT=$(echo "scale=1; $cpu_cores * 0.75" | bc)

# Frontend limits (10% of system memory, 25% of CPU cores)
FRONTEND_MEMORY_LIMIT=$((total_memory * 10 / 100))m
FRONTEND_CPU_LIMIT=$(echo "scale=1; $cpu_cores * 0.25" | bc)

# Traefik limits (5% of system memory, 10% of CPU cores)
TRAEFIK_MEMORY_LIMIT=$((total_memory * 5 / 100))m
TRAEFIK_CPU_LIMIT=$(echo "scale=1; $cpu_cores * 0.1" | bc)

# Cache and cleanup thresholds
CACHE_SIZE_THRESHOLD=$((total_memory * 2))  # 2x system memory in MB
DISK_WARNING_THRESHOLD=80
DISK_CRITICAL_THRESHOLD=90
EOF
    
    log "INFO" "Resource limits configured in $env_file"
}

# Setup monitoring and alerting
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        log "INFO" "Monitoring disabled, skipping setup"
        return
    fi
    
    log "INFO" "Setting up monitoring and alerting..."
    
    # Create monitoring configuration
    local monitoring_config="${PROJECT_ROOT}/config/monitoring.yml"
    mkdir -p "$(dirname "$monitoring_config")"
    
    cat > "$monitoring_config" << EOF
# Monitoring configuration for Qwen2
monitoring:
  enabled: true
  interval: 30s
  
  disk_monitoring:
    enabled: true
    warning_threshold: 80
    critical_threshold: 90
    
  gpu_monitoring:
    enabled: true
    memory_threshold: 85
    
  log_monitoring:
    enabled: true
    max_size: 100MB
    rotation: daily
    
  alerts:
    webhook_url: "${ALERT_WEBHOOK_URL:-}"
    email: "${ALERT_EMAIL:-}"
EOF
    
    # Start monitoring services
    if [[ "$DRY_RUN" != "true" ]]; then
        log "INFO" "Starting monitoring services..."
        
        # Start disk space monitor in background
        nohup "${PROJECT_ROOT}/scripts/disk-space-monitor.sh" monitor 300 > "${PROJECT_ROOT}/logs/disk-monitor.log" 2>&1 &
        echo $! > "${PROJECT_ROOT}/logs/disk-monitor.pid"
        
        log "INFO" "Disk space monitor started (PID: $(cat "${PROJECT_ROOT}/logs/disk-monitor.pid"))"
    fi
    
    log "INFO" "Monitoring setup completed"
}

# Setup automatic cleanup
setup_cleanup() {
    if [[ "$ENABLE_CLEANUP" != "true" ]]; then
        log "INFO" "Automatic cleanup disabled, skipping setup"
        return
    fi
    
    log "INFO" "Setting up automatic cleanup..."
    
    # Create cleanup cron job
    local cron_job="0 */6 * * * ${PROJECT_ROOT}/scripts/resource-cleanup.sh --verbose >> ${PROJECT_ROOT}/logs/cleanup-cron.log 2>&1"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Add to crontab if not already present
        (crontab -l 2>/dev/null | grep -v "resource-cleanup.sh"; echo "$cron_job") | crontab -
        log "INFO" "Cleanup cron job added (runs every 6 hours)"
    else
        log "INFO" "[DRY RUN] Would add cron job: $cron_job"
    fi
    
    log "INFO" "Automatic cleanup setup completed"
}

# Deploy with Docker Compose
deploy_application() {
    log "INFO" "Deploying application with resource management..."
    
    local compose_files=("-f" "docker-compose.yml")
    
    # Add environment-specific compose file
    case $ENVIRONMENT in
        "development"|"dev")
            compose_files+=("-f" "docker-compose.dev.yml")
            ;;
        "production"|"prod")
            compose_files+=("-f" "docker-compose.prod.yml")
            ;;
    esac
    
    # Add resource-optimized configuration
    if [[ "$ENABLE_RESOURCE_LIMITS" == "true" ]]; then
        compose_files+=("-f" "docker-compose.resource-optimized.yml")
    fi
    
    log "INFO" "Using compose files: ${compose_files[*]}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would run: docker-compose ${compose_files[*]} up -d"
        return
    fi
    
    # Pull latest images
    log "INFO" "Pulling latest images..."
    docker-compose "${compose_files[@]}" pull
    
    # Build images
    log "INFO" "Building images..."
    docker-compose "${compose_files[@]}" build
    
    # Start services
    log "INFO" "Starting services..."
    docker-compose "${compose_files[@]}" up -d
    
    # Wait for services to be healthy
    log "INFO" "Waiting for services to be healthy..."
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        local unhealthy_services=$(docker-compose "${compose_files[@]}" ps --services --filter "health=unhealthy" 2>/dev/null || true)
        
        if [[ -z "$unhealthy_services" ]]; then
            log "INFO" "All services are healthy"
            break
        fi
        
        log "INFO" "Waiting for services to be healthy... (${wait_time}s/${max_wait}s)"
        sleep 10
        wait_time=$((wait_time + 10))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        log "WARN" "Some services may not be healthy after ${max_wait}s"
    fi
    
    log "INFO" "Application deployment completed"
}

# Verify deployment
verify_deployment() {
    log "INFO" "Verifying deployment..."
    
    # Check service status
    log "INFO" "Service status:"
    docker-compose -f docker-compose.yml ps
    
    # Check resource usage
    log "INFO" "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    
    # Check disk space
    log "INFO" "Disk space:"
    "${PROJECT_ROOT}/scripts/disk-space-monitor.sh" check
    
    # Check GPU status if available
    if command -v nvidia-smi &> /dev/null; then
        log "INFO" "GPU status:"
        "${PROJECT_ROOT}/scripts/gpu-memory-manager.sh" status
    fi
    
    # Test API endpoint
    log "INFO" "Testing API endpoint..."
    local api_url="http://localhost:8000/health"
    if curl -f "$api_url" &> /dev/null; then
        log "INFO" "API endpoint is responding"
    else
        log "WARN" "API endpoint is not responding"
    fi
    
    log "INFO" "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log "INFO" "Cleaning up deployment resources..."
    
    # Stop monitoring processes
    if [[ -f "${PROJECT_ROOT}/logs/disk-monitor.pid" ]]; then
        local pid=$(cat "${PROJECT_ROOT}/logs/disk-monitor.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Stopping disk monitor (PID: $pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "${PROJECT_ROOT}/logs/disk-monitor.pid"
    fi
    
    log "INFO" "Cleanup completed"
}

# Main deployment function
main() {
    local action=${1:-"deploy"}
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "INFO" "Starting enhanced deployment with resource management"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Monitoring: $ENABLE_MONITORING"
    log "INFO" "Cleanup: $ENABLE_CLEANUP"
    log "INFO" "Resource Limits: $ENABLE_RESOURCE_LIMITS"
    log "INFO" "Dry Run: $DRY_RUN"
    
    case $action in
        "deploy")
            check_prerequisites
            setup_directories
            configure_resource_limits
            setup_monitoring
            setup_cleanup
            deploy_application
            verify_deployment
            ;;
        "verify")
            verify_deployment
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            log "ERROR" "Unknown action: $action"
            show_help
            exit 1
            ;;
    esac
    
    log "INFO" "Enhanced deployment completed successfully"
}

# Help function
show_help() {
    cat << EOF
Enhanced Deployment Script with Resource Management for Qwen2

Usage: $0 [ACTION] [OPTIONS]

Actions:
    deploy                          Full deployment with resource management (default)
    verify                          Verify existing deployment
    cleanup                         Cleanup deployment resources
    
Environment Variables:
    ENVIRONMENT                     Deployment environment (development/production)
    ENABLE_MONITORING              Enable monitoring services (true/false)
    ENABLE_CLEANUP                 Enable automatic cleanup (true/false)
    ENABLE_RESOURCE_LIMITS         Enable resource limits (true/false)
    DRY_RUN                        Show what would be done (true/false)
    ALERT_WEBHOOK_URL              Webhook URL for alerts (optional)
    ALERT_EMAIL                    Email for alerts (optional)
    
Examples:
    $0                              Deploy with default settings
    $0 deploy                       Full deployment
    $0 verify                       Verify deployment
    ENVIRONMENT=development $0      Deploy in development mode
    DRY_RUN=true $0                 Show what would be deployed

EOF
}

# Set up signal handlers
trap cleanup EXIT
trap 'log "ERROR" "Script interrupted"; exit 1' INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Run main function
main "$@"