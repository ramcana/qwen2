#!/bin/bash

# Deployment script for Qwen-Image API Docker container
# Handles building, running, and managing the containerized API

set -e

# Configuration
COMPOSE_FILE="docker-compose.api.yml"
SERVICE_NAME="qwen-api"
IMAGE_NAME="qwen-image-api"
DEFAULT_TAG="gpu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  start     Start the API service"
    echo "  stop      Stop the API service"
    echo "  restart   Restart the API service"
    echo "  logs      Show service logs"
    echo "  status    Show service status"
    echo "  clean     Clean up containers and images"
    echo "  shell     Open shell in running container"
    echo "  health    Check service health"
    echo ""
    echo "Options:"
    echo "  --cpu     Use CPU-only build/deployment"
    echo "  --gpu     Use GPU build/deployment (default)"
    echo "  --tag     Specify image tag"
    echo "  --follow  Follow logs (for logs command)"
    echo "  --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build --gpu"
    echo "  $0 start"
    echo "  $0 logs --follow"
    echo "  $0 restart"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check for GPU support if needed
    if [[ "$USE_GPU" == "true" ]]; then
        if command -v nvidia-smi &> /dev/null; then
            log_success "NVIDIA GPU detected"
        else
            log_warning "NVIDIA GPU not detected, but GPU mode requested"
        fi
    fi
    
    log_success "Requirements check passed"
}

build_image() {
    log_info "Building Docker image..."
    
    if [[ "$USE_GPU" == "true" ]]; then
        log_info "Building with GPU support"
        docker-compose -f "$COMPOSE_FILE" build --build-arg ENABLE_GPU=true
    else
        log_info "Building with CPU-only support"
        docker-compose -f "$COMPOSE_FILE" build --build-arg ENABLE_GPU=false
    fi
    
    log_success "Image built successfully"
}

start_service() {
    log_info "Starting API service..."
    
    # Create necessary directories
    mkdir -p models cache generated_images uploads offload logs
    
    # Start the service
    docker-compose -f "$COMPOSE_FILE" up -d "$SERVICE_NAME"
    
    log_success "Service started"
    log_info "API will be available at http://localhost:8000"
    log_info "Use '$0 logs --follow' to monitor startup"
}

stop_service() {
    log_info "Stopping API service..."
    docker-compose -f "$COMPOSE_FILE" stop "$SERVICE_NAME"
    log_success "Service stopped"
}

restart_service() {
    log_info "Restarting API service..."
    docker-compose -f "$COMPOSE_FILE" restart "$SERVICE_NAME"
    log_success "Service restarted"
}

show_logs() {
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        log_info "Following logs (Ctrl+C to exit)..."
        docker-compose -f "$COMPOSE_FILE" logs -f "$SERVICE_NAME"
    else
        log_info "Showing recent logs..."
        docker-compose -f "$COMPOSE_FILE" logs --tail=50 "$SERVICE_NAME"
    fi
}

show_status() {
    log_info "Service status:"
    docker-compose -f "$COMPOSE_FILE" ps "$SERVICE_NAME"
    
    echo ""
    log_info "Container stats:"
    if docker-compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME" | xargs docker stats --no-stream 2>/dev/null; then
        :
    else
        log_warning "Service is not running"
    fi
}

clean_up() {
    log_warning "This will remove containers and images. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleaning up..."
        docker-compose -f "$COMPOSE_FILE" down --rmi all --volumes --remove-orphans
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

open_shell() {
    log_info "Opening shell in container..."
    if docker-compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME" | grep -q .; then
        docker-compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" /bin/bash
    else
        log_error "Service is not running"
        exit 1
    fi
}

check_health() {
    log_info "Checking service health..."
    
    if docker-compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME" | grep -q .; then
        # Check container health
        health_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME") 2>/dev/null || echo "unknown")
        
        case "$health_status" in
            "healthy")
                log_success "Service is healthy"
                ;;
            "unhealthy")
                log_error "Service is unhealthy"
                ;;
            "starting")
                log_warning "Service is starting up"
                ;;
            *)
                log_warning "Health status unknown"
                ;;
        esac
        
        # Try to reach the API
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_success "API endpoint is responding"
        else
            log_warning "API endpoint is not responding"
        fi
    else
        log_error "Service is not running"
    fi
}

# Parse arguments
COMMAND=""
USE_GPU="true"
TAG="$DEFAULT_TAG"
FOLLOW_LOGS="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        build|start|stop|restart|logs|status|clean|shell|health)
            COMMAND="$1"
            shift
            ;;
        --cpu)
            USE_GPU="false"
            TAG="cpu"
            shift
            ;;
        --gpu)
            USE_GPU="true"
            TAG="gpu"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --follow)
            FOLLOW_LOGS="true"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command was provided
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_usage
    exit 1
fi

# Check requirements
check_requirements

# Execute command
case "$COMMAND" in
    build)
        build_image
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    clean)
        clean_up
        ;;
    shell)
        open_shell
        ;;
    health)
        check_health
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac