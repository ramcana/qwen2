#!/bin/bash

# DiffSynth Enhanced UI Deployment Script
# This script handles the complete deployment of the DiffSynth Enhanced UI system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"
COMPOSE_FILE="docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check NVIDIA Docker (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        log_warning "NVIDIA Docker runtime not available. GPU acceleration will not work."
    fi
    
    # Check available disk space (minimum 20GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 20971520 ]; then  # 20GB in KB
        log_warning "Less than 20GB disk space available. Consider freeing up space."
    fi
    
    log_success "Prerequisites check completed"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up environment for: $ENVIRONMENT"
    
    cd "$PROJECT_ROOT"
    
    # Set compose file based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        COMPOSE_FILE="docker-compose.prod.yml"
        log_info "Using production configuration"
    else
        COMPOSE_FILE="docker-compose.yml"
        log_info "Using development configuration"
    fi
    
    # Create necessary directories
    log_info "Creating directories..."
    mkdir -p generated_images uploads cache models offload logs
    mkdir -p cache/huggingface cache/torch cache/diffsynth cache/controlnet
    mkdir -p logs/api logs/frontend logs/traefik
    mkdir -p monitoring/prometheus monitoring/grafana/provisioning
    
    # Set permissions
    chmod 755 generated_images uploads cache models offload logs
    chmod -R 755 cache logs
    
    # Create acme.json for SSL certificates (production only)
    if [ "$ENVIRONMENT" = "production" ]; then
        touch acme.json
        chmod 600 acme.json
    fi
    
    log_success "Environment setup completed"
}

# Function to download models
download_models() {
    log_info "Downloading required models..."
    
    # Check if models exist
    if [ ! -d "models/Qwen-Image" ] && [ ! -f "models/Qwen-Image" ]; then
        log_info "Downloading Qwen models..."
        python tools/download_models.py --model qwen-image
    fi
    
    # Download DiffSynth models if needed
    if [ ! -d "models/diffsynth" ]; then
        log_info "Downloading DiffSynth models..."
        mkdir -p models/diffsynth
        # Models will be downloaded on first use
    fi
    
    # Download ControlNet models if needed
    if [ ! -d "models/controlnet" ]; then
        log_info "Downloading ControlNet models..."
        mkdir -p models/controlnet
        # Models will be downloaded on first use
    fi
    
    log_success "Model download completed"
}

# Function to build and start services
start_services() {
    log_info "Building and starting services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest images
    log_info "Pulling latest base images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build services
    log_info "Building services..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Start services
    log_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_success "Services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for API service
    log_info "Waiting for API service..."
    timeout=300  # 5 minutes
    counter=0
    
    while [ $counter -lt $timeout ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API service is ready"
            break
        fi
        
        if [ $counter -eq 0 ]; then
            log_info "API service starting up (this may take a few minutes)..."
        fi
        
        sleep 5
        counter=$((counter + 5))
        
        if [ $((counter % 30)) -eq 0 ]; then
            log_info "Still waiting for API service... ($counter/$timeout seconds)"
        fi
    done
    
    if [ $counter -ge $timeout ]; then
        log_error "API service failed to start within $timeout seconds"
        return 1
    fi
    
    # Wait for frontend service
    log_info "Waiting for frontend service..."
    counter=0
    
    while [ $counter -lt 60 ]; do  # 1 minute for frontend
        if curl -f http://localhost:3000 &> /dev/null || curl -f http://localhost/health &> /dev/null; then
            log_success "Frontend service is ready"
            break
        fi
        
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ $counter -ge 60 ]; then
        log_warning "Frontend service may not be ready yet"
    fi
    
    log_success "Services are ready"
}

# Function to run deployment validation
validate_deployment() {
    log_info "Validating deployment..."
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    # Health check
    if ! curl -f http://localhost:8000/health &> /dev/null; then
        log_error "API health check failed"
        return 1
    fi
    
    # Status check
    if ! curl -f http://localhost:8000/status &> /dev/null; then
        log_error "API status check failed"
        return 1
    fi
    
    # DiffSynth endpoints (if available)
    if curl -f http://localhost:8000/diffsynth/status &> /dev/null; then
        log_success "DiffSynth service is available"
    else
        log_warning "DiffSynth service may not be ready yet"
    fi
    
    # ControlNet endpoints (if available)
    if curl -f http://localhost:8000/controlnet/types &> /dev/null; then
        log_success "ControlNet service is available"
    else
        log_warning "ControlNet service may not be ready yet"
    fi
    
    # Test frontend
    if curl -f http://localhost:3000 &> /dev/null || curl -f http://localhost &> /dev/null; then
        log_success "Frontend is accessible"
    else
        log_warning "Frontend may not be accessible"
    fi
    
    log_success "Deployment validation completed"
}

# Function to show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    
    # Show running containers
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    
    # Show service URLs
    log_info "Service URLs:"
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "  Frontend: https://yourdomain.com"
        echo "  API: https://api.yourdomain.com"
        echo "  Metrics: https://metrics.yourdomain.com"
        echo "  Dashboard: https://dashboard.yourdomain.com"
    else
        echo "  Frontend: http://localhost:3000 or http://frontend.localhost"
        echo "  API: http://localhost:8000 or http://api.localhost"
        echo "  Traefik Dashboard: http://localhost:8080"
    fi
    echo
    
    # Show resource usage
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down
    log_info "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting DiffSynth Enhanced UI deployment..."
    log_info "Environment: $ENVIRONMENT"
    echo
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    # Run deployment steps
    check_prerequisites
    setup_environment
    download_models
    start_services
    wait_for_services
    validate_deployment
    
    # Show final status
    echo
    log_success "Deployment completed successfully!"
    echo
    show_status
    
    # Show next steps
    echo
    log_info "Next Steps:"
    echo "1. Access the web interface using the URLs shown above"
    echo "2. Initialize the models on first use (may take a few minutes)"
    echo "3. Check logs if you encounter any issues: docker-compose -f $COMPOSE_FILE logs"
    echo "4. Monitor resource usage and adjust as needed"
    echo
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "Production Notes:"
        echo "- Update domain names in docker-compose.prod.yml"
        echo "- Configure SSL certificates"
        echo "- Set up monitoring and alerting"
        echo "- Configure backup procedures"
        echo
    fi
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [development|production]"
    echo
    echo "Examples:"
    echo "  $0 development    # Deploy for development"
    echo "  $0 production     # Deploy for production"
    echo
    exit 1
fi

# Run main function
main "$@"