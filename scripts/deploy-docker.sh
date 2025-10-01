#!/bin/bash

# Docker Deployment Script for Qwen2 Image Generation Application
# Usage: ./scripts/deploy-docker.sh [dev|prod|staging] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
BUILD_IMAGES=false
PULL_IMAGES=false
FORCE_RECREATE=false
DETACHED=true
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Docker Deployment Script for Qwen2 Application

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    dev         Development environment (default)
    prod        Production environment
    staging     Staging environment

OPTIONS:
    --build         Build images before deployment
    --pull          Pull latest images before deployment
    --force         Force recreate containers
    --foreground    Run in foreground (don't detach)
    --verbose       Enable verbose output
    --help          Show this help message

EXAMPLES:
    $0 dev --build                    # Deploy development with fresh build
    $0 prod --pull --force           # Deploy production with latest images
    $0 staging --build --foreground  # Deploy staging and watch logs

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        dev|prod|staging)
            ENVIRONMENT="$1"
            shift
            ;;
        --build)
            BUILD_IMAGES=true
            shift
            ;;
        --pull)
            PULL_IMAGES=true
            shift
            ;;
        --force)
            FORCE_RECREATE=true
            shift
            ;;
        --foreground)
            DETACHED=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set compose file based on environment
case $ENVIRONMENT in
    dev)
        COMPOSE_FILE="docker-compose.dev.yml"
        ;;
    prod)
        COMPOSE_FILE="docker-compose.prod.yml"
        ;;
    staging)
        COMPOSE_FILE="docker-compose.yml"
        ;;
    *)
        print_error "Invalid environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Check if compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    print_error "Compose file not found: $COMPOSE_FILE"
    exit 1
fi

print_status "Deploying Qwen2 application in $ENVIRONMENT environment"
print_status "Using compose file: $COMPOSE_FILE"

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p models cache generated_images uploads offload logs/api logs/traefik data/redis data/prometheus data/grafana

# Set proper permissions
chmod 755 models cache generated_images uploads offload
chmod 755 logs logs/api logs/traefik
chmod 755 data data/redis data/prometheus data/grafana

# Pull images if requested
if [[ "$PULL_IMAGES" == "true" ]]; then
    print_status "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" pull
fi

# Build images if requested
if [[ "$BUILD_IMAGES" == "true" ]]; then
    print_status "Building images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
fi

# Prepare docker-compose command
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE up"

if [[ "$FORCE_RECREATE" == "true" ]]; then
    COMPOSE_CMD="$COMPOSE_CMD --force-recreate"
fi

if [[ "$DETACHED" == "true" ]]; then
    COMPOSE_CMD="$COMPOSE_CMD -d"
fi

# Deploy the application
print_status "Starting services..."
eval $COMPOSE_CMD

if [[ "$DETACHED" == "true" ]]; then
    print_success "Services started successfully!"
    print_status "Checking service health..."
    
    # Wait a moment for services to start
    sleep 5
    
    # Show service status
    docker-compose -f "$COMPOSE_FILE" ps
    
    print_status "Application URLs:"
    case $ENVIRONMENT in
        dev)
            echo "  Frontend: http://localhost:3000"
            echo "  API: http://localhost:8000"
            echo "  Traefik Dashboard: http://localhost:8080"
            ;;
        prod|staging)
            echo "  Application: https://your-domain.com"
            echo "  Traefik Dashboard: https://traefik.your-domain.com"
            ;;
    esac
    
    print_status "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
    print_status "To stop services: docker-compose -f $COMPOSE_FILE down"
else
    print_status "Running in foreground mode. Press Ctrl+C to stop."
fi