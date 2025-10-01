#!/bin/bash

# =============================================================================
# Docker Build Script for Qwen2 Frontend
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
BUILD_ARGS=""
TAG="qwen-frontend"
DOCKERFILE="Dockerfile"
PUSH=false
CACHE_FROM=""

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
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENVIRONMENT     Build environment (production|development) [default: production]"
    echo "  -t, --tag TAG            Docker image tag [default: qwen-frontend]"
    echo "  -f, --file DOCKERFILE    Dockerfile to use [default: Dockerfile]"
    echo "  -p, --push               Push image to registry after build"
    echo "  --cache-from IMAGE       Use image as cache source"
    echo "  --no-cache               Build without using cache"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e production -t qwen-frontend:latest"
    echo "  $0 -e development -f Dockerfile.dev"
    echo "  $0 --no-cache -p"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --cache-from)
            CACHE_FROM="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        -h|--help)
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

# Validate environment
if [[ "$ENVIRONMENT" != "production" && "$ENVIRONMENT" != "development" ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be 'production' or 'development'"
    exit 1
fi

# Set environment-specific build arguments
case $ENVIRONMENT in
    production)
        BUILD_ARGS="$BUILD_ARGS --build-arg NODE_ENV=production"
        BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_OPTIMIZATION=true"
        BUILD_ARGS="$BUILD_ARGS --build-arg GENERATE_SOURCEMAP=false"
        BUILD_ARGS="$BUILD_ARGS --build-arg REACT_APP_API_URL=/api"
        BUILD_ARGS="$BUILD_ARGS --build-arg REACT_APP_WS_URL=/ws"
        ;;
    development)
        BUILD_ARGS="$BUILD_ARGS --build-arg NODE_ENV=development"
        BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_OPTIMIZATION=false"
        BUILD_ARGS="$BUILD_ARGS --build-arg GENERATE_SOURCEMAP=true"
        BUILD_ARGS="$BUILD_ARGS --build-arg REACT_APP_API_URL=http://localhost:8000/api"
        BUILD_ARGS="$BUILD_ARGS --build-arg REACT_APP_WS_URL=ws://localhost:8000/ws"
        DOCKERFILE="Dockerfile.dev"
        ;;
esac

# Add cache from if specified
if [[ -n "$CACHE_FROM" ]]; then
    BUILD_ARGS="$BUILD_ARGS --cache-from $CACHE_FROM"
fi

# Build command
BUILD_CMD="docker build $BUILD_ARGS -f $DOCKERFILE -t $TAG ."

print_status "Building Docker image for $ENVIRONMENT environment..."
print_status "Tag: $TAG"
print_status "Dockerfile: $DOCKERFILE"
print_status "Build command: $BUILD_CMD"

# Pre-build validation
print_status "Running pre-build validation..."
if [ -f "./validate-build.sh" ]; then
    if ./validate-build.sh; then
        print_success "Pre-build validation passed"
    else
        print_warning "Pre-build validation failed, continuing with build..."
    fi
else
    print_warning "validate-build.sh not found, skipping pre-build validation"
fi

# Execute build with better error handling
print_status "Executing Docker build..."
if eval $BUILD_CMD; then
    print_success "Docker image built successfully: $TAG"
    
    # Post-build validation
    print_status "Running post-build validation..."
    if docker run --rm --name test-container-$$ -d -p 0:80 "$TAG" > /dev/null; then
        CONTAINER_ID=$(docker ps -q --filter "name=test-container-$$")
        sleep 5
        
        if docker exec "$CONTAINER_ID" test -f /usr/share/nginx/html/index.html; then
            print_success "Container validation passed"
        else
            print_warning "Container validation failed - index.html not found"
        fi
        
        docker stop "$CONTAINER_ID" > /dev/null
    else
        print_warning "Could not start test container for validation"
    fi
else
    print_error "Docker build failed"
    print_error "Common issues to check:"
    print_error "  1. TypeScript dependency resolution"
    print_error "  2. Missing source files (src/index.tsx, public/index.html)"
    print_error "  3. Invalid nginx configuration"
    print_error "  4. Build process memory issues"
    print_error ""
    print_error "Run './validate-build.sh --full' for detailed diagnostics"
    exit 1
fi

# Push if requested
if [[ "$PUSH" == true ]]; then
    print_status "Pushing image to registry..."
    if docker push "$TAG"; then
        print_success "Image pushed successfully: $TAG"
    else
        print_error "Failed to push image"
        exit 1
    fi
fi

# Show image info
print_status "Image information:"
docker images "$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

print_success "Build process completed successfully!"