#!/bin/bash
# =============================================================================
# Optimized Docker Build Script for Frontend with Enhanced Caching
# =============================================================================

set -e

# Configuration
IMAGE_NAME="qwen-frontend"
TAG="${1:-latest}"
BUILD_CONTEXT="."
DOCKERFILE="Dockerfile.prod"

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

# Build arguments with environment variable handling
BUILD_ARGS=(
    "--build-arg" "NODE_ENV=production"
    "--build-arg" "REACT_APP_API_URL=${REACT_APP_API_URL:-/api}"
    "--build-arg" "REACT_APP_WS_URL=${REACT_APP_WS_URL:-/ws}"
    "--build-arg" "REACT_APP_BACKEND_HOST=${REACT_APP_BACKEND_HOST:-qwen-api}"
    "--build-arg" "REACT_APP_BACKEND_PORT=${REACT_APP_BACKEND_PORT:-8000}"
    "--build-arg" "REACT_APP_VERSION=${REACT_APP_VERSION:-2.0.0}"
    "--build-arg" "REACT_APP_BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    "--build-arg" "GENERATE_SOURCEMAP=${GENERATE_SOURCEMAP:-false}"
    "--build-arg" "BUILD_OPTIMIZATION=${BUILD_OPTIMIZATION:-true}"
    "--build-arg" "INLINE_RUNTIME_CHUNK=${INLINE_RUNTIME_CHUNK:-false}"
)

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if BuildKit is enabled for advanced caching
if ! docker buildx version &> /dev/null; then
    log_warning "Docker BuildKit not available, using standard build"
    BUILDX_ENABLED=false
else
    log_info "Docker BuildKit detected, enabling advanced caching"
    BUILDX_ENABLED=true
fi

log_info "Starting optimized Docker build for ${IMAGE_NAME}:${TAG}"
log_info "Build context: ${BUILD_CONTEXT}"
log_info "Dockerfile: ${DOCKERFILE}"

# Display build configuration
log_info "Build configuration:"
echo "  - NODE_ENV: production"
echo "  - API_URL: ${REACT_APP_API_URL:-/api}"
echo "  - WS_URL: ${REACT_APP_WS_URL:-/ws}"
echo "  - Backend Host: ${REACT_APP_BACKEND_HOST:-qwen-api}"
echo "  - Backend Port: ${REACT_APP_BACKEND_PORT:-8000}"
echo "  - Version: ${REACT_APP_VERSION:-2.0.0}"
echo "  - Source Maps: ${GENERATE_SOURCEMAP:-false}"
echo "  - Build Optimization: ${BUILD_OPTIMIZATION:-true}"

# Build with or without BuildKit
if [ "$BUILDX_ENABLED" = true ]; then
    log_info "Building with Docker BuildKit and advanced caching..."
    
    # Create builder instance if it doesn't exist
    if ! docker buildx inspect qwen-builder &> /dev/null; then
        log_info "Creating BuildKit builder instance..."
        docker buildx create --name qwen-builder --use
    else
        docker buildx use qwen-builder
    fi
    
    # Build with BuildKit and cache mounts
    docker buildx build \
        --platform linux/amd64 \
        --cache-from type=local,src=/tmp/.buildx-cache \
        --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max \
        "${BUILD_ARGS[@]}" \
        -t "${IMAGE_NAME}:${TAG}" \
        -f "${DOCKERFILE}" \
        --load \
        "${BUILD_CONTEXT}"
    
    # Move cache to avoid cache bloat
    if [ -d "/tmp/.buildx-cache-new" ]; then
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
    fi
else
    log_info "Building with standard Docker build..."
    
    # Standard Docker build
    docker build \
        "${BUILD_ARGS[@]}" \
        -t "${IMAGE_NAME}:${TAG}" \
        -f "${DOCKERFILE}" \
        "${BUILD_CONTEXT}"
fi

# Verify the build
log_info "Verifying build..."
if docker image inspect "${IMAGE_NAME}:${TAG}" &> /dev/null; then
    log_success "Build completed successfully!"
    
    # Display image information
    IMAGE_SIZE=$(docker image inspect "${IMAGE_NAME}:${TAG}" --format='{{.Size}}' | numfmt --to=iec)
    log_info "Image size: ${IMAGE_SIZE}"
    
    # Display layers information
    log_info "Image layers:"
    docker history "${IMAGE_NAME}:${TAG}" --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
    
else
    log_error "Build verification failed!"
    exit 1
fi

# Optional: Test the container
if [ "${TEST_CONTAINER:-false}" = "true" ]; then
    log_info "Testing container startup..."
    
    CONTAINER_ID=$(docker run -d -p 8080:80 "${IMAGE_NAME}:${TAG}")
    
    # Wait for container to start
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "Container health check passed!"
    else
        log_error "Container health check failed!"
        docker logs "$CONTAINER_ID"
        docker stop "$CONTAINER_ID" &> /dev/null
        docker rm "$CONTAINER_ID" &> /dev/null
        exit 1
    fi
    
    # Cleanup test container
    docker stop "$CONTAINER_ID" &> /dev/null
    docker rm "$CONTAINER_ID" &> /dev/null
    log_info "Test container cleaned up"
fi

log_success "Optimized Docker build completed successfully!"
log_info "To run the container: docker run -p 80:80 ${IMAGE_NAME}:${TAG}"