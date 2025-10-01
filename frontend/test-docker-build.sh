#!/bin/bash

# =============================================================================
# Docker Build Test Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
TEST_TAG="frontend-test-$(date +%s)"
TEST_PORT="3001"
CLEANUP=true

# Cleanup function
cleanup() {
    if [ "$CLEANUP" = true ]; then
        print_status "Cleaning up test resources..."
        docker stop "$TEST_TAG" 2>/dev/null || true
        docker rm "$TEST_TAG" 2>/dev/null || true
        docker rmi "$TEST_TAG" 2>/dev/null || true
        print_success "Cleanup completed"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Function to test Docker build
test_docker_build() {
    print_status "Testing Docker build process..."
    
    # Build the image
    if docker build -f Dockerfile.prod -t "$TEST_TAG" .; then
        print_success "Docker build completed successfully"
    else
        print_error "Docker build failed"
        return 1
    fi
    
    # Check image size
    IMAGE_SIZE=$(docker images "$TEST_TAG" --format "{{.Size}}")
    print_status "Image size: $IMAGE_SIZE"
    
    return 0
}

# Function to test container startup
test_container_startup() {
    print_status "Testing container startup..."
    
    # Start container
    if docker run -d --name "$TEST_TAG" -p "$TEST_PORT:80" "$TEST_TAG"; then
        print_success "Container started successfully"
    else
        print_error "Container startup failed"
        return 1
    fi
    
    # Wait for container to be ready
    print_status "Waiting for container to be ready..."
    sleep 10
    
    # Check if container is running
    if docker ps | grep -q "$TEST_TAG"; then
        print_success "Container is running"
    else
        print_error "Container is not running"
        docker logs "$TEST_TAG"
        return 1
    fi
    
    return 0
}

# Function to test HTTP endpoints
test_http_endpoints() {
    print_status "Testing HTTP endpoints..."
    
    # Test health endpoint
    if curl -f "http://localhost:$TEST_PORT/health" > /dev/null 2>&1; then
        print_success "Health endpoint is accessible"
    else
        print_error "Health endpoint is not accessible"
        return 1
    fi
    
    # Test main page
    if curl -f "http://localhost:$TEST_PORT/" > /dev/null 2>&1; then
        print_success "Main page is accessible"
    else
        print_error "Main page is not accessible"
        return 1
    fi
    
    # Test static assets (if they exist)
    if curl -f "http://localhost:$TEST_PORT/static/css/" > /dev/null 2>&1; then
        print_success "Static assets are accessible"
    else
        print_warning "Static assets may not be accessible (this might be normal)"
    fi
    
    return 0
}

# Function to test container health
test_container_health() {
    print_status "Testing container health..."
    
    # Check container health status
    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$TEST_TAG" 2>/dev/null || echo "unknown")
    
    if [ "$HEALTH_STATUS" = "healthy" ]; then
        print_success "Container health check passed"
    elif [ "$HEALTH_STATUS" = "starting" ]; then
        print_warning "Container is still starting, waiting..."
        sleep 15
        HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$TEST_TAG" 2>/dev/null || echo "unknown")
        if [ "$HEALTH_STATUS" = "healthy" ]; then
            print_success "Container health check passed after waiting"
        else
            print_error "Container health check failed: $HEALTH_STATUS"
            return 1
        fi
    else
        print_error "Container health check failed: $HEALTH_STATUS"
        return 1
    fi
    
    return 0
}

# Function to show container logs
show_container_logs() {
    print_status "Container logs:"
    docker logs "$TEST_TAG" 2>&1 | tail -20
}

# Main test function
main() {
    print_status "Starting Docker build test..."
    
    local exit_code=0
    
    # Check if Docker is available
    if ! command -v docker > /dev/null 2>&1; then
        print_error "Docker is not available"
        exit 1
    fi
    
    # Check if port is available
    if netstat -tuln 2>/dev/null | grep -q ":$TEST_PORT "; then
        print_error "Port $TEST_PORT is already in use"
        exit 1
    fi
    
    # Run tests
    test_docker_build || exit_code=1
    
    if [ $exit_code -eq 0 ]; then
        test_container_startup || exit_code=1
    fi
    
    if [ $exit_code -eq 0 ]; then
        test_container_health || exit_code=1
    fi
    
    if [ $exit_code -eq 0 ]; then
        test_http_endpoints || exit_code=1
    fi
    
    # Show logs regardless of success/failure
    show_container_logs
    
    if [ $exit_code -eq 0 ]; then
        print_success "All Docker build tests passed!"
        print_status "You can access the test frontend at: http://localhost:$TEST_PORT"
        print_status "Press Ctrl+C to stop the test and cleanup"
        
        # Keep container running for manual testing if requested
        if [ "$1" = "--keep-running" ]; then
            CLEANUP=false
            print_status "Container will keep running (cleanup disabled)"
            print_status "Manual cleanup: docker stop $TEST_TAG && docker rm $TEST_TAG && docker rmi $TEST_TAG"
        fi
    else
        print_error "Some Docker build tests failed"
    fi
    
    return $exit_code
}

# Show usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--keep-running] [--help]"
    echo ""
    echo "Options:"
    echo "  --keep-running    Keep the test container running after tests"
    echo "  --help           Show this help message"
    echo ""
    echo "This script tests the Docker build process and container functionality."
    exit 0
fi

# Run main function
main "$@"