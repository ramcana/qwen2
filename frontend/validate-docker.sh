#!/bin/bash

# =============================================================================
# Docker Validation Script for Qwen2 Frontend
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
IMAGE_NAME="qwen-frontend-test"
CONTAINER_NAME="qwen-frontend-validation"
TEST_PORT="8080"

print_status "Starting Docker validation for Qwen2 Frontend..."

# Clean up any existing test containers
print_status "Cleaning up existing test containers..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Run the container
print_status "Starting container for validation..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --port "$TEST_PORT:80" \
    "$IMAGE_NAME"

# Wait for container to start
print_status "Waiting for container to start..."
sleep 5

# Test health check endpoint
print_status "Testing health check endpoint..."
if curl -f "http://localhost:$TEST_PORT/health" > /dev/null 2>&1; then
    print_success "Health check endpoint is working"
else
    print_error "Health check endpoint failed"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

# Test main application
print_status "Testing main application..."
if curl -f "http://localhost:$TEST_PORT/" > /dev/null 2>&1; then
    print_success "Main application is accessible"
else
    print_error "Main application is not accessible"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

# Check security headers
print_status "Testing security headers..."
HEADERS=$(curl -I "http://localhost:$TEST_PORT/" 2>/dev/null)

if echo "$HEADERS" | grep -q "X-Frame-Options"; then
    print_success "Security headers are present"
else
    print_error "Security headers are missing"
fi

# Check gzip compression
print_status "Testing gzip compression..."
if curl -H "Accept-Encoding: gzip" -I "http://localhost:$TEST_PORT/" 2>/dev/null | grep -q "Content-Encoding: gzip"; then
    print_success "Gzip compression is working"
else
    print_error "Gzip compression is not working"
fi

# Clean up
print_status "Cleaning up test container..."
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"

print_success "Docker validation completed successfully!"