#!/bin/bash
# =============================================================================
# Build Validation Script for Optimized Frontend Docker Build
# =============================================================================

set -e

# Configuration
IMAGE_NAME="${1:-qwen-frontend:latest}"
CONTAINER_NAME="qwen-frontend-test-$(date +%s)"
TEST_PORT="8080"

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

# Cleanup function
cleanup() {
    if [ -n "$CONTAINER_ID" ]; then
        log_info "Cleaning up test container..."
        docker stop "$CONTAINER_ID" &> /dev/null || true
        docker rm "$CONTAINER_ID" &> /dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT

log_info "Starting validation for Docker image: $IMAGE_NAME"

# 1. Verify image exists
log_info "1. Verifying image exists..."
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    log_error "Image $IMAGE_NAME not found!"
    exit 1
fi
log_success "Image exists"

# 2. Check image size and layers
log_info "2. Analyzing image characteristics..."
IMAGE_SIZE=$(docker image inspect "$IMAGE_NAME" --format='{{.Size}}' | numfmt --to=iec)
LAYER_COUNT=$(docker history "$IMAGE_NAME" --quiet | wc -l)
log_info "Image size: $IMAGE_SIZE"
log_info "Layer count: $LAYER_COUNT"

# Check if image size is reasonable (should be under 200MB for optimized build)
IMAGE_SIZE_BYTES=$(docker image inspect "$IMAGE_NAME" --format='{{.Size}}')
if [ "$IMAGE_SIZE_BYTES" -gt 209715200 ]; then  # 200MB in bytes
    log_warning "Image size is larger than expected (>200MB). Consider further optimization."
else
    log_success "Image size is optimized"
fi

# 3. Start container for testing
log_info "3. Starting test container..."
CONTAINER_ID=$(docker run -d -p "$TEST_PORT:80" --name "$CONTAINER_NAME" "$IMAGE_NAME")
log_info "Container started with ID: ${CONTAINER_ID:0:12}"

# 4. Wait for container to be ready
log_info "4. Waiting for container to be ready..."
RETRY_COUNT=0
MAX_RETRIES=30

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec "$CONTAINER_ID" sh -c "curl -f http://localhost/health" &> /dev/null; then
        log_success "Container is ready"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        log_error "Container failed to become ready within timeout"
        docker logs "$CONTAINER_ID"
        exit 1
    fi
    
    sleep 2
done

# 5. Test health endpoints
log_info "5. Testing health endpoints..."

# Test /health endpoint
if curl -f "http://localhost:$TEST_PORT/health" &> /dev/null; then
    log_success "Health endpoint (/health) is accessible"
else
    log_error "Health endpoint (/health) is not accessible"
    exit 1
fi

# Test root endpoint
if curl -f "http://localhost:$TEST_PORT/" &> /dev/null; then
    log_success "Root endpoint (/) is accessible"
else
    log_error "Root endpoint (/) is not accessible"
    exit 1
fi

# 6. Test static asset serving
log_info "6. Testing static asset serving..."

# Check if static directory exists in container
if docker exec "$CONTAINER_ID" test -d "/usr/share/nginx/html/static"; then
    log_success "Static assets directory exists"
    
    # Test CSS files
    if docker exec "$CONTAINER_ID" find "/usr/share/nginx/html/static" -name "*.css" | head -1 | grep -q .; then
        log_success "CSS files found"
        
        # Test if compressed versions exist
        if docker exec "$CONTAINER_ID" find "/usr/share/nginx/html/static" -name "*.css.gz" | head -1 | grep -q .; then
            log_success "Gzip compressed CSS files found"
        else
            log_warning "No gzip compressed CSS files found"
        fi
    else
        log_warning "No CSS files found in static directory"
    fi
    
    # Test JS files
    if docker exec "$CONTAINER_ID" find "/usr/share/nginx/html/static" -name "*.js" | head -1 | grep -q .; then
        log_success "JavaScript files found"
        
        # Test if compressed versions exist
        if docker exec "$CONTAINER_ID" find "/usr/share/nginx/html/static" -name "*.js.gz" | head -1 | grep -q .; then
            log_success "Gzip compressed JavaScript files found"
        else
            log_warning "No gzip compressed JavaScript files found"
        fi
    else
        log_warning "No JavaScript files found in static directory"
    fi
else
    log_warning "Static assets directory not found"
fi

# 7. Test nginx configuration
log_info "7. Testing nginx configuration..."

# Test nginx config syntax
if docker exec "$CONTAINER_ID" nginx -t &> /dev/null; then
    log_success "Nginx configuration is valid"
else
    log_error "Nginx configuration is invalid"
    docker exec "$CONTAINER_ID" nginx -t
    exit 1
fi

# 8. Test security headers
log_info "8. Testing security headers..."

RESPONSE_HEADERS=$(curl -I "http://localhost:$TEST_PORT/" 2>/dev/null)

# Check for important security headers
if echo "$RESPONSE_HEADERS" | grep -i "x-frame-options" &> /dev/null; then
    log_success "X-Frame-Options header present"
else
    log_warning "X-Frame-Options header missing"
fi

if echo "$RESPONSE_HEADERS" | grep -i "x-content-type-options" &> /dev/null; then
    log_success "X-Content-Type-Options header present"
else
    log_warning "X-Content-Type-Options header missing"
fi

if echo "$RESPONSE_HEADERS" | grep -i "content-security-policy" &> /dev/null; then
    log_success "Content-Security-Policy header present"
else
    log_warning "Content-Security-Policy header missing"
fi

# 9. Test caching headers
log_info "9. Testing caching headers..."

# Test HTML caching (should be no-cache)
HTML_CACHE=$(curl -I "http://localhost:$TEST_PORT/" 2>/dev/null | grep -i "cache-control" || echo "")
if echo "$HTML_CACHE" | grep -i "no-cache" &> /dev/null; then
    log_success "HTML files have correct no-cache headers"
else
    log_warning "HTML files may not have correct caching headers"
fi

# 10. Test container resource usage
log_info "10. Testing container resource usage..."

# Get container stats
CONTAINER_STATS=$(docker stats "$CONTAINER_ID" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}")
log_info "Container resource usage:"
echo "$CONTAINER_STATS"

# 11. Test build info endpoint (if available)
log_info "11. Testing build information..."

if curl -f "http://localhost:$TEST_PORT/build-info.json" &> /dev/null; then
    BUILD_INFO=$(curl -s "http://localhost:$TEST_PORT/build-info.json")
    log_success "Build information available:"
    echo "$BUILD_INFO" | jq . 2>/dev/null || echo "$BUILD_INFO"
else
    log_info "Build information endpoint not available (optional)"
fi

# 12. Test container logs for errors
log_info "12. Checking container logs for errors..."

CONTAINER_LOGS=$(docker logs "$CONTAINER_ID" 2>&1)
if echo "$CONTAINER_LOGS" | grep -i "error" &> /dev/null; then
    log_warning "Errors found in container logs:"
    echo "$CONTAINER_LOGS" | grep -i "error"
else
    log_success "No errors found in container logs"
fi

# 13. Performance test (basic)
log_info "13. Running basic performance test..."

# Test response time
RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" "http://localhost:$TEST_PORT/")
log_info "Response time: ${RESPONSE_TIME}s"

if (( $(echo "$RESPONSE_TIME < 1.0" | bc -l) )); then
    log_success "Response time is good (<1s)"
else
    log_warning "Response time is slow (>1s)"
fi

# Final summary
log_success "==================================="
log_success "Build validation completed successfully!"
log_success "==================================="
log_info "Summary:"
log_info "- Image: $IMAGE_NAME"
log_info "- Size: $IMAGE_SIZE"
log_info "- Layers: $LAYER_COUNT"
log_info "- Response time: ${RESPONSE_TIME}s"
log_info "- Container ID: ${CONTAINER_ID:0:12}"

log_info "To access the test container: http://localhost:$TEST_PORT"
log_info "To view logs: docker logs $CONTAINER_NAME"
log_info "To stop and remove: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"