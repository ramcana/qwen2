#!/bin/bash

# Test script for production Docker build
set -e

echo "🏗️  Testing production Docker build..."

# Clean up any previous builds
echo "Cleaning up previous builds..."
rm -rf build build-dev dist || true

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.prod \
  --build-arg NODE_ENV=production \
  --build-arg BUILD_PATH=build \
  --build-arg GENERATE_SOURCEMAP=false \
  --build-arg REACT_APP_API_URL=/api \
  --build-arg REACT_APP_WS_URL=/ws \
  --build-arg REACT_APP_BACKEND_HOST=qwen-api \
  --build-arg REACT_APP_BACKEND_PORT=8000 \
  --build-arg REACT_APP_VERSION=2.0.0 \
  -t qwen-frontend:test-build \
  .

echo "✅ Docker build completed successfully!"

# Test the container
echo "Testing container startup..."
docker run --rm -d --name qwen-frontend-test -p 3001:80 qwen-frontend:test-build

# Wait a moment for startup
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:3001/health || (echo "❌ Health check failed" && exit 1)

# Test main page
echo "Testing main page..."
curl -f http://localhost:3001/ > /dev/null || (echo "❌ Main page failed" && exit 1)

# Clean up
echo "Cleaning up test container..."
docker stop qwen-frontend-test || true

echo "🎉 All tests passed! Production build is working correctly."