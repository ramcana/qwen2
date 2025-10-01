#!/bin/bash

# Test just the build step in Docker
set -e

echo "🏗️  Testing build step only..."

# Create a simple Dockerfile just for testing the build
cat > Dockerfile.test << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY .npmrc* ./

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy source
COPY . .

# Set environment variables
ENV NODE_ENV=production
ENV GENERATE_SOURCEMAP=false
ENV BUILD_PATH=build
ENV SKIP_PREFLIGHT_CHECK=true
ENV TSC_COMPILE_ON_ERROR=true

# Run build
RUN npm run build

# List what was created
RUN ls -la
RUN ls -la build/ || echo "No build directory"

CMD ["echo", "Build test complete"]
EOF

# Build and test
docker build -f Dockerfile.test -t qwen-frontend:build-test .

# Clean up
rm Dockerfile.test

echo "✅ Build test completed!"