#!/bin/bash

# Build script for Qwen-Image API Docker container
# Supports both CPU and GPU builds with optimization

set -e

# Default values
IMAGE_NAME="qwen-image-api"
TAG="latest"
BUILD_ARGS=""
PLATFORM=""
NO_CACHE=""
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            BUILD_ARGS="$BUILD_ARGS --build-arg ENABLE_GPU=true"
            TAG="gpu"
            shift
            ;;
        --cpu)
            BUILD_ARGS="$BUILD_ARGS --build-arg ENABLE_GPU=false"
            TAG="cpu"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --verbose)
            VERBOSE="--progress=plain"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu              Build with GPU support (default)"
            echo "  --cpu              Build with CPU-only support"
            echo "  --tag TAG          Set image tag (default: latest)"
            echo "  --name NAME        Set image name (default: qwen-image-api)"
            echo "  --platform ARCH    Set target platform (e.g., linux/amd64)"
            echo "  --no-cache         Build without using cache"
            echo "  --verbose          Show verbose build output"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --gpu --tag v1.0"
            echo "  $0 --cpu --no-cache"
            echo "  $0 --platform linux/amd64 --verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default to GPU if no specific option provided
if [[ "$BUILD_ARGS" != *"ENABLE_GPU"* ]]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg ENABLE_GPU=true"
    if [[ "$TAG" == "latest" ]]; then
        TAG="gpu"
    fi
fi

# Build information
echo "Building Qwen-Image API Docker container..."
echo "Image name: $IMAGE_NAME:$TAG"
echo "Build args: $BUILD_ARGS"
echo "Platform: ${PLATFORM:-default}"
echo "Cache: ${NO_CACHE:-enabled}"
echo ""

# Check if Dockerfile exists
if [[ ! -f "Dockerfile.api" ]]; then
    echo "Error: Dockerfile.api not found in current directory"
    exit 1
fi

# Check if requirements file exists
if [[ ! -f "requirements-docker.txt" ]]; then
    echo "Warning: requirements-docker.txt not found, using requirements.txt"
    if [[ ! -f "requirements.txt" ]]; then
        echo "Error: No requirements file found"
        exit 1
    fi
fi

# Build the Docker image
echo "Starting Docker build..."
docker build \
    $PLATFORM \
    $NO_CACHE \
    $VERBOSE \
    $BUILD_ARGS \
    -f Dockerfile.api \
    -t "$IMAGE_NAME:$TAG" \
    .

# Check build success
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✓ Build completed successfully!"
    echo "Image: $IMAGE_NAME:$TAG"
    echo ""
    echo "To run the container:"
    if [[ "$TAG" == *"gpu"* ]]; then
        echo "  docker run --gpus all -p 8000:8000 $IMAGE_NAME:$TAG"
    else
        echo "  docker run -p 8000:8000 $IMAGE_NAME:$TAG"
    fi
    echo ""
    echo "To run with volumes:"
    echo "  docker run --gpus all -p 8000:8000 \\"
    echo "    -v \$(pwd)/models:/app/models \\"
    echo "    -v \$(pwd)/cache:/app/cache \\"
    echo "    -v \$(pwd)/generated_images:/app/generated_images \\"
    echo "    $IMAGE_NAME:$TAG"
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi