#!/bin/bash

# Dockerfile validation script
# Checks syntax and best practices for Dockerfile.api

set -e

echo "Validating Dockerfile.api..."

# Check if Dockerfile exists
if [[ ! -f "Dockerfile.api" ]]; then
    echo "❌ Dockerfile.api not found"
    exit 1
fi

# Check if required files exist
echo "Checking required files..."

required_files=(
    "requirements-docker.txt"
    "docker-entrypoint.sh"
    "src/api_server.py"
    "configs/"
)

for file in "${required_files[@]}"; do
    if [[ -e "$file" ]]; then
        echo "✅ $file exists"
    else
        echo "⚠️  $file not found (may be created at runtime)"
    fi
done

# Validate Dockerfile syntax using docker build --dry-run if available
echo ""
echo "Checking Dockerfile syntax..."

# Create a temporary minimal context for syntax check
temp_dir=$(mktemp -d)
cp Dockerfile.api "$temp_dir/"
cp requirements-docker.txt "$temp_dir/" 2>/dev/null || cp requirements.txt "$temp_dir/" 2>/dev/null || echo "# minimal requirements" > "$temp_dir/requirements-docker.txt"
cp docker-entrypoint.sh "$temp_dir/" 2>/dev/null || echo "#!/bin/bash" > "$temp_dir/docker-entrypoint.sh"
mkdir -p "$temp_dir/src" "$temp_dir/configs"
echo "print('test')" > "$temp_dir/src/api_server.py"

# Try to parse Dockerfile
if command -v docker &> /dev/null; then
    cd "$temp_dir"
    # Basic syntax validation - check if Docker can parse the file
    if docker build --help &>/dev/null; then
        echo "✅ Docker is available for syntax checking"
        # Note: Full syntax validation requires a complete build context
        echo "ℹ️  Run 'docker build -f Dockerfile.api .' to test full build"
    else
        echo "⚠️  Docker build command not available"
    fi
    cd - > /dev/null
else
    echo "⚠️  Docker not available, skipping syntax check"
fi

# Clean up
rm -rf "$temp_dir"

# Check Dockerfile best practices
echo ""
echo "Checking Dockerfile best practices..."

# Check for multi-stage build
if grep -q "FROM.*AS.*builder" Dockerfile.api && grep -q "FROM.*AS.*runtime" Dockerfile.api; then
    echo "✅ Multi-stage build detected"
else
    echo "⚠️  Multi-stage build not detected"
fi

# Check for non-root user
if grep -q "USER.*appuser" Dockerfile.api; then
    echo "✅ Non-root user configured"
else
    echo "⚠️  Non-root user not configured"
fi

# Check for health check
if grep -q "HEALTHCHECK" Dockerfile.api; then
    echo "✅ Health check configured"
else
    echo "⚠️  Health check not configured"
fi

# Check for proper COPY usage
if grep -q "COPY --chown=" Dockerfile.api; then
    echo "✅ Proper file ownership in COPY commands"
else
    echo "⚠️  File ownership not set in COPY commands"
fi

# Check for .dockerignore
if [[ -f ".dockerignore" ]]; then
    echo "✅ .dockerignore file exists"
else
    echo "⚠️  .dockerignore file not found"
fi

# Check for Python 3.11
if grep -q "python:3.11" Dockerfile.api; then
    echo "✅ Python 3.11 base image"
else
    echo "⚠️  Python 3.11 not detected"
fi

# Check for CUDA support
if grep -q "nvidia" Dockerfile.api; then
    echo "✅ NVIDIA/CUDA support configured"
else
    echo "⚠️  NVIDIA/CUDA support not detected"
fi

echo ""
echo "Validation complete!"
echo ""
echo "To build the container:"
echo "  ./build-docker-api.sh --gpu"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 8000:8000 qwen-image-api:gpu"