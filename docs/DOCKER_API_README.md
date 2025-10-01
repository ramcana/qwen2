# Qwen-Image API Docker Container

This document describes the optimized Docker container for the Qwen-Image API server with DiffSynth integration.

## Features

- **Python 3.11-slim** base image for compatibility and security
- **Multi-stage build** for optimized image size and build performance
- **DiffSynth-Studio integration** with proper git submodule handling
- **CUDA/GPU support** with nvidia-docker runtime
- **Comprehensive caching** for HuggingFace, PyTorch, and DiffSynth models
- **Security hardening** with non-root user execution
- **Health checks** and monitoring capabilities
- **Memory optimization** for large model inference

## Quick Start

### Prerequisites

- Docker Engine 20.10+ with BuildKit support
- NVIDIA Docker runtime (for GPU support)
- At least 8GB RAM (16GB+ recommended for GPU)
- 20GB+ free disk space for models and cache

### Build the Container

#### Using PowerShell (Windows)

```powershell
# GPU build (recommended)
.\build-docker-api.ps1 -Gpu

# CPU-only build
.\build-docker-api.ps1 -Cpu

# Custom tag
.\build-docker-api.ps1 -Gpu -Tag v1.0
```

#### Using Bash (Linux/WSL)

```bash
# GPU build (recommended)
./build-docker-api.sh --gpu

# CPU-only build
./build-docker-api.sh --cpu

# Custom tag
./build-docker-api.sh --gpu --tag v1.0
```

#### Manual Build

```bash
# Basic build
docker build -f Dockerfile.api -t qwen-image-api:gpu .

# With build arguments
docker build \
  --build-arg ENABLE_GPU=true \
  -f Dockerfile.api \
  -t qwen-image-api:gpu .
```

### Run the Container

#### Basic Run (GPU)

```bash
docker run --gpus all -p 8000:8000 qwen-image-api:gpu
```

#### With Persistent Volumes

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/generated_images:/app/generated_images \
  -v $(pwd)/uploads:/app/uploads \
  qwen-image-api:gpu
```

#### With Environment Variables

```bash
docker run --gpus all -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MEMORY_OPTIMIZATION=true \
  -e ENABLE_DIFFSYNTH=true \
  -e ENABLE_CONTROLNET=true \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  qwen-image-api:gpu
```

## Configuration

### Environment Variables

| Variable                     | Default                  | Description                     |
| ---------------------------- | ------------------------ | ------------------------------- |
| `CUDA_VISIBLE_DEVICES`       | `0`                      | GPU device selection            |
| `HF_HOME`                    | `/app/cache/huggingface` | HuggingFace cache directory     |
| `TORCH_HOME`                 | `/app/cache/torch`       | PyTorch cache directory         |
| `DIFFSYNTH_CACHE_DIR`        | `/app/cache/diffsynth`   | DiffSynth model cache           |
| `CONTROLNET_CACHE_DIR`       | `/app/cache/controlnet`  | ControlNet model cache          |
| `ENABLE_DIFFSYNTH`           | `true`                   | Enable DiffSynth functionality  |
| `ENABLE_CONTROLNET`          | `true`                   | Enable ControlNet functionality |
| `MEMORY_OPTIMIZATION`        | `true`                   | Enable memory optimizations     |
| `TILED_PROCESSING_THRESHOLD` | `1024`                   | Memory management threshold     |

### Volume Mounts

| Container Path          | Purpose            | Recommended Host Path |
| ----------------------- | ------------------ | --------------------- |
| `/app/models`           | Pre-trained models | `./models`            |
| `/app/cache`            | Model cache        | `./cache`             |
| `/app/generated_images` | Generated outputs  | `./generated_images`  |
| `/app/uploads`          | User uploads       | `./uploads`           |
| `/app/offload`          | Model offloading   | `./offload`           |

## Architecture

### Multi-Stage Build

1. **Builder Stage**: Installs dependencies in isolated environment
2. **Runtime Stage**: Copies only necessary files for minimal image size

### Security Features

- Non-root user execution (`appuser`)
- Minimal base image (python:3.11-slim)
- Read-only filesystem where possible
- Proper file permissions and ownership

### Performance Optimizations

- Layer caching for faster rebuilds
- Optimized dependency installation
- Memory management for large models
- GPU memory optimization
- Efficient model loading and caching

## Health Checks

The container includes comprehensive health monitoring:

```dockerfile
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

Access health status:

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

#### Build Failures

1. **Out of disk space**: Ensure 20GB+ free space
2. **Network timeouts**: Use `--no-cache` flag or check internet connection
3. **Permission errors**: Ensure Docker daemon is running with proper permissions

#### Runtime Issues

1. **GPU not detected**: Verify nvidia-docker runtime installation
2. **Out of memory**: Increase Docker memory limits or enable memory optimization
3. **Model download failures**: Check internet connection and HuggingFace access

### Debug Commands

```bash
# Check container logs
docker logs <container_id>

# Interactive shell
docker exec -it <container_id> /bin/bash

# Check GPU status
docker exec <container_id> nvidia-smi

# Check Python environment
docker exec <container_id> python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Monitoring

```bash
# Container resource usage
docker stats <container_id>

# GPU usage
docker exec <container_id> nvidia-smi -l 1

# Memory usage
docker exec <container_id> free -h
```

## Development

### Local Development

For development, mount the source code:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  qwen-image-api:gpu
```

### Debugging

Enable verbose logging:

```bash
docker run --gpus all -p 8000:8000 \
  -e PYTHONPATH=/app/src:/app/DiffSynth-Studio \
  -e CUDA_LAUNCH_BLOCKING=1 \
  qwen-image-api:gpu
```

## Production Deployment

### Resource Requirements

- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB+ (32GB+ for large models)
- **GPU**: 8GB+ VRAM (RTX 3080/4080 or better)
- **Storage**: 50GB+ SSD for models and cache

### Scaling

Use Docker Compose or Kubernetes for horizontal scaling:

```yaml
# docker-compose.yml example
version: "3.8"
services:
  api:
    image: qwen-image-api:gpu
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Security Considerations

- Run with non-root user
- Use secrets management for sensitive environment variables
- Regularly update base images for security patches
- Implement network policies for container isolation
- Monitor container logs for security events

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review container logs for error messages
3. Verify system requirements and dependencies
4. Test with minimal configuration first
