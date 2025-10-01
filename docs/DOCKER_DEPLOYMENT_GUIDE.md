# Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Qwen2 Image Generation Application using Docker containers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Deployment Scripts](#deployment-scripts)
- [Volume Management](#volume-management)
- [Common Operations](#common-operations)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

## Prerequisites

### System Requirements

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher (or docker-compose v1.29+)
- **GPU Support** (optional): NVIDIA Docker runtime for GPU acceleration
- **Memory**: Minimum 8GB RAM (16GB+ recommended for GPU usage)
- **Storage**: At least 20GB free space for models and cache

### GPU Support (Optional)

For GPU acceleration, install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### 1. Environment Setup

Run the setup script to prepare your Docker environment:

```bash
# Make setup script executable and run
chmod +x scripts/setup-docker-env.sh
./scripts/setup-docker-env.sh
```

This script will:

- Check Docker installation and requirements
- Create necessary directories and permissions
- Generate environment configuration files
- Set up Traefik configuration
- Create management scripts

### 2. Deploy Development Environment

```bash
# Deploy with fresh build
./scripts/deploy-docker.sh dev --build

# Or deploy with existing images
./scripts/deploy-docker.sh dev
```

### 3. Access the Application

Once deployed, access the application at:

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **Traefik Dashboard**: http://localhost:8080

## Environment Setup

### Configuration Files

The setup creates several configuration files:

#### `.env.docker` - Development Configuration

```bash
# Application Settings
APP_NAME=qwen2-app
API_PORT=8000
FRONTEND_PORT=3000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
ENABLE_GPU=true

# Model Paths
HF_HOME=/app/cache/huggingface
TORCH_HOME=/app/cache/torch
DIFFSYNTH_CACHE_DIR=/app/cache/diffsynth

# Feature Flags
ENABLE_DIFFSYNTH=true
ENABLE_CONTROLNET=true
```

#### `.env.docker.prod` - Production Template

Production configuration with enhanced security and performance settings.

### Directory Structure

The setup creates the following directory structure:

```
qwen2/
├── models/                 # Pre-trained model storage
├── cache/                  # HuggingFace and PyTorch cache
├── generated_images/       # Generated image outputs
├── uploads/               # User uploaded files
├── offload/               # Model offloading directory
├── logs/                  # Application logs
│   ├── api/              # API service logs
│   └── traefik/          # Traefik proxy logs
├── data/                  # Persistent data
│   ├── redis/            # Redis data
│   ├── prometheus/       # Prometheus metrics
│   └── grafana/          # Grafana dashboards
├── config/               # Configuration files
│   └── traefik/          # Traefik configuration
└── ssl/                  # SSL certificates
```

## Deployment Scripts

### Main Deployment Script

`./scripts/deploy-docker.sh` - Primary deployment script

**Usage:**

```bash
./scripts/deploy-docker.sh [ENVIRONMENT] [OPTIONS]
```

**Environments:**

- `dev` - Development environment (default)
- `prod` - Production environment
- `staging` - Staging environment

**Options:**

- `--build` - Build images before deployment
- `--pull` - Pull latest images before deployment
- `--force` - Force recreate containers
- `--foreground` - Run in foreground (don't detach)
- `--verbose` - Enable verbose output

**Examples:**

```bash
# Development with fresh build
./scripts/deploy-docker.sh dev --build

# Production with latest images
./scripts/deploy-docker.sh prod --pull --force

# Staging in foreground mode
./scripts/deploy-docker.sh staging --build --foreground
```

### Docker Operations Script

`./scripts/docker-ops.sh` - Simplified Docker management

**Common Commands:**

```bash
# Start services
./scripts/docker-ops.sh start dev

# View logs
./scripts/docker-ops.sh logs api

# Check status
./scripts/docker-ops.sh status

# Open shell in API container
./scripts/docker-ops.sh shell api

# Execute command in container
./scripts/docker-ops.sh exec api "pip list"

# Backup data
./scripts/docker-ops.sh backup

# Clean up
./scripts/docker-ops.sh clean
```

## Volume Management

### Persistent Volumes

The application uses several persistent volumes to maintain data across container restarts:

#### Model Storage (`models/`)

- **Purpose**: Store downloaded pre-trained models
- **Size**: 10-50GB depending on models
- **Backup**: Critical - models are expensive to re-download

#### Cache Storage (`cache/`)

- **Purpose**: HuggingFace, PyTorch, and DiffSynth cache
- **Size**: 5-20GB
- **Backup**: Recommended - improves startup time

#### Generated Images (`generated_images/`)

- **Purpose**: Store generated image outputs
- **Size**: Varies based on usage
- **Backup**: User preference

#### Uploads (`uploads/`)

- **Purpose**: User uploaded files for editing
- **Size**: Varies based on usage
- **Backup**: Recommended

### Volume Backup and Restore

#### Create Backup

```bash
# Using docker-ops script
./scripts/docker-ops.sh backup

# Manual backup
tar -czf qwen2-backup-$(date +%Y%m%d).tar.gz models cache generated_images uploads data
```

#### Restore from Backup

```bash
# Using docker-ops script
./scripts/docker-ops.sh restore qwen2-backup-20241201.tar.gz

# Manual restore
tar -xzf qwen2-backup-20241201.tar.gz
```

### Volume Cleanup

#### Clean Generated Images

```bash
# Remove images older than 7 days
find generated_images/ -name "*.png" -o -name "*.jpg" -mtime +7 -delete
```

#### Clean Cache

```bash
# Clear HuggingFace cache
rm -rf cache/huggingface/transformers/
rm -rf cache/huggingface/hub/

# Clear PyTorch cache
rm -rf cache/torch/
```

## Common Operations

### Building Images

#### Build All Images

```bash
# Using deployment script
./scripts/deploy-docker.sh dev --build

# Using docker-compose directly
docker-compose -f docker-compose.dev.yml build --no-cache
```

#### Build Specific Service

```bash
# Build only API service
docker-compose -f docker-compose.dev.yml build api

# Build only frontend
docker-compose -f docker-compose.dev.yml build frontend
```

### Managing Services

#### Start Services

```bash
# All services
docker-compose -f docker-compose.dev.yml up -d

# Specific service
docker-compose -f docker-compose.dev.yml up -d api
```

#### Stop Services

```bash
# All services
docker-compose -f docker-compose.dev.yml down

# Specific service
docker-compose -f docker-compose.dev.yml stop api
```

#### Restart Services

```bash
# All services
docker-compose -f docker-compose.dev.yml restart

# Specific service
docker-compose -f docker-compose.dev.yml restart api
```

### Viewing Logs

#### All Services

```bash
docker-compose -f docker-compose.dev.yml logs -f
```

#### Specific Service

```bash
# API logs
docker-compose -f docker-compose.dev.yml logs -f api

# Frontend logs
docker-compose -f docker-compose.dev.yml logs -f frontend

# Traefik logs
docker-compose -f docker-compose.dev.yml logs -f traefik
```

#### Log Management

```bash
# View last 100 lines
docker-compose logs --tail=100 api

# View logs since specific time
docker-compose logs --since="2024-01-01T00:00:00" api
```

### Container Access

#### Open Shell

```bash
# API container
docker-compose exec api /bin/bash

# Frontend container (if running)
docker-compose exec frontend /bin/sh
```

#### Execute Commands

```bash
# Check Python packages in API
docker-compose exec api pip list

# Check disk usage
docker-compose exec api df -h

# View environment variables
docker-compose exec api env
```

## Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check what's using port 8000
sudo lsof -i :8000

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:8000)
```

#### GPU Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU visibility in container
docker-compose exec api nvidia-smi
```

#### Out of Memory

```bash
# Check container memory usage
docker stats

# Check system memory
free -h

# Restart with memory limits
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.dev.yml up -d
```

#### Permission Issues

```bash
# Fix volume permissions
sudo chown -R $USER:$USER models cache generated_images uploads

# Fix Docker socket permissions (if needed)
sudo chmod 666 /var/run/docker.sock
```

### Service Health Checks

#### Check Service Status

```bash
# All services
docker-compose ps

# Specific service health
docker inspect qwen2-api --format='{{.State.Health.Status}}'
```

#### Manual Health Checks

```bash
# API health endpoint
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/health

# Traefik dashboard
curl http://localhost:8080/dashboard/
```

### Log Analysis

#### Error Patterns

```bash
# Search for errors in API logs
docker-compose logs api | grep -i error

# Search for GPU issues
docker-compose logs api | grep -i "cuda\|gpu"

# Search for memory issues
docker-compose logs api | grep -i "memory\|oom"
```

## Production Deployment

### Production Configuration

#### Environment Variables

```bash
# Copy production template
cp .env.docker.prod .env.docker

# Edit production settings
nano .env.docker
```

#### SSL Configuration

```bash
# Update Traefik domain
TRAEFIK_DOMAIN=your-domain.com
TRAEFIK_EMAIL=admin@your-domain.com
ENABLE_SSL=true
```

#### Resource Limits

```yaml
# In docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: "4"
        reservations:
          memory: 4G
          cpus: "2"
```

### Production Deployment Steps

1. **Prepare Environment**

   ```bash
   ./scripts/setup-docker-env.sh
   cp .env.docker.prod .env.docker
   # Edit .env.docker with production values
   ```

2. **Deploy Production**

   ```bash
   ./scripts/deploy-docker.sh prod --build
   ```

3. **Verify Deployment**

   ```bash
   ./scripts/docker-ops.sh status
   curl https://your-domain.com/health
   ```

4. **Set up Monitoring**

   ```bash
   # Access Grafana dashboard
   open https://grafana.your-domain.com

   # Check Prometheus metrics
   open https://prometheus.your-domain.com
   ```

### Production Monitoring

#### Health Monitoring

```bash
# Set up health check monitoring
curl -f https://your-domain.com/health || exit 1
```

#### Log Monitoring

```bash
# Set up log aggregation
docker-compose logs --since="1h" | grep ERROR
```

#### Resource Monitoring

```bash
# Monitor resource usage
docker stats --no-stream
```

### Backup Strategy

#### Automated Backups

```bash
# Create backup script
cat > /etc/cron.daily/qwen2-backup << 'EOF'
#!/bin/bash
cd /path/to/qwen2
./scripts/docker-ops.sh backup
# Upload to cloud storage
EOF

chmod +x /etc/cron.daily/qwen2-backup
```

## Security Considerations

### Network Security

- Services communicate via internal Docker networks
- Only Traefik exposes ports to the host
- SSL/TLS termination at the proxy level

### Container Security

- Non-root users where possible
- Read-only filesystems for immutable containers
- Regular security updates for base images

### Data Security

- Proper file permissions on volumes
- Encrypted storage for sensitive data
- Regular backups with encryption

## Performance Optimization

### Image Optimization

- Multi-stage builds to minimize image size
- Layer caching for faster builds
- Minimal base images (alpine, slim)

### Runtime Optimization

- Resource limits to prevent resource exhaustion
- Health checks for automatic recovery
- Proper logging configuration

### Storage Optimization

- Efficient volume mounting
- Cache optimization for models
- Regular cleanup of temporary files

## Support and Maintenance

### Regular Maintenance Tasks

1. **Update Images**

   ```bash
   ./scripts/deploy-docker.sh prod --pull
   ```

2. **Clean Up**

   ```bash
   docker system prune -f
   ```

3. **Backup Data**

   ```bash
   ./scripts/docker-ops.sh backup
   ```

4. **Monitor Logs**
   ```bash
   ./scripts/docker-ops.sh logs | grep ERROR
   ```

### Getting Help

- Check logs for error messages
- Verify configuration files
- Test with minimal configuration
- Check Docker and system resources
- Review this documentation

For additional support, refer to the main project documentation or create an issue in the project repository.
