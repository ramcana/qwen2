# Docker Setup Guide for Qwen2 Image Generation

This guide provides comprehensive instructions for setting up and running the Qwen2 Image Generation application using Docker containers.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Environments](#deployment-environments)
- [Service Management](#service-management)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## 🎯 Overview

The Docker setup provides a complete containerized environment with:

- **Traefik Reverse Proxy**: SSL termination, load balancing, and service discovery
- **FastAPI Backend**: ML model inference with GPU support and DiffSynth integration
- **React Frontend**: Optimized production build with Nginx
- **Persistent Storage**: Models, cache, and generated content preservation
- **Development Tools**: Hot reloading, debugging, and monitoring services

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Traefik       │    │   Frontend      │    │   API Server    │
│  (Port 80/443)  │◄──►│   (React+Nginx) │◄──►│   (FastAPI+GPU) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Shared Volumes │
                    │  - Models       │
                    │  - Cache        │
                    │  - Generated    │
                    │  - Uploads      │
                    └─────────────────┘
```

## 🔧 Prerequisites

### Required Software

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)

   - Version 20.10+ recommended
   - Docker Compose v2.0+ (included with Docker Desktop)

2. **NVIDIA Docker Runtime** (for GPU support)

   - Required for model inference acceleration
   - Install nvidia-docker2 package on Linux
   - GPU support is automatic on Windows/Mac with Docker Desktop

3. **Git** (for cloning and submodule management)

### System Requirements

- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 50GB+ free space for models and cache
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Network**: Stable internet connection for model downloads

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd qwen2

# Run initial setup
./docker-deploy.sh setup
# or on Windows:
.\docker-deploy.ps1 setup
```

### 2. Configure Environment

```bash
# Copy and edit environment configuration
cp .env.example .env
# Edit .env file with your preferred settings
```

### 3. Start Development Environment

```bash
# Start all services in development mode
./docker-deploy.sh start dev

# or on Windows:
.\docker-deploy.ps1 start dev
```

### 4. Access the Application

- **Frontend**: http://qwen.localhost
- **API**: http://api.localhost
- **Traefik Dashboard**: http://traefik.localhost:8080

## ⚙️ Configuration

### Environment Variables

The `.env` file contains all configuration options:

```bash
# Core settings
NODE_ENV=development
CUDA_VISIBLE_DEVICES=0
ENABLE_DIFFSYNTH=true
ENABLE_CONTROLNET=true

# Performance settings
MEMORY_OPTIMIZATION=true
TILED_PROCESSING_THRESHOLD=2048
MAX_BATCH_SIZE=4

# Security settings (production)
CORS_ORIGINS=https://yourdomain.com
SECURE_COOKIES=true
HTTPS_ONLY=true
```

### Volume Configuration

Data persistence is handled through Docker volumes:

- `models/`: Pre-trained model storage
- `cache/`: HuggingFace and PyTorch cache
- `generated_images/`: Output image storage
- `uploads/`: User upload storage
- `logs/`: Application logs

## 🌍 Deployment Environments

### Development Environment

Features:

- Hot reloading for frontend and backend
- Debug logging and monitoring
- Direct port access for debugging
- Development tools (Redis, PostgreSQL, Mailhog)

```bash
# Start development environment
./docker-deploy.sh start dev

# Start with additional services
./docker-deploy.sh start dev "database,email,tools"
```

### Production Environment

Features:

- SSL/TLS termination with Let's Encrypt
- Enhanced security configurations
- Resource limits and monitoring
- High availability and scaling

```bash
# Build production images
./docker-deploy.sh build prod

# Start production environment
./docker-deploy.sh start prod

# Start with monitoring
./docker-deploy.sh start prod monitoring
```

## 🔧 Service Management

### Basic Commands

```bash
# Check service status
./docker-deploy.sh status

# View logs
./docker-deploy.sh logs              # All services
./docker-deploy.sh logs api          # Specific service
./docker-deploy.sh logs api true     # Follow logs

# Restart services
./docker-deploy.sh restart dev       # All services
./docker-deploy.sh restart dev api   # Specific service

# Stop services
./docker-deploy.sh stop dev
```

### Docker Compose Commands

```bash
# Direct Docker Compose usage
docker-compose ps                    # Service status
docker-compose logs -f api          # Follow API logs
docker-compose exec api bash        # Shell into API container
docker-compose restart frontend     # Restart frontend
```

### Service Profiles

Enable additional services using profiles:

- `monitoring`: Prometheus + Grafana
- `database`: PostgreSQL database
- `email`: Mailhog for email testing
- `tools`: File browser and utilities

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Start multiple profiles
docker-compose --profile monitoring --profile database up -d
```

## 📊 Monitoring and Logging

### Built-in Monitoring

- **Traefik Dashboard**: Service discovery and routing
- **Health Checks**: Automatic service health monitoring
- **Structured Logging**: JSON-formatted logs with rotation

### Advanced Monitoring (Production)

Enable the monitoring profile for:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Node Exporter**: System metrics

```bash
# Start with monitoring
./docker-deploy.sh start prod monitoring

# Access monitoring
# Grafana: http://monitoring.yourdomain.com
# Prometheus: http://prometheus.yourdomain.com
```

### Log Management

```bash
# View logs by service
docker-compose logs api
docker-compose logs frontend
docker-compose logs traefik

# Follow logs in real-time
docker-compose logs -f --tail=100 api

# Log files location
ls logs/api/        # API server logs
ls logs/traefik/    # Traefik proxy logs
```

## 🔍 Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Verify GPU access in container
docker-compose exec api nvidia-smi
```

#### 2. Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :80
netstat -tulpn | grep :8000

# Modify ports in .env file
API_PORT=8001
FRONTEND_PORT=3001
```

#### 3. Memory Issues

```bash
# Check container memory usage
docker stats

# Adjust memory limits in docker-compose.yml
mem_limit: 8g  # Reduce if needed
```

#### 4. Model Download Issues

```bash
# Check model download progress
docker-compose logs -f api | grep -i download

# Manual model download
docker-compose exec api python -c "from src.model_download_manager import download_models; download_models()"
```

### Service Health Checks

```bash
# Check service health
curl http://localhost:8000/health    # API health
curl http://localhost/health         # Frontend health

# Traefik service discovery
curl http://localhost:8080/api/http/services
```

### Container Debugging

```bash
# Shell into containers
docker-compose exec api bash
docker-compose exec frontend sh
docker-compose exec traefik sh

# Check container logs
docker-compose logs --tail=50 api
docker-compose logs --since=1h frontend

# Inspect container configuration
docker inspect qwen-api
docker inspect qwen-frontend
```

## 🔧 Advanced Configuration

### Custom Domains

For production deployment with custom domains:

1. Update `.env` file:

```bash
FRONTEND_DOMAIN=yourdomain.com
API_DOMAIN=api.yourdomain.com
TRAEFIK_DOMAIN=dashboard.yourdomain.com
LETSENCRYPT_EMAIL=your-email@yourdomain.com
```

2. Configure DNS records:

```
yourdomain.com        A    YOUR_SERVER_IP
api.yourdomain.com    A    YOUR_SERVER_IP
dashboard.yourdomain.com A YOUR_SERVER_IP
```

### SSL Configuration

#### Development (Self-signed)

```bash
# Generate self-signed certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/localhost.key -out ssl/localhost.crt
```

#### Production (Let's Encrypt)

```bash
# Automatic SSL with Traefik
# Configure in traefik.prod.yml
certificatesResolvers:
  letsencrypt:
    acme:
      email: your-email@yourdomain.com
      storage: /acme.json
      httpChallenge:
        entryPoint: web
```

### Resource Optimization

#### GPU Memory Management

```yaml
# In docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - MEMORY_OPTIMIZATION=true
  - TILED_PROCESSING_THRESHOLD=4096
```

#### Container Resource Limits

```yaml
# Production resource limits
deploy:
  resources:
    limits:
      memory: 12G
      cpus: "6.0"
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Scaling Configuration

#### Horizontal Scaling

```yaml
# Scale API service
deploy:
  replicas: 3
  update_config:
    parallelism: 1
    delay: 30s
```

#### Load Balancing

```yaml
# Traefik load balancing
labels:
  - "traefik.http.services.api.loadbalancer.server.port=8000"
  - "traefik.http.services.api.loadbalancer.sticky.cookie=true"
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

## 🆘 Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review container logs: `./docker-deploy.sh logs`
3. Check service status: `./docker-deploy.sh status`
4. Create an issue in the project repository

## 📝 License

This Docker configuration is part of the Qwen2 Image Generation project and follows the same license terms.
