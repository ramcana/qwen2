# Qwen2 Docker Deployment

Complete Docker containerization solution for the Qwen2 Image Generation Application, designed to resolve WSL host/port issues and provide a seamless development and deployment experience.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script to prepare your Docker environment
./scripts/setup-docker-env.sh
```

### 2. Deploy Development Environment

```bash
# Deploy with fresh build
./scripts/deploy-docker.sh dev --build

# Or deploy with existing images
./scripts/deploy-docker.sh dev
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **Traefik Dashboard**: http://localhost:8080

## 📋 Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **GPU Support** (optional): NVIDIA Docker runtime
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: At least 20GB free space

## 🛠 Available Scripts

### Deployment Scripts

| Script                | Purpose            | Usage                                         |
| --------------------- | ------------------ | --------------------------------------------- |
| `setup-docker-env.sh` | Environment setup  | `./scripts/setup-docker-env.sh`               |
| `deploy-docker.sh`    | Main deployment    | `./scripts/deploy-docker.sh [env] [options]`  |
| `deploy-docker.ps1`   | Windows deployment | `.\scripts\deploy-docker.ps1 [env] [options]` |
| `docker-ops.sh`       | Docker operations  | `./scripts/docker-ops.sh [command]`           |

### Common Operations

```bash
# Start development environment
./scripts/deploy-docker.sh dev --build

# Start production environment
./scripts/deploy-docker.sh prod --pull

# View logs
./scripts/docker-ops.sh logs api

# Open shell in API container
./scripts/docker-ops.sh shell api

# Check service status
./scripts/docker-ops.sh status

# Backup data
./scripts/docker-ops.sh backup

# Clean up resources
./scripts/docker-ops.sh clean
```

## 🏗 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Traefik       │    │   Frontend      │    │   API Server    │
│  (Reverse Proxy)│◄──►│   (React)       │◄──►│   (FastAPI)     │
│   Port 80/443   │    │   Port 3000     │    │   Port 8000     │
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

## 📁 Directory Structure

After running the setup script, the following structure is created:

```
qwen2/
├── models/                 # Pre-trained model storage (10-50GB)
├── cache/                  # Framework caches (5-20GB)
├── generated_images/       # Generated outputs
├── uploads/               # User uploads
├── offload/               # Model offloading
├── logs/                  # Application logs
│   ├── api/              # API service logs
│   └── traefik/          # Traefik logs
├── data/                  # Persistent data
│   ├── redis/            # Redis data
│   ├── prometheus/       # Metrics
│   └── grafana/          # Dashboards
├── config/               # Configuration
│   └── traefik/          # Traefik config
├── ssl/                  # SSL certificates
└── scripts/              # Deployment scripts
```

## 🔧 Configuration

### Environment Files

The setup creates configuration files for different environments:

- **`.env.docker`** - Development configuration
- **`.env.docker.prod`** - Production template

### Key Configuration Options

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
ENABLE_GPU=true

# Feature Flags
ENABLE_DIFFSYNTH=true
ENABLE_CONTROLNET=true
ENABLE_QWEN_EDIT=true

# Performance Settings
MEMORY_OPTIMIZATION=true
TILED_PROCESSING_THRESHOLD=1024
MAX_WORKERS=4

# Security
CORS_ORIGINS=http://localhost:3000
ALLOWED_HOSTS=localhost,127.0.0.1
```

## 🌍 Environment Deployment

### Development Environment

```bash
# Features:
# - Hot reloading
# - Debug logging
# - Development ports
# - Single replica

./scripts/deploy-docker.sh dev --build
```

### Production Environment

```bash
# Features:
# - SSL/TLS termination
# - Multiple replicas
# - Resource limits
# - Enhanced security

./scripts/deploy-docker.sh prod --pull
```

### Staging Environment

```bash
# Features:
# - Production-like setup
# - Testing configuration
# - Monitoring enabled

./scripts/deploy-docker.sh staging --build
```

## 💾 Volume Management

### Critical Volumes

| Volume              | Purpose            | Size     | Backup Priority |
| ------------------- | ------------------ | -------- | --------------- |
| `models/`           | Pre-trained models | 10-50GB  | Critical        |
| `cache/`            | Framework cache    | 5-20GB   | Important       |
| `generated_images/` | User outputs       | Variable | Important       |
| `uploads/`          | User inputs        | Variable | Critical        |

### Backup Operations

```bash
# Create full backup
./scripts/docker-ops.sh backup

# Manual backup
tar -czf backup.tar.gz models cache generated_images uploads data

# Restore from backup
./scripts/docker-ops.sh restore backup.tar.gz
```

## 📊 Monitoring and Logging

### Service Monitoring

```bash
# Check service status
docker-compose ps

# View resource usage
docker stats

# Check service health
curl http://localhost:8000/health
```

### Log Management

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api

# Search logs for errors
docker-compose logs api | grep ERROR
```

## 🔍 Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check what's using port 8000
sudo lsof -i :8000

# Kill conflicting process
sudo kill -9 $(sudo lsof -t -i:8000)
```

#### GPU Not Available

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU in container
docker-compose exec api nvidia-smi
```

#### Out of Memory

```bash
# Check memory usage
docker stats

# Restart with clean state
docker-compose down && docker-compose up -d
```

#### Permission Issues

```bash
# Fix volume permissions
sudo chown -R $USER:$USER models cache generated_images uploads
```

### Recovery Procedures

```bash
# Force recreate containers
docker-compose up -d --force-recreate

# Clean restart
docker-compose down -v && docker-compose up -d --build

# Emergency cleanup
docker system prune -a -f --volumes
```

## 🚀 Production Deployment

### Pre-deployment Checklist

- [ ] Update `.env.docker` with production values
- [ ] Configure SSL certificates
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy
- [ ] Test deployment in staging environment

### Production Deployment Steps

1. **Prepare Environment**

   ```bash
   cp .env.docker.prod .env.docker
   # Edit .env.docker with production values
   ```

2. **Deploy Services**

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
   # Access monitoring dashboards
   open https://grafana.your-domain.com
   open https://prometheus.your-domain.com
   ```

## 📚 Documentation

Comprehensive documentation is available:

- **[Docker Deployment Guide](docs/DOCKER_DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[Volume Management Guide](docs/DOCKER_VOLUME_MANAGEMENT.md)** - Data persistence and backup strategies
- **[Commands Reference](docs/DOCKER_COMMANDS_REFERENCE.md)** - Complete Docker commands reference

## 🔐 Security Considerations

### Network Security

- Services communicate via internal Docker networks
- Only Traefik exposes ports to the host
- SSL/TLS termination at proxy level

### Container Security

- Non-root users where possible
- Read-only filesystems for immutable containers
- Regular security updates for base images

### Data Security

- Proper file permissions on volumes
- Encrypted storage for sensitive data
- Regular backups with encryption

## 🎯 Performance Optimization

### Image Optimization

- Multi-stage builds for minimal image size
- Layer caching for faster builds
- Optimized base images

### Runtime Optimization

- Resource limits to prevent exhaustion
- Health checks for automatic recovery
- Efficient volume mounting

### Storage Optimization

- Model deduplication
- Cache cleanup automation
- Log rotation

## 🆘 Support

### Getting Help

1. **Check Logs**: `./scripts/docker-ops.sh logs`
2. **Verify Configuration**: `docker-compose config`
3. **Test Connectivity**: `curl http://localhost:8000/health`
4. **Check Resources**: `docker stats`
5. **Review Documentation**: See docs/ directory

### Common Commands Quick Reference

```bash
# Essential operations
./scripts/deploy-docker.sh dev --build    # Deploy development
./scripts/docker-ops.sh status           # Check status
./scripts/docker-ops.sh logs api         # View API logs
./scripts/docker-ops.sh shell api        # Access API container
./scripts/docker-ops.sh backup           # Backup data
docker-compose down                       # Stop all services

# Troubleshooting
docker-compose ps                         # Service status
docker stats                              # Resource usage
docker system df                          # Disk usage
docker system prune -f                    # Cleanup
```

## 📄 License

This Docker deployment configuration is part of the Qwen2 Image Generation Application. See the main project LICENSE for details.

---

**Ready to get started?** Run `./scripts/setup-docker-env.sh` to begin your Docker deployment journey! 🐳
