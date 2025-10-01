# Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Qwen2 Image Generation application using Docker containers in both development and production environments.

## Overview

The Docker containerization solution includes:

- **Multi-environment support**: Separate configurations for development and production
- **Optimized resource management**: Environment-specific resource limits and monitoring
- **Simplified deployment**: Automated scripts for easy deployment and management
- **Comprehensive logging**: Different logging levels for development and production
- **Security**: Production-ready security configurations
- **Scalability**: Support for horizontal scaling in production

## Quick Start

### Development Environment

1. **Prerequisites**

   ```bash
   # Install Docker and Docker Compose
   # For Ubuntu/Debian:
   sudo apt update
   sudo apt install docker.io docker-compose

   # For Windows/Mac: Install Docker Desktop
   ```

2. **Start Development Environment**

   ```bash
   # Simple one-command start
   ./scripts/deploy-dev.sh start

   # Or step by step:
   cp .env.dev.example .env
   ./scripts/deploy-dev.sh build
   ./scripts/deploy-dev.sh start
   ```

3. **Access Services**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Traefik Dashboard: http://localhost:8080

### Production Environment

1. **Prerequisites**

   ```bash
   # Ensure GPU support (for production)
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Configure Production Environment**

   ```bash
   # Copy and edit production configuration
   cp .env.prod.example .env.prod

   # Edit the following required fields:
   # - DOMAIN settings
   # - All passwords (search for CHANGE_THIS)
   # - SSL_EMAIL
   nano .env.prod
   ```

3. **Deploy Production Environment**
   ```bash
   ./scripts/deploy-prod.sh start
   ```

## Architecture

### Service Architecture

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

### Container Components

1. **Traefik (Reverse Proxy)**

   - Load balancing and SSL termination
   - Automatic service discovery
   - Dashboard for monitoring

2. **API Server (FastAPI + DiffSynth)**

   - GPU-accelerated model inference
   - RESTful API endpoints
   - Health monitoring

3. **Frontend (React + Nginx)**

   - User interface
   - Static asset serving
   - Production optimizations

4. **Support Services (Production)**
   - Redis for caching
   - Prometheus for metrics
   - Grafana for visualization

## Environment Configurations

### Development Environment

**Resource Limits:**

- API: 8GB RAM, 4 CPU cores
- Frontend: 2GB RAM, 2 CPU cores
- Traefik: 512MB RAM, 1 CPU core

**Features:**

- Hot reloading for development
- Debug logging (verbose)
- Direct port access for debugging
- Source code mounting
- Development tools integration

**Logging:**

- Uncompressed logs for easier debugging
- Extended retention (10 files, 200MB each)
- Debug-level logging
- Service tagging

### Production Environment

**Resource Limits:**

- API: 12GB RAM limit, 6 CPU cores (8GB/4 cores reserved)
- Frontend: 256MB RAM limit, 0.5 CPU cores (128MB/0.25 cores reserved)
- Traefik: 256MB RAM limit, 0.5 CPU cores (128MB/0.25 cores reserved)

**Features:**

- SSL/TLS with Let's Encrypt
- Enhanced security headers
- Rate limiting and compression
- Monitoring and metrics
- Automated backups

**Logging:**

- Compressed logs for storage efficiency
- Limited retention (3-10 files, 25-50MB each)
- INFO-level logging
- Structured JSON format

## Deployment Scripts

### Development Script (`scripts/deploy-dev.sh`)

```bash
# Start development environment
./scripts/deploy-dev.sh start

# View logs
./scripts/deploy-dev.sh logs [service]

# Restart specific service
./scripts/deploy-dev.sh restart api

# Stop environment
./scripts/deploy-dev.sh stop

# Clean up (including volumes)
./scripts/deploy-dev.sh clean --volumes
```

### Production Script (`scripts/deploy-prod.sh`)

```bash
# Start production environment
./scripts/deploy-prod.sh start

# Scale API service
./scripts/deploy-prod.sh scale api=3

# Update services
./scripts/deploy-prod.sh update

# Create backup
./scripts/deploy-prod.sh backup

# View status and resource usage
./scripts/deploy-prod.sh status
```

## Configuration Management

### Environment Variables

Both environments support extensive configuration through environment variables:

**Core Settings:**

- `NODE_ENV`: Environment mode (development/production)
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging verbosity

**API Configuration:**

- `API_WORKERS`: Number of API worker processes
- `MEMORY_OPTIMIZATION`: Enable memory optimizations
- `MAX_BATCH_SIZE`: Maximum batch size for processing

**Security Settings:**

- `CORS_ORIGINS`: Allowed CORS origins
- `SECURE_COOKIES`: Enable secure cookie settings
- `HTTPS_ONLY`: Force HTTPS redirects

**Performance Settings:**

- `TILED_PROCESSING_THRESHOLD`: Memory management threshold
- `CACHE_TTL`: Cache time-to-live
- `MODEL_CACHE_SIZE`: Model cache size limit

### Volume Management

**Persistent Volumes:**

- `models/`: Pre-trained model storage
- `cache/`: HuggingFace and PyTorch cache
- `generated_images/`: Output image storage
- `uploads/`: User upload storage
- `logs/`: Application logs

**Cache Optimization:**

- Separate cache volumes for different model types
- Shared cache between container restarts
- Configurable cache size limits

## Monitoring and Logging

### Development Monitoring

- **Traefik Dashboard**: http://localhost:8080
- **Direct API Access**: http://localhost:8000
- **Verbose Logging**: All services log at DEBUG level
- **Hot Reloading**: Automatic code reload on changes

### Production Monitoring

- **Grafana Dashboard**: https://monitoring.yourdomain.com
- **Prometheus Metrics**: Automated metrics collection
- **Structured Logging**: JSON format for log aggregation
- **Health Checks**: Automated service health monitoring

### Log Management

**Development:**

```bash
# View all logs
./scripts/deploy-dev.sh logs

# View specific service logs
./scripts/deploy-dev.sh logs api

# Follow logs in real-time
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f api
```

**Production:**

```bash
# View production logs
./scripts/deploy-prod.sh logs

# View resource usage
./scripts/deploy-prod.sh status

# Create backup before maintenance
./scripts/deploy-prod.sh backup
```

## Security Considerations

### Development Security

- Relaxed CORS settings for development
- Direct port access for debugging
- Basic authentication for Traefik dashboard
- Non-production SSL certificates

### Production Security

- Strict CORS configuration
- SSL/TLS encryption with Let's Encrypt
- Security headers (HSTS, CSP, etc.)
- Rate limiting and DDoS protection
- Non-root container users
- Read-only filesystems where possible

## Troubleshooting

### Common Issues

1. **GPU Not Detected**

   ```bash
   # Check GPU support
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

   # Install nvidia-docker if needed
   sudo apt install nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Port Conflicts**

   ```bash
   # Check port usage
   sudo netstat -tulpn | grep :8000

   # Stop conflicting services
   sudo systemctl stop apache2  # or nginx
   ```

3. **Memory Issues**

   ```bash
   # Check available memory
   free -h

   # Adjust resource limits in compose files
   # Reduce MAX_BATCH_SIZE in environment variables
   ```

4. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER models/ cache/ generated_images/
   chmod 755 models/ cache/ generated_images/
   ```

### Debug Commands

```bash
# Check container status
docker ps -a

# View container logs
docker logs qwen-api

# Execute commands in container
docker exec -it qwen-api bash

# Check resource usage
docker stats

# Inspect container configuration
docker inspect qwen-api
```

## Performance Optimization

### Development Optimizations

- Source code mounting for hot reloading
- Relaxed resource limits for debugging
- Uncompressed logs for easier reading
- Development-specific build targets

### Production Optimizations

- Multi-stage builds for smaller images
- Resource reservations and limits
- Compressed logging
- GPU memory optimization
- CDN-ready static asset serving

## Scaling and High Availability

### Horizontal Scaling

```bash
# Scale API service
./scripts/deploy-prod.sh scale api=3

# Scale with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale api=3
```

### Load Balancing

- Traefik automatically load balances across API instances
- Health checks ensure traffic only goes to healthy containers
- Rolling updates with zero downtime

### Data Persistence

- All critical data stored in persistent volumes
- Automatic backup scripts for production
- Database replication support (when enabled)

## Maintenance

### Regular Maintenance Tasks

1. **Update Images**

   ```bash
   ./scripts/deploy-prod.sh update
   ```

2. **Create Backups**

   ```bash
   ./scripts/deploy-prod.sh backup
   ```

3. **Monitor Resource Usage**

   ```bash
   ./scripts/deploy-prod.sh status
   ```

4. **Clean Up Old Images**
   ```bash
   docker image prune -a
   docker volume prune
   ```

### Backup and Restore

```bash
# Create backup
./scripts/deploy-prod.sh backup

# Restore from backup
./scripts/deploy-prod.sh restore /path/to/backup/directory
```

## Support

For additional support:

1. Check the troubleshooting section above
2. Review container logs for error messages
3. Verify environment configuration
4. Ensure all prerequisites are met
5. Check Docker and GPU driver versions

## Requirements Compliance

This Docker containerization solution addresses the following requirements:

- **Requirement 4.1**: Simple deployment commands for different environments
- **Requirement 4.2**: Environment-specific settings via environment variables
- **Requirement 1.1**: Complete containerized application stack
- **Requirement 2.1**: Optimized Docker images with proper caching
- **Requirement 2.2**: Appropriate resource limits and health checks
- **Requirement 3.1**: Persistent data volumes
- **Requirement 5.1**: Network isolation and service discovery
