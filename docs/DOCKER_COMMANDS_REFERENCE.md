# Docker Commands Reference

This reference provides comprehensive Docker commands for managing the Qwen2 application deployment, organized by common use cases and operations.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Build Commands](#build-commands)
- [Service Management](#service-management)
- [Logging and Debugging](#logging-and-debugging)
- [Volume Operations](#volume-operations)
- [Network Management](#network-management)
- [Monitoring and Health](#monitoring-and-health)
- [Cleanup and Maintenance](#cleanup-and-maintenance)
- [Troubleshooting Commands](#troubleshooting-commands)

## Quick Reference

### Essential Commands

```bash
# Start development environment
./scripts/deploy-docker.sh dev --build

# Stop all services
docker-compose down

# View all service logs
docker-compose logs -f

# Check service status
docker-compose ps

# Open shell in API container
docker-compose exec api /bin/bash

# Rebuild and restart
docker-compose up --build -d
```

### Environment-Specific Commands

```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Production
docker-compose -f docker-compose.prod.yml up -d

# Staging
docker-compose -f docker-compose.yml up -d
```

## Build Commands

### Building Images

#### Build All Services

```bash
# Build all services with no cache
docker-compose build --no-cache

# Build all services with cache
docker-compose build

# Build with specific compose file
docker-compose -f docker-compose.dev.yml build
```

#### Build Specific Services

```bash
# Build only API service
docker-compose build api

# Build only frontend service
docker-compose build frontend

# Build with custom tag
docker build -t qwen2-api:custom -f Dockerfile.api .
```

#### Advanced Build Options

```bash
# Build with build arguments
docker-compose build --build-arg PYTHON_VERSION=3.11 api

# Build with progress output
docker-compose build --progress=plain

# Build in parallel
docker-compose build --parallel
```

### Image Management

#### List Images

```bash
# List all Qwen2 images
docker images | grep qwen2

# List all images with sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# List dangling images
docker images -f "dangling=true"
```

#### Remove Images

```bash
# Remove specific image
docker rmi qwen2-api:latest

# Remove all Qwen2 images
docker images | grep qwen2 | awk '{print $3}' | xargs docker rmi

# Remove dangling images
docker image prune -f
```

#### Image Inspection

```bash
# Inspect image details
docker inspect qwen2-api:latest

# View image history
docker history qwen2-api:latest

# Check image layers
docker inspect qwen2-api:latest | jq '.[0].RootFS.Layers'
```

## Service Management

### Starting Services

#### Start All Services

```bash
# Start in detached mode
docker-compose up -d

# Start in foreground
docker-compose up

# Start with specific compose file
docker-compose -f docker-compose.dev.yml up -d
```

#### Start Specific Services

```bash
# Start only API service
docker-compose up -d api

# Start API and dependencies
docker-compose up -d api redis

# Start with recreate
docker-compose up -d --force-recreate api
```

#### Advanced Start Options

```bash
# Start with build
docker-compose up -d --build

# Start with scale
docker-compose up -d --scale api=3

# Start with timeout
docker-compose up -d --timeout 60
```

### Stopping Services

#### Stop All Services

```bash
# Stop all services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop with timeout
docker-compose stop -t 30
```

#### Stop Specific Services

```bash
# Stop only API service
docker-compose stop api

# Stop multiple services
docker-compose stop api frontend
```

#### Advanced Stop Options

```bash
# Stop and remove volumes
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all

# Stop and remove orphans
docker-compose down --remove-orphans
```

### Restarting Services

#### Restart All Services

```bash
# Restart all services
docker-compose restart

# Restart with timeout
docker-compose restart -t 30
```

#### Restart Specific Services

```bash
# Restart API service
docker-compose restart api

# Restart multiple services
docker-compose restart api frontend
```

### Scaling Services

#### Scale Services

```bash
# Scale API service to 3 replicas
docker-compose up -d --scale api=3

# Scale multiple services
docker-compose up -d --scale api=3 --scale frontend=2

# Check scaled services
docker-compose ps
```

## Logging and Debugging

### Viewing Logs

#### All Service Logs

```bash
# Follow all logs
docker-compose logs -f

# Show last 100 lines
docker-compose logs --tail=100

# Show logs since timestamp
docker-compose logs --since="2024-01-01T00:00:00"
```

#### Specific Service Logs

```bash
# API service logs
docker-compose logs -f api

# Frontend service logs
docker-compose logs -f frontend

# Traefik service logs
docker-compose logs -f traefik
```

#### Advanced Log Options

```bash
# Show timestamps
docker-compose logs -f -t

# Show logs with service names
docker-compose logs -f --no-log-prefix=false

# Filter logs by level (if structured)
docker-compose logs api | grep ERROR
```

### Container Access

#### Execute Commands

```bash
# Open bash shell in API container
docker-compose exec api /bin/bash

# Execute single command
docker-compose exec api python --version

# Execute as specific user
docker-compose exec --user root api /bin/bash
```

#### File Operations

```bash
# Copy file from container
docker-compose exec api cat /app/config.json > local_config.json

# Copy file to container
docker cp local_file.txt $(docker-compose ps -q api):/app/

# Edit file in container
docker-compose exec api nano /app/config.json
```

#### Process Monitoring

```bash
# Show running processes in container
docker-compose exec api ps aux

# Show container resource usage
docker stats $(docker-compose ps -q)

# Show container details
docker-compose exec api cat /proc/meminfo
```

## Volume Operations

### Volume Management

#### List Volumes

```bash
# List all volumes
docker volume ls

# List Qwen2 volumes
docker volume ls | grep qwen2

# Inspect volume details
docker volume inspect qwen2_models
```

#### Create Volumes

```bash
# Create named volume
docker volume create qwen2_models

# Create volume with driver options
docker volume create --driver local --opt type=none --opt device=/path/to/models --opt o=bind qwen2_models
```

#### Remove Volumes

```bash
# Remove specific volume
docker volume rm qwen2_models

# Remove unused volumes
docker volume prune -f

# Remove all Qwen2 volumes
docker volume ls | grep qwen2 | awk '{print $2}' | xargs docker volume rm
```

### Volume Inspection

#### Check Volume Usage

```bash
# Show volume mount points
docker-compose exec api df -h

# Show directory sizes in container
docker-compose exec api du -sh /app/*

# Check volume permissions
docker-compose exec api ls -la /app/models
```

#### Volume Backup

```bash
# Backup volume to tar
docker run --rm -v qwen2_models:/data -v $(pwd):/backup alpine tar czf /backup/models_backup.tar.gz -C /data .

# Restore volume from tar
docker run --rm -v qwen2_models:/data -v $(pwd):/backup alpine tar xzf /backup/models_backup.tar.gz -C /data
```

## Network Management

### Network Operations

#### List Networks

```bash
# List all networks
docker network ls

# List Qwen2 networks
docker network ls | grep qwen2

# Inspect network details
docker network inspect qwen2_default
```

#### Create Networks

```bash
# Create custom network
docker network create qwen2-network

# Create network with specific driver
docker network create --driver bridge qwen2-network

# Create network with subnet
docker network create --subnet=172.20.0.0/16 qwen2-network
```

#### Network Connectivity

```bash
# Test connectivity between containers
docker-compose exec api ping frontend

# Check network configuration in container
docker-compose exec api ip addr show

# Show network connections
docker-compose exec api netstat -tlnp
```

### Port Management

#### Check Port Usage

```bash
# Show container port mappings
docker-compose port api 8000

# List all port mappings
docker-compose ps --format "table {{.Name}}\t{{.Ports}}"

# Check what's using a port on host
sudo lsof -i :8000
```

#### Test Connectivity

```bash
# Test API endpoint
curl http://localhost:8000/health

# Test frontend
curl http://localhost:3000

# Test Traefik dashboard
curl http://localhost:8080/dashboard/
```

## Monitoring and Health

### Service Status

#### Check Service Health

```bash
# Show service status
docker-compose ps

# Show detailed service info
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Check specific service health
docker inspect $(docker-compose ps -q api) --format='{{.State.Health.Status}}'
```

#### Resource Usage

```bash
# Show real-time resource usage
docker stats

# Show resource usage for Qwen2 containers only
docker stats $(docker-compose ps -q)

# Show resource limits
docker inspect $(docker-compose ps -q api) | jq '.[0].HostConfig.Memory'
```

### Performance Monitoring

#### Container Metrics

```bash
# Show container processes
docker-compose top

# Show container events
docker events --filter container=$(docker-compose ps -q api)

# Show system info
docker system info
```

#### Application Metrics

```bash
# Check API health endpoint
curl http://localhost:8000/health | jq

# Check API metrics (if available)
curl http://localhost:8000/metrics

# Monitor log output
docker-compose logs -f api | grep -E "(ERROR|WARNING|INFO)"
```

## Cleanup and Maintenance

### System Cleanup

#### Remove Unused Resources

```bash
# Clean everything unused
docker system prune -a -f

# Clean containers only
docker container prune -f

# Clean images only
docker image prune -a -f

# Clean volumes only
docker volume prune -f
```

#### Specific Cleanup

```bash
# Remove stopped containers
docker rm $(docker ps -a -q)

# Remove dangling images
docker rmi $(docker images -f "dangling=true" -q)

# Remove unused networks
docker network prune -f
```

### Maintenance Tasks

#### Update Images

```bash
# Pull latest images
docker-compose pull

# Rebuild with latest base images
docker-compose build --pull --no-cache

# Update and restart
docker-compose pull && docker-compose up -d
```

#### Log Rotation

```bash
# Truncate container logs
truncate -s 0 $(docker inspect --format='{{.LogPath}}' $(docker-compose ps -q api))

# Configure log rotation (in compose file)
# logging:
#   driver: "json-file"
#   options:
#     max-size: "10m"
#     max-file: "3"
```

## Troubleshooting Commands

### Diagnostic Commands

#### Container Diagnostics

```bash
# Check container exit codes
docker-compose ps -a

# Show container logs with timestamps
docker-compose logs -t api

# Inspect container configuration
docker inspect $(docker-compose ps -q api) | jq
```

#### System Diagnostics

```bash
# Check Docker daemon status
systemctl status docker

# Check Docker daemon logs
journalctl -u docker.service

# Check system resources
free -h && df -h
```

### Debug Mode

#### Enable Debug Logging

```bash
# Set debug mode in environment
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Run with debug output
docker-compose --verbose up -d
```

#### Container Debugging

```bash
# Run container in debug mode
docker run -it --rm qwen2-api:latest /bin/bash

# Override entrypoint for debugging
docker run -it --rm --entrypoint /bin/bash qwen2-api:latest

# Debug networking
docker-compose exec api nslookup frontend
docker-compose exec api telnet frontend 3000
```

### Recovery Commands

#### Service Recovery

```bash
# Force recreate containers
docker-compose up -d --force-recreate

# Restart with clean state
docker-compose down -v && docker-compose up -d

# Reset to clean state
docker-compose down --rmi all -v --remove-orphans
docker-compose up -d --build
```

#### Data Recovery

```bash
# Backup current state before recovery
docker run --rm -v qwen2_models:/data -v $(pwd):/backup alpine tar czf /backup/emergency_backup.tar.gz -C /data .

# Restore from backup
docker run --rm -v qwen2_models:/data -v $(pwd):/backup alpine tar xzf /backup/models_backup.tar.gz -C /data
```

### Performance Troubleshooting

#### Memory Issues

```bash
# Check memory usage
docker stats --no-stream

# Check for OOM kills
dmesg | grep -i "killed process"

# Increase memory limits (in compose file)
# deploy:
#   resources:
#     limits:
#       memory: 8G
```

#### CPU Issues

```bash
# Check CPU usage
docker stats --no-stream

# Check container processes
docker-compose exec api top

# Limit CPU usage (in compose file)
# deploy:
#   resources:
#     limits:
#       cpus: '2.0'
```

#### Disk Issues

```bash
# Check disk usage
docker system df

# Check container disk usage
docker-compose exec api df -h

# Clean up disk space
docker system prune -a -f --volumes
```

### GPU Troubleshooting

#### GPU Diagnostics

```bash
# Check GPU availability in container
docker-compose exec api nvidia-smi

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU memory usage
docker-compose exec api nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Advanced Operations

### Multi-Environment Management

#### Environment Switching

```bash
# Switch to development
export COMPOSE_FILE=docker-compose.dev.yml

# Switch to production
export COMPOSE_FILE=docker-compose.prod.yml

# Use multiple compose files
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### Configuration Management

```bash
# Use different env files
docker-compose --env-file .env.dev up -d

# Override specific variables
API_PORT=9000 docker-compose up -d

# Check resolved configuration
docker-compose config
```

### CI/CD Integration

#### Automated Deployment

```bash
# Build and test
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Deploy if tests pass
if [ $? -eq 0 ]; then
    docker-compose -f docker-compose.prod.yml up -d
fi
```

#### Health Checks

```bash
# Wait for services to be healthy
docker-compose up -d
while ! docker-compose exec api curl -f http://localhost:8000/health; do
    sleep 5
done
```

This comprehensive command reference covers all essential Docker operations for managing the Qwen2 application deployment effectively.
