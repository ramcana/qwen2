# Build Variants System

This document describes the comprehensive build variants system implemented for the Qwen2 frontend, providing development, staging, and production build configurations with environment-specific optimizations.

## Overview

The build variants system provides:

- **Environment-specific configurations** for development, staging, and production
- **Optimized Docker builds** with multi-stage builds and caching
- **Environment-specific nginx configurations** with appropriate security and performance settings
- **Automated validation and testing** for each build variant
- **Comprehensive build management tools** for easy deployment

## Build Variants

### Development Variant

**Purpose**: Fast development with hot reloading and debugging features

**Key Features**:

- Hot module replacement (HMR) enabled
- Source maps enabled for debugging
- Relaxed security headers for easier development
- Verbose logging and error reporting
- No minification for faster builds
- Development-specific API endpoints

**Configuration Files**:

- `Dockerfile.dev` - Development Docker configuration
- `.env.development` - Development environment variables
- `nginx.dev.conf` - Development nginx configuration
- `docker-compose.dev.yml` - Development Docker Compose

**Build Command**:

```bash
npm run build:development
# or
npm run build:variant:dev
```

**Docker Build**:

```bash
npm run docker:build:dev
# or
node scripts/build-variants.js build development --docker
```

### Staging Variant

**Purpose**: Production-like environment with debugging capabilities for testing

**Key Features**:

- Production optimizations with debugging enabled
- Source maps enabled for debugging
- Enhanced security headers
- Moderate compression and caching
- Bundle analysis enabled
- Staging-specific API endpoints

**Configuration Files**:

- `Dockerfile.prod` - Production Docker configuration (with staging args)
- `.env.staging` - Staging environment variables
- `nginx.staging.conf` - Staging nginx configuration
- `docker-compose.staging.yml` - Staging Docker Compose

**Build Command**:

```bash
npm run build:staging
# or
npm run build:variant:staging
```

**Docker Build**:

```bash
npm run docker:build:staging
# or
node scripts/build-variants.js build staging --docker
```

### Production Variant

**Purpose**: Fully optimized production build with maximum performance and security

**Key Features**:

- Maximum minification and compression
- Source maps disabled for security
- Strict security headers and CSP
- Aggressive code splitting
- Maximum caching strategies
- Production API endpoints

**Configuration Files**:

- `Dockerfile.prod` - Production Docker configuration
- `.env.production` - Production environment variables
- `nginx.prod.conf` - Production nginx configuration
- `docker-compose.optimized.yml` - Production Docker Compose

**Build Command**:

```bash
npm run build:production
# or
npm run build:variant:prod
```

**Docker Build**:

```bash
npm run docker:build:prod
# or
node scripts/build-variants.js build production --docker
```

## Environment Variables

### Development Environment (`.env.development`)

```bash
NODE_ENV=development
GENERATE_SOURCEMAP=true
REACT_APP_DEBUG=true
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_WS_URL=ws://localhost:8000/ws
FAST_REFRESH=true
CHOKIDAR_USEPOLLING=true
```

### Staging Environment (`.env.staging`)

```bash
NODE_ENV=production
GENERATE_SOURCEMAP=true
REACT_APP_DEBUG=true
REACT_APP_ENVIRONMENT=staging
REACT_APP_API_URL=/api
REACT_APP_WS_URL=/ws
BUILD_OPTIMIZATION=true
```

### Production Environment (`.env.production`)

```bash
NODE_ENV=production
GENERATE_SOURCEMAP=false
REACT_APP_DEBUG=false
REACT_APP_ENVIRONMENT=production
REACT_APP_API_URL=/api
REACT_APP_WS_URL=/ws
BUILD_OPTIMIZATION=true
```

## Build Management Scripts

### Build Variants Manager

The `scripts/build-variants.js` script provides comprehensive build management:

```bash
# Build specific variant
node scripts/build-variants.js build <variant> [options]

# Test specific variant
node scripts/build-variants.js test <variant>

# Deploy specific variant
node scripts/build-variants.js deploy <variant>

# Validate variant configuration
node scripts/build-variants.js validate <variant>
```

**Options**:

- `--docker` - Use Docker build
- `--local` - Use local build
- `--no-cache` - Disable build cache
- `--tag=<name>` - Custom Docker tag

### Environment Validation

The `scripts/validate-env.js` script validates environment configurations:

```bash
# Validate all environments
npm run validate:env

# Or directly
node scripts/validate-env.js
```

### Build Variants Validation

The `validate-build-variants.js` script tests all build variants:

```bash
# Validate all variants
npm run validate:variants

# Validate specific variant
npm run validate:variants:dev
npm run validate:variants:staging
npm run validate:variants:prod
```

## Docker Configurations

### Multi-Stage Build Strategy

All Docker builds use multi-stage builds for optimization:

1. **Dependencies Stage**: Install and cache dependencies
2. **Builder Stage**: Build the application with environment-specific optimizations
3. **Production Stage**: Serve with nginx with optimized configuration

### Build Caching

Docker builds leverage BuildKit caching for faster builds:

```bash
# Enable BuildKit caching
export DOCKER_BUILDKIT=1

# Build with cache
docker build --cache-from qwen-frontend:cache -t qwen-frontend:prod .
```

## Nginx Configurations

### Development (`nginx.dev.conf`)

- Relaxed security headers
- CORS enabled for all origins
- Verbose logging
- No compression
- Source maps allowed

### Staging (`nginx.staging.conf`)

- Enhanced security headers
- Controlled CORS
- Moderate compression
- Source maps enabled
- Rate limiting

### Production (`nginx.prod.conf`)

- Maximum security headers
- Strict CSP
- Maximum compression
- Source maps disabled
- Aggressive caching

## Performance Optimizations

### Code Splitting

Environment-specific code splitting configurations:

- **Development**: Minimal splitting for faster builds
- **Staging**: Moderate splitting for testing
- **Production**: Aggressive splitting for optimal loading

### Compression

- **Development**: No compression
- **Staging**: Gzip compression
- **Production**: Gzip + Brotli compression

### Caching Strategies

- **Development**: No caching for hot reloading
- **Staging**: Moderate caching for testing
- **Production**: Aggressive caching for performance

## Usage Examples

### Local Development

```bash
# Start development server
npm run start:dev

# Build development variant
npm run build:development

# Validate development configuration
node scripts/build-variants.js validate development
```

### Staging Deployment

```bash
# Build staging variant
npm run build:staging

# Build staging Docker image
npm run docker:build:staging

# Deploy staging environment
npm run deploy:variant:staging
```

### Production Deployment

```bash
# Build production variant
npm run build:production

# Build production Docker image with no cache
node scripts/build-variants.js build production --docker --no-cache

# Deploy production environment
npm run deploy:variant:prod
```

### Testing and Validation

```bash
# Validate all environments
npm run validate:env

# Test all build variants
npm run validate:variants

# Test specific variant
node scripts/build-variants.js test staging
```

## Troubleshooting

### Common Issues

1. **TypeScript Errors**: Ensure TypeScript is in dependencies, not devDependencies
2. **Environment Variables**: Check that all required variables are set in environment files
3. **Docker Build Failures**: Verify Docker BuildKit is enabled
4. **Nginx Configuration**: Validate nginx configuration syntax

### Debug Commands

```bash
# Check environment configuration
npm run validate:env

# Validate specific variant
node scripts/build-variants.js validate <variant>

# Test build without Docker
npm run build:<variant>

# Check Docker build logs
docker build --progress=plain -f Dockerfile.<variant> .
```

### Log Locations

- Build logs: `build.log`
- Docker logs: `docker logs <container-name>`
- Nginx logs: `/var/log/nginx/` (in container)
- Build reports: `build-report-*.json`

## Best Practices

1. **Always validate** configurations before building
2. **Use appropriate variant** for each environment
3. **Test builds locally** before Docker deployment
4. **Monitor build performance** and optimize as needed
5. **Keep environment files** in sync with requirements
6. **Use build caching** for faster development cycles
7. **Validate security headers** in staging and production

## Integration with CI/CD

The build variants system integrates well with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Validate Environment
  run: npm run validate:env

- name: Build Development
  run: node scripts/build-variants.js build development --local

- name: Build Staging
  run: node scripts/build-variants.js build staging --docker

- name: Test Build Variants
  run: npm run validate:variants

- name: Deploy Production
  run: node scripts/build-variants.js deploy production
```

This comprehensive build variants system ensures consistent, optimized builds across all deployment environments while maintaining flexibility for development and debugging needs.
