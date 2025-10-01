# Frontend Build Environments

This document describes the environment-specific build configurations for the Qwen2 React frontend, including development, staging, and production variants with their respective optimizations.

## Overview

The frontend build system supports three distinct environments:

- **Development**: Optimized for fast development with hot reloading, source maps, and debugging tools
- **Staging**: Balanced configuration for testing with production-like optimizations but debugging enabled
- **Production**: Fully optimized for deployment with minification, compression, and security hardening

## Environment Configurations

### Development Environment

**Purpose**: Local development with maximum developer experience

**Key Features**:

- Hot module reloading (HMR)
- Source maps enabled
- Debug mode enabled
- Relaxed security for local development
- Verbose logging
- Fast build times

**Configuration File**: `.env.development`

**Build Command**: `npm run build:development`

**Docker**: `Dockerfile.dev` with development server

### Staging Environment

**Purpose**: Pre-production testing with production-like optimizations

**Key Features**:

- Production build with source maps
- Debug mode enabled for testing
- Moderate security settings
- Balanced performance optimizations
- Testing-friendly configurations

**Configuration File**: `.env.staging`

**Build Command**: `npm run build:staging`

**Docker**: `Dockerfile.prod` with staging configuration

### Production Environment

**Purpose**: Optimized deployment build

**Key Features**:

- Full minification and compression
- Source maps disabled for security
- Debug mode disabled
- Maximum security settings
- Optimized bundle sizes
- Pre-compressed assets (gzip/brotli)

**Configuration File**: `.env.production`

**Build Command**: `npm run build:production`

**Docker**: `Dockerfile.prod` with production optimizations

## Build Scripts

### NPM Scripts

```bash
# Environment-specific builds
npm run build:development    # Build for development
npm run build:staging       # Build for staging
npm run build:production    # Build for production

# Environment-specific development servers
npm run start:dev           # Start development server
npm run start:staging       # Start with staging config

# Docker builds
npm run docker:build:dev    # Build development Docker image
npm run docker:build:staging # Build staging Docker image
npm run docker:build:prod   # Build production Docker image

# Environment validation
npm run validate:env        # Validate all environment configurations
```

### Custom Build Scripts

#### Environment-Specific Build Script

```bash
node scripts/build-env.js [environment]
```

**Examples**:

```bash
node scripts/build-env.js development
node scripts/build-env.js staging
node scripts/build-env.js production
```

**Features**:

- Environment validation
- Optimized build configurations
- Post-build optimizations
- Build analysis and reporting
- Automatic compression for staging/production

#### Docker Build Script

```bash
node scripts/docker-build-env.js [command] [environment] [options]
```

**Commands**:

- `build`: Build Docker image
- `test`: Test Docker image
- `build-and-test`: Build and test
- `push`: Push to registry
- `compose`: Generate docker-compose file

**Examples**:

```bash
# Build development image
node scripts/docker-build-env.js build development

# Build and test staging image
node scripts/docker-build-env.js build-and-test staging

# Build production image and push to registry
node scripts/docker-build-env.js build production --push --registry=your-registry.com
```

## Docker Configurations

### Development Docker Setup

**File**: `Dockerfile.dev`

**Features**:

- Node.js development server
- Hot reloading support
- Source code mounting
- Development dependencies included
- Debugging tools enabled

**Ports**: `3000:3000`

**Usage**:

```bash
docker-compose -f docker-compose.dev.yml up
```

### Staging Docker Setup

**File**: `Dockerfile.prod` (with staging args)

**Features**:

- Production build with source maps
- Nginx serving with staging configuration
- Moderate security settings
- Testing-friendly setup

**Ports**: `3002:80`

**Usage**:

```bash
docker-compose -f docker-compose.staging.yml up
```

### Production Docker Setup

**File**: `Dockerfile.prod`

**Features**:

- Multi-stage optimized build
- Nginx with production configuration
- Security hardening
- Asset compression
- Health checks

**Ports**: `3001:80`

**Usage**:

```bash
docker-compose -f docker-compose.optimized.yml up
```

## Environment Variables

### Common Variables

All environments include these base variables:

```env
NODE_ENV=<environment>
REACT_APP_API_URL=<api-url>
REACT_APP_WS_URL=<websocket-url>
REACT_APP_BACKEND_HOST=<backend-host>
REACT_APP_BACKEND_PORT=<backend-port>
REACT_APP_VERSION=<version>
REACT_APP_ENVIRONMENT=<environment>
```

### Development-Specific Variables

```env
GENERATE_SOURCEMAP=true
FAST_REFRESH=true
CHOKIDAR_USEPOLLING=true
REACT_APP_DEBUG=true
REACT_APP_HOT_RELOAD=true
REACT_APP_ENABLE_PROFILER=true
```

### Staging-Specific Variables

```env
GENERATE_SOURCEMAP=true
BUILD_OPTIMIZATION=true
REACT_APP_DEBUG=true
REACT_APP_LOG_LEVEL=info
```

### Production-Specific Variables

```env
GENERATE_SOURCEMAP=false
BUILD_OPTIMIZATION=true
REACT_APP_DEBUG=false
REACT_APP_HTTPS_ONLY=true
REACT_APP_LOG_LEVEL=error
```

## Nginx Configurations

### Development Nginx (`nginx.dev.conf`)

- Verbose logging for debugging
- Permissive CORS settings
- No asset caching
- Debug headers enabled
- Direct backend connection

### Staging Nginx (`nginx.staging.conf`)

- Balanced logging
- Moderate CORS restrictions
- Short-term asset caching
- Security headers
- Staging backend connection

### Production Nginx (`nginx.prod.conf`)

- Minimal logging
- Strict CORS settings
- Long-term asset caching
- Full security headers
- Production backend connection
- Asset compression

## Build Optimizations

### Development Optimizations

- **Fast builds**: Minimal optimizations for speed
- **Source maps**: Full source maps for debugging
- **Hot reloading**: Instant updates during development
- **No compression**: Faster build times

### Staging Optimizations

- **Balanced builds**: Some optimizations with debugging
- **Source maps**: Enabled for testing
- **Moderate compression**: Gzip compression
- **Testing features**: Debug mode enabled

### Production Optimizations

- **Full optimization**: Maximum minification and tree-shaking
- **No source maps**: Security and size optimization
- **Asset compression**: Gzip and Brotli pre-compression
- **Bundle analysis**: Size optimization reporting
- **Security hardening**: Strict CSP and security headers

## Validation and Testing

### Environment Validation

```bash
npm run validate:env
```

Validates:

- Required environment variables
- Environment-specific settings
- Configuration consistency
- Security settings

### Build Testing

```bash
# Test specific environment build
node scripts/build-env.js [environment]

# Test Docker image
node scripts/docker-build-env.js test [environment]
```

### Health Checks

Each environment includes health check endpoints:

- **Development**: `http://localhost:3000/health`
- **Staging**: `http://localhost:3002/health`
- **Production**: `http://localhost:3001/health`

## Deployment Workflows

### Development Workflow

1. Start development server: `npm run start:dev`
2. Make changes with hot reloading
3. Test with: `npm run test:dev`
4. Build for testing: `npm run build:development`

### Staging Workflow

1. Build staging image: `npm run docker:build:staging`
2. Deploy to staging environment
3. Run integration tests
4. Validate with staging configuration

### Production Workflow

1. Validate environment: `npm run validate:env`
2. Build production image: `npm run docker:build:prod`
3. Test production build locally
4. Deploy to production environment
5. Monitor health checks

## Troubleshooting

### Common Issues

1. **Environment variables not loaded**

   - Check `.env.[environment]` file exists
   - Verify variable names and values
   - Run `npm run validate:env`

2. **Build failures**

   - Check Node.js memory limits
   - Verify dependencies are installed
   - Check for TypeScript errors

3. **Docker build issues**
   - Ensure Docker BuildKit is enabled
   - Check Dockerfile syntax
   - Verify build context

### Debug Commands

```bash
# Check environment configuration
npm run validate:env

# Build with verbose output
NODE_OPTIONS="--max-old-space-size=4096" npm run build:development

# Test Docker image with logs
node scripts/docker-build-env.js test development --logs --no-cleanup
```

## Performance Metrics

### Build Times (Approximate)

- **Development**: 30-60 seconds
- **Staging**: 2-4 minutes
- **Production**: 3-6 minutes

### Bundle Sizes (Approximate)

- **Development**: 15-25 MB (uncompressed)
- **Staging**: 3-5 MB (compressed)
- **Production**: 2-3 MB (compressed)

### Docker Image Sizes

- **Development**: 800MB-1.2GB (includes dev dependencies)
- **Staging**: 150-200MB (nginx + assets)
- **Production**: 100-150MB (optimized nginx + assets)

## Best Practices

1. **Use appropriate environment** for each stage of development
2. **Validate configurations** before building
3. **Test Docker images** locally before deployment
4. **Monitor build performance** and optimize as needed
5. **Keep environment files** in version control (except secrets)
6. **Use BuildKit** for Docker builds for better caching
7. **Regularly update dependencies** and validate builds

## Migration Guide

### From Single Build to Multi-Environment

1. Create environment-specific `.env` files
2. Update package.json scripts
3. Configure Docker files for each environment
4. Update CI/CD pipelines
5. Test each environment configuration

### Updating Existing Builds

1. Backup current configuration
2. Run environment validation
3. Test new build process
4. Update deployment scripts
5. Monitor for issues
