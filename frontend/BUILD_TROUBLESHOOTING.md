# Frontend Build Troubleshooting Guide

This guide helps resolve common issues with the React frontend Docker build process.

## Common Build Issues

### 1. TypeScript Module Not Found

**Error**: `Cannot resolve module 'typescript'` or `Module not found: Can't resolve 'typescript'`

**Causes**:

- TypeScript is in devDependencies but not available during Docker build
- npm ci failed to install dependencies properly
- Node modules cache corruption

**Solutions**:

1. **Verify TypeScript Installation**:

   ```bash
   npm list typescript
   npx tsc --version
   ```

2. **Clear npm cache**:

   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Check package.json dependencies**:
   - Ensure TypeScript is in `dependencies` (not just `devDependencies`)
   - Verify react-scripts version compatibility

### 2. Docker Build Memory Issues

**Error**: `JavaScript heap out of memory` or build process killed

**Solutions**:

1. **Increase Node.js memory limit** (already configured in Dockerfile):

   ```dockerfile
   ENV NODE_OPTIONS="--max-old-space-size=4096"
   ```

2. **Build with more Docker memory**:
   ```bash
   docker build --memory=4g -f Dockerfile.prod -t frontend .
   ```

### 3. Dependency Installation Failures

**Error**: `npm ci` fails with network or permission errors

**Solutions**:

1. **Use build script with retry logic**:

   ```bash
   ./build-docker.sh --no-cache
   ```

2. **Check npm configuration**:
   ```bash
   npm config list
   npm config set registry https://registry.npmjs.org/
   ```

### 4. Missing Source Files

**Error**: `Module not found: Can't resolve './src/index.tsx'`

**Solutions**:

1. **Verify file structure**:

   ```
   frontend/
   ├── src/
   │   ├── index.tsx
   │   └── ...
   ├── public/
   │   ├── index.html
   │   └── ...
   └── package.json
   ```

2. **Check .dockerignore**:
   - Ensure src/ and public/ are not excluded

### 5. Nginx Configuration Issues

**Error**: `nginx: [emerg] invalid parameter` or container fails to start

**Solutions**:

1. **Validate nginx config**:

   ```bash
   nginx -t -c nginx.prod.conf
   ```

2. **Check file permissions**:
   ```bash
   ls -la nginx*.conf
   ```

## Build Validation Tools

### 1. Pre-build Validation

```bash
./validate-build.sh
```

### 2. Full Build Test

```bash
./validate-build.sh --full
```

### 3. Docker Build with Validation

```bash
./build-docker.sh -e production --no-cache
```

## Debugging Steps

### 1. Local Build Test

```bash
# Test local build first
npm install
npm run build:production
```

### 2. Docker Build Debug

```bash
# Build with verbose output
docker build -f Dockerfile.prod -t debug-frontend . --progress=plain --no-cache
```

### 3. Container Inspection

```bash
# Run container interactively
docker run -it --entrypoint /bin/sh frontend-image

# Check build output
ls -la /usr/share/nginx/html/
cat /usr/share/nginx/html/index.html
```

## Environment-Specific Issues

### Development vs Production

- Development builds include source maps and debugging info
- Production builds are optimized and minified
- Use appropriate Dockerfile (Dockerfile.dev vs Dockerfile.prod)

### API Configuration

- Check REACT_APP_API_URL environment variable
- Verify nginx proxy configuration for backend communication
- Ensure CORS settings are correct

## Performance Optimization

### Build Speed

1. **Use Docker layer caching**:

   ```bash
   docker build --cache-from previous-image -t new-image .
   ```

2. **Optimize package.json**:
   - Remove unused dependencies
   - Use exact versions to avoid resolution conflicts

### Image Size

1. **Multi-stage builds** (already implemented)
2. **Minimize installed packages**
3. **Use alpine base images**

## Getting Help

If issues persist:

1. **Check logs**:

   ```bash
   docker logs container-name
   ```

2. **Run validation script**:

   ```bash
   ./validate-build.sh --full
   ```

3. **Review build output** for specific error messages

4. **Check system resources** (memory, disk space)

## Quick Fixes

### Reset Everything

```bash
# Clean Docker
docker system prune -a

# Clean npm
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Rebuild
./build-docker.sh --no-cache
```

### Emergency Build

```bash
# Minimal build without optimizations
docker build -f Dockerfile.dev -t frontend-emergency .
```
