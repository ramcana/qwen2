# =============================================================================
# Multi-stage optimized Dockerfile for Qwen2 React Frontend
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Dependencies (for better caching)
# -----------------------------------------------------------------------------
FROM node:18-alpine AS dependencies

# Set working directory
WORKDIR /app

# Install system dependencies for better performance
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    curl \
    && rm -rf /var/cache/apk/*

# Copy package files for dependency caching
COPY package*.json ./
COPY .npmrc* ./

# Configure npm for optimal performance and security
RUN npm config set registry https://registry.npmjs.org/ && \
    npm config set fetch-retries 3 && \
    npm config set fetch-retry-factor 2 && \
    npm config set fetch-retry-mintimeout 10000 && \
    npm config set fetch-retry-maxtimeout 60000 && \
    npm config set audit-level moderate && \
    npm config set fund false

# Install all dependencies with caching optimization
RUN --mount=type=cache,target=/root/.npm \
    --mount=type=cache,target=/app/.npm \
    npm ci --legacy-peer-deps --no-audit --no-fund --prefer-offline

# -----------------------------------------------------------------------------
# Stage 2: Build (optimized for different environments)
# -----------------------------------------------------------------------------
FROM node:18-alpine AS builder

WORKDIR /app

# Install build-time dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    && rm -rf /var/cache/apk/*

# Copy dependencies from previous stage
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=dependencies /app/package*.json ./

# Copy source code (use .dockerignore to exclude unnecessary files)
COPY . .

# Build arguments for environment-specific optimizations
ARG NODE_ENV=production
ARG REACT_APP_API_URL=/api
ARG REACT_APP_WS_URL=/ws
ARG BUILD_OPTIMIZATION=true
ARG GENERATE_SOURCEMAP=false
ARG BUILD_PATH=build

# Set environment variables for build optimization
ENV NODE_ENV=${NODE_ENV}
ENV REACT_APP_API_URL=${REACT_APP_API_URL}
ENV REACT_APP_WS_URL=${REACT_APP_WS_URL}
ENV GENERATE_SOURCEMAP=${GENERATE_SOURCEMAP}
ENV BUILD_PATH=${BUILD_PATH}
ENV CI=true
ENV INLINE_RUNTIME_CHUNK=false

# Optimize build based on environment with memory management
RUN if [ "$BUILD_OPTIMIZATION" = "true" ]; then \
        # Production optimizations with memory limits
        export NODE_OPTIONS="--max-old-space-size=4096" && \
        npm run build -- --production; \
    else \
        # Development build
        npm run build; \
    fi

# Remove source maps in production for security
RUN if [ "$NODE_ENV" = "production" ] && [ "$GENERATE_SOURCEMAP" = "false" ]; then \
        find ${BUILD_PATH} -name "*.map" -type f -delete; \
    fi

# Pre-compress static assets for better performance
RUN find ${BUILD_PATH} -type f \( -name "*.js" -o -name "*.css" -o -name "*.html" -o -name "*.json" \) \
    -exec gzip -k -9 {} \; && \
    find ${BUILD_PATH} -type f \( -name "*.js" -o -name "*.css" -o -name "*.html" -o -name "*.json" \) \
    -exec brotli -k -q 11 {} \; 2>/dev/null || true

# Create health check file
RUN echo 'healthy' > ${BUILD_PATH}/health

# -----------------------------------------------------------------------------
# Stage 3: Production Runtime with Nginx
# -----------------------------------------------------------------------------
FROM nginx:1.25-alpine AS production

# Build arguments for runtime configuration
ARG BUILD_PATH=build

# Install additional tools for health checks, security, and compression
RUN apk add --no-cache \
    curl \
    ca-certificates \
    tzdata \
    brotli \
    && rm -rf /var/cache/apk/*

# Create nginx user and set permissions for security
RUN addgroup -g 1001 -S nginx-app && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx-app -g nginx-app nginx-app

# Copy built application from builder stage with proper ownership
COPY --from=builder --chown=nginx-app:nginx-app /app/${BUILD_PATH} /usr/share/nginx/html

# Copy API error page for backend connection issues
COPY --chown=nginx-app:nginx-app api_error.html /usr/share/nginx/html/api_error.html

# Copy nginx configurations with proper ownership
COPY --chown=nginx-app:nginx-app nginx.prod.conf /etc/nginx/nginx.conf
COPY --chown=nginx-app:nginx-app nginx-security.conf /etc/nginx/conf.d/security.conf

# Create necessary directories and set proper permissions
RUN mkdir -p /var/cache/nginx /var/log/nginx /var/run /tmp/nginx && \
    chown -R nginx-app:nginx-app /var/cache/nginx /var/log/nginx /var/run /tmp/nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html && \
    chmod 755 /var/cache/nginx /var/log/nginx /var/run /tmp/nginx

# Update nginx configuration for container environment
RUN sed -i 's/user nginx;/user nginx-app;/' /etc/nginx/nginx.conf && \
    sed -i 's/api:8000/qwen-api:8000/g' /etc/nginx/nginx.conf && \
    sed -i 's|/var/run/nginx.pid|/tmp/nginx/nginx.pid|g' /etc/nginx/nginx.conf

# Security: Remove default nginx files and set proper permissions
RUN rm -rf /etc/nginx/conf.d/default.conf /usr/share/nginx/html/index.html && \
    chmod 644 /etc/nginx/nginx.conf /etc/nginx/conf.d/security.conf

# Update nginx configuration for container environment
RUN sed -i 's|pid /var/run/nginx.pid;|pid /tmp/nginx/nginx.pid;|g' /etc/nginx/nginx.conf

# Copy health check script
COPY --chown=nginx-app:nginx-app health-check.js /usr/local/bin/health-check.js
RUN chmod +x /usr/local/bin/health-check.js

# Health check configuration with comprehensive checking
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD node /usr/local/bin/health-check.js simple || exit 1

# Labels for better container management and metadata
LABEL maintainer="Qwen2 Team" \
      version="2.0.0" \
      description="Optimized React frontend for Qwen2 Image Generation with multi-stage build" \
      org.opencontainers.image.title="Qwen2 Frontend" \
      org.opencontainers.image.description="Multi-stage optimized React frontend with Nginx, compression, and security headers" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="Qwen2 Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/qwen2/image-generation"

# Expose port
EXPOSE 80

# Switch to non-root user for security
USER nginx-app

# Start nginx with optimized configuration and error handling
CMD ["sh", "-c", "nginx -t && nginx -g 'daemon off;'"]