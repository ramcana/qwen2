#!/bin/bash

# Traefik Setup Script
# This script initializes the required directories and files for Traefik configuration

set -e

echo "🚀 Setting up Traefik configuration..."

# Create required directories
echo "📁 Creating directories..."
mkdir -p config/traefik
mkdir -p ssl/certs
mkdir -p ssl/private
mkdir -p logs/traefik
mkdir -p data/traefik

# Set proper permissions for SSL directory
chmod 700 ssl
chmod 600 ssl/private 2>/dev/null || true

# Create ACME storage files
echo "🔐 Creating ACME storage files..."
touch ssl/acme.json
touch ssl/acme-dev.json
chmod 600 ssl/acme.json
chmod 600 ssl/acme-dev.json

# Generate self-signed certificates for development
echo "🔑 Generating self-signed certificates for development..."
if [ ! -f ssl/certs/localhost.crt ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/private/localhost.key \
        -out ssl/certs/localhost.crt \
        -subj "/C=US/ST=Local/L=Local/O=Development/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,DNS:*.localhost,DNS:app.localhost,DNS:api.localhost,DNS:traefik.localhost,IP:127.0.0.1"
    
    echo "✅ Self-signed certificate generated"
else
    echo "ℹ️  Self-signed certificate already exists"
fi

# Create network if it doesn't exist
echo "🌐 Creating Docker networks..."
docker network create qwen-network 2>/dev/null || echo "ℹ️  Network qwen-network already exists"
docker network create traefik-public 2>/dev/null || echo "ℹ️  Network traefik-public already exists"

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# Traefik Configuration
ACME_EMAIL=admin@localhost
TRAEFIK_DOMAIN=traefik.localhost
APP_DOMAIN=app.localhost
API_DOMAIN=api.localhost

# Production Configuration (uncomment for production)
# ACME_EMAIL=your-email@example.com
# TRAEFIK_DOMAIN=traefik.yourdomain.com
# APP_DOMAIN=app.yourdomain.com
# API_DOMAIN=api.yourdomain.com
# DNS_PROVIDER=cloudflare
# TRAEFIK_AUTH_USER=admin:\$2y\$10\$2b2cu/0P6dvFYOTwMZQo4OEbb.Npsb.2bJ65v2Oy7SHq6UfXVo0n2

# Optional: Jaeger tracing
# JAEGER_AGENT=jaeger:6831
EOF
    echo "✅ .env template created"
else
    echo "ℹ️  .env file already exists"
fi

# Set proper log directory permissions
chmod 755 logs/traefik

# Create basic auth file for dashboard (development)
echo "🔒 Creating basic auth for dashboard..."
if command -v htpasswd >/dev/null 2>&1; then
    # Use htpasswd if available
    htpasswd -cb config/traefik/.htpasswd admin admin 2>/dev/null || true
else
    # Fallback to openssl
    echo "admin:$(openssl passwd -apr1 admin)" > config/traefik/.htpasswd
fi

echo "✅ Traefik setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review and update .env file with your domain settings"
echo "2. For production, update ACME_EMAIL and domain settings"
echo "3. Start Traefik with: docker-compose -f docker-compose.traefik.yml up -d"
echo "4. Access dashboard at: https://traefik.localhost:8080 (admin/admin)"
echo ""
echo "🔧 Available commands:"
echo "  Development: docker-compose -f docker-compose.traefik.yml up -d"
echo "  Production:  docker-compose -f docker-compose.traefik.prod.yml up -d"
echo "  Logs:        docker-compose -f docker-compose.traefik.yml logs -f traefik"
echo "  Stop:        docker-compose -f docker-compose.traefik.yml down"