# Traefik Configuration for Qwen2 Docker Containerization

This directory contains the Traefik reverse proxy configuration for the Qwen2 image generation application. Traefik provides automatic service discovery, load balancing, SSL termination, and monitoring capabilities.

## 📁 File Structure

```
config/traefik/
├── README.md                 # This documentation
├── traefik.yml              # Main Traefik configuration (development)
├── traefik.prod.yml         # Production Traefik configuration
├── dynamic.yml              # Dynamic routing and middleware (development)
├── dynamic.prod.yml         # Production dynamic configuration
└── .htpasswd               # Basic auth credentials (generated)
```

## 🚀 Quick Start

### Development Setup

1. **Initialize Traefik configuration:**

   ```bash
   # Linux/macOS
   ./scripts/setup_traefik.sh

   # Windows PowerShell
   .\scripts\setup_traefik.ps1
   ```

2. **Start Traefik:**

   ```bash
   docker-compose -f docker-compose.traefik.yml up -d
   ```

3. **Access services:**
   - Dashboard: https://traefik.localhost:8080 (admin/admin)
   - Application: https://app.localhost
   - API: https://api.localhost

### Production Setup

1. **Configure environment variables:**

   ```bash
   # Update .env file with your domain settings
   ACME_EMAIL=your-email@example.com
   TRAEFIK_DOMAIN=traefik.yourdomain.com
   APP_DOMAIN=app.yourdomain.com
   API_DOMAIN=api.yourdomain.com
   ```

2. **Start production Traefik:**
   ```bash
   docker-compose -f docker-compose.traefik.prod.yml up -d
   ```

## 🔧 Configuration Details

### Main Configuration (traefik.yml)

The main configuration file defines:

- **Entry Points**: HTTP (80), HTTPS (443), Dashboard (8080)
- **Certificate Resolvers**: Let's Encrypt for automatic SSL
- **Providers**: Docker service discovery and file-based configuration
- **Logging**: Structured JSON logging with access logs
- **Metrics**: Prometheus metrics collection

### Dynamic Configuration (dynamic.yml)

The dynamic configuration includes:

- **Routers**: URL routing rules for services
- **Services**: Load balancer configuration with health checks
- **Middleware**: Security headers, rate limiting, CORS, compression
- **TLS**: SSL/TLS security settings

### Key Features

#### 🔒 Security Features

- **Automatic HTTPS**: Let's Encrypt certificate management
- **Security Headers**: HSTS, CSP, XSS protection, etc.
- **Rate Limiting**: Configurable rate limits per service
- **Basic Authentication**: Dashboard protection
- **CORS**: Cross-origin resource sharing configuration

#### ⚖️ Load Balancing

- **Health Checks**: Automatic health monitoring
- **Sticky Sessions**: Session affinity for API services
- **Multiple Backends**: Support for horizontal scaling
- **Circuit Breaker**: Automatic failure recovery

#### 📊 Monitoring

- **Dashboard**: Web-based management interface
- **Metrics**: Prometheus metrics endpoint
- **Access Logs**: Detailed request logging
- **Tracing**: Optional Jaeger integration

## 🌐 Service Discovery

Traefik automatically discovers services through Docker labels:

```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.app.rule=Host(`app.localhost`)"
  - "traefik.http.routers.app.entrypoints=websecure"
  - "traefik.http.routers.app.tls.certresolver=letsencrypt"
```

## 🔐 SSL Certificate Management

### Development (Self-Signed)

- Certificates generated automatically by setup script
- Stored in `ssl/certs/` and `ssl/private/`
- Valid for `*.localhost` domains

### Production (Let's Encrypt)

- Automatic certificate provisioning and renewal
- HTTP-01 challenge (default) or DNS-01 challenge
- Certificates stored in `ssl/acme.json`

### Custom Certificates

Place custom certificates in:

- `ssl/certs/yourdomain.crt`
- `ssl/private/yourdomain.key`

## 📈 Monitoring and Metrics

### Prometheus Metrics

Available at: `https://traefik.yourdomain.com/metrics`

Key metrics:

- Request duration and count
- Response status codes
- Service health status
- Certificate expiration

### Dashboard

Access the Traefik dashboard at:

- Development: https://traefik.localhost:8080
- Production: https://traefik.yourdomain.com:8080

Default credentials: `admin/admin` (change in production)

## 🔧 Customization

### Environment Variables

| Variable            | Description                 | Default             |
| ------------------- | --------------------------- | ------------------- |
| `ACME_EMAIL`        | Email for Let's Encrypt     | `admin@localhost`   |
| `TRAEFIK_DOMAIN`    | Traefik dashboard domain    | `traefik.localhost` |
| `APP_DOMAIN`        | Application domain          | `app.localhost`     |
| `API_DOMAIN`        | API domain                  | `api.localhost`     |
| `DNS_PROVIDER`      | DNS provider for challenges | -                   |
| `TRAEFIK_AUTH_USER` | Dashboard auth credentials  | `admin:hash`        |

### Adding New Services

1. **Add Docker labels to your service:**

   ```yaml
   labels:
     - "traefik.enable=true"
     - "traefik.http.routers.myservice.rule=Host(`myservice.localhost`)"
     - "traefik.http.routers.myservice.entrypoints=websecure"
     - "traefik.http.routers.myservice.tls.certresolver=letsencrypt"
   ```

2. **Configure middleware (optional):**
   ```yaml
   labels:
     - "traefik.http.routers.myservice.middlewares=security-headers@file,rate-limit@file"
   ```

### Custom Middleware

Add custom middleware to `dynamic.yml`:

```yaml
http:
  middlewares:
    my-middleware:
      headers:
        customRequestHeaders:
          X-Custom-Header: "value"
```

## 🚨 Troubleshooting

### Common Issues

1. **Certificate errors:**

   ```bash
   # Check certificate status
   docker-compose -f docker-compose.traefik.yml logs traefik | grep -i cert

   # Reset ACME storage
   rm ssl/acme.json && touch ssl/acme.json && chmod 600 ssl/acme.json
   ```

2. **Service not found:**

   ```bash
   # Check service discovery
   docker-compose -f docker-compose.traefik.yml logs traefik | grep -i "service"

   # Verify Docker labels
   docker inspect <container_name> | grep -A 10 Labels
   ```

3. **Dashboard access issues:**

   ```bash
   # Check dashboard configuration
   curl -k https://traefik.localhost:8080/api/rawdata

   # Verify authentication
   cat config/traefik/.htpasswd
   ```

### Debug Mode

Enable debug logging in `traefik.yml`:

```yaml
log:
  level: DEBUG
api:
  debug: true
```

### Health Checks

Check Traefik health:

```bash
# Container health
docker-compose -f docker-compose.traefik.yml ps

# Service health
curl -k https://traefik.localhost:8080/ping

# API health
curl -k https://api.localhost/health
```

## 📚 Additional Resources

- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [Docker Provider](https://doc.traefik.io/traefik/providers/docker/)
- [Let's Encrypt](https://doc.traefik.io/traefik/https/acme/)
- [Middleware Reference](https://doc.traefik.io/traefik/middlewares/overview/)

## 🔄 Updates and Maintenance

### Updating Traefik

1. **Update image version in docker-compose files**
2. **Review configuration changes in release notes**
3. **Test in development environment first**
4. **Deploy to production with rolling update**

### Certificate Renewal

Let's Encrypt certificates are automatically renewed. Monitor logs for renewal status:

```bash
docker-compose -f docker-compose.traefik.yml logs traefik | grep -i "renew"
```

### Backup

Important files to backup:

- `ssl/acme.json` (certificates)
- `config/traefik/` (configuration)
- `.env` (environment variables)

## 📞 Support

For issues related to this Traefik configuration:

1. Check the troubleshooting section above
2. Review Traefik logs for error messages
3. Verify Docker service labels and network configuration
4. Consult the official Traefik documentation
