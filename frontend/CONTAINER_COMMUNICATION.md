# Frontend-Backend Container Communication

This document explains how the frontend and backend containers communicate in the Qwen2 Image Generation system.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Browser       │    │   Frontend      │    │   Backend       │
│                 │    │   Container     │    │   Container     │
│                 │    │   (nginx)       │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │ HTTP/WebSocket        │ Container Network     │
         │ requests              │ (qwen-network)        │
         └───────────────────────┼───────────────────────┘
                                 │
                                 │ Proxy to
                                 │ qwen-api:8000
                                 │
```

## Communication Flow

### 1. Browser to Frontend Container

- **Protocol**: HTTP/HTTPS
- **Port**: 80 (HTTP) / 443 (HTTPS)
- **Routing**: Handled by Traefik reverse proxy

### 2. Frontend to Backend (via Nginx Proxy)

- **Internal Network**: `qwen-network`
- **Backend Service**: `qwen-api:8000`
- **Proxy Configuration**: nginx.prod.conf

## Configuration Files

### Environment Variables

#### Production Container Environment

```bash
REACT_APP_CONTAINER_MODE=true
REACT_APP_API_URL=/api
REACT_APP_WS_URL=/ws
REACT_APP_BACKEND_HOST=qwen-api
REACT_APP_BACKEND_PORT=8000
```

#### Development Environment

```bash
REACT_APP_API_URL=http://qwen-api:8000/api
REACT_APP_WS_URL=ws://qwen-api:8000/ws
REACT_APP_BACKEND_HOST=qwen-api
REACT_APP_BACKEND_PORT=8000
```

### Nginx Proxy Configuration

The nginx configuration in `nginx.prod.conf` handles:

1. **API Proxying**: Routes `/api/*` requests to `qwen-api:8000`
2. **WebSocket Support**: Routes `/ws` connections to `qwen-api:8000/ws`
3. **CORS Headers**: Enables cross-origin requests
4. **Error Handling**: Shows custom error pages for backend issues

#### Key Proxy Settings

```nginx
# API proxy with CORS support
location /api/ {
    proxy_pass http://qwen-api:8000/;
    proxy_http_version 1.1;

    # CORS headers
    add_header Access-Control-Allow-Origin $cors_origin always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS, PATCH" always;
    add_header Access-Control-Allow-Headers "Accept, Authorization, Cache-Control, Content-Type, DNT, If-Modified-Since, Keep-Alive, Origin, User-Agent, X-Requested-With, X-CSRF-Token, X-API-Key" always;
    add_header Access-Control-Allow-Credentials true always;

    # Handle preflight OPTIONS requests
    if ($request_method = 'OPTIONS') {
        return 204;
    }
}
```

## API Client Configuration

The frontend uses a configuration-based approach for API communication:

### Configuration Service (`src/config/api.ts`)

```typescript
export const apiConfig = {
  baseUrl: process.env.REACT_APP_API_URL || "/api",
  wsUrl: process.env.REACT_APP_WS_URL || "/ws",
  timeout: 120000,
  retries: 3,
  backendHost: process.env.REACT_APP_BACKEND_HOST || "qwen-api",
  backendPort: parseInt(process.env.REACT_APP_BACKEND_PORT || "8000", 10),
};
```

### Axios Configuration

```typescript
const api = axios.create({
  baseURL: apiConfig.baseUrl,
  timeout: apiConfig.timeout,
  withCredentials: true, // Enable credentials for CORS
});
```

## CORS Configuration

### Frontend CORS Settings

- **Credentials**: Enabled (`withCredentials: true`)
- **Origins**: Configured via nginx proxy
- **Methods**: GET, POST, PUT, DELETE, OPTIONS, PATCH
- **Headers**: Standard headers plus custom API headers

### Backend CORS Requirements

The backend should be configured to accept requests from:

- Frontend container hostname
- Nginx proxy headers
- WebSocket upgrade requests

## Error Handling

### Backend Connection Errors

When the backend is unavailable, nginx serves a custom error page (`api_error.html`) that:

- Shows a user-friendly error message
- Automatically retries every 30 seconds
- Provides manual retry option

### Health Checks

Multiple health check endpoints are available:

- `/health` - Frontend container health
- `/api/health` - Backend health through proxy
- `/api/status` - Backend status information

## Testing Container Communication

Use the provided test script to validate communication:

```bash
# Test from within frontend container
node test-container-communication.js

# Test with custom configuration
node test-container-communication.js --api-host localhost --api-port 8000
```

## Troubleshooting

### Common Issues

1. **CORS Errors**

   - Check nginx CORS configuration
   - Verify `withCredentials` setting in axios
   - Ensure backend accepts preflight requests

2. **Connection Refused**

   - Verify backend container is running
   - Check Docker network configuration
   - Validate service names in docker-compose

3. **Timeout Errors**
   - Increase nginx proxy timeouts
   - Check backend response times
   - Verify network connectivity

### Debug Commands

```bash
# Check container network
docker network inspect qwen-network

# Test internal connectivity
docker exec qwen-frontend curl -f http://qwen-api:8000/health

# Check nginx configuration
docker exec qwen-frontend nginx -t

# View nginx logs
docker logs qwen-frontend

# View backend logs
docker logs qwen-api
```

## Security Considerations

1. **Network Isolation**: Containers communicate on internal Docker network
2. **CORS Restrictions**: Origins are validated by nginx
3. **No Direct Backend Access**: Backend is not exposed to external network
4. **Secure Headers**: Security headers are added by nginx
5. **Rate Limiting**: API requests are rate-limited by nginx

## Performance Optimizations

1. **Connection Pooling**: HTTP/1.1 with keep-alive
2. **Compression**: Gzip/Brotli compression enabled
3. **Caching**: Static assets cached with appropriate headers
4. **Buffering**: Disabled for real-time WebSocket connections
5. **Timeouts**: Optimized for different request types

## Monitoring

The system includes monitoring capabilities:

- Health check endpoints
- Request/response logging
- Error tracking
- Performance metrics
- Container status monitoring

For more detailed monitoring, integrate with Prometheus and Grafana using the provided configurations in the `monitoring/` directory.
