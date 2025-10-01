/**
 * API Configuration for Container Communication
 *
 * This configuration handles different environments and container communication patterns:
 * - Development: Direct API calls to localhost
 * - Production: Proxied calls through nginx
 * - Container: Internal container-to-container communication
 */

export interface ApiConfig {
  baseUrl: string;
  wsUrl: string;
  timeout: number;
  retries: number;
  backendHost: string;
  backendPort: number;
}

// Environment detection
const isDevelopment = process.env.NODE_ENV === "development";
const isContainer = process.env.REACT_APP_CONTAINER_MODE === "true";

// Default configuration for different environments
const getApiConfig = (): ApiConfig => {
  // Container environment (production with nginx proxy)
  if (
    isContainer ||
    (!isDevelopment && !process.env.REACT_APP_API_URL?.startsWith("http"))
  ) {
    return {
      baseUrl: process.env.REACT_APP_API_URL || "/api",
      wsUrl: process.env.REACT_APP_WS_URL || "/ws",
      timeout: 120000, // 2 minutes
      retries: 3,
      backendHost: process.env.REACT_APP_BACKEND_HOST || "qwen-api",
      backendPort: parseInt(process.env.REACT_APP_BACKEND_PORT || "8000", 10),
    };
  }

  // Development environment (direct API calls)
  if (isDevelopment) {
    return {
      baseUrl: process.env.REACT_APP_API_URL || "http://localhost:8000/api",
      wsUrl: process.env.REACT_APP_WS_URL || "ws://localhost:8000/ws",
      timeout: 120000,
      retries: 3,
      backendHost: "localhost",
      backendPort: 8000,
    };
  }

  // Production fallback
  return {
    baseUrl: process.env.REACT_APP_API_URL || "/api",
    wsUrl: process.env.REACT_APP_WS_URL || "/ws",
    timeout: 120000,
    retries: 3,
    backendHost: process.env.REACT_APP_BACKEND_HOST || "qwen-api",
    backendPort: parseInt(process.env.REACT_APP_BACKEND_PORT || "8000", 10),
  };
};

export const apiConfig = getApiConfig();

// Helper functions for URL construction
export const getApiUrl = (endpoint: string): string => {
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint.slice(1) : endpoint;
  const cleanBaseUrl = apiConfig.baseUrl.endsWith("/")
    ? apiConfig.baseUrl.slice(0, -1)
    : apiConfig.baseUrl;
  return `${cleanBaseUrl}/${cleanEndpoint}`;
};

export const getImageUrl = (imagePath: string): string => {
  const filename = imagePath.split("/").pop() || imagePath;
  return getApiUrl(`images/${filename}`);
};

export const getWebSocketUrl = (): string => {
  return apiConfig.wsUrl;
};

// CORS configuration for different environments
export const getCorsConfig = () => {
  if (isDevelopment) {
    return {
      credentials: true,
      origin: ["http://localhost:3000", "http://127.0.0.1:3000"],
    };
  }

  return {
    credentials: true,
    origin: true, // Allow all origins in container environment (handled by nginx)
  };
};

// Health check configuration
export const healthCheckConfig = {
  interval: 30000, // 30 seconds
  timeout: 10000, // 10 seconds
  retries: 3,
  endpoints: ["/health", "/health/live", "/status"],
};

// Export configuration for debugging
export const debugConfig = () => {
  console.log("🔧 API Configuration:", {
    environment: process.env.NODE_ENV,
    isDevelopment,
    isContainer,
    config: apiConfig,
  });
};
