import axios from "axios";
import {
  AspectRatioResponse,
  GenerationRequest,
  GenerationResponse,
  ImageToImageRequest,
  MemoryResponse,
  QueueResponse,
  StatusResponse,
} from "../types/api";
import {
  apiConfig,
  getImageUrl as getConfigImageUrl,
  debugConfig,
} from "../config/api";

// Debug configuration in development
if (process.env.NODE_ENV === "development") {
  debugConfig();
}

const api = axios.create({
  baseURL: apiConfig.baseUrl,
  timeout: apiConfig.timeout,
  withCredentials: true, // Enable credentials for CORS
});

// Request interceptor for logging
api.interceptors.request.use((config) => {
  console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error(
      `❌ API Error: ${error.response?.status} ${error.config?.url}`,
      error.response?.data
    );
    return Promise.reject(error);
  }
);

export const getStatus = async (): Promise<StatusResponse> => {
  const response = await api.get("/status");
  return response.data;
};

export const initializeModel = async (): Promise<{
  success: boolean;
  message: string;
}> => {
  const response = await api.post("/initialize");
  return response.data;
};

export const generateTextToImage = async (
  request: GenerationRequest
): Promise<GenerationResponse> => {
  const response = await api.post("/generate/text-to-image", request);
  return response.data;
};

export const generateImageToImage = async (
  request: ImageToImageRequest
): Promise<GenerationResponse> => {
  const response = await api.post("/generate/image-to-image", request);
  return response.data;
};

export const getAspectRatios = async (): Promise<AspectRatioResponse> => {
  const response = await api.get("/aspect-ratios");
  return response.data;
};

export const getQueue = async (): Promise<QueueResponse> => {
  const response = await api.get("/queue");
  return response.data;
};

export const cancelJob = async (
  jobId: string
): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete(`/queue/${jobId}`);
  return response.data;
};

export const clearMemory = async (): Promise<MemoryResponse> => {
  const response = await api.get("/memory/clear");
  return response.data;
};

export const getMemoryStatus = async (): Promise<MemoryResponse> => {
  const response = await api.get("/memory/status");
  return response.data;
};

export const getImageUrl = (imagePath: string): string => {
  // Use the configuration-based image URL function
  return getConfigImageUrl(imagePath);
};

export const healthCheck = async (): Promise<any> => {
  const response = await api.get("/health");
  return response.data;
};
