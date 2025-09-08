import axios from 'axios';
import {
    AspectRatioResponse,
    GenerationRequest,
    GenerationResponse,
    QueueResponse,
    StatusResponse
} from '../types/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes timeout for generation requests
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
    console.error(`❌ API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
    return Promise.reject(error);
  }
);

export const getStatus = async (): Promise<StatusResponse> => {
  const response = await api.get('/status');
  return response.data;
};

export const initializeModel = async (): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/initialize');
  return response.data;
};

export const generateTextToImage = async (request: GenerationRequest): Promise<GenerationResponse> => {
  const response = await api.post('/generate/text-to-image', request);
  return response.data;
};

export const getAspectRatios = async (): Promise<AspectRatioResponse> => {
  const response = await api.get('/aspect-ratios');
  return response.data;
};

export const getQueue = async (): Promise<QueueResponse> => {
  const response = await api.get('/queue');
  return response.data;
};

export const cancelJob = async (jobId: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete(`/queue/${jobId}`);
  return response.data;
};

export const clearMemory = async (): Promise<{ success: boolean; message: string; memory_info: any }> => {
  const response = await api.get('/memory/clear');
  return response.data;
};

export const getImageUrl = (imagePath: string): string => {
  // Extract filename from path
  const filename = imagePath.split('/').pop() || imagePath;
  return `${API_BASE_URL}/images/${filename}`;
};

export const healthCheck = async (): Promise<any> => {
  const response = await api.get('/health');
  return response.data;
};