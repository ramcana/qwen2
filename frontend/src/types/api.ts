export interface GenerationRequest {
  prompt: string;
  negative_prompt?: string;
  width: number;
  height: number;
  num_inference_steps: number;
  cfg_scale: number;
  seed: number;
  language: 'en' | 'zh';
  enhance_prompt: boolean;
  aspect_ratio?: string;
}

export interface ImageToImageRequest extends GenerationRequest {
  init_image_path: string;
  strength: number;
}

export interface GenerationResponse {
  success: boolean;
  image_path?: string;
  message: string;
  generation_time?: number;
  parameters?: GenerationParameters;
  job_id?: string;
}

export interface GenerationParameters {
  prompt: string;
  negative_prompt?: string;
  width: number;
  height: number;
  num_inference_steps: number;
  cfg_scale: number;
  seed: number;
  language: 'en' | 'zh';
  enhance_prompt: boolean;
  aspect_ratio?: string;
}

export interface StatusResponse {
  model_loaded: boolean;
  device: string;
  memory_info?: {
    total_memory: number;
    allocated_memory: number;
    cached_memory: number;
    free_memory: number;
    device_name?: string;
    total_memory_gb?: number;
    allocated_memory_gb?: number;
    free_memory_gb?: number;
    memory_usage_percent?: number;
  };
  current_generation?: string | null;
  queue_length: number;
  initialization?: {
    status: string;
    message: string;
    progress: number;
    model_loaded: boolean;
    error?: string;
  };
}

export interface AspectRatioResponse {
  ratios: Record<string, [number, number]>;
}

export interface QueueItem {
  job_id: string;
  type: 'text-to-image' | 'image-to-image';
  request: GenerationRequest | ImageToImageRequest;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  image_path?: string;
  generation_time?: number;
  error?: string;
  message?: string;
}

export interface QueueResponse {
  queue: Record<string, QueueItem>;
  current: string | null;
}

export interface MemoryInfo {
  device_name?: string;
  total_memory: number;
  allocated_memory: number;
  cached_memory: number;
  free_memory: number;
  total_memory_gb?: number;
  allocated_memory_gb?: number;
  cached_memory_gb?: number;
  free_memory_gb?: number;
  memory_usage_percent?: number;
  freed_allocated?: number;
  freed_cached?: number;
}

export interface MemoryResponse {
  success: boolean;
  message: string;
  memory_info?: MemoryInfo;
}