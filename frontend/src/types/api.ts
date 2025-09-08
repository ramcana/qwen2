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

export interface GenerationResponse {
  success: boolean;
  image_path?: string;
  message: string;
  generation_time?: number;
  parameters?: any;
  job_id?: string;
}

export interface StatusResponse {
  model_loaded: boolean;
  gpu_available: boolean;
  memory_info?: {
    allocated_gb: number;
    total_gb: number;
    usage_percent: number;
    device_name?: string;
  };
  queue_size: number;
  is_generating: boolean;
}

export interface AspectRatioResponse {
  ratios: Record<string, [number, number]>;
}

export interface QueueItem {
  job_id: string;
  type: string;
  request: any;
  timestamp: string;
}

export interface QueueResponse {
  queue_size: number;
  is_generating: boolean;
  queue: QueueItem[];
}