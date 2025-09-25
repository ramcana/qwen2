import { AlertCircle, CheckCircle, Loader, Play, Trash2, XCircle, Download, Cpu, HardDrive } from 'lucide-react';
import React from 'react';
import { toast } from 'react-hot-toast';
import { useMutation } from 'react-query';
import { clearMemory, initializeModel } from '../services/api';
import { StatusResponse } from '../types/api';

interface StatusBarProps {
  status?: StatusResponse;
  isLoading: boolean;
  error?: any;
  onRetry?: () => void;
}

const StatusBar: React.FC<StatusBarProps> = React.memo(({ status, isLoading, error, onRetry }) => {
  const initializeMutation = useMutation(initializeModel, {
    onSuccess: (data) => {
      toast.success('Model initialized successfully!');
      if (onRetry) onRetry(); // Refresh status instead of full reload
    },
    onError: (error: any) => {
      toast.error(`Initialization failed: ${error.response?.data?.detail || error.message}`);
    }
  });

  const clearMemoryMutation = useMutation(clearMemory, {
    onSuccess: (data) => {
      const freedMB = data.memory_info?.freed_allocated ? (data.memory_info.freed_allocated / 1e6).toFixed(0) : 'N/A';
      toast.success(`Memory cleared! Freed ${freedMB}MB`);
      if (onRetry) onRetry(); // Refresh status
    },
    onError: (error: any) => {
      toast.error(`Memory clear failed: ${error.response?.data?.detail || error.message}`);
    }
  });

  // Show detailed initialization status
  if (isLoading || (status?.initialization && status.initialization.status !== 'ready' && status.initialization.status !== 'model_ready')) {
    const init = status?.initialization;
    
    return (
      <div className="card p-4 border-blue-200 bg-blue-50">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {init?.status === 'loading_model' || init?.status === 'downloading_model' || init?.status === 'loading_to_gpu' ? (
                <Download className="w-5 h-5 animate-pulse text-blue-600" />
              ) : (
                <Loader className="w-5 h-5 animate-spin text-blue-600" />
              )}
              <div>
                <span className="text-blue-900 font-medium">
                  {init?.message || 'Initializing Qwen Model...'}
                </span>
                <p className="text-sm text-blue-700">
                  {init?.status === 'downloading_model' ? 'Downloading model files (first time only)' :
                   init?.status === 'loading_to_gpu' ? 'Loading model to GPU memory' :
                   'This may take a few minutes on first startup'}
                </p>
              </div>
            </div>
            {onRetry && (
              <button
                onClick={onRetry}
                className="btn btn-secondary px-3 py-1 text-sm"
              >
                Retry
              </button>
            )}
          </div>
          
          {/* Progress bar */}
          {init?.progress !== undefined && (
            <div className="space-y-1">
              <div className="flex justify-between text-sm text-blue-700">
                <span>Progress</span>
                <span>{init.progress}%</span>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${init.progress}%` }}
                />
              </div>
            </div>
          )}
          
          {/* Status indicators */}
          <div className="flex items-center space-x-4 text-xs text-blue-600">
            <div className="flex items-center space-x-1">
              <Cpu className="w-3 h-3" />
              <span>GPU: {status?.device || 'Detecting...'}</span>
            </div>
            {status?.memory_info?.device_name && (
              <div className="flex items-center space-x-1">
                <HardDrive className="w-3 h-3" />
                <span>{status.memory_info.device_name}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="card p-4 border-red-200 bg-red-50">
        <div className="flex items-center space-x-3">
          <XCircle className="w-5 h-5 text-red-500" />
          <span className="text-red-700">API Connection Failed</span>
        </div>
      </div>
    );
  }

  const getStatusColor = () => {
    if (!status.gpu_available) return 'red';
    if (!status.model_loaded) return 'yellow';
    return 'green';
  };

  const getStatusIcon = () => {
    const color = getStatusColor();
    if (color === 'red') return <XCircle className="w-5 h-5 text-red-500" />;
    if (color === 'yellow') return <AlertCircle className="w-5 h-5 text-yellow-500" />;
    return <CheckCircle className="w-5 h-5 text-green-500" />;
  };

  const getMemoryColor = () => {
    if (!status.memory_info) return 'gray';
    const usage = status.memory_info.usage_percent;
    if (usage > 85) return 'red';
    if (usage > 70) return 'yellow';
    return 'green';
  };

  return (
    <div className="space-y-4">
      {/* System Status */}
      <div className="card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <div>
              <h3 className="font-semibold text-gray-900">System Status</h3>
              <p className="text-sm text-gray-600">
                {status.model_loaded ? 'Ready for generation' : 'Model not loaded'}
              </p>
            </div>
          </div>
          
          {!status.model_loaded && (
            <button
              onClick={() => initializeMutation.mutate()}
              disabled={initializeMutation.isLoading}
              className="btn btn-primary px-3 py-2"
            >
              {initializeMutation.isLoading ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              <span className="ml-1">Initialize</span>
            </button>
          )}
        </div>
      </div>

      {/* GPU Memory */}
      {status.memory_info && (
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-900">GPU Memory</h3>
            <button
              onClick={() => clearMemoryMutation.mutate()}
              disabled={clearMemoryMutation.isLoading}
              className="btn btn-secondary px-2 py-1 text-xs"
            >
              {clearMemoryMutation.isLoading ? (
                <Loader className="w-3 h-3 animate-spin" />
              ) : (
                <Trash2 className="w-3 h-3" />
              )}
              <span className="ml-1">Clear</span>
            </button>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Used: {status.memory_info.allocated_gb}GB</span>
              <span>Available: {status.memory_info.available_gb || 'N/A'}GB</span>
            </div>
            <div className="text-xs text-gray-500 text-center">
              Total: {status.memory_info.total_gb}GB
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  getMemoryColor() === 'red' ? 'bg-red-500' :
                  getMemoryColor() === 'yellow' ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${status.memory_info.usage_percent}%` }}
              />
            </div>
            
            <div className="flex justify-between text-xs text-gray-500">
              <span>{status.memory_info.usage_percent}% used</span>
              {status.memory_info.device_name && (
                <span>{status.memory_info.device_name}</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Queue Status */}
      {(status.queue_size > 0 || status.is_generating) && (
        <div className="card p-4">
          <h3 className="font-semibold text-gray-900 mb-2">Generation Queue</h3>
          <div className="space-y-2">
            {status.is_generating && (
              <div className="flex items-center space-x-2 text-sm">
                <Loader className="w-4 h-4 animate-spin text-blue-500" />
                <span className="text-blue-600">Currently generating...</span>
              </div>
            )}
            {status.queue_size > 0 && (
              <div className="text-sm text-gray-600">
                {status.queue_size} request{status.queue_size !== 1 ? 's' : ''} in queue
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

export default StatusBar;