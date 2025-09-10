import { AlertCircle, CheckCircle, Loader, Play, Trash2, XCircle } from 'lucide-react';
import React from 'react';
import { toast } from 'react-hot-toast';
import { useMutation } from 'react-query';
import { clearMemory, initializeModel } from '../services/api';
import { StatusResponse } from '../types/api';

interface StatusBarProps {
  status?: StatusResponse;
  isLoading: boolean;
}

const StatusBar: React.FC<StatusBarProps> = ({ status, isLoading }) => {
  const initializeMutation = useMutation(initializeModel, {
    onSuccess: () => {
      toast.success('Model initialized successfully!');
      window.location.reload();
    },
    onError: (error: any) => {
      toast.error(`Initialization failed: ${error.response?.data?.detail || error.message}`);
    }
  });

  const clearMemoryMutation = useMutation(clearMemory, {
    onSuccess: (data) => {
      toast.success(`Memory cleared! Available: ${data.memory_info?.available_gb || 'N/A'}GB`);
    },
    onError: (error: any) => {
      toast.error(`Memory clear failed: ${error.response?.data?.detail || error.message}`);
    }
  });

  if (isLoading) {
    return (
      <div className="card p-4">
        <div className="flex items-center space-x-3">
          <Loader className="w-5 h-5 animate-spin text-blue-500" />
          <span className="text-gray-600">Connecting to API...</span>
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
              <span>Total: {status.memory_info.total_gb}GB</span>
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
};

export default StatusBar;
