import {
  AlertCircle,
  CheckCircle,
  Loader,
  Play,
  Trash2,
  XCircle,
  Download,
  Cpu,
  HardDrive,
  Clock,
} from "lucide-react";
import React from "react";
import { toast } from "react-hot-toast";
import { useMutation } from "@tanstack/react-query";
import { clearMemory, initializeModel } from "../services/api";
import { StatusResponse } from "../types/api";

interface StatusBarProps {
  status?: StatusResponse;
  isLoading: boolean;
  error?: any;
  onRetry?: () => void;
}

const StatusBar: React.FC<StatusBarProps> = React.memo(
  ({ status, isLoading, error, onRetry }) => {
    const initializeMutation = useMutation(initializeModel, {
      onSuccess: (data) => {
        toast.success("Model initialized successfully!");
        if (onRetry) onRetry(); // Refresh status instead of full reload
      },
      onError: (error: any) => {
        toast.error(
          `Initialization failed: ${
            error.response?.data?.detail || error.message
          }`
        );
      },
    });

    const clearMemoryMutation = useMutation(clearMemory, {
      onSuccess: (data) => {
        const freedMB = data.memory_info?.freed_allocated
          ? (data.memory_info.freed_allocated / 1e6).toFixed(0)
          : "N/A";
        toast.success(`Memory cleared! Freed ${freedMB}MB`);
        if (onRetry) onRetry(); // Refresh status
      },
      onError: (error: any) => {
        toast.error(
          `Memory clear failed: ${
            error.response?.data?.detail || error.message
          }`
        );
      },
    });

    // Show detailed initialization status
    if (
      isLoading ||
      (status?.initialization &&
        status.initialization.status !== "ready" &&
        status.initialization.status !== "model_ready")
    ) {
      const init = status?.initialization;

      const getInitializationSteps = () => {
        const steps = [
          {
            key: "starting",
            label: "Starting API Server",
            completed: true,
            description: "Backend services are online",
          },
          {
            key: "initializing",
            label: "Initializing Services",
            completed: (init?.progress || 0) >= 10,
            description: "Setting up AI processing pipeline",
          },
          {
            key: "creating_directories",
            label: "Preparing Storage",
            completed: (init?.progress || 0) >= 20,
            description: "Creating cache and model directories",
          },
          {
            key: "loading_model",
            label: "Loading Qwen Model",
            completed: (init?.progress || 0) >= 50,
            description: "Loading AI model components and tokenizer",
          },
          {
            key: "downloading_model",
            label: "Downloading Model Files",
            completed: (init?.progress || 0) >= 70,
            description: "First-time download: ~2-5GB (cached for future use)",
          },
          {
            key: "loading_to_gpu",
            label: "GPU Memory Loading",
            completed: (init?.progress || 0) >= 90,
            description: "Transferring model to GPU for fast inference",
          },
          {
            key: "model_ready",
            label: "Ready for Generation",
            completed: (init?.progress || 0) >= 100,
            description: "All systems ready for image generation",
          },
        ];
        return steps;
      };

      const currentStep = getInitializationSteps().find(
        (step) => init?.status === step.key
      );

      return (
        <div className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/30 border border-blue-200 dark:border-blue-800 rounded-xl shadow-lg">
          <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center shadow-md">
                  {init?.status === "downloading_model" ? (
                    <Download className="w-6 h-6 text-white animate-pulse" />
                  ) : (
                    <Loader className="w-6 h-6 text-white animate-spin" />
                  )}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-blue-900 dark:text-blue-100">
                    {init?.message || "Initializing Qwen AI Model"}
                  </h3>
                  <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                    {currentStep?.description ||
                      "Setting up AI image generation system..."}
                  </p>
                </div>
              </div>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="btn btn-secondary px-4 py-2 text-sm"
                >
                  Refresh
                </button>
              )}
            </div>

            {/* Progress bar */}
            {init?.progress !== undefined && (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                    Overall Progress
                  </span>
                  <span className="text-lg font-bold text-blue-700 dark:text-blue-300">
                    {init.progress}%
                  </span>
                </div>
                <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-4 shadow-inner">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-indigo-500 h-4 rounded-full transition-all duration-500 ease-out shadow-sm"
                    style={{ width: `${init.progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Detailed step progress */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-200">
                Initialization Steps:
              </h4>
              <div className="space-y-2">
                {getInitializationSteps().map((step, index) => (
                  <div
                    key={step.key}
                    className={`flex items-start space-x-3 p-3 rounded-lg transition-all ${
                      init?.status === step.key
                        ? "bg-blue-100 dark:bg-blue-800/50 border border-blue-300 dark:border-blue-600"
                        : step.completed
                        ? "bg-green-50 dark:bg-green-900/20"
                        : "bg-gray-50 dark:bg-gray-800/50"
                    }`}
                  >
                    <div
                      className={`w-4 h-4 rounded-full flex items-center justify-center mt-0.5 flex-shrink-0 ${
                        step.completed
                          ? "bg-green-500"
                          : init?.status === step.key
                          ? "bg-blue-500"
                          : "bg-gray-300"
                      }`}
                    >
                      {step.completed ? (
                        <CheckCircle className="w-3 h-3 text-white" />
                      ) : init?.status === step.key ? (
                        <Loader className="w-3 h-3 text-white animate-spin" />
                      ) : (
                        <div className="w-2 h-2 bg-white rounded-full" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <span
                          className={`text-sm font-medium ${
                            step.completed
                              ? "text-green-800 dark:text-green-200"
                              : init?.status === step.key
                              ? "text-blue-800 dark:text-blue-200"
                              : "text-gray-600 dark:text-gray-400"
                          }`}
                        >
                          {step.label}
                        </span>
                        {init?.status === step.key && (
                          <span className="text-xs bg-blue-500 text-white px-2 py-0.5 rounded-full">
                            Active
                          </span>
                        )}
                      </div>
                      <p
                        className={`text-xs mt-1 ${
                          step.completed
                            ? "text-green-600 dark:text-green-400"
                            : init?.status === step.key
                            ? "text-blue-600 dark:text-blue-400"
                            : "text-gray-500 dark:text-gray-500"
                        }`}
                      >
                        {step.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* System info footer */}
            <div className="flex items-center justify-between pt-4 border-t border-blue-200 dark:border-blue-700">
              <div className="flex items-center space-x-4 text-xs text-blue-600 dark:text-blue-400">
                <div className="flex items-center space-x-1">
                  <Cpu className="w-3 h-3" />
                  <span>GPU: {status?.device || "Detecting..."}</span>
                </div>
                {status?.memory_info?.device_name && (
                  <div className="flex items-center space-x-1">
                    <HardDrive className="w-3 h-3" />
                    <span>{status.memory_info.device_name}</span>
                  </div>
                )}
              </div>
              <div className="flex items-center space-x-1 text-xs text-blue-600 dark:text-blue-400">
                <Clock className="w-3 h-3" />
                <span>Initializing: {new Date().toLocaleTimeString()}</span>
              </div>
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
      // Check if GPU is available based on device field and memory_info
      const gpuAvailable =
        status.device !== "cpu" &&
        status.device !== "unknown" &&
        status.memory_info &&
        !status.memory_info.error;
      if (!gpuAvailable) return "red";
      if (!status.model_loaded) return "yellow";
      return "green";
    };

    const getStatusIcon = () => {
      const color = getStatusColor();
      if (color === "red") return <XCircle className="w-5 h-5 text-red-500" />;
      if (color === "yellow")
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    };

    // Helper functions to calculate memory values from raw bytes
    const getMemoryValues = () => {
      if (!status.memory_info || status.memory_info.error) {
        return {
          allocated_gb: 0,
          available_gb: 0,
          total_gb: 0,
          usage_percent: 0,
        };
      }

      const { total_memory, allocated_memory, free_memory } =
        status.memory_info;
      const total_gb = Math.round((total_memory / 1024 ** 3) * 10) / 10;
      const allocated_gb = Math.round((allocated_memory / 1024 ** 3) * 10) / 10;
      const available_gb = Math.round((free_memory / 1024 ** 3) * 10) / 10;
      const usage_percent = Math.round((allocated_memory / total_memory) * 100);

      return { allocated_gb, available_gb, total_gb, usage_percent };
    };

    const getMemoryColor = () => {
      if (!status.memory_info) return "gray";
      const { usage_percent } = getMemoryValues();
      if (usage_percent > 85) return "red";
      if (usage_percent > 70) return "yellow";
      return "green";
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
                  {status.model_loaded
                    ? "Ready for generation"
                    : "Model not loaded"}
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
              {(() => {
                const { allocated_gb, available_gb, total_gb, usage_percent } =
                  getMemoryValues();
                return (
                  <>
                    <div className="flex justify-between text-sm">
                      <span>Used: {allocated_gb}GB</span>
                      <span>Available: {available_gb}GB</span>
                    </div>
                    <div className="text-xs text-gray-500 text-center">
                      Total: {total_gb}GB
                    </div>

                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          getMemoryColor() === "red"
                            ? "bg-red-500"
                            : getMemoryColor() === "yellow"
                            ? "bg-yellow-500"
                            : "bg-green-500"
                        }`}
                        style={{ width: `${usage_percent}%` }}
                      />
                    </div>

                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{usage_percent}% used</span>
                      {status.memory_info.device_name && (
                        <span>{status.memory_info.device_name}</span>
                      )}
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        )}

        {/* Queue Status */}
        {(status.queue_length > 0 || status.current_generation) && (
          <div className="card p-4">
            <h3 className="font-semibold text-gray-900 mb-2">
              Generation Queue
            </h3>
            <div className="space-y-2">
              {status.current_generation && (
                <div className="flex items-center space-x-2 text-sm">
                  <Loader className="w-4 h-4 animate-spin text-blue-500" />
                  <span className="text-blue-600">Currently generating...</span>
                </div>
              )}
              {status.queue_length > 0 && (
                <div className="text-sm text-gray-600">
                  {status.queue_length} request
                  {status.queue_length !== 1 ? "s" : ""} in queue
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }
);

export default StatusBar;
