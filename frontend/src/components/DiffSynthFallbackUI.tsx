/**
 * DiffSynth Fallback UI Components
 * Provides alternative UI when DiffSynth features are unavailable
 */

import React, { useState } from "react";
import {
  AlertCircle,
  Cpu,
  Image,
  Zap,
  Settings,
  Info,
  CheckCircle,
  XCircle,
} from "lucide-react";

interface FallbackUIProps {
  fallbackType:
    | "service_unavailable"
    | "memory_limited"
    | "feature_disabled"
    | "processing_error";
  availableFeatures: string[];
  onFeatureSelect?: (feature: string) => void;
  onRetryOriginal?: () => void;
  limitations?: string[];
}

interface ServiceStatusProps {
  services: {
    name: string;
    status: "available" | "unavailable" | "degraded";
    description: string;
    fallbackAvailable: boolean;
  }[];
}

interface FeatureComparisonProps {
  originalFeatures: string[];
  fallbackFeatures: string[];
  currentMode: "original" | "fallback";
}

// Main Fallback UI Component
export const DiffSynthFallbackUI: React.FC<FallbackUIProps> = ({
  fallbackType,
  availableFeatures,
  onFeatureSelect,
  onRetryOriginal,
  limitations = [],
}) => {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const getFallbackConfig = () => {
    switch (fallbackType) {
      case "service_unavailable":
        return {
          title: "DiffSynth Service Unavailable",
          description:
            "Advanced editing features are temporarily unavailable. You can still use basic image generation.",
          icon: <AlertCircle className="w-6 h-6 text-orange-500" />,
          color: "orange",
          suggestions: [
            "Use Qwen text-to-image generation",
            "Try basic image processing tools",
            "Check back later for full functionality",
          ],
        };

      case "memory_limited":
        return {
          title: "Memory Optimization Mode",
          description:
            "Running in memory-optimized mode with reduced capabilities to ensure stable operation.",
          icon: <Cpu className="w-6 h-6 text-blue-500" />,
          color: "blue",
          suggestions: [
            "Lower resolution processing available",
            "CPU-based processing enabled",
            "Tiled processing for large images",
          ],
        };

      case "feature_disabled":
        return {
          title: "Feature Temporarily Disabled",
          description:
            "Some advanced features are disabled to maintain system stability.",
          icon: <Settings className="w-6 h-6 text-yellow-500" />,
          color: "yellow",
          suggestions: [
            "Core functionality remains available",
            "Alternative processing methods enabled",
            "Reduced feature set for stability",
          ],
        };

      case "processing_error":
        return {
          title: "Processing Error Recovery",
          description:
            "An error occurred during processing. Alternative methods are available.",
          icon: <Zap className="w-6 h-6 text-red-500" />,
          color: "red",
          suggestions: [
            "Simplified processing available",
            "Basic operations still functional",
            "Error recovery mechanisms active",
          ],
        };

      default:
        return {
          title: "Fallback Mode Active",
          description: "Operating in fallback mode with limited functionality.",
          icon: <Info className="w-6 h-6 text-gray-500" />,
          color: "gray",
          suggestions: ["Limited functionality available"],
        };
    }
  };

  const config = getFallbackConfig();
  const colorClasses = {
    orange: "bg-orange-50 border-orange-200 text-orange-800",
    blue: "bg-blue-50 border-blue-200 text-blue-800",
    yellow: "bg-yellow-50 border-yellow-200 text-yellow-800",
    red: "bg-red-50 border-red-200 text-red-800",
    gray: "bg-gray-50 border-gray-200 text-gray-800",
  };

  const handleFeatureSelect = (feature: string) => {
    setSelectedFeature(feature);
    if (onFeatureSelect) {
      onFeatureSelect(feature);
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div
        className={`border rounded-lg p-4 ${colorClasses[config.color as keyof typeof colorClasses]}`}
      >
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">{config.icon}</div>
          <div className="flex-1">
            <h3 className="font-semibold text-lg mb-1">{config.title}</h3>
            <p className="mb-3">{config.description}</p>

            {config.suggestions.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Available Options:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  {config.suggestions.map((suggestion, index) => (
                    <li key={index}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Available Features */}
      {availableFeatures.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="font-semibold text-lg mb-4 flex items-center space-x-2">
            <Image className="w-5 h-5 text-green-500" />
            <span>Available Features</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {availableFeatures.map((feature, index) => (
              <button
                key={index}
                onClick={() => handleFeatureSelect(feature)}
                className={`p-4 border rounded-lg text-left transition-all ${
                  selectedFeature === feature
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
                }`}
              >
                <div className="flex items-center space-x-2">
                  <CheckCircle
                    className={`w-4 h-4 ${
                      selectedFeature === feature
                        ? "text-blue-500"
                        : "text-green-500"
                    }`}
                  />
                  <span className="font-medium">{feature}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Limitations */}
      {limitations.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800 mb-2 flex items-center space-x-2">
            <Info className="w-4 h-4" />
            <span>Current Limitations</span>
          </h4>
          <ul className="list-disc list-inside space-y-1 text-sm text-yellow-700">
            {limitations.map((limitation, index) => (
              <li key={index}>{limitation}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Retry Option */}
      {onRetryOriginal && (
        <div className="flex justify-center">
          <button
            onClick={onRetryOriginal}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <Zap className="w-4 h-4" />
            <span>Try Full Features Again</span>
          </button>
        </div>
      )}
    </div>
  );
};

// Service Status Component
export const ServiceStatusIndicator: React.FC<ServiceStatusProps> = ({
  services,
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="font-semibold text-lg mb-4">Service Status</h3>

      <div className="space-y-3">
        {services.map((service, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
          >
            <div className="flex items-center space-x-3">
              <div
                className={`w-3 h-3 rounded-full ${
                  service.status === "available"
                    ? "bg-green-500"
                    : service.status === "degraded"
                      ? "bg-yellow-500"
                      : "bg-red-500"
                }`}
              />
              <div>
                <h4 className="font-medium">{service.name}</h4>
                <p className="text-sm text-gray-600">{service.description}</p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {service.status === "available" && (
                <CheckCircle className="w-4 h-4 text-green-500" />
              )}
              {service.status === "degraded" && (
                <AlertCircle className="w-4 h-4 text-yellow-500" />
              )}
              {service.status === "unavailable" && (
                <XCircle className="w-4 h-4 text-red-500" />
              )}

              {service.fallbackAvailable && service.status !== "available" && (
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  Fallback Available
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Feature Comparison Component
export const FeatureComparison: React.FC<FeatureComparisonProps> = ({
  originalFeatures,
  fallbackFeatures,
  currentMode,
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <h3 className="font-semibold text-lg mb-4">Feature Comparison</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Original Features */}
        <div>
          <h4 className="font-medium mb-3 flex items-center space-x-2">
            <span
              className={`w-3 h-3 rounded-full ${
                currentMode === "original" ? "bg-green-500" : "bg-gray-300"
              }`}
            />
            <span>Full Features</span>
          </h4>
          <ul className="space-y-2">
            {originalFeatures.map((feature, index) => (
              <li key={index} className="flex items-center space-x-2">
                <CheckCircle
                  className={`w-4 h-4 ${
                    currentMode === "original"
                      ? "text-green-500"
                      : "text-gray-400"
                  }`}
                />
                <span
                  className={
                    currentMode === "original"
                      ? "text-gray-900"
                      : "text-gray-500"
                  }
                >
                  {feature}
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* Fallback Features */}
        <div>
          <h4 className="font-medium mb-3 flex items-center space-x-2">
            <span
              className={`w-3 h-3 rounded-full ${
                currentMode === "fallback" ? "bg-blue-500" : "bg-gray-300"
              }`}
            />
            <span>Fallback Mode</span>
          </h4>
          <ul className="space-y-2">
            {fallbackFeatures.map((feature, index) => (
              <li key={index} className="flex items-center space-x-2">
                <CheckCircle
                  className={`w-4 h-4 ${
                    currentMode === "fallback"
                      ? "text-blue-500"
                      : "text-gray-400"
                  }`}
                />
                <span
                  className={
                    currentMode === "fallback"
                      ? "text-gray-900"
                      : "text-gray-500"
                  }
                >
                  {feature}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

// Simplified Edit Panel for Fallback Mode
export const SimplifiedEditPanel: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleGenerate = async () => {
    setIsProcessing(true);
    // Simulate processing
    setTimeout(() => {
      setIsProcessing(false);
    }, 2000);
  };

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <Info className="w-4 h-4 text-blue-500" />
          <span className="font-medium text-blue-800">Simplified Mode</span>
        </div>
        <p className="text-blue-700 text-sm">
          Advanced editing features are unavailable. Using basic text-to-image
          generation.
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            placeholder="Describe the image you want to generate..."
          />
        </div>

        <button
          onClick={handleGenerate}
          disabled={isProcessing || !prompt.trim()}
          className="w-full py-3 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
        >
          {isProcessing ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Generating...</span>
            </>
          ) : (
            <>
              <Image className="w-4 h-4" />
              <span>Generate Image</span>
            </>
          )}
        </button>
      </div>

      <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
        <h4 className="font-medium mb-1">Available in Simplified Mode:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Basic text-to-image generation</li>
          <li>Standard resolution output</li>
          <li>Simple prompt processing</li>
        </ul>
      </div>
    </div>
  );
};

export default DiffSynthFallbackUI;
