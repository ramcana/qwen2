import React from "react";
import { Loader2, AlertCircle } from "lucide-react";
import { ControlType } from "./ControlTypeSelector";

interface ControlPreviewProps {
  controlType: ControlType;
  isProcessing?: boolean;
  previewImage?: string;
  error?: string;
  onRetry?: () => void;
}

const ControlPreview: React.FC<ControlPreviewProps> = ({
  controlType,
  isProcessing = false,
  previewImage,
  error,
  onRetry,
}) => {
  const getControlTypeLabel = (type: ControlType): string => {
    switch (type) {
      case "auto":
        return "Auto-Detected Control";
      case "canny":
        return "Canny Edge Detection";
      case "depth":
        return "Depth Map";
      case "pose":
        return "Pose Detection";
      case "normal":
        return "Normal Map";
      case "segmentation":
        return "Segmentation Map";
      default:
        return "Control Preview";
    }
  };

  const getControlDescription = (type: ControlType): string => {
    switch (type) {
      case "auto":
        return "Automatically detected the best control features from your image";
      case "canny":
        return "Detected edges and structural lines that will guide generation";
      case "depth":
        return "Estimated depth information for 3D-aware generation";
      case "pose":
        return "Detected human poses and skeletal structure";
      case "normal":
        return "Generated surface normal vectors for lighting control";
      case "segmentation":
        return "Segmented objects and regions in the image";
      default:
        return "Control features extracted from your image";
    }
  };

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center space-x-2 text-red-700 mb-2">
          <AlertCircle size={20} />
          <h4 className="font-medium">Control Detection Failed</h4>
        </div>
        <p className="text-sm text-red-600 mb-3">{error}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-sm bg-red-100 hover:bg-red-200 text-red-700 px-3 py-1 rounded transition-colors"
          >
            Try Again
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium text-purple-900">
          {getControlTypeLabel(controlType)}
        </h4>
        {isProcessing && (
          <div className="flex items-center space-x-2 text-purple-600">
            <Loader2 size={16} className="animate-spin" />
            <span className="text-sm">Processing...</span>
          </div>
        )}
      </div>

      {isProcessing ? (
        <div className="flex items-center justify-center h-32 bg-purple-100 rounded-lg">
          <div className="text-center">
            <Loader2
              size={32}
              className="animate-spin text-purple-500 mx-auto mb-2"
            />
            <p className="text-sm text-purple-600">
              Analyzing image for{" "}
              {controlType === "auto" ? "optimal" : controlType} control
              features...
            </p>
          </div>
        </div>
      ) : previewImage ? (
        <div className="space-y-3">
          <img
            src={previewImage}
            alt="Control Preview"
            className="w-full max-h-48 object-contain rounded-lg bg-white shadow-sm"
          />
          <p className="text-sm text-purple-700">
            {getControlDescription(controlType)}
          </p>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <p className="text-sm">
            Upload a control image to see the detected features
          </p>
        </div>
      )}

      {previewImage && !isProcessing && (
        <div className="mt-3 p-3 bg-purple-100 rounded-lg">
          <h5 className="text-sm font-medium text-purple-900 mb-1">
            Control Strength Tips:
          </h5>
          <ul className="text-xs text-purple-700 space-y-1">
            <li>• Higher strength = more faithful to control features</li>
            <li>• Lower strength = more creative freedom</li>
            <li>
              • Adjust based on how closely you want to follow the control
            </li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default ControlPreview;
