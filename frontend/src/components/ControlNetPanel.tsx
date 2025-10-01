import React, { useState } from "react";
import { Play, RotateCcw, Zap } from "lucide-react";
import ControlTypeSelector, { ControlType } from "./ControlTypeSelector";
import ControlImageUpload from "./ControlImageUpload";
import ControlPreview from "./ControlPreview";
import {
  useWorkspaceState,
  ControlNetWorkspaceState,
} from "../hooks/useWorkspaceState";

interface ControlNetPanelProps {
  onGenerate?: (params: any) => void;
  onDetectControl?: (image: File, type: ControlType) => Promise<string>;
  isGenerating?: boolean;
  isDetecting?: boolean;
}

const ControlNetPanel: React.FC<ControlNetPanelProps> = ({
  onGenerate,
  onDetectControl,
  isGenerating = false,
  isDetecting = false,
}) => {
  const { getCurrentState, updateCurrentState, resetCurrentState } =
    useWorkspaceState();
  const state = getCurrentState() as ControlNetWorkspaceState;

  const [showPreview, setShowPreview] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [detectionError, setDetectionError] = useState<string | null>(null);

  const handleTypeChange = (type: ControlType) => {
    updateCurrentState({ controlType: type });
    // Clear preview when changing type
    setPreviewImage(null);
    setDetectionError(null);
  };

  const handleImageSelect = async (file: File | null) => {
    updateCurrentState({ controlImage: file || undefined });
    setPreviewImage(null);
    setDetectionError(null);

    // Auto-detect control features if image is uploaded and auto mode is selected
    if (file && state.controlType === "auto" && onDetectControl) {
      try {
        const preview = await onDetectControl(file, "auto");
        setPreviewImage(preview);
      } catch (error) {
        setDetectionError(
          error instanceof Error
            ? error.message
            : "Failed to detect control features"
        );
      }
    }
  };

  const handleDetectControl = async () => {
    if (!state.controlImage || !onDetectControl) return;

    setDetectionError(null);
    try {
      const preview = await onDetectControl(
        state.controlImage,
        state.controlType
      );
      setPreviewImage(preview);
      setShowPreview(true);
    } catch (error) {
      setDetectionError(
        error instanceof Error
          ? error.message
          : "Failed to detect control features"
      );
    }
  };

  const handleControlChange = (field: string, value: any) => {
    updateCurrentState({ [field]: value });
  };

  const handleGenerate = () => {
    if (!state.controlImage) {
      alert("Please upload a control image first.");
      return;
    }

    if (!state.prompt.trim()) {
      alert("Please enter a prompt.");
      return;
    }

    const params = {
      controlType: state.controlType,
      controlImage: state.controlImage,
      prompt: state.prompt,
      controlStrength: state.controlStrength,
      steps: state.steps,
      guidance: state.guidance,
    };

    onGenerate?.(params);
  };

  const handleReset = () => {
    if (confirm("Are you sure you want to reset all settings?")) {
      resetCurrentState();
      setPreviewImage(null);
      setDetectionError(null);
      setShowPreview(false);
    }
  };

  const canGenerate = state.controlImage && state.prompt.trim();
  const canDetect = state.controlImage && state.controlType !== "auto";

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">
            ControlNet Generation
          </h2>
          <button
            onClick={handleReset}
            disabled={isGenerating || isDetecting}
            className="flex items-center space-x-2 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Reset all settings"
          >
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 space-y-6">
        {/* Control Type Selection */}
        <ControlTypeSelector
          currentType={state.controlType}
          onTypeChange={handleTypeChange}
          disabled={isGenerating || isDetecting}
        />

        {/* Control Image Upload */}
        <ControlImageUpload
          onFileSelect={handleImageSelect}
          currentFile={state.controlImage}
          disabled={isGenerating || isDetecting}
          showPreview={showPreview}
          onPreviewToggle={() => setShowPreview(!showPreview)}
        />

        {/* Manual Control Detection Button */}
        {canDetect && (
          <button
            onClick={handleDetectControl}
            disabled={isDetecting || isGenerating}
            className="w-full flex items-center justify-center space-x-2 py-2 px-4 border border-purple-300 text-purple-700 rounded-lg hover:bg-purple-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Zap size={18} />
            <span>
              {isDetecting
                ? "Detecting Control Features..."
                : `Detect ${state.controlType} Features`}
            </span>
          </button>
        )}

        {/* Control Preview */}
        {(showPreview || previewImage || detectionError || isDetecting) && (
          <ControlPreview
            controlType={state.controlType}
            isProcessing={isDetecting}
            previewImage={previewImage || undefined}
            error={detectionError || undefined}
            onRetry={handleDetectControl}
          />
        )}

        {/* Prompt Input */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Prompt
          </label>
          <textarea
            value={state.prompt}
            onChange={(e) => handleControlChange("prompt", e.target.value)}
            placeholder="Describe what you want to generate using the control guidance..."
            disabled={isGenerating || isDetecting}
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
        </div>

        {/* Control Strength */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Control Strength
          </label>
          <div className="space-y-1">
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={state.controlStrength}
              onChange={(e) =>
                handleControlChange(
                  "controlStrength",
                  parseFloat(e.target.value)
                )
              }
              disabled={isGenerating || isDetecting}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0.1 (Subtle)</span>
              <span className="font-medium">{state.controlStrength}</span>
              <span>2.0 (Strong)</span>
            </div>
            <p className="text-xs text-gray-500">
              How closely to follow the control features
            </p>
          </div>
        </div>

        {/* Steps Control */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Inference Steps
          </label>
          <div className="space-y-1">
            <input
              type="range"
              min="10"
              max="50"
              step="5"
              value={state.steps}
              onChange={(e) =>
                handleControlChange("steps", parseInt(e.target.value))
              }
              disabled={isGenerating || isDetecting}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>10 (Fast)</span>
              <span className="font-medium">{state.steps}</span>
              <span>50 (Quality)</span>
            </div>
          </div>
        </div>

        {/* Guidance Scale */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Guidance Scale
          </label>
          <div className="space-y-1">
            <input
              type="range"
              min="1"
              max="20"
              step="0.5"
              value={state.guidance}
              onChange={(e) =>
                handleControlChange("guidance", parseFloat(e.target.value))
              }
              disabled={isGenerating || isDetecting}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1 (Creative)</span>
              <span className="font-medium">{state.guidance}</span>
              <span>20 (Precise)</span>
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={!canGenerate || isGenerating || isDetecting}
          className={`
            w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all duration-200
            ${
              canGenerate && !isGenerating && !isDetecting
                ? "bg-purple-600 hover:bg-purple-700 text-white shadow-sm hover:shadow-md"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }
          `}
        >
          <Play size={20} />
          <span>
            {isGenerating
              ? "Generating with ControlNet..."
              : "Generate with ControlNet"}
          </span>
        </button>

        {/* Help Text */}
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h4 className="font-medium text-purple-900 mb-2">ControlNet Tips</h4>
          <ul className="text-sm text-purple-800 space-y-1">
            <li>
              • ControlNet uses your image to guide the generation process
            </li>
            <li>• Auto-detect finds the best control features automatically</li>
            <li>• Manual types let you choose specific control methods</li>
            <li>• Higher control strength follows your image more closely</li>
            <li>• Lower strength allows more creative interpretation</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ControlNetPanel;
