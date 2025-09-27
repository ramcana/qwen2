import React, { useState } from "react";
import { Play, RotateCcw } from "lucide-react";
import EditModeSelector, { EditMode } from "./EditModeSelector";
import ImageUploadArea from "./ImageUploadArea";
import EditControls from "./EditControls";
import {
  useWorkspaceState,
  EditWorkspaceState,
} from "../hooks/useWorkspaceState";

interface EditPanelProps {
  onGenerate?: (params: any) => void;
  isGenerating?: boolean;
}

const EditPanel: React.FC<EditPanelProps> = ({
  onGenerate,
  isGenerating = false,
}) => {
  const { getCurrentState, updateCurrentState, resetCurrentState } =
    useWorkspaceState();
  const state = getCurrentState() as EditWorkspaceState;

  const handleModeChange = (mode: EditMode) => {
    updateCurrentState({ editMode: mode });
  };

  const handleFileSelect =
    (field: "sourceImage" | "maskImage" | "styleImage") =>
    (file: File | null) => {
      updateCurrentState({ [field]: file });
    };

  const handleControlChange = (field: string, value: any) => {
    updateCurrentState({ [field]: value });
  };

  const handleGenerate = () => {
    if (!state.sourceImage) {
      alert("Please upload a source image first.");
      return;
    }

    if (state.editMode === "inpaint" && !state.maskImage) {
      alert("Please upload a mask image for inpainting.");
      return;
    }

    if (state.editMode === "style-transfer" && !state.styleImage) {
      alert("Please upload a style reference image.");
      return;
    }

    if (!state.prompt.trim()) {
      alert("Please enter a prompt.");
      return;
    }

    const params = {
      editMode: state.editMode,
      sourceImage: state.sourceImage,
      maskImage: state.maskImage,
      styleImage: state.styleImage,
      prompt: state.prompt,
      strength: state.strength,
      steps: state.steps,
      guidance: state.guidance,
    };

    onGenerate?.(params);
  };

  const handleReset = () => {
    if (confirm("Are you sure you want to reset all settings?")) {
      resetCurrentState();
    }
  };

  const canGenerate =
    state.sourceImage &&
    state.prompt.trim() &&
    (state.editMode !== "inpaint" || state.maskImage) &&
    (state.editMode !== "style-transfer" || state.styleImage);

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">Image Editing</h2>
          <button
            onClick={handleReset}
            disabled={isGenerating}
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
        {/* Edit Mode Selection */}
        <EditModeSelector
          currentMode={state.editMode}
          onModeChange={handleModeChange}
          disabled={isGenerating}
        />

        {/* Source Image Upload */}
        <ImageUploadArea
          label="Source Image"
          onFileSelect={handleFileSelect("sourceImage")}
          currentFile={state.sourceImage}
          placeholder="Upload the image you want to edit"
          required
          disabled={isGenerating}
        />

        {/* Conditional Uploads based on Edit Mode */}
        {state.editMode === "inpaint" && (
          <ImageUploadArea
            label="Mask Image"
            onFileSelect={handleFileSelect("maskImage")}
            currentFile={state.maskImage}
            placeholder="Upload a mask (white = edit area, black = keep original)"
            required
            disabled={isGenerating}
          />
        )}

        {state.editMode === "style-transfer" && (
          <ImageUploadArea
            label="Style Reference Image"
            onFileSelect={handleFileSelect("styleImage")}
            currentFile={state.styleImage}
            placeholder="Upload an image with the desired style"
            required
            disabled={isGenerating}
          />
        )}

        {/* Edit Controls */}
        <EditControls
          editMode={state.editMode}
          values={{
            prompt: state.prompt,
            strength: state.strength,
            steps: state.steps,
            guidance: state.guidance,
          }}
          onChange={handleControlChange}
          disabled={isGenerating}
        />

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={!canGenerate || isGenerating}
          className={`
            w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all duration-200
            ${
              canGenerate && !isGenerating
                ? "bg-blue-600 hover:bg-blue-700 text-white shadow-sm hover:shadow-md"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }
          `}
        >
          <Play size={20} />
          <span>
            {isGenerating
              ? "Editing Image..."
              : `Start ${
                  state.editMode === "inpaint"
                    ? "Inpainting"
                    : state.editMode === "outpaint"
                      ? "Outpainting"
                      : "Style Transfer"
                }`}
          </span>
        </button>

        {/* Help Text */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2">
            {state.editMode === "inpaint" && "Inpainting Tips"}
            {state.editMode === "outpaint" && "Outpainting Tips"}
            {state.editMode === "style-transfer" && "Style Transfer Tips"}
          </h4>
          <ul className="text-sm text-blue-800 space-y-1">
            {state.editMode === "inpaint" && (
              <>
                <li>
                  • White areas in the mask will be replaced with AI-generated
                  content
                </li>
                <li>• Black areas will be preserved from the original image</li>
                <li>
                  • Use lower strength for subtle changes, higher for complete
                  replacement
                </li>
              </>
            )}
            {state.editMode === "outpaint" && (
              <>
                <li>
                  • The AI will extend your image beyond its current boundaries
                </li>
                <li>• Describe what should appear in the extended areas</li>
                <li>• Lower strength creates more seamless blending</li>
              </>
            )}
            {state.editMode === "style-transfer" && (
              <>
                <li>
                  • Upload a reference image with the desired artistic style
                </li>
                <li>• The AI will apply that style to your source image</li>
                <li>
                  • Adjust strength to control how strongly the style is applied
                </li>
              </>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EditPanel;
