import React from "react";
import { EditMode } from "./EditModeSelector";

interface EditControlsProps {
  editMode: EditMode;
  values: {
    prompt: string;
    strength: number;
    steps: number;
    guidance: number;
  };
  onChange: (field: string, value: any) => void;
  disabled?: boolean;
}

const EditControls: React.FC<EditControlsProps> = ({
  editMode,
  values,
  onChange,
  disabled = false,
}) => {
  const getPromptPlaceholder = () => {
    switch (editMode) {
      case "inpaint":
        return "Describe what should fill the masked area...";
      case "outpaint":
        return "Describe what should extend beyond the image...";
      case "style-transfer":
        return "Describe the desired style or content...";
      default:
        return "Enter your prompt...";
    }
  };

  const getStrengthLabel = () => {
    switch (editMode) {
      case "inpaint":
        return "Inpaint Strength";
      case "outpaint":
        return "Outpaint Strength";
      case "style-transfer":
        return "Style Strength";
      default:
        return "Strength";
    }
  };

  const getStrengthDescription = () => {
    switch (editMode) {
      case "inpaint":
        return "How much to change the masked area (0.1 = subtle, 1.0 = complete replacement)";
      case "outpaint":
        return "How much to blend with existing image (0.1 = seamless, 1.0 = distinct)";
      case "style-transfer":
        return "How strongly to apply the style (0.1 = subtle, 1.0 = strong)";
      default:
        return "Strength of the effect";
    }
  };

  return (
    <div className="space-y-6">
      {/* Prompt Input */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Prompt
        </label>
        <textarea
          value={values.prompt}
          onChange={(e) => onChange("prompt", e.target.value)}
          placeholder={getPromptPlaceholder()}
          disabled={disabled}
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
        />
      </div>

      {/* Strength Control */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          {getStrengthLabel()}
        </label>
        <div className="space-y-1">
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={values.strength}
            onChange={(e) => onChange("strength", parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>0.1 (Subtle)</span>
            <span className="font-medium">{values.strength}</span>
            <span>1.0 (Strong)</span>
          </div>
          <p className="text-xs text-gray-500">{getStrengthDescription()}</p>
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
            value={values.steps}
            onChange={(e) => onChange("steps", parseInt(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>10 (Fast)</span>
            <span className="font-medium">{values.steps}</span>
            <span>50 (Quality)</span>
          </div>
          <p className="text-xs text-gray-500">
            More steps = better quality but slower generation
          </p>
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
            value={values.guidance}
            onChange={(e) => onChange("guidance", parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>1 (Creative)</span>
            <span className="font-medium">{values.guidance}</span>
            <span>20 (Precise)</span>
          </div>
          <p className="text-xs text-gray-500">
            How closely to follow the prompt
          </p>
        </div>
      </div>
    </div>
  );
};

export default EditControls;
