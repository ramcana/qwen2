import React from "react";
import { Paintbrush, Expand, Palette } from "lucide-react";

export type EditMode = "inpaint" | "outpaint" | "style-transfer";

interface EditModeSelectorProps {
  currentMode: EditMode;
  onModeChange: (mode: EditMode) => void;
  disabled?: boolean;
}

const EditModeSelector: React.FC<EditModeSelectorProps> = ({
  currentMode,
  onModeChange,
  disabled = false,
}) => {
  const modes = [
    {
      id: "inpaint" as EditMode,
      label: "Inpaint",
      icon: Paintbrush,
      description: "Fill masked areas with AI-generated content",
    },
    {
      id: "outpaint" as EditMode,
      label: "Outpaint",
      icon: Expand,
      description: "Extend image beyond its current boundaries",
    },
    {
      id: "style-transfer" as EditMode,
      label: "Style Transfer",
      icon: Palette,
      description: "Apply artistic style from reference image",
    },
  ];

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Edit Operation
      </label>
      <div className="grid grid-cols-1 gap-2">
        {modes.map((mode) => {
          const Icon = mode.icon;
          const isActive = currentMode === mode.id;

          return (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id)}
              disabled={disabled}
              className={`
                flex items-center space-x-3 p-3 rounded-lg border transition-all duration-200 text-left
                ${
                  isActive
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50"
                }
                ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
              `}
              title={mode.description}
            >
              <Icon
                size={20}
                className={isActive ? "text-blue-600" : "text-gray-500"}
              />
              <div>
                <div className="font-medium">{mode.label}</div>
                <div className="text-xs text-gray-500">{mode.description}</div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default EditModeSelector;
