import React from "react";
import { Palette, Edit3, Layers } from "lucide-react";

export type WorkflowMode = "generate" | "edit" | "controlnet";

interface ModeSelectorProps {
  currentMode: WorkflowMode;
  onModeChange: (mode: WorkflowMode) => void;
  disabled?: boolean;
}

const ModeSelector: React.FC<ModeSelectorProps> = ({
  currentMode,
  onModeChange,
  disabled = false,
}) => {
  const modes = [
    {
      id: "generate" as WorkflowMode,
      label: "Generate",
      icon: Palette,
      description: "Text-to-image generation",
    },
    {
      id: "edit" as WorkflowMode,
      label: "Edit",
      icon: Edit3,
      description: "Advanced image editing",
    },
    {
      id: "controlnet" as WorkflowMode,
      label: "ControlNet",
      icon: Layers,
      description: "Guided generation with control",
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border p-1">
      <div className="flex space-x-1">
        {modes.map((mode) => {
          const Icon = mode.icon;
          const isActive = currentMode === mode.id;

          return (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id)}
              disabled={disabled}
              className={`
                flex items-center space-x-2 px-4 py-2 rounded-md transition-all duration-200
                ${
                  isActive
                    ? "bg-blue-500 text-white shadow-sm"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }
                ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
              `}
              title={mode.description}
            >
              <Icon size={18} />
              <span className="font-medium">{mode.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ModeSelector;
