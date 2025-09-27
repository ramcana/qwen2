import React from "react";
import { Zap, Scissors, Eye, User, Layers, Grid } from "lucide-react";

export type ControlType =
  | "auto"
  | "canny"
  | "depth"
  | "pose"
  | "normal"
  | "segmentation";

interface ControlTypeSelectorProps {
  currentType: ControlType;
  onTypeChange: (type: ControlType) => void;
  disabled?: boolean;
}

const ControlTypeSelector: React.FC<ControlTypeSelectorProps> = ({
  currentType,
  onTypeChange,
  disabled = false,
}) => {
  const controlTypes = [
    {
      id: "auto" as ControlType,
      label: "Auto Detect",
      icon: Zap,
      description: "Automatically detect the best control type",
    },
    {
      id: "canny" as ControlType,
      label: "Canny Edge",
      icon: Scissors,
      description: "Edge detection for structural control",
    },
    {
      id: "depth" as ControlType,
      label: "Depth Map",
      icon: Eye,
      description: "3D depth information for spatial control",
    },
    {
      id: "pose" as ControlType,
      label: "Pose Detection",
      icon: User,
      description: "Human pose and skeleton detection",
    },
    {
      id: "normal" as ControlType,
      label: "Normal Map",
      icon: Layers,
      description: "Surface normal vectors for lighting control",
    },
    {
      id: "segmentation" as ControlType,
      label: "Segmentation",
      icon: Grid,
      description: "Object and region segmentation",
    },
  ];

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Control Type
      </label>
      <div className="grid grid-cols-2 gap-2">
        {controlTypes.map((type) => {
          const Icon = type.icon;
          const isActive = currentType === type.id;

          return (
            <button
              key={type.id}
              onClick={() => onTypeChange(type.id)}
              disabled={disabled}
              className={`
                flex items-center space-x-2 p-3 rounded-lg border transition-all duration-200 text-left
                ${
                  isActive
                    ? "border-purple-500 bg-purple-50 text-purple-700"
                    : "border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50"
                }
                ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
              `}
              title={type.description}
            >
              <Icon
                size={18}
                className={isActive ? "text-purple-600" : "text-gray-500"}
              />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">{type.label}</div>
                <div className="text-xs text-gray-500 truncate">
                  {type.description}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ControlTypeSelector;
