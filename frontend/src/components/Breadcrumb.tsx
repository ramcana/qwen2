import React from "react";
import { ChevronRight, Home } from "lucide-react";
import { WorkflowMode } from "./ModeSelector";

interface BreadcrumbItem {
  label: string;
  path?: string;
  current?: boolean;
}

interface BreadcrumbProps {
  currentMode: WorkflowMode;
  currentStep?: string;
}

const Breadcrumb: React.FC<BreadcrumbProps> = ({
  currentMode,
  currentStep,
}) => {
  const getModeLabel = (mode: WorkflowMode): string => {
    switch (mode) {
      case "generate":
        return "Text-to-Image Generation";
      case "edit":
        return "Image Editing";
      case "controlnet":
        return "ControlNet Generation";
      default:
        return "Unknown Mode";
    }
  };

  const items: BreadcrumbItem[] = [
    { label: "Home", path: "/" },
    { label: getModeLabel(currentMode), current: !currentStep },
    ...(currentStep ? [{ label: currentStep, current: true }] : []),
  ];

  return (
    <nav className="flex items-center space-x-2 text-sm text-gray-600 mb-4">
      <Home size={16} className="text-gray-400" />
      {items.map((item, index) => (
        <React.Fragment key={index}>
          {index > 0 && <ChevronRight size={14} className="text-gray-400" />}
          <span
            className={`
              ${
                item.current
                  ? "text-blue-600 font-medium"
                  : "text-gray-600 hover:text-gray-900"
              }
              ${item.path ? "cursor-pointer" : ""}
            `}
          >
            {item.label}
          </span>
        </React.Fragment>
      ))}
    </nav>
  );
};

export default Breadcrumb;
