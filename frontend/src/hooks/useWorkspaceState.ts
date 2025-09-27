import { useState, useCallback, useRef } from "react";
import { WorkflowMode } from "../components/ModeSelector";

interface GenerateWorkspaceState {
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed?: number;
}

interface EditWorkspaceState {
  editMode: "inpaint" | "outpaint" | "style-transfer";
  sourceImage?: File;
  maskImage?: File;
  styleImage?: File;
  prompt: string;
  strength: number;
  steps: number;
  guidance: number;
}

interface ControlNetWorkspaceState {
  controlType: "auto" | "canny" | "depth" | "pose" | "normal" | "segmentation";
  controlImage?: File;
  prompt: string;
  controlStrength: number;
  steps: number;
  guidance: number;
}

interface WorkspaceStates {
  generate: GenerateWorkspaceState;
  edit: EditWorkspaceState;
  controlnet: ControlNetWorkspaceState;
}

const defaultStates: WorkspaceStates = {
  generate: {
    prompt: "",
    negativePrompt: "",
    width: 512,
    height: 512,
    steps: 20,
    guidance: 7.5,
  },
  edit: {
    editMode: "inpaint",
    prompt: "",
    strength: 0.8,
    steps: 20,
    guidance: 7.5,
  },
  controlnet: {
    controlType: "auto",
    prompt: "",
    controlStrength: 1.0,
    steps: 20,
    guidance: 7.5,
  },
};

export const useWorkspaceState = () => {
  const [currentMode, setCurrentMode] = useState<WorkflowMode>("generate");
  const workspaceStates = useRef<WorkspaceStates>({
    generate: { ...defaultStates.generate },
    edit: { ...defaultStates.edit },
    controlnet: { ...defaultStates.controlnet },
  });

  const getCurrentState = useCallback(() => {
    return workspaceStates.current[currentMode];
  }, [currentMode]);

  const updateCurrentState = useCallback(
    (updates: Partial<any>) => {
      workspaceStates.current[currentMode] = {
        ...workspaceStates.current[currentMode],
        ...updates,
      };
    },
    [currentMode]
  );

  const switchMode = useCallback((newMode: WorkflowMode) => {
    setCurrentMode(newMode);
  }, []);

  const resetCurrentState = useCallback(() => {
    workspaceStates.current[currentMode] = { ...defaultStates[currentMode] };
  }, [currentMode]);

  const getAllStates = useCallback(() => {
    return workspaceStates.current;
  }, []);

  return {
    currentMode,
    switchMode,
    getCurrentState,
    updateCurrentState,
    resetCurrentState,
    getAllStates,
  };
};

export type {
  GenerateWorkspaceState,
  EditWorkspaceState,
  ControlNetWorkspaceState,
};
