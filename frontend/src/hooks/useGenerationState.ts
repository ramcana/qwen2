import { useState, useCallback, useRef } from "react";
import { GenerationResponse } from "../types/api";

export interface GeneratedImage {
  id: string;
  url: string;
  prompt: string;
  negativePrompt?: string;
  parameters: {
    width: number;
    height: number;
    num_inference_steps: number;
    cfg_scale: number;
    seed: number;
    aspect_ratio: string;
    enhance_prompt: boolean;
    language: string;
  };
  generationTime: number;
  timestamp: Date;
  mode: "text-to-image" | "image-to-image";
  isFavorite?: boolean;
}

export interface GenerationState {
  currentImage: GeneratedImage | null;
  imageHistory: GeneratedImage[];
  isGenerating: boolean;
  generationProgress: {
    startTime: number | null;
    estimatedTime: number | null;
    currentStep: number;
    totalSteps: number;
  };
  lastError: string | null;
}

const initialState: GenerationState = {
  currentImage: null,
  imageHistory: [],
  isGenerating: false,
  generationProgress: {
    startTime: null,
    estimatedTime: null,
    currentStep: 0,
    totalSteps: 0,
  },
  lastError: null,
};

export const useGenerationState = () => {
  const [state, setState] = useState<GenerationState>(initialState);
  const generationIdRef = useRef<string | null>(null);

  const startGeneration = useCallback((jobId: string, totalSteps: number) => {
    generationIdRef.current = jobId;
    setState((prev) => ({
      ...prev,
      isGenerating: true,
      lastError: null,
      generationProgress: {
        startTime: Date.now(),
        estimatedTime: totalSteps * 2, // Rough estimate: 2 seconds per step
        currentStep: 0,
        totalSteps,
      },
    }));
  }, []);

  const updateProgress = useCallback((currentStep: number) => {
    setState((prev) => ({
      ...prev,
      generationProgress: {
        ...prev.generationProgress,
        currentStep,
      },
    }));
  }, []);

  const completeGeneration = useCallback(
    (response: GenerationResponse, requestData: any) => {
      if (!response.success || !response.image_path) {
        setState((prev) => ({
          ...prev,
          isGenerating: false,
          lastError: response.message,
          generationProgress: {
            startTime: null,
            estimatedTime: null,
            currentStep: 0,
            totalSteps: 0,
          },
        }));
        return;
      }

      const newImage: GeneratedImage = {
        id: response.job_id || Date.now().toString(),
        url: `/api/images/${response.image_path.split("/").pop()}`,
        prompt: requestData.prompt,
        negativePrompt: requestData.negative_prompt,
        parameters: {
          width: requestData.width,
          height: requestData.height,
          num_inference_steps: requestData.num_inference_steps,
          cfg_scale: requestData.cfg_scale,
          seed: requestData.seed,
          aspect_ratio: requestData.aspect_ratio,
          enhance_prompt: requestData.enhance_prompt,
          language: requestData.language,
        },
        generationTime: response.generation_time || 0,
        timestamp: new Date(),
        mode: requestData.init_image_path ? "image-to-image" : "text-to-image",
        isFavorite: false,
      };

      setState((prev) => ({
        ...prev,
        currentImage: newImage,
        imageHistory: [newImage, ...prev.imageHistory].slice(0, 50), // Keep last 50 images
        isGenerating: false,
        generationProgress: {
          startTime: null,
          estimatedTime: null,
          currentStep: 0,
          totalSteps: 0,
        },
        lastError: null,
      }));

      generationIdRef.current = null;
    },
    []
  );

  const failGeneration = useCallback((error: string) => {
    setState((prev) => ({
      ...prev,
      isGenerating: false,
      lastError: error,
      generationProgress: {
        startTime: null,
        estimatedTime: null,
        currentStep: 0,
        totalSteps: 0,
      },
    }));
    generationIdRef.current = null;
  }, []);

  const selectImage = useCallback((imageId: string) => {
    setState((prev) => {
      const image = prev.imageHistory.find((img) => img.id === imageId);
      if (image) {
        return {
          ...prev,
          currentImage: image,
        };
      }
      return prev;
    });
  }, []);

  const toggleFavorite = useCallback((imageId: string) => {
    setState((prev) => ({
      ...prev,
      imageHistory: prev.imageHistory.map((img) =>
        img.id === imageId ? { ...img, isFavorite: !img.isFavorite } : img
      ),
      currentImage:
        prev.currentImage?.id === imageId
          ? { ...prev.currentImage, isFavorite: !prev.currentImage.isFavorite }
          : prev.currentImage,
    }));
  }, []);

  const deleteImage = useCallback((imageId: string) => {
    setState((prev) => ({
      ...prev,
      imageHistory: prev.imageHistory.filter((img) => img.id !== imageId),
      currentImage:
        prev.currentImage?.id === imageId ? null : prev.currentImage,
    }));
  }, []);

  const clearHistory = useCallback(() => {
    setState((prev) => ({
      ...prev,
      imageHistory: [],
      currentImage: null,
    }));
  }, []);

  const getCurrentGenerationId = useCallback(() => {
    return generationIdRef.current;
  }, []);

  return {
    ...state,
    startGeneration,
    updateProgress,
    completeGeneration,
    failGeneration,
    selectImage,
    toggleFavorite,
    deleteImage,
    clearHistory,
    getCurrentGenerationId,
  };
};
