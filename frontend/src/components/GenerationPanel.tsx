import {
  Settings,
  Shuffle,
  Wand2,
  Image as ImageIcon,
  Clock,
  Zap,
  XCircle,
} from "lucide-react";
import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "react-hot-toast";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  generateTextToImage,
  generateImageToImage,
  getAspectRatios,
} from "../services/api";
import { GenerationRequest } from "../types/api";
import { GenerationState } from "../hooks/useGenerationState";
import ImageUpload from "./ImageUpload";

interface GenerationFormData extends Omit<GenerationRequest, "seed"> {
  seed: string; // Form uses string, convert to number
}

interface GenerationPanelProps {
  generationState: GenerationState & {
    startGeneration: (jobId: string, totalSteps: number) => void;
    completeGeneration: (response: any, requestData: any) => void;
    failGeneration: (error: string) => void;
  };
}

const GenerationPanel: React.FC<GenerationPanelProps> = ({
  generationState,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(true); // Show advanced by default
  const [generationMode, setGenerationMode] = useState<
    "text-to-image" | "image-to-image"
  >("text-to-image");
  const [uploadedImage, setUploadedImage] = useState<{
    file: File;
    previewUrl: string;
  } | null>(null);
  const [strength, setStrength] = useState(0.7);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<GenerationFormData>({
    defaultValues: {
      prompt: "",
      negative_prompt: "",
      width: 1344,
      height: 768,
      num_inference_steps: 50,
      cfg_scale: 4.0,
      seed: "-1",
      language: "en",
      enhance_prompt: true,
      aspect_ratio: "16:9",
    },
  });

  const { data: aspectRatios } = useQuery(["aspect-ratios"], getAspectRatios, {
    retry: 3,
    retryDelay: 1000,
    staleTime: 300000, // 5 minutes
    onError: (error) => {
      console.warn("Failed to load aspect ratios:", error);
    },
  });

  const selectedAspectRatio = watch("aspect_ratio");

  // Update dimensions when aspect ratio changes
  useEffect(() => {
    if (
      aspectRatios?.ratios &&
      selectedAspectRatio &&
      aspectRatios.ratios[selectedAspectRatio]
    ) {
      const [width, height] = aspectRatios.ratios[selectedAspectRatio];
      console.log(
        `Updating dimensions for ${selectedAspectRatio}: ${width}x${height}`
      );
      setValue("width", width);
      setValue("height", height);
    }
  }, [selectedAspectRatio, aspectRatios, setValue]);

  // Watch for dimension changes to show real-time updates
  const currentWidth = watch("width");
  const currentHeight = watch("height");

  const generateMutation = useMutation(
    async (data: GenerationFormData) => {
      const request: GenerationRequest = {
        ...data,
        seed: data.seed === "-1" ? -1 : parseInt(data.seed) || -1,
      };

      // Start generation tracking
      const jobId = `gen_${Date.now()}`;
      generationState.startGeneration(jobId, data.num_inference_steps);

      if (generationMode === "image-to-image") {
        if (!uploadedImage) {
          throw new Error(
            "Please upload an image for image-to-image generation"
          );
        }

        // For now, we'll use the preview URL as the image path
        // In a real implementation, you'd upload the image to the server first
        const img2imgRequest = {
          ...request,
          init_image_path: uploadedImage.previewUrl,
          strength: strength,
        };

        return {
          response: await generateImageToImage(img2imgRequest),
          requestData: img2imgRequest,
        };
      } else {
        return {
          response: await generateTextToImage(request),
          requestData: request,
        };
      }
    },
    {
      onSuccess: ({ response, requestData }) => {
        if (response.success) {
          generationState.completeGeneration(response, requestData);
          toast.success(
            `Image generated in ${response.generation_time?.toFixed(1)}s`
          );
        } else {
          generationState.failGeneration(response.message);
          toast.error(response.message);
        }
      },
      onError: (error: any) => {
        const errorMessage = error.response?.data?.detail || error.message;
        generationState.failGeneration(errorMessage);
        toast.error(`Generation failed: ${errorMessage}`);
      },
    }
  );

  // Timer state for real-time updates
  const [currentTime, setCurrentTime] = useState(Date.now());

  // Update timer every second during generation
  useEffect(() => {
    let interval: any;
    if (
      generationState.isGenerating &&
      generationState.generationProgress.startTime
    ) {
      interval = setInterval(() => {
        setCurrentTime(Date.now());
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [
    generationState.isGenerating,
    generationState.generationProgress.startTime,
  ]);

  const onSubmit = (data: GenerationFormData) => {
    generateMutation.mutate(data);
  };

  const randomizeSeed = () => {
    setValue("seed", Math.floor(Math.random() * 1000000).toString());
  };

  const quickPresets = [
    { name: "Fast Preview", steps: 20, cfg: 3.0 },
    { name: "Balanced", steps: 50, cfg: 4.0 },
    { name: "High Quality", steps: 80, cfg: 7.0 },
  ];

  const examplePrompts = [
    "A futuristic coffee shop with neon signs reading 'AI Café' and 'Welcome' in both English and Chinese, cyberpunk style",
    "A beautiful landscape painting with text overlay reading 'Qwen Mountain Resort - Est. 2025', traditional Chinese painting style",
    "A modern poster design with the text 'Innovation Summit 2025' in bold letters, minimalist design, blue and white color scheme",
  ];

  return (
    <div className="space-y-6">
      {/* Generation Form */}
      <div className="card p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Generate Image</h2>

        {/* Generation Mode Toggle */}
        <div className="flex space-x-2 mb-4">
          <button
            type="button"
            onClick={() => setGenerationMode("text-to-image")}
            className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              generationMode === "text-to-image"
                ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800"
                : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
            }`}
          >
            <Wand2 className="w-4 h-4 mr-2" />
            Text to Image
          </button>
          <button
            type="button"
            onClick={() => setGenerationMode("image-to-image")}
            className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              generationMode === "image-to-image"
                ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800"
                : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
            }`}
          >
            <ImageIcon className="w-4 h-4 mr-2" />
            Image to Image
          </button>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Image Upload for img2img */}
          {generationMode === "image-to-image" && (
            <div>
              <ImageUpload
                onImageUpload={(file, previewUrl) =>
                  setUploadedImage({ file, previewUrl })
                }
                currentImage={uploadedImage?.previewUrl}
                onClearImage={() => setUploadedImage(null)}
                disabled={generateMutation.isLoading}
              />

              {/* Strength Slider */}
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Strength: {strength.toFixed(1)} (how much to change the image)
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={strength}
                  onChange={(e) => setStrength(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Subtle changes</span>
                  <span>Major changes</span>
                </div>
              </div>
            </div>
          )}

          {/* Prompt */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prompt
            </label>
            <textarea
              {...register("prompt", { required: "Prompt is required" })}
              className="input min-h-[100px] resize-none"
              placeholder="A coffee shop entrance with a chalkboard sign reading 'Qwen Coffee ☕ $2 per cup'..."
            />
            {errors.prompt && (
              <p className="text-red-500 text-sm mt-1">
                {errors.prompt.message}
              </p>
            )}
          </div>

          {/* Language and Enhancement */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Language
              </label>
              <select {...register("language")} className="input">
                <option value="en">English</option>
                <option value="zh">中文</option>
              </select>
            </div>
            <div className="flex items-center pt-8">
              <input
                {...register("enhance_prompt")}
                type="checkbox"
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label className="ml-2 text-sm text-gray-700">
                Enhance prompt
              </label>
            </div>
          </div>

          {/* Aspect Ratio and Dimensions */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Aspect Ratio
              </label>
              <select {...register("aspect_ratio")} className="input">
                {aspectRatios?.ratios &&
                  Object.keys(aspectRatios.ratios).map((ratio) => (
                    <option key={ratio} value={ratio}>
                      {ratio.replace(":", ":")} ({aspectRatios.ratios[ratio][0]}
                      ×{aspectRatios.ratios[ratio][1]})
                    </option>
                  ))}
              </select>
            </div>

            {/* Current Dimensions Display */}
            <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                  Current Size:
                </span>
                <span className="text-sm font-mono text-blue-700 dark:text-blue-300">
                  {currentWidth} × {currentHeight} px
                </span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-blue-600 dark:text-blue-400">
                  Aspect Ratio:
                </span>
                <span className="text-xs font-mono text-blue-600 dark:text-blue-400">
                  {selectedAspectRatio || "Custom"}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Width
                </label>
                <input
                  {...register("width", { min: 512, max: 2048 })}
                  type="number"
                  step="64"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Height
                </label>
                <input
                  {...register("height", { min: 512, max: 2048 })}
                  type="number"
                  step="64"
                  className="input"
                />
              </div>
            </div>
          </div>

          {/* Advanced Settings */}
          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center text-sm text-gray-600 hover:text-gray-900"
            >
              <Settings className="w-4 h-4 mr-1" />
              Advanced Settings
            </button>
          </div>

          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t">
              {/* Negative Prompt */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Negative Prompt
                </label>
                <textarea
                  {...register("negative_prompt")}
                  className="input min-h-[60px] resize-none"
                  placeholder="blurry, low quality, distorted..."
                />
              </div>

              {/* Steps and CFG */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Steps: {watch("num_inference_steps")}
                  </label>
                  <input
                    {...register("num_inference_steps", { min: 10, max: 100 })}
                    type="range"
                    min="10"
                    max="100"
                    step="5"
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    CFG Scale: {watch("cfg_scale")}
                  </label>
                  <input
                    {...register("cfg_scale", { min: 1, max: 20 })}
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    className="w-full"
                  />
                </div>
              </div>

              {/* Seed */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Seed
                </label>
                <div className="flex space-x-2">
                  <input
                    {...register("seed")}
                    className="input flex-1"
                    placeholder="-1 for random"
                  />
                  <button
                    type="button"
                    onClick={randomizeSeed}
                    className="btn btn-secondary px-3"
                  >
                    <Shuffle className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Quick Presets */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Presets
            </label>
            <div className="flex space-x-2">
              {quickPresets.map((preset) => (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => {
                    setValue("num_inference_steps", preset.steps);
                    setValue("cfg_scale", preset.cfg);
                  }}
                  className="btn btn-secondary text-xs px-3 py-1"
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <button
            type="submit"
            disabled={generationState.isGenerating}
            className="w-full btn btn-primary py-3 text-lg"
          >
            {generationState.isGenerating ? (
              <div className="flex items-center justify-center">
                <Wand2 className="w-5 h-5 mr-2 animate-spin" />
                Generating...
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <Wand2 className="w-5 h-5 mr-2" />
                Generate Image
              </div>
            )}
          </button>
        </form>

        {/* Generation Progress Display */}
        {generationState.isGenerating && (
          <div className="mt-4 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 border border-blue-200 dark:border-blue-800 rounded-xl shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center mr-3">
                  <Wand2 className="w-5 h-5 text-white animate-spin" />
                </div>
                <div>
                  <span className="font-semibold text-blue-900 dark:text-blue-100 text-lg">
                    Generating Image
                  </span>
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    AI is creating your image...
                  </p>
                </div>
              </div>
              {generationState.generationProgress.startTime && (
                <div className="text-right">
                  <div className="flex items-center text-lg font-mono font-bold text-blue-700 dark:text-blue-300">
                    <Clock className="w-4 h-4 mr-1" />
                    {Math.floor(
                      (currentTime -
                        generationState.generationProgress.startTime) /
                        1000
                    )}
                    s
                  </div>
                  <p className="text-xs text-blue-600 dark:text-blue-400">
                    elapsed
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-3">
              <div className="flex justify-between text-sm text-blue-700 dark:text-blue-300">
                <span className="font-medium">
                  Processing {generationState.generationProgress.totalSteps}{" "}
                  inference steps
                </span>
                {generationState.generationProgress.estimatedTime &&
                  generationState.generationProgress.startTime && (
                    <span className="font-mono">
                      ~
                      {Math.max(
                        0,
                        generationState.generationProgress.estimatedTime -
                          Math.floor(
                            (currentTime -
                              generationState.generationProgress.startTime) /
                              1000
                          )
                      )}
                      s remaining
                    </span>
                  )}
              </div>

              <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-3 shadow-inner">
                <div
                  className="bg-gradient-to-r from-blue-500 to-indigo-500 h-3 rounded-full transition-all duration-1000 ease-out shadow-sm"
                  style={{
                    width: generationState.generationProgress.startTime
                      ? `${Math.min(
                          100,
                          ((currentTime -
                            generationState.generationProgress.startTime) /
                            1000 /
                            (generationState.generationProgress.estimatedTime ||
                              60)) *
                            100
                        )}%`
                      : "0%",
                  }}
                />
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-blue-200 dark:border-blue-700">
                <div className="flex items-center space-x-4 text-xs text-blue-600 dark:text-blue-400">
                  <div className="flex items-center">
                    <Zap className="w-3 h-3 mr-1" />
                    <span className="font-mono">
                      {currentWidth}×{currentHeight}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span>CFG: {watch("cfg_scale")}</span>
                  </div>
                  <div className="flex items-center">
                    <span>Steps: {watch("num_inference_steps")}</span>
                  </div>
                </div>
                <span className="text-xs font-mono font-bold text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-800 px-2 py-1 rounded">
                  {generationMode === "image-to-image" ? "IMG2IMG" : "TXT2IMG"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {generationState.lastError && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-center">
              <XCircle className="w-5 h-5 text-red-600 dark:text-red-400 mr-2" />
              <span className="font-medium text-red-900 dark:text-red-100">
                Generation Failed
              </span>
            </div>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              {generationState.lastError}
            </p>
          </div>
        )}
      </div>

      {/* Example Prompts */}
      <div className="card p-6">
        <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
          <span className="w-6 h-6 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-full flex items-center justify-center text-sm font-bold mr-2">
            ✨
          </span>
          Example Prompts
        </h3>
        <div className="space-y-3">
          {examplePrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => setValue("prompt", prompt)}
              className="w-full text-left p-4 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 hover:from-blue-50 hover:to-indigo-50 dark:hover:from-blue-900/20 dark:hover:to-indigo-900/20 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-blue-200 dark:hover:border-blue-700 text-gray-700 dark:text-gray-300 hover:text-blue-800 dark:hover:text-blue-200 transition-all duration-200 group"
            >
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-400 to-pink-400 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform">
                  <span className="text-white text-xs font-bold">
                    {index === 0 ? "🏢" : index === 1 ? "🎨" : "🚀"}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium mb-1">
                    {index === 0
                      ? "Business & Professional"
                      : index === 1
                      ? "Artistic & Creative"
                      : "Futuristic & Tech"}
                  </p>
                  <p className="text-xs opacity-75 line-clamp-2">
                    {prompt.length > 100
                      ? `${prompt.substring(0, 100)}...`
                      : prompt}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
          <p className="text-xs text-blue-700 dark:text-blue-300">
            💡 <strong>Tip:</strong> Click any example to use it as your prompt,
            then modify it to match your vision!
          </p>
        </div>
      </div>
    </div>
  );
};

export default GenerationPanel;
