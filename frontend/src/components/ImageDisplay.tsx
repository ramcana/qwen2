import { Clock, Settings } from "lucide-react";
import React, { useState } from "react";
import { toast } from "react-hot-toast";
import { getImageUrl } from "../services/api";
import { GenerationState } from "../hooks/useGenerationState";
import ImageWorkspace from "./ImageWorkspace";
import { ImageData } from "./ComparisonView";
import { ImageVersion } from "./ImageVersioning";

interface ImageDisplayProps {
  generationState: GenerationState & {
    selectImage: (imageId: string) => void;
    toggleFavorite: (imageId: string) => void;
    deleteImage: (imageId: string) => void;
  };
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ generationState }) => {
  // Convert current image to ImageData format
  const currentImage: ImageData | undefined = generationState.currentImage
    ? {
        id: generationState.currentImage.id,
        url: generationState.currentImage.url,
        label: `Generated Image - ${generationState.currentImage.mode}`,
        timestamp: generationState.currentImage.timestamp,
        metadata: {
          prompt: generationState.currentImage.prompt,
          mode: generationState.currentImage.mode,
          parameters: generationState.currentImage.parameters,
        },
      }
    : undefined;

  // Convert image history to version format
  const imageVersions: ImageVersion[] = generationState.imageHistory.map(
    (img, index) => ({
      id: img.id,
      url: img.url,
      label: `${img.prompt.substring(0, 30)}${
        img.prompt.length > 30 ? "..." : ""
      }`,
      timestamp: img.timestamp,
      metadata: {
        prompt: img.prompt,
        mode: img.mode,
        parameters: img.parameters,
      },
      isFavorite: img.isFavorite || false,
    })
  );

  const handleVersionSelect = (version: ImageVersion) => {
    generationState.selectImage(version.id);
  };

  const handleVersionDelete = (versionId: string) => {
    generationState.deleteImage(versionId);
    toast.success("Image deleted");
  };

  const handleVersionFavorite = (versionId: string, isFavorite: boolean) => {
    generationState.toggleFavorite(versionId);
    toast.success(isFavorite ? "Added to favorites" : "Removed from favorites");
  };

  const handleImageDownload = (image: ImageData) => {
    const link = document.createElement("a");
    link.href = image.url;
    link.download = `${image.label.replace(/\s+/g, "_")}_${
      image.timestamp.toISOString().split("T")[0]
    }.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success("Image downloaded!");
  };

  const handleImageShare = async (image: ImageData) => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: image.label,
          text: image.metadata?.prompt || "Generated image",
          url: image.url,
        });
      } else {
        await navigator.clipboard.writeText(image.url);
        toast.success("Image URL copied to clipboard!");
      }
    } catch (error) {
      toast.error("Failed to share image");
    }
  };

  return (
    <div className="space-y-6">
      {currentImage ? (
        <ImageWorkspace
          currentImage={currentImage}
          versions={imageVersions}
          onVersionSelect={handleVersionSelect}
          onVersionDelete={handleVersionDelete}
          onVersionFavorite={handleVersionFavorite}
          onImageDownload={handleImageDownload}
          onImageShare={handleImageShare}
        />
      ) : (
        <div className="space-y-6">
          {/* Quick Start Card */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-8">
            <div className="text-center">
              <div className="w-20 h-20 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
                <Settings className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                Ready to Create Amazing Images
              </h2>
              <p className="text-gray-600 dark:text-gray-300 text-lg mb-6">
                Use the panel on the left to start generating your first image
              </p>

              {/* Quick Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center items-center">
                <div className="flex items-center text-sm text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30 px-4 py-2 rounded-lg">
                  <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold mr-2">
                    1
                  </span>
                  Write a prompt describing your image
                </div>
                <div className="flex items-center text-sm text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30 px-4 py-2 rounded-lg">
                  <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold mr-2">
                    2
                  </span>
                  Click "Generate Image"
                </div>
              </div>
            </div>
          </div>

          {/* Example Gallery */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Example Prompts to Get Started
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-white text-xs font-bold">🎨</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-green-900 dark:text-green-100 mb-1">
                        Artistic Portrait
                      </h4>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        "Professional headshot of a confident person, studio
                        lighting, clean background, photorealistic"
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-white text-xs font-bold">🏔️</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                        Scenic Landscape
                      </h4>
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        "Serene mountain lake at golden hour, reflection in
                        water, dramatic clouds, high detail"
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="p-4 bg-gradient-to-r from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-purple-500 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-white text-xs font-bold">🚀</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-1">
                        Futuristic Scene
                      </h4>
                      <p className="text-sm text-purple-700 dark:text-purple-300">
                        "Futuristic cityscape with flying cars, neon lights,
                        cyberpunk aesthetic, night scene"
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-white text-xs font-bold">🎭</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-orange-900 dark:text-orange-100 mb-1">
                        Creative Art
                      </h4>
                      <p className="text-sm text-orange-700 dark:text-orange-300">
                        "Abstract digital art, vibrant colors, geometric
                        patterns, modern style, high resolution"
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Overview */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Available Generation Modes
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gradient-to-b from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/30 rounded-lg border border-green-200 dark:border-green-800">
                <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-3 shadow-md">
                  <span className="text-white font-bold text-sm">T2I</span>
                </div>
                <h4 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                  Text to Image
                </h4>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Create images from text descriptions with advanced AI models
                </p>
              </div>

              <div className="text-center p-4 bg-gradient-to-b from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/30 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-3 shadow-md">
                  <span className="text-white font-bold text-sm">I2I</span>
                </div>
                <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                  Image to Image
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  Transform and modify existing images with AI guidance
                </p>
              </div>

              <div className="text-center p-4 bg-gradient-to-b from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-900/30 rounded-lg border border-purple-200 dark:border-purple-800">
                <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-3 shadow-md">
                  <span className="text-white font-bold text-sm">CN</span>
                </div>
                <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                  ControlNet
                </h4>
                <p className="text-sm text-purple-700 dark:text-purple-300">
                  Precise control over image composition and structure
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Generation Details (when image exists) */}
      {generationState.currentImage && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="font-semibold text-gray-900 mb-4">
            Generation Details
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Generation Time:</span>
                <span className="text-sm font-medium flex items-center">
                  <Clock className="w-3 h-3 mr-1" />
                  {generationState.currentImage.generationTime?.toFixed(1)}s
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Dimensions:</span>
                <span className="text-sm font-medium">
                  {generationState.currentImage.parameters.width} ×{" "}
                  {generationState.currentImage.parameters.height}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Steps:</span>
                <span className="text-sm font-medium">
                  {generationState.currentImage.parameters.num_inference_steps}
                </span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">CFG Scale:</span>
                <span className="text-sm font-medium">
                  {generationState.currentImage.parameters.cfg_scale}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Seed:</span>
                <span className="text-sm font-medium">
                  {generationState.currentImage.parameters.seed}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Enhanced:</span>
                <span className="text-sm font-medium">
                  {generationState.currentImage.parameters.enhance_prompt
                    ? "Yes"
                    : "No"}
                </span>
              </div>
            </div>
          </div>

          {/* Prompt Display */}
          <div className="mt-6 pt-4 border-t">
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-700">Prompt:</span>
            </div>
            <div className="bg-gray-50 p-3 rounded-lg border text-sm text-gray-700">
              {generationState.currentImage.prompt}
            </div>
          </div>

          {/* Negative Prompt */}
          {generationState.currentImage.negativePrompt && (
            <div className="mt-4">
              <div className="mb-2">
                <span className="text-sm font-medium text-gray-700">
                  Negative Prompt:
                </span>
              </div>
              <div className="bg-red-50 p-3 rounded-lg border text-sm text-gray-700">
                {generationState.currentImage.negativePrompt}
              </div>
            </div>
          )}

          {/* Generation Mode */}
          <div className="mt-4 pt-4 border-t">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">Mode:</span>
              <span className="text-sm font-medium capitalize bg-blue-100 text-blue-800 px-2 py-1 rounded">
                {generationState.currentImage.mode.replace("-", " ")}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageDisplay;
