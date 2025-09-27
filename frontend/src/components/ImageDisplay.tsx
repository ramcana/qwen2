import { Clock, Settings } from "lucide-react";
import React, { useState } from "react";
import { toast } from "react-hot-toast";
import { getImageUrl } from "../services/api";
import ImageWorkspace from "./ImageWorkspace";
import { ImageData } from "./ComparisonView";
import { ImageVersion } from "./ImageVersioning";

interface ImageDisplayProps {
  // Will be connected to generation state
}

const ImageDisplay: React.FC<ImageDisplayProps> = () => {
  const [generatedImage, setGeneratedImage] = useState<any>(null);
  const [imageVersions, setImageVersions] = useState<ImageVersion[]>([]);
  const [currentImageId, setCurrentImageId] = useState<string | null>(null);

  // Mock data for now - will be replaced with actual state management
  const mockGeneration = {
    success: true,
    image_path: "generated_images/api_generated_20241207_123456.png",
    message: "Image generated successfully",
    generation_time: 23.5,
    parameters: {
      prompt: "A futuristic coffee shop with neon signs",
      width: 1664,
      height: 928,
      num_inference_steps: 50,
      cfg_scale: 4.0,
      seed: 12345,
    },
  };

  // Use mock data for demonstration and convert to ImageData format
  const displayImage = generatedImage || mockGeneration;

  // Convert to ImageData format
  const currentImage: ImageData | undefined = displayImage
    ? {
        id: "current",
        url: getImageUrl(displayImage.image_path),
        label: "Generated Image",
        timestamp: new Date(),
        metadata: {
          prompt: displayImage.parameters?.prompt,
          mode: "generate",
          parameters: displayImage.parameters,
        },
      }
    : undefined;

  // Mock version history - in real app this would come from state management
  const mockVersions: ImageVersion[] = [
    {
      id: "v1",
      url: getImageUrl("generated_images/api_generated_20241207_123456.png"),
      label: "Futuristic Coffee Shop v1",
      timestamp: new Date(Date.now() - 3600000), // 1 hour ago
      metadata: {
        prompt: "A futuristic coffee shop with neon signs",
        mode: "generate",
      },
      isFavorite: true,
    },
    {
      id: "v2",
      url: getImageUrl("generated_images/api_generated_20241207_123456.png"),
      label: "Futuristic Coffee Shop v2",
      timestamp: new Date(Date.now() - 1800000), // 30 minutes ago
      metadata: {
        prompt: "A futuristic coffee shop with neon signs and robots",
        mode: "edit",
      },
    },
  ];

  const handleVersionSelect = (version: ImageVersion) => {
    setCurrentImageId(version.id);
    // In real app, this would update the current image display
    console.log("Selected version:", version);
  };

  const handleVersionDelete = (versionId: string) => {
    setImageVersions((prev) => prev.filter((v) => v.id !== versionId));
    toast.success("Version deleted");
  };

  const handleVersionFavorite = (versionId: string, isFavorite: boolean) => {
    setImageVersions((prev) =>
      prev.map((v) => (v.id === versionId ? { ...v, isFavorite } : v))
    );
    toast.success(isFavorite ? "Added to favorites" : "Removed from favorites");
  };

  const handleImageDownload = (image: ImageData) => {
    const link = document.createElement("a");
    link.href = image.url;
    link.download = `${image.label.replace(/\s+/g, "_")}_${image.timestamp.toISOString().split("T")[0]}.png`;
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
          versions={imageVersions.length > 0 ? imageVersions : mockVersions}
          onVersionSelect={handleVersionSelect}
          onVersionDelete={handleVersionDelete}
          onVersionFavorite={handleVersionFavorite}
          onImageDownload={handleImageDownload}
          onImageShare={handleImageShare}
        />
      ) : (
        <div className="bg-white rounded-lg shadow-sm border p-8">
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <Settings className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-500 font-medium">
                No image generated yet
              </p>
              <p className="text-gray-400 text-sm">
                Use the panel on the left to generate your first image
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Generation Details (when image exists) */}
      {displayImage && (
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
                  {displayImage.generation_time?.toFixed(1)}s
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Dimensions:</span>
                <span className="text-sm font-medium">
                  {displayImage.parameters?.width} ×{" "}
                  {displayImage.parameters?.height}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Steps:</span>
                <span className="text-sm font-medium">
                  {displayImage.parameters?.num_inference_steps}
                </span>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">CFG Scale:</span>
                <span className="text-sm font-medium">
                  {displayImage.parameters?.cfg_scale}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Seed:</span>
                <span className="text-sm font-medium">
                  {displayImage.parameters?.seed}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Enhanced:</span>
                <span className="text-sm font-medium">
                  {displayImage.parameters?.enhance_prompt ? "Yes" : "No"}
                </span>
              </div>
            </div>
          </div>

          {/* Prompt Display */}
          {displayImage.parameters?.prompt && (
            <div className="mt-6 pt-4 border-t">
              <div className="mb-2">
                <span className="text-sm font-medium text-gray-700">
                  Prompt:
                </span>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg border text-sm text-gray-700">
                {displayImage.parameters.prompt}
              </div>
            </div>
          )}

          {/* Negative Prompt */}
          {displayImage.parameters?.negative_prompt && (
            <div className="mt-4">
              <div className="mb-2">
                <span className="text-sm font-medium text-gray-700">
                  Negative Prompt:
                </span>
              </div>
              <div className="bg-red-50 p-3 rounded-lg border text-sm text-gray-700">
                {displayImage.parameters.negative_prompt}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageDisplay;
