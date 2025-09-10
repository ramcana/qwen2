import { Clock, Copy, Download, Maximize2, Settings } from 'lucide-react';
import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { getImageUrl } from '../services/api';

interface ImageDisplayProps {
  // Will be connected to generation state
}

const ImageDisplay: React.FC<ImageDisplayProps> = () => {
  const [generatedImage, setGeneratedImage] = useState<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Mock data for now - will be replaced with actual state management
  const mockGeneration = {
    success: true,
    image_path: 'generated_images/api_generated_20241207_123456.png',
    message: 'Image generated successfully',
    generation_time: 23.5,
    parameters: {
      prompt: 'A futuristic coffee shop with neon signs',
      width: 1664,
      height: 928,
      num_inference_steps: 50,
      cfg_scale: 4.0,
      seed: 12345
    }
  };

  const handleDownload = () => {
    if (!generatedImage?.image_path) return;

    const link = document.createElement('a');
    link.href = getImageUrl(generatedImage.image_path);
    link.download = `qwen-image-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    toast.success('Image downloaded!');
  };

  const handleCopyToClipboard = async () => {
    if (!generatedImage?.parameters?.prompt) return;

    try {
      await navigator.clipboard.writeText(generatedImage.parameters.prompt);
      toast.success('Prompt copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy prompt');
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // Use mock data for demonstration
  const displayImage = generatedImage || mockGeneration;

  return (
    <div className="space-y-6">
      {/* Main Image Display */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Generated Image</h2>

          {displayImage && (
            <div className="flex items-center space-x-2">
              <button
                onClick={handleCopyToClipboard}
                className="btn btn-secondary px-3 py-2"
                title="Copy prompt"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={toggleFullscreen}
                className="btn btn-secondary px-3 py-2"
                title="Fullscreen"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
              <button
                onClick={handleDownload}
                className="btn btn-primary px-3 py-2"
                title="Download"
              >
                <Download className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        {/* Image Container */}
        <div className="relative">
          {displayImage ? (
            <div className="relative group">
              <img
                src={getImageUrl(displayImage.image_path)}
                alt="Generated"
                className={`w-full rounded-lg shadow-lg transition-all duration-200 ${
                  isFullscreen ? 'fixed inset-4 z-50 object-contain bg-black bg-opacity-90' : 'max-h-[600px] object-contain'
                }`}
                onClick={toggleFullscreen}
              />

              {/* Overlay info */}
              <div className="absolute top-2 right-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm opacity-0 group-hover:opacity-100 transition-opacity">
                {displayImage.parameters?.width} × {displayImage.parameters?.height}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-center">
                <Settings className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-500 font-medium">No image generated yet</p>
                <p className="text-gray-400 text-sm">Use the panel on the left to generate your first image</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Generation Info */}
      {displayImage && (
        <div className="card p-4">
          <h3 className="font-semibold text-gray-900 mb-3">Generation Details</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
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
                  {displayImage.parameters?.width} × {displayImage.parameters?.height}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Steps:</span>
                <span className="text-sm font-medium">
                  {displayImage.parameters?.num_inference_steps}
                </span>
              </div>
            </div>

            <div className="space-y-2">
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
                  {displayImage.parameters?.enhance_prompt ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          </div>

          {/* Prompt Display */}
          {displayImage.parameters?.prompt && (
            <div className="mt-4 pt-4 border-t">
              <div className="mb-2">
                <span className="text-sm font-medium text-gray-700">Prompt:</span>
              </div>
              <div className="bg-gray-50 p-3 rounded border text-sm text-gray-700">
                {displayImage.parameters.prompt}
              </div>
            </div>
          )}

          {/* Negative Prompt */}
          {displayImage.parameters?.negative_prompt && (
            <div className="mt-3">
              <div className="mb-2">
                <span className="text-sm font-medium text-gray-700">Negative Prompt:</span>
              </div>
              <div className="bg-red-50 p-3 rounded border text-sm text-gray-700">
                {displayImage.parameters.negative_prompt}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent Generations */}
      <div className="card p-4">
        <h3 className="font-semibold text-gray-900 mb-3">Recent Generations</h3>
        <div className="grid grid-cols-3 gap-3">
          {/* Placeholder for recent images */}
          {[1, 2, 3].map(i => (
            <div key={i} className="aspect-square bg-gray-100 rounded border-2 border-dashed border-gray-300 flex items-center justify-center">
              <span className="text-gray-400 text-xs">Empty</span>
            </div>
          ))}
        </div>
      </div>

      {/* Fullscreen Overlay */}
      {isFullscreen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-90 z-40"
          onClick={toggleFullscreen}
        />
      )}
    </div>
  );
};

export default ImageDisplay;
