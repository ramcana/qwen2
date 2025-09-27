import React, { useState } from "react";
import { Eye, EyeOff, RotateCcw, Download, Maximize2 } from "lucide-react";

interface ImageData {
  id: string;
  url: string;
  label: string;
  timestamp: Date;
  metadata?: {
    prompt?: string;
    mode?: string;
    parameters?: Record<string, any>;
  };
}

interface ComparisonViewProps {
  originalImage?: ImageData;
  editedImage?: ImageData;
  onDownload?: (image: ImageData) => void;
  onFullscreen?: (image: ImageData) => void;
  className?: string;
}

const ComparisonView: React.FC<ComparisonViewProps> = ({
  originalImage,
  editedImage,
  onDownload,
  onFullscreen,
  className = "",
}) => {
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

  const handleDownload = (image: ImageData) => {
    if (onDownload) {
      onDownload(image);
    } else {
      // Default download behavior
      const link = document.createElement("a");
      link.href = image.url;
      link.download = `${image.label}_${image.timestamp.toISOString().split("T")[0]}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const ImagePanel: React.FC<{
    image: ImageData;
    title: string;
    isOverlay?: boolean;
    opacity?: number;
  }> = ({ image, title, isOverlay = false, opacity = 1 }) => (
    <div className={`relative ${isOverlay ? "absolute inset-0" : ""}`}>
      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        {/* Header */}
        <div className="px-4 py-2 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="font-medium text-gray-900">{title}</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => onFullscreen?.(image)}
              className="p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded transition-colors"
              title="View fullscreen"
            >
              <Maximize2 size={16} />
            </button>
            <button
              onClick={() => handleDownload(image)}
              className="p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded transition-colors"
              title="Download image"
            >
              <Download size={16} />
            </button>
          </div>
        </div>

        {/* Image */}
        <div className="relative bg-gray-100">
          <img
            src={image.url}
            alt={image.label}
            className="w-full h-auto max-h-96 object-contain"
            style={{ opacity: isOverlay ? opacity : 1 }}
          />
        </div>

        {/* Metadata */}
        {image.metadata && (
          <div className="px-4 py-2 bg-gray-50 border-t">
            <div className="text-xs text-gray-600 space-y-1">
              {image.metadata.prompt && (
                <div>
                  <span className="font-medium">Prompt:</span>{" "}
                  {image.metadata.prompt}
                </div>
              )}
              {image.metadata.mode && (
                <div>
                  <span className="font-medium">Mode:</span>{" "}
                  {image.metadata.mode}
                </div>
              )}
              <div>
                <span className="font-medium">Created:</span>{" "}
                {image.timestamp.toLocaleString()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  if (!originalImage && !editedImage) {
    return (
      <div
        className={`bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center ${className}`}
      >
        <div className="text-gray-500">
          <Eye size={48} className="mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No Images to Compare</h3>
          <p className="text-sm">Generate or edit images to see them here</p>
        </div>
      </div>
    );
  }

  const canCompare = originalImage && editedImage;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Comparison Controls */}
      {canCompare && (
        <div className="flex items-center justify-between bg-white rounded-lg shadow-sm border p-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Image Comparison
          </h2>
          <div className="flex items-center space-x-4">
            {/* Overlay Toggle */}
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className={`
                flex items-center space-x-2 px-3 py-1.5 rounded-md transition-colors
                ${
                  showOverlay
                    ? "bg-blue-100 text-blue-700"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }
              `}
            >
              {showOverlay ? <EyeOff size={16} /> : <Eye size={16} />}
              <span className="text-sm">Overlay</span>
            </button>

            {/* Opacity Slider (when overlay is active) */}
            {showOverlay && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Opacity:</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={overlayOpacity}
                  onChange={(e) =>
                    setOverlayOpacity(parseFloat(e.target.value))
                  }
                  className="w-20 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600 w-8">
                  {Math.round(overlayOpacity * 100)}%
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Images Display */}
      {canCompare ? (
        showOverlay ? (
          /* Overlay Mode */
          <div className="relative">
            <ImagePanel image={originalImage} title="Original" />
            <ImagePanel
              image={editedImage}
              title="Edited"
              isOverlay={true}
              opacity={overlayOpacity}
            />
          </div>
        ) : (
          /* Side-by-Side Mode */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ImagePanel image={originalImage} title="Original" />
            <ImagePanel image={editedImage} title="Edited" />
          </div>
        )
      ) : (
        /* Single Image Mode */
        <div className="max-w-2xl mx-auto">
          {originalImage && (
            <ImagePanel image={originalImage} title="Original" />
          )}
          {editedImage && <ImagePanel image={editedImage} title="Generated" />}
        </div>
      )}
    </div>
  );
};

export default ComparisonView;
export type { ImageData };
