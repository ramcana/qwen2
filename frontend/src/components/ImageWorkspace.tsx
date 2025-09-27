import React, { useState, useCallback } from "react";
import {
  Maximize2,
  Minimize2,
  Grid,
  List,
  Download,
  Share2,
} from "lucide-react";
import ComparisonView, { ImageData } from "./ComparisonView";
import ImageVersioning, { ImageVersion } from "./ImageVersioning";

interface ImageWorkspaceProps {
  currentImage?: ImageData;
  originalImage?: ImageData;
  versions: ImageVersion[];
  onVersionSelect: (version: ImageVersion) => void;
  onVersionDelete?: (versionId: string) => void;
  onVersionFavorite?: (versionId: string, isFavorite: boolean) => void;
  onImageDownload?: (image: ImageData) => void;
  onImageShare?: (image: ImageData) => void;
  className?: string;
}

type ViewMode = "comparison" | "versions" | "fullscreen";
type Layout = "horizontal" | "vertical";

const ImageWorkspace: React.FC<ImageWorkspaceProps> = ({
  currentImage,
  originalImage,
  versions,
  onVersionSelect,
  onVersionDelete,
  onVersionFavorite,
  onImageDownload,
  onImageShare,
  className = "",
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>("comparison");
  const [layout, setLayout] = useState<Layout>("horizontal");
  const [fullscreenImage, setFullscreenImage] = useState<ImageData | null>(
    null
  );
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });

  const handleFullscreen = useCallback((image: ImageData) => {
    setFullscreenImage(image);
    setViewMode("fullscreen");
    setZoomLevel(1);
    setPanPosition({ x: 0, y: 0 });
  }, []);

  const handleExitFullscreen = useCallback(() => {
    setFullscreenImage(null);
    setViewMode("comparison");
  }, []);

  const handleZoom = useCallback((delta: number) => {
    setZoomLevel((prev) => Math.max(0.1, Math.min(5, prev + delta)));
  }, []);

  const handlePan = useCallback((deltaX: number, deltaY: number) => {
    setPanPosition((prev) => ({
      x: prev.x + deltaX,
      y: prev.y + deltaY,
    }));
  }, []);

  const handleDownloadAll = useCallback(() => {
    if (currentImage) onImageDownload?.(currentImage);
    if (originalImage) onImageDownload?.(originalImage);
  }, [currentImage, originalImage, onImageDownload]);

  const handleShare = useCallback(
    (image: ImageData) => {
      if (onImageShare) {
        onImageShare(image);
      } else {
        // Default share behavior
        if (navigator.share) {
          navigator.share({
            title: image.label,
            text: image.metadata?.prompt || "Generated image",
            url: image.url,
          });
        } else {
          // Fallback: copy URL to clipboard
          navigator.clipboard.writeText(image.url);
          alert("Image URL copied to clipboard");
        }
      }
    },
    [onImageShare]
  );

  // Fullscreen Modal
  if (viewMode === "fullscreen" && fullscreenImage) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center">
        {/* Controls */}
        <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
          <div className="flex items-center space-x-4">
            <button
              onClick={handleExitFullscreen}
              className="p-2 bg-black bg-opacity-50 text-white rounded-lg hover:bg-opacity-70 transition-colors"
            >
              <Minimize2 size={20} />
            </button>
            <h3 className="text-white font-medium">{fullscreenImage.label}</h3>
          </div>

          <div className="flex items-center space-x-2">
            {/* Zoom Controls */}
            <button
              onClick={() => handleZoom(-0.2)}
              className="px-3 py-1 bg-black bg-opacity-50 text-white rounded hover:bg-opacity-70 transition-colors"
            >
              -
            </button>
            <span className="text-white text-sm">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => handleZoom(0.2)}
              className="px-3 py-1 bg-black bg-opacity-50 text-white rounded hover:bg-opacity-70 transition-colors"
            >
              +
            </button>

            {/* Reset */}
            <button
              onClick={() => {
                setZoomLevel(1);
                setPanPosition({ x: 0, y: 0 });
              }}
              className="p-2 bg-black bg-opacity-50 text-white rounded hover:bg-opacity-70 transition-colors"
              title="Reset zoom and position"
            >
              <Grid size={16} />
            </button>

            {/* Download */}
            <button
              onClick={() => onImageDownload?.(fullscreenImage)}
              className="p-2 bg-black bg-opacity-50 text-white rounded hover:bg-opacity-70 transition-colors"
              title="Download image"
            >
              <Download size={16} />
            </button>
          </div>
        </div>

        {/* Image */}
        <div
          className="relative overflow-hidden cursor-move"
          onMouseDown={(e) => {
            const startX = e.clientX - panPosition.x;
            const startY = e.clientY - panPosition.y;

            const handleMouseMove = (e: MouseEvent) => {
              handlePan(
                e.clientX - startX - panPosition.x,
                e.clientY - startY - panPosition.y
              );
            };

            const handleMouseUp = () => {
              document.removeEventListener("mousemove", handleMouseMove);
              document.removeEventListener("mouseup", handleMouseUp);
            };

            document.addEventListener("mousemove", handleMouseMove);
            document.addEventListener("mouseup", handleMouseUp);
          }}
        >
          <img
            src={fullscreenImage.url}
            alt={fullscreenImage.label}
            className="max-w-none"
            style={{
              transform: `scale(${zoomLevel}) translate(${panPosition.x}px, ${panPosition.y}px)`,
              transformOrigin: "center center",
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Toolbar */}
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Image Workspace
            </h2>

            {/* View Mode Toggle */}
            <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode("comparison")}
                className={`
                  px-3 py-1 text-sm rounded-md transition-colors
                  ${
                    viewMode === "comparison"
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-600 hover:text-gray-900"
                  }
                `}
              >
                Compare
              </button>
              <button
                onClick={() => setViewMode("versions")}
                className={`
                  px-3 py-1 text-sm rounded-md transition-colors
                  ${
                    viewMode === "versions"
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-600 hover:text-gray-900"
                  }
                `}
              >
                Versions
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* Layout Toggle (for comparison mode) */}
            {viewMode === "comparison" && (currentImage || originalImage) && (
              <button
                onClick={() =>
                  setLayout(layout === "horizontal" ? "vertical" : "horizontal")
                }
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors"
                title={`Switch to ${layout === "horizontal" ? "vertical" : "horizontal"} layout`}
              >
                {layout === "horizontal" ? (
                  <List size={16} />
                ) : (
                  <Grid size={16} />
                )}
              </button>
            )}

            {/* Download All */}
            {(currentImage || originalImage) && (
              <button
                onClick={handleDownloadAll}
                className="flex items-center space-x-2 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                title="Download all images"
              >
                <Download size={16} />
                <span>Download</span>
              </button>
            )}

            {/* Share */}
            {currentImage && (
              <button
                onClick={() => handleShare(currentImage)}
                className="flex items-center space-x-2 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                title="Share image"
              >
                <Share2 size={16} />
                <span>Share</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div
        className={`${layout === "vertical" ? "space-y-4" : "grid grid-cols-1 lg:grid-cols-3 gap-4"}`}
      >
        {/* Main View */}
        <div className={layout === "vertical" ? "" : "lg:col-span-2"}>
          {viewMode === "comparison" ? (
            <ComparisonView
              originalImage={originalImage}
              editedImage={currentImage}
              onDownload={onImageDownload}
              onFullscreen={handleFullscreen}
            />
          ) : (
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="text-center text-gray-500">
                <Grid size={48} className="mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">Version View</h3>
                <p className="text-sm">
                  Select a version from the history to view it
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Version History Sidebar */}
        <div className={layout === "vertical" ? "" : "lg:col-span-1"}>
          <ImageVersioning
            versions={versions}
            currentVersion={currentImage?.id}
            onVersionSelect={onVersionSelect}
            onVersionDelete={onVersionDelete}
            onVersionFavorite={onVersionFavorite}
            onVersionDownload={onImageDownload}
          />
        </div>
      </div>
    </div>
  );
};

export default ImageWorkspace;
