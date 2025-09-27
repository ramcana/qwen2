import React, { useState } from "react";
import { Clock, Trash2, Download, Eye, RotateCcw, Star } from "lucide-react";
import { ImageData } from "./ComparisonView";

interface ImageVersion extends ImageData {
  isFavorite?: boolean;
  parentId?: string;
}

interface ImageVersioningProps {
  versions: ImageVersion[];
  currentVersion?: string;
  onVersionSelect: (version: ImageVersion) => void;
  onVersionDelete?: (versionId: string) => void;
  onVersionFavorite?: (versionId: string, isFavorite: boolean) => void;
  onVersionDownload?: (version: ImageVersion) => void;
  maxVersions?: number;
}

const ImageVersioning: React.FC<ImageVersioningProps> = ({
  versions,
  currentVersion,
  onVersionSelect,
  onVersionDelete,
  onVersionFavorite,
  onVersionDownload,
  maxVersions = 10,
}) => {
  const [showAll, setShowAll] = useState(false);

  const sortedVersions = [...versions].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const displayVersions = showAll
    ? sortedVersions
    : sortedVersions.slice(0, maxVersions);

  const handleDownload = (version: ImageVersion) => {
    if (onVersionDownload) {
      onVersionDownload(version);
    } else {
      // Default download behavior
      const link = document.createElement("a");
      link.href = version.url;
      link.download = `${version.label}_v${version.id}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const formatTimestamp = (timestamp: Date): string => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return timestamp.toLocaleDateString();
  };

  if (versions.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="text-center text-gray-500">
          <Clock size={48} className="mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No Version History</h3>
          <p className="text-sm">
            Generate images to start building version history
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">
            Version History
          </h3>
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Clock size={16} />
            <span>
              {versions.length} version{versions.length !== 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </div>

      {/* Version List */}
      <div className="max-h-96 overflow-y-auto">
        <div className="space-y-2 p-4">
          {displayVersions.map((version, index) => {
            const isActive = currentVersion === version.id;
            const isLatest = index === 0;

            return (
              <div
                key={version.id}
                className={`
                  relative flex items-center space-x-3 p-3 rounded-lg border transition-all duration-200 cursor-pointer
                  ${
                    isActive
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
                  }
                `}
                onClick={() => onVersionSelect(version)}
              >
                {/* Thumbnail */}
                <div className="flex-shrink-0">
                  <img
                    src={version.url}
                    alt={`Version ${version.id}`}
                    className="w-16 h-16 object-cover rounded-md border"
                  />
                </div>

                {/* Version Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <h4 className="text-sm font-medium text-gray-900 truncate">
                      {version.label}
                    </h4>
                    {isLatest && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                        Latest
                      </span>
                    )}
                    {version.isFavorite && (
                      <Star
                        size={14}
                        className="text-yellow-500 fill-current"
                      />
                    )}
                  </div>

                  <p className="text-xs text-gray-500 mt-1">
                    {formatTimestamp(version.timestamp)}
                  </p>

                  {version.metadata?.prompt && (
                    <p className="text-xs text-gray-600 mt-1 truncate">
                      {version.metadata.prompt}
                    </p>
                  )}

                  {version.metadata?.mode && (
                    <span className="inline-block mt-1 px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded">
                      {version.metadata.mode}
                    </span>
                  )}
                </div>

                {/* Actions */}
                <div className="flex-shrink-0 flex items-center space-x-1">
                  {/* Favorite Toggle */}
                  {onVersionFavorite && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onVersionFavorite(version.id, !version.isFavorite);
                      }}
                      className={`
                        p-1 rounded transition-colors
                        ${
                          version.isFavorite
                            ? "text-yellow-500 hover:text-yellow-600"
                            : "text-gray-400 hover:text-yellow-500"
                        }
                      `}
                      title={
                        version.isFavorite
                          ? "Remove from favorites"
                          : "Add to favorites"
                      }
                    >
                      <Star
                        size={14}
                        className={version.isFavorite ? "fill-current" : ""}
                      />
                    </button>
                  )}

                  {/* View Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onVersionSelect(version);
                    }}
                    className="p-1 text-gray-400 hover:text-blue-500 rounded transition-colors"
                    title="View this version"
                  >
                    <Eye size={14} />
                  </button>

                  {/* Download Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(version);
                    }}
                    className="p-1 text-gray-400 hover:text-green-500 rounded transition-colors"
                    title="Download this version"
                  >
                    <Download size={14} />
                  </button>

                  {/* Delete Button */}
                  {onVersionDelete && !isLatest && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (
                          confirm(
                            "Are you sure you want to delete this version?"
                          )
                        ) {
                          onVersionDelete(version.id);
                        }
                      }}
                      className="p-1 text-gray-400 hover:text-red-500 rounded transition-colors"
                      title="Delete this version"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Show More/Less Button */}
      {versions.length > maxVersions && (
        <div className="px-6 py-3 border-t border-gray-200">
          <button
            onClick={() => setShowAll(!showAll)}
            className="w-full text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            {showAll
              ? `Show Less (${maxVersions} of ${versions.length})`
              : `Show All (${versions.length} versions)`}
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageVersioning;
export type { ImageVersion };
