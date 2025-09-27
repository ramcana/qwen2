import React, { useState } from "react";

interface QualityPreset {
  name: string;
  description: string;
  config: {
    eliGenMode: string;
    eliGenEntityDetection: boolean;
    eliGenQualityEnhancement: boolean;
    eliGenDetailEnhancement: number;
    eliGenColorEnhancement: number;
  };
}

interface QualityPresetsProps {
  currentPreset: string;
  onPresetChange: (preset: QualityPreset) => void;
  onSaveCustomPreset: (name: string, config: any) => void;
  customPresets?: QualityPreset[];
}

const QualityPresets: React.FC<QualityPresetsProps> = ({
  currentPreset,
  onPresetChange,
  onSaveCustomPreset,
  customPresets = [],
}) => {
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newPresetName, setNewPresetName] = useState("");

  const defaultPresets: QualityPreset[] = [
    {
      name: "Fast",
      description: "Quick processing with minimal enhancements",
      config: {
        eliGenMode: "fast",
        eliGenEntityDetection: false,
        eliGenQualityEnhancement: true,
        eliGenDetailEnhancement: 0.2,
        eliGenColorEnhancement: 0.1,
      },
    },
    {
      name: "Balanced",
      description: "Good quality with reasonable processing time",
      config: {
        eliGenMode: "balanced",
        eliGenEntityDetection: true,
        eliGenQualityEnhancement: true,
        eliGenDetailEnhancement: 0.5,
        eliGenColorEnhancement: 0.3,
      },
    },
    {
      name: "Quality",
      description: "High quality with entity detection and enhancement",
      config: {
        eliGenMode: "quality",
        eliGenEntityDetection: true,
        eliGenQualityEnhancement: true,
        eliGenDetailEnhancement: 0.7,
        eliGenColorEnhancement: 0.4,
      },
    },
    {
      name: "Ultra",
      description: "Maximum quality with multi-pass processing",
      config: {
        eliGenMode: "ultra",
        eliGenEntityDetection: true,
        eliGenQualityEnhancement: true,
        eliGenDetailEnhancement: 0.8,
        eliGenColorEnhancement: 0.5,
      },
    },
  ];

  const allPresets = [...defaultPresets, ...customPresets];

  const handleSavePreset = () => {
    if (newPresetName.trim()) {
      // Get current configuration (this would come from parent component)
      const currentConfig = {
        eliGenMode: "balanced", // This should come from current state
        eliGenEntityDetection: true,
        eliGenQualityEnhancement: true,
        eliGenDetailEnhancement: 0.5,
        eliGenColorEnhancement: 0.3,
      };

      onSaveCustomPreset(newPresetName.trim(), currentConfig);
      setNewPresetName("");
      setShowSaveDialog(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">Quality Presets</h3>
        <button
          onClick={() => setShowSaveDialog(true)}
          className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
        >
          Save Current
        </button>
      </div>

      {/* Preset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {allPresets.map((preset) => (
          <div
            key={preset.name}
            className={`p-4 border rounded-lg cursor-pointer transition-all ${
              currentPreset === preset.name
                ? "border-blue-500 bg-blue-50 shadow-md"
                : "border-gray-300 hover:border-gray-400 hover:shadow-sm"
            }`}
            onClick={() => onPresetChange(preset)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="font-medium text-gray-900">{preset.name}</h4>
                <p className="text-sm text-gray-600 mt-1">
                  {preset.description}
                </p>

                {/* Preset Details */}
                <div className="mt-2 space-y-1">
                  <div className="flex items-center space-x-2 text-xs text-gray-500">
                    <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                    <span>Mode: {preset.config.eliGenMode}</span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs text-gray-500">
                    <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                    <span>
                      Detail:{" "}
                      {(preset.config.eliGenDetailEnhancement * 100).toFixed(0)}
                      %
                    </span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs text-gray-500">
                    <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
                    <span>
                      Color:{" "}
                      {(preset.config.eliGenColorEnhancement * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Features Icons */}
              <div className="flex flex-col space-y-1 ml-2">
                {preset.config.eliGenEntityDetection && (
                  <div
                    className="w-6 h-6 bg-green-100 rounded flex items-center justify-center"
                    title="Entity Detection"
                  >
                    <svg
                      className="w-3 h-3 text-green-600"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                )}
                {preset.config.eliGenQualityEnhancement && (
                  <div
                    className="w-6 h-6 bg-blue-100 rounded flex items-center justify-center"
                    title="Quality Enhancement"
                  >
                    <svg
                      className="w-3 h-3 text-blue-600"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Save Preset Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 max-w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Save Custom Preset
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Preset Name
                </label>
                <input
                  type="text"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  placeholder="Enter preset name..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                />
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Current Settings
                </h4>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>• Entity Detection: Enabled</div>
                  <div>• Quality Enhancement: Enabled</div>
                  <div>• Detail Enhancement: 50%</div>
                  <div>• Color Enhancement: 30%</div>
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSavePreset}
                disabled={!newPresetName.trim()}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Save Preset
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Usage Tips */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
        <div className="flex items-start space-x-2">
          <svg
            className="w-5 h-5 text-yellow-600 mt-0.5"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
          <div>
            <h4 className="text-sm font-medium text-yellow-800">Preset Tips</h4>
            <ul className="text-xs text-yellow-700 mt-1 space-y-1">
              <li>• Fast: Best for quick iterations and previews</li>
              <li>• Balanced: Recommended for most use cases</li>
              <li>• Quality: Use when detail is important</li>
              <li>• Ultra: For final high-quality outputs (slower)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QualityPresets;
