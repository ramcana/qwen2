import React from "react";

interface EliGenControlsProps {
  eliGenEnabled: boolean;
  eliGenMode: string;
  eliGenEntityDetection: boolean;
  eliGenQualityEnhancement: boolean;
  eliGenDetailEnhancement: number;
  eliGenColorEnhancement: number;
  onEliGenEnabledChange: (enabled: boolean) => void;
  onEliGenModeChange: (mode: string) => void;
  onEliGenEntityDetectionChange: (enabled: boolean) => void;
  onEliGenQualityEnhancementChange: (enabled: boolean) => void;
  onEliGenDetailEnhancementChange: (value: number) => void;
  onEliGenColorEnhancementChange: (value: number) => void;
}

const EliGenControls: React.FC<EliGenControlsProps> = ({
  eliGenEnabled,
  eliGenMode,
  eliGenEntityDetection,
  eliGenQualityEnhancement,
  eliGenDetailEnhancement,
  eliGenColorEnhancement,
  onEliGenEnabledChange,
  onEliGenModeChange,
  onEliGenEntityDetectionChange,
  onEliGenQualityEnhancementChange,
  onEliGenDetailEnhancementChange,
  onEliGenColorEnhancementChange,
}) => {
  const qualityModes = [
    {
      value: "fast",
      label: "Fast",
      description: "Quick processing with basic enhancements",
    },
    {
      value: "balanced",
      label: "Balanced",
      description: "Good quality with reasonable speed",
    },
    {
      value: "quality",
      label: "Quality",
      description: "High quality with entity detection",
    },
    {
      value: "ultra",
      label: "Ultra",
      description: "Maximum quality with multi-pass refinement",
    },
  ];

  return (
    <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">
          EliGen Enhancement
        </h3>
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-600">Enable EliGen</label>
          <input
            type="checkbox"
            checked={eliGenEnabled}
            onChange={(e) => onEliGenEnabledChange(e.target.checked)}
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
          />
        </div>
      </div>

      {eliGenEnabled && (
        <div className="space-y-4">
          {/* Quality Mode Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quality Mode
            </label>
            <div className="grid grid-cols-2 gap-2">
              {qualityModes.map((mode) => (
                <div
                  key={mode.value}
                  className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                    eliGenMode === mode.value
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-300 hover:border-gray-400"
                  }`}
                  onClick={() => onEliGenModeChange(mode.value)}
                >
                  <div className="font-medium text-sm">{mode.label}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {mode.description}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">
                  Entity Detection
                </label>
                <p className="text-xs text-gray-500">
                  Automatically detect faces, objects, and regions
                </p>
              </div>
              <input
                type="checkbox"
                checked={eliGenEntityDetection}
                onChange={(e) =>
                  onEliGenEntityDetectionChange(e.target.checked)
                }
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">
                  Quality Enhancement
                </label>
                <p className="text-xs text-gray-500">
                  Apply post-processing quality improvements
                </p>
              </div>
              <input
                type="checkbox"
                checked={eliGenQualityEnhancement}
                onChange={(e) =>
                  onEliGenQualityEnhancementChange(e.target.checked)
                }
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Enhancement Sliders */}
          {eliGenQualityEnhancement && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Detail Enhancement:{" "}
                  {(eliGenDetailEnhancement * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={eliGenDetailEnhancement}
                  onChange={(e) =>
                    onEliGenDetailEnhancementChange(parseFloat(e.target.value))
                  }
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Subtle</span>
                  <span>Strong</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Color Enhancement: {(eliGenColorEnhancement * 100).toFixed(0)}
                  %
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={eliGenColorEnhancement}
                  onChange={(e) =>
                    onEliGenColorEnhancementChange(parseFloat(e.target.value))
                  }
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Natural</span>
                  <span>Vibrant</span>
                </div>
              </div>
            </div>
          )}

          {/* Quality Mode Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="text-sm text-blue-800">
              <strong>
                Current Mode:{" "}
                {qualityModes.find((m) => m.value === eliGenMode)?.label}
              </strong>
            </div>
            <div className="text-xs text-blue-600 mt-1">
              {qualityModes.find((m) => m.value === eliGenMode)?.description}
            </div>
            {eliGenMode === "ultra" && (
              <div className="text-xs text-blue-600 mt-2">
                ⚠️ Ultra mode uses multi-pass processing and may take longer
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EliGenControls;
