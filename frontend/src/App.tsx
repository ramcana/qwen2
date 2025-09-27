import React, { useState } from "react";
import { toast } from "react-hot-toast";
import { useQuery } from "react-query";
import GenerationPanel from "./components/GenerationPanel";
import EditPanel from "./components/EditPanel";
import ControlNetPanel from "./components/ControlNetPanel";
import Header from "./components/Header";
import ImageDisplay from "./components/ImageDisplay";
import StatusBar from "./components/StatusBar";
import ModeSelector, { WorkflowMode } from "./components/ModeSelector";
import Breadcrumb from "./components/Breadcrumb";
import DiffSynthErrorBoundary from "./components/DiffSynthErrorBoundary";
import {
  DiffSynthFallbackUI,
  SimplifiedEditPanel,
} from "./components/DiffSynthFallbackUI";
import { useWorkspaceState } from "./hooks/useWorkspaceState";
import { getStatus } from "./services/api";
import { useErrorReporting } from "./services/errorReporting";

const App: React.FC = () => {
  const [showInitializationDetails, setShowInitializationDetails] =
    useState(false);
  const [fallbackMode, setFallbackMode] = useState(false);
  const { currentMode, switchMode } = useWorkspaceState();
  const { reportError } = useErrorReporting();

  const {
    data: status,
    isLoading,
    error,
    refetch,
  } = useQuery("status", getStatus, {
    refetchInterval: (data) => {
      // More frequent updates during initialization
      if (
        data?.initialization?.status === "loading_model" ||
        data?.initialization?.status === "downloading_model" ||
        data?.initialization?.status === "loading_to_gpu"
      ) {
        return 2000; // 2 seconds during model loading
      }
      // If model is loaded, check less frequently
      return data?.model_loaded ? 30000 : 5000;
    },
    refetchOnWindowFocus: false,
    staleTime: 5000, // Shorter stale time for better initialization feedback
    retry: (failureCount, error) => {
      return failureCount < 3;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * attemptIndex, 5000),
    onError: (error: any) => {
      // Only show error toast if it's not a connection error
      if (
        !error?.message?.includes("Network Error") &&
        !error?.code?.includes("ERR_NETWORK")
      ) {
        toast.error("Failed to connect to API server");
      }
    },
  });

  const handleError = (error: Error, errorInfo: React.ErrorInfo) => {
    reportError(error, errorInfo, {
      currentMode,
      userAgent: navigator.userAgent,
      timestamp: Date.now(),
    });
  };

  const handleRetry = () => {
    refetch();
  };

  const handleFallback = () => {
    setFallbackMode(true);
  };

  const fallbackComponent = fallbackMode ? (
    <DiffSynthFallbackUI
      fallbackType="service_unavailable"
      availableFeatures={[
        "Basic text-to-image generation",
        "Standard resolution output",
        "Simple prompt processing",
      ]}
      limitations={[
        "Advanced editing features unavailable",
        "No ControlNet support",
        "Limited to basic generation",
      ]}
      onRetryOriginal={() => {
        setFallbackMode(false);
        refetch();
      }}
    />
  ) : null;

  return (
    <DiffSynthErrorBoundary
      onError={handleError}
      onRetry={handleRetry}
      onFallback={handleFallback}
      enableFallback={true}
      fallbackComponent={fallbackComponent}
      showTechnicalDetails={process.env.NODE_ENV === "development"}
    >
      <div className="min-h-screen bg-gray-50">
        <Header />

        <main className="container mx-auto px-4 py-6">
          {/* Mode Selection and Navigation */}
          <div className="mb-6 space-y-4">
            <ModeSelector
              currentMode={currentMode}
              onModeChange={switchMode}
              disabled={isLoading}
            />
            <Breadcrumb currentMode={currentMode} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Panel - Controls */}
            <div className="lg:col-span-1 space-y-6">
              <StatusBar
                status={status}
                isLoading={isLoading}
                error={error}
                onRetry={() => refetch()}
              />

              {/* Render different panels based on current mode */}
              {currentMode === "generate" && (
                <DiffSynthErrorBoundary
                  onError={handleError}
                  enableFallback={true}
                  fallbackComponent={<SimplifiedEditPanel />}
                >
                  <GenerationPanel />
                </DiffSynthErrorBoundary>
              )}

              {currentMode === "edit" && (
                <DiffSynthErrorBoundary
                  onError={handleError}
                  enableFallback={true}
                  fallbackComponent={<SimplifiedEditPanel />}
                >
                  <EditPanel
                    onGenerate={(params) => {
                      console.log("Edit generation requested:", params);
                      // TODO: Implement actual edit generation API call
                    }}
                    isGenerating={false}
                  />
                </DiffSynthErrorBoundary>
              )}

              {currentMode === "controlnet" && (
                <DiffSynthErrorBoundary
                  onError={handleError}
                  enableFallback={true}
                  fallbackComponent={<SimplifiedEditPanel />}
                >
                  <ControlNetPanel
                    onGenerate={(params) => {
                      console.log("ControlNet generation requested:", params);
                      // TODO: Implement actual ControlNet generation API call
                    }}
                    onDetectControl={async (image, type) => {
                      console.log("Control detection requested:", {
                        image: image.name,
                        type,
                      });
                      // TODO: Implement actual control detection API call
                      // For now, return a placeholder
                      return "data:image/png;base64,placeholder";
                    }}
                    isGenerating={false}
                    isDetecting={false}
                  />
                </DiffSynthErrorBoundary>
              )}
            </div>

            {/* Right Panel - Results */}
            <div className="lg:col-span-2">
              <DiffSynthErrorBoundary
                onError={handleError}
                enableFallback={true}
              >
                <ImageDisplay />
              </DiffSynthErrorBoundary>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t mt-12">
          <div className="container mx-auto px-4 py-6 text-center text-gray-600">
            <p>Qwen-Image Generator v2.0 | Professional AI Image Generation</p>
            <p className="text-sm mt-1">
              Optimized for RTX 4080 • Memory-Managed • React + FastAPI
            </p>
          </div>
        </footer>
      </div>
    </DiffSynthErrorBoundary>
  );
};

export default App;
