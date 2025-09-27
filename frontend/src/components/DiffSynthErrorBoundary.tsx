/**
 * DiffSynth Error Boundary Component
 * Provides graceful error handling for DiffSynth-related operations
 */

import React, { Component, ReactNode } from "react";
import {
  AlertTriangle,
  RefreshCw,
  Settings,
  HelpCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface DiffSynthErrorInfo {
  errorType: string;
  severity: "recoverable" | "degraded" | "critical" | "fatal";
  message: string;
  suggestedFixes: string[];
  limitations: string[];
  fallbackAvailable: boolean;
  userFriendlyMessage: string;
}

interface Props {
  children: ReactNode;
  fallbackComponent?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  onRetry?: () => void;
  onFallback?: () => void;
  enableFallback?: boolean;
  showTechnicalDetails?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  diffSynthError: DiffSynthErrorInfo | null;
  showDetails: boolean;
  retryCount: number;
  fallbackActivated: boolean;
}

class DiffSynthErrorBoundary extends Component<Props, State> {
  private maxRetries = 3;
  private retryTimeout: NodeJS.Timeout | null = null;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      diffSynthError: null,
      showDetails: false,
      retryCount: 0,
      fallbackActivated: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error(
      "DiffSynth Error Boundary caught an error:",
      error,
      errorInfo
    );

    // Parse DiffSynth-specific error information
    const diffSynthError = this.parseDiffSynthError(error);

    this.setState({
      errorInfo,
      diffSynthError,
    });

    // Call external error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Report error to monitoring service
    this.reportError(error, errorInfo, diffSynthError);
  }

  private parseDiffSynthError(error: Error): DiffSynthErrorInfo | null {
    const errorMessage = error.message.toLowerCase();
    const errorStack = error.stack?.toLowerCase() || "";

    // Check for DiffSynth-specific error patterns
    if (
      errorMessage.includes("diffsynth") ||
      errorStack.includes("diffsynth")
    ) {
      // Memory errors
      if (
        errorMessage.includes("memory") ||
        errorMessage.includes("cuda out of memory")
      ) {
        return {
          errorType: "memory_allocation",
          severity: "degraded",
          message: error.message,
          suggestedFixes: [
            "Reduce image resolution",
            "Enable CPU offloading",
            "Close other applications using GPU",
            "Try tiled processing for large images",
          ],
          limitations: [
            "May process at lower quality",
            "Slower processing speed",
          ],
          fallbackAvailable: true,
          userFriendlyMessage:
            "GPU memory is full. We can try processing with reduced settings or use CPU instead.",
        };
      }

      // Service initialization errors
      if (
        errorMessage.includes("initialization") ||
        errorMessage.includes("import")
      ) {
        return {
          errorType: "service_initialization",
          severity: "critical",
          message: error.message,
          suggestedFixes: [
            "Check if DiffSynth is properly installed",
            "Verify system requirements",
            "Restart the application",
            "Check GPU drivers",
          ],
          limitations: [
            "DiffSynth features unavailable",
            "Limited to basic image generation",
          ],
          fallbackAvailable: true,
          userFriendlyMessage:
            "DiffSynth service is not available. You can still use basic image generation features.",
        };
      }

      // ControlNet errors
      if (
        errorMessage.includes("controlnet") ||
        errorMessage.includes("control")
      ) {
        return {
          errorType: "controlnet_processing",
          severity: "recoverable",
          message: error.message,
          suggestedFixes: [
            "Check control image format",
            "Try a different ControlNet type",
            "Verify image dimensions",
            "Use automatic detection",
          ],
          limitations: [
            "ControlNet guidance not available",
            "Standard generation only",
          ],
          fallbackAvailable: true,
          userFriendlyMessage:
            "ControlNet processing failed. We can generate without structural guidance.",
        };
      }

      // Image processing errors
      if (
        errorMessage.includes("processing") ||
        errorMessage.includes("generation")
      ) {
        return {
          errorType: "image_processing",
          severity: "recoverable",
          message: error.message,
          suggestedFixes: [
            "Try different generation parameters",
            "Check input image format",
            "Reduce complexity of the prompt",
            "Use a different random seed",
          ],
          limitations: [
            "May need to adjust settings",
            "Some features may be disabled",
          ],
          fallbackAvailable: true,
          userFriendlyMessage:
            "Image processing encountered an issue. Let's try with different settings.",
        };
      }

      // Generic DiffSynth error
      return {
        errorType: "diffsynth_general",
        severity: "degraded",
        message: error.message,
        suggestedFixes: [
          "Try refreshing the page",
          "Check your internet connection",
          "Verify system resources",
          "Contact support if issue persists",
        ],
        limitations: ["Some advanced features may be unavailable"],
        fallbackAvailable: true,
        userFriendlyMessage:
          "DiffSynth encountered an issue. Basic features are still available.",
      };
    }

    return null;
  }

  private reportError(
    error: Error,
    errorInfo: React.ErrorInfo,
    diffSynthError: DiffSynthErrorInfo | null
  ) {
    // Report to error tracking service
    try {
      // This would integrate with your error reporting service
      console.log("Error reported:", {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        diffSynthError,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      });
    } catch (reportingError) {
      console.error("Failed to report error:", reportingError);
    }
  }

  private handleRetry = () => {
    if (this.state.retryCount >= this.maxRetries) {
      return;
    }

    this.setState((prevState) => ({
      hasError: false,
      error: null,
      errorInfo: null,
      diffSynthError: null,
      retryCount: prevState.retryCount + 1,
    }));

    if (this.props.onRetry) {
      this.props.onRetry();
    }
  };

  private handleFallback = () => {
    this.setState({
      fallbackActivated: true,
    });

    if (this.props.onFallback) {
      this.props.onFallback();
    }
  };

  private toggleDetails = () => {
    this.setState((prevState) => ({
      showDetails: !prevState.showDetails,
    }));
  };

  private getSeverityColor(severity: string): string {
    switch (severity) {
      case "recoverable":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "degraded":
        return "text-orange-600 bg-orange-50 border-orange-200";
      case "critical":
        return "text-red-600 bg-red-50 border-red-200";
      case "fatal":
        return "text-red-800 bg-red-100 border-red-300";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  }

  private getSeverityIcon(severity: string) {
    switch (severity) {
      case "recoverable":
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case "degraded":
        return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      case "critical":
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case "fatal":
        return <AlertTriangle className="w-5 h-5 text-red-600" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-gray-500" />;
    }
  }

  render() {
    if (this.state.hasError) {
      const { diffSynthError, showDetails, retryCount, fallbackActivated } =
        this.state;
      const canRetry = retryCount < this.maxRetries;
      const canFallback =
        this.props.enableFallback &&
        diffSynthError?.fallbackAvailable &&
        !fallbackActivated;

      // If fallback is activated and a fallback component is provided
      if (fallbackActivated && this.props.fallbackComponent) {
        return (
          <div className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Settings className="w-5 h-5 text-blue-500" />
                <span className="text-blue-700 font-medium">
                  Fallback Mode Active
                </span>
              </div>
              <p className="text-blue-600 text-sm mt-1">
                Using alternative processing method with limited features.
              </p>
            </div>
            {this.props.fallbackComponent}
          </div>
        );
      }

      return (
        <div className="min-h-[400px] flex items-center justify-center p-6">
          <div className="max-w-2xl w-full space-y-6">
            {/* Main Error Display */}
            <div
              className={`border rounded-lg p-6 ${diffSynthError ? this.getSeverityColor(diffSynthError.severity) : "text-red-600 bg-red-50 border-red-200"}`}
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  {diffSynthError ? (
                    this.getSeverityIcon(diffSynthError.severity)
                  ) : (
                    <AlertTriangle className="w-5 h-5 text-red-500" />
                  )}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg mb-2">
                    {diffSynthError ? "DiffSynth Error" : "Application Error"}
                  </h3>
                  <p className="mb-4">
                    {diffSynthError?.userFriendlyMessage ||
                      "An unexpected error occurred. Please try again."}
                  </p>

                  {/* Suggested Fixes */}
                  {diffSynthError?.suggestedFixes &&
                    diffSynthError.suggestedFixes.length > 0 && (
                      <div className="mb-4">
                        <h4 className="font-medium mb-2">
                          Suggested Solutions:
                        </h4>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                          {diffSynthError.suggestedFixes.map((fix, index) => (
                            <li key={index}>{fix}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {/* Limitations */}
                  {diffSynthError?.limitations &&
                    diffSynthError.limitations.length > 0 && (
                      <div className="mb-4">
                        <h4 className="font-medium mb-2">
                          Current Limitations:
                        </h4>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                          {diffSynthError.limitations.map(
                            (limitation, index) => (
                              <li key={index}>{limitation}</li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                  {/* Action Buttons */}
                  <div className="flex flex-wrap gap-3">
                    {canRetry && (
                      <button
                        onClick={this.handleRetry}
                        className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                      >
                        <RefreshCw className="w-4 h-4" />
                        <span>
                          Try Again ({this.maxRetries - retryCount} left)
                        </span>
                      </button>
                    )}

                    {canFallback && (
                      <button
                        onClick={this.handleFallback}
                        className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                      >
                        <Settings className="w-4 h-4" />
                        <span>Use Fallback Mode</span>
                      </button>
                    )}

                    <button
                      onClick={() => window.location.reload()}
                      className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
                    >
                      <RefreshCw className="w-4 h-4" />
                      <span>Refresh Page</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Technical Details (Collapsible) */}
            {(this.props.showTechnicalDetails ||
              process.env.NODE_ENV === "development") && (
              <div className="border border-gray-200 rounded-lg">
                <button
                  onClick={this.toggleDetails}
                  className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-2">
                    <HelpCircle className="w-4 h-4 text-gray-500" />
                    <span className="font-medium text-gray-700">
                      Technical Details
                    </span>
                  </div>
                  {showDetails ? (
                    <ChevronUp className="w-4 h-4 text-gray-500" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-gray-500" />
                  )}
                </button>

                {showDetails && (
                  <div className="border-t border-gray-200 p-4 bg-gray-50">
                    <div className="space-y-4 text-sm">
                      {diffSynthError && (
                        <div>
                          <h5 className="font-medium text-gray-700 mb-1">
                            Error Type:
                          </h5>
                          <p className="text-gray-600 font-mono">
                            {diffSynthError.errorType}
                          </p>
                        </div>
                      )}

                      <div>
                        <h5 className="font-medium text-gray-700 mb-1">
                          Error Message:
                        </h5>
                        <p className="text-gray-600 font-mono break-all">
                          {this.state.error?.message || "Unknown error"}
                        </p>
                      </div>

                      {this.state.error?.stack && (
                        <div>
                          <h5 className="font-medium text-gray-700 mb-1">
                            Stack Trace:
                          </h5>
                          <pre className="text-xs text-gray-600 bg-white p-2 rounded border overflow-x-auto">
                            {this.state.error.stack}
                          </pre>
                        </div>
                      )}

                      <div>
                        <h5 className="font-medium text-gray-700 mb-1">
                          Retry Count:
                        </h5>
                        <p className="text-gray-600">
                          {retryCount} / {this.maxRetries}
                        </p>
                      </div>

                      <div>
                        <h5 className="font-medium text-gray-700 mb-1">
                          Timestamp:
                        </h5>
                        <p className="text-gray-600">
                          {new Date().toISOString()}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Help Text */}
            <div className="text-center text-gray-500 text-sm">
              <p>
                If this problem persists, please{" "}
                <a
                  href="mailto:support@example.com"
                  className="text-blue-600 hover:text-blue-800 underline"
                >
                  contact support
                </a>{" "}
                with the technical details above.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default DiffSynthErrorBoundary;
