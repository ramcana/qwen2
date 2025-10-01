/**
 * Error Reporting and Feedback Service
 * Handles error reporting, user feedback collection, and error analytics
 */

interface ErrorReport {
  id: string;
  timestamp: string;
  errorType: string;
  severity: "low" | "medium" | "high" | "critical";
  message: string;
  stack?: string;
  componentStack?: string;
  userAgent: string;
  url: string;
  userId?: string;
  sessionId: string;
  buildVersion?: string;
  additionalContext?: Record<string, any>;
}

interface UserFeedback {
  errorId: string;
  rating: 1 | 2 | 3 | 4 | 5;
  feedback: string;
  helpfulSuggestions?: string[];
  wouldRecommendFallback: boolean;
  timestamp: string;
}

interface ErrorAnalytics {
  errorCount: number;
  errorsByType: Record<string, number>;
  errorsBySeverity: Record<string, number>;
  averageResolutionTime: number;
  fallbackUsageRate: number;
  userSatisfactionScore: number;
}

class ErrorReportingService {
  private apiEndpoint: string;
  private sessionId: string;
  private userId?: string;
  private buildVersion?: string;
  private errorQueue: ErrorReport[] = [];
  private feedbackQueue: UserFeedback[] = [];
  private isOnline: boolean = navigator.onLine;
  private maxQueueSize: number = 100;

  constructor(
    config: {
      apiEndpoint?: string;
      userId?: string;
      buildVersion?: string;
    } = {}
  ) {
    this.apiEndpoint = config.apiEndpoint || "/api/error-reporting";
    this.userId = config.userId;
    this.buildVersion = config.buildVersion || process.env.REACT_APP_VERSION;
    this.sessionId = this.generateSessionId();

    // Listen for online/offline events
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.flushQueues();
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
    });

    // Flush queues periodically
    setInterval(() => {
      if (this.isOnline) {
        this.flushQueues();
      }
    }, 30000); // Every 30 seconds

    // Flush queues before page unload
    window.addEventListener("beforeunload", () => {
      this.flushQueues();
    });
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Report an error to the error tracking service
   */
  async reportError(
    error: Error,
    errorInfo?: React.ErrorInfo,
    additionalContext?: Record<string, any>
  ): Promise<string> {
    const errorId = `error_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    const errorReport: ErrorReport = {
      id: errorId,
      timestamp: new Date().toISOString(),
      errorType: this.classifyError(error),
      severity: this.determineSeverity(error),
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo?.componentStack || undefined,
      userAgent: navigator.userAgent,
      url: window.location.href,
      userId: this.userId,
      sessionId: this.sessionId,
      buildVersion: this.buildVersion,
      additionalContext: {
        ...additionalContext,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
        },
        timestamp: Date.now(),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
    };

    // Add to queue
    this.errorQueue.push(errorReport);

    // Limit queue size
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue = this.errorQueue.slice(-this.maxQueueSize);
    }

    // Try to send immediately if online
    if (this.isOnline) {
      await this.flushErrorQueue();
    }

    console.log("Error reported:", errorId, errorReport);
    return errorId;
  }

  /**
   * Collect user feedback about an error
   */
  async submitFeedback(
    feedback: Omit<UserFeedback, "timestamp">
  ): Promise<void> {
    const feedbackWithTimestamp: UserFeedback = {
      ...feedback,
      timestamp: new Date().toISOString(),
    };

    this.feedbackQueue.push(feedbackWithTimestamp);

    // Try to send immediately if online
    if (this.isOnline) {
      await this.flushFeedbackQueue();
    }

    console.log("Feedback submitted:", feedbackWithTimestamp);
  }

  /**
   * Get error analytics and statistics
   */
  async getErrorAnalytics(timeRange?: {
    start: string;
    end: string;
  }): Promise<ErrorAnalytics> {
    try {
      const params = new URLSearchParams();
      if (timeRange) {
        params.append("start", timeRange.start);
        params.append("end", timeRange.end);
      }
      if (this.userId) {
        params.append("userId", this.userId);
      }

      const response = await fetch(`${this.apiEndpoint}/analytics?${params}`);

      if (!response.ok) {
        throw new Error(`Analytics request failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Failed to fetch error analytics:", error);

      // Return default analytics
      return {
        errorCount: 0,
        errorsByType: {},
        errorsBySeverity: {},
        averageResolutionTime: 0,
        fallbackUsageRate: 0,
        userSatisfactionScore: 0,
      };
    }
  }

  /**
   * Check if a specific error has been resolved
   */
  async checkErrorStatus(errorId: string): Promise<{
    status: "open" | "investigating" | "resolved" | "closed";
    resolution?: string;
    estimatedFixTime?: string;
  }> {
    try {
      const response = await fetch(`${this.apiEndpoint}/status/${errorId}`);

      if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Failed to check error status:", error);
      return { status: "open" };
    }
  }

  private classifyError(error: Error): string {
    const message = error.message.toLowerCase();
    const stack = error.stack?.toLowerCase() || "";

    // DiffSynth-specific errors
    if (message.includes("diffsynth") || stack.includes("diffsynth")) {
      if (
        message.includes("memory") ||
        message.includes("cuda out of memory")
      ) {
        return "diffsynth_memory_error";
      }
      if (message.includes("initialization") || message.includes("import")) {
        return "diffsynth_initialization_error";
      }
      if (message.includes("controlnet")) {
        return "diffsynth_controlnet_error";
      }
      if (message.includes("processing") || message.includes("generation")) {
        return "diffsynth_processing_error";
      }
      return "diffsynth_general_error";
    }

    // Network errors
    if (message.includes("network") || message.includes("fetch")) {
      return "network_error";
    }

    // React errors
    if (stack.includes("react") || message.includes("component")) {
      return "react_component_error";
    }

    // JavaScript errors
    if (error instanceof TypeError) {
      return "type_error";
    }
    if (error instanceof ReferenceError) {
      return "reference_error";
    }
    if (error instanceof SyntaxError) {
      return "syntax_error";
    }

    return "unknown_error";
  }

  private determineSeverity(
    error: Error
  ): "low" | "medium" | "high" | "critical" {
    const message = error.message.toLowerCase();

    // Critical errors that break core functionality
    if (
      message.includes("initialization") ||
      message.includes("service unavailable") ||
      message.includes("fatal")
    ) {
      return "critical";
    }

    // High severity errors that significantly impact user experience
    if (
      message.includes("memory") ||
      message.includes("processing failed") ||
      message.includes("generation failed")
    ) {
      return "high";
    }

    // Medium severity errors that cause feature degradation
    if (
      message.includes("controlnet") ||
      message.includes("fallback") ||
      message.includes("degraded")
    ) {
      return "medium";
    }

    // Low severity errors that have minimal impact
    return "low";
  }

  private async flushQueues(): Promise<void> {
    await Promise.all([this.flushErrorQueue(), this.flushFeedbackQueue()]);
  }

  private async flushErrorQueue(): Promise<void> {
    if (this.errorQueue.length === 0) return;

    try {
      const errors = [...this.errorQueue];
      this.errorQueue = [];

      const response = await fetch(`${this.apiEndpoint}/errors`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ errors }),
      });

      if (!response.ok) {
        // Put errors back in queue if request failed
        this.errorQueue.unshift(...errors);
        throw new Error(`Error reporting failed: ${response.status}`);
      }

      console.log(`Successfully reported ${errors.length} errors`);
    } catch (error) {
      console.error("Failed to flush error queue:", error);
    }
  }

  private async flushFeedbackQueue(): Promise<void> {
    if (this.feedbackQueue.length === 0) return;

    try {
      const feedback = [...this.feedbackQueue];
      this.feedbackQueue = [];

      const response = await fetch(`${this.apiEndpoint}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ feedback }),
      });

      if (!response.ok) {
        // Put feedback back in queue if request failed
        this.feedbackQueue.unshift(...feedback);
        throw new Error(`Feedback submission failed: ${response.status}`);
      }

      console.log(`Successfully submitted ${feedback.length} feedback items`);
    } catch (error) {
      console.error("Failed to flush feedback queue:", error);
    }
  }

  /**
   * Get queued items count (for debugging)
   */
  getQueueStatus(): {
    errorQueue: number;
    feedbackQueue: number;
    isOnline: boolean;
  } {
    return {
      errorQueue: this.errorQueue.length,
      feedbackQueue: this.feedbackQueue.length,
      isOnline: this.isOnline,
    };
  }

  /**
   * Clear all queued items (for testing)
   */
  clearQueues(): void {
    this.errorQueue = [];
    this.feedbackQueue = [];
  }
}

// Create singleton instance
export const errorReportingService = new ErrorReportingService({
  apiEndpoint:
    process.env.REACT_APP_ERROR_REPORTING_ENDPOINT || "/api/error-reporting",
  buildVersion: process.env.REACT_APP_VERSION,
});

// React hook for error reporting
export const useErrorReporting = () => {
  const reportError = async (
    error: Error,
    errorInfo?: React.ErrorInfo,
    additionalContext?: Record<string, any>
  ) => {
    return await errorReportingService.reportError(
      error,
      errorInfo,
      additionalContext
    );
  };

  const submitFeedback = async (feedback: Omit<UserFeedback, "timestamp">) => {
    return await errorReportingService.submitFeedback(feedback);
  };

  const getAnalytics = async (timeRange?: { start: string; end: string }) => {
    return await errorReportingService.getErrorAnalytics(timeRange);
  };

  const checkStatus = async (errorId: string) => {
    return await errorReportingService.checkErrorStatus(errorId);
  };

  return {
    reportError,
    submitFeedback,
    getAnalytics,
    checkStatus,
    getQueueStatus: () => errorReportingService.getQueueStatus(),
  };
};

export default errorReportingService;
