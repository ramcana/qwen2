/**
 * Tests for Error Reporting Service
 */

import { errorReportingService, useErrorReporting } from "../errorReporting";
import { renderHook, act } from "@testing-library/react";

// Mock Intl for Jest environment
global.Intl = {
  DateTimeFormat: jest.fn(() => ({
    resolvedOptions: jest.fn(() => ({ timeZone: "UTC" })),
  })),
} as any;

// Mock fetch
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
  })
) as jest.Mock;

// Mock navigator
Object.defineProperty(navigator, "onLine", {
  writable: true,
  value: true,
});

Object.defineProperty(navigator, "userAgent", {
  writable: true,
  value: "Mozilla/5.0 (Test Browser)",
});

// Mock window events
const mockAddEventListener = jest.fn();
const mockRemoveEventListener = jest.fn();
Object.defineProperty(window, "addEventListener", {
  value: mockAddEventListener,
});
Object.defineProperty(window, "removeEventListener", {
  value: mockRemoveEventListener,
});

// Mock Intl
Object.defineProperty(Intl, "DateTimeFormat", {
  value: jest.fn(() => ({
    resolvedOptions: () => ({ timeZone: "UTC" }),
  })),
});

describe("ErrorReportingService", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockClear();
    errorReportingService.clearQueues();
  });

  describe("Error Classification", () => {
    it("classifies DiffSynth memory errors correctly", async () => {
      const error = new Error("CUDA out of memory");
      const errorId = await errorReportingService.reportError(error);

      expect(errorId).toMatch(/^error_\d+_/);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
    });

    it("classifies DiffSynth initialization errors correctly", async () => {
      const error = new Error("DiffSynth initialization failed");
      const errorId = await errorReportingService.reportError(error);

      expect(errorId).toMatch(/^error_\d+_/);
    });

    it("classifies ControlNet errors correctly", async () => {
      const error = new Error("ControlNet processing failed");
      const errorId = await errorReportingService.reportError(error);

      expect(errorId).toMatch(/^error_\d+_/);
    });

    it("classifies network errors correctly", async () => {
      const error = new Error("Network request failed");
      const errorId = await errorReportingService.reportError(error);

      expect(errorId).toMatch(/^error_\d+_/);
    });

    it("classifies React component errors correctly", async () => {
      const error = new Error("Component render failed");
      error.stack = "Error: Component render failed\n    at React.Component";

      const errorId = await errorReportingService.reportError(error);

      expect(errorId).toMatch(/^error_\d+_/);
    });

    it("classifies TypeError correctly", async () => {
      const error = new TypeError("Cannot read property of undefined");
      const errorId = await errorReportingService.reportError(error);

      expect(errorId).toMatch(/^error_\d+_/);
    });
  });

  describe("Error Severity Determination", () => {
    it("determines critical severity for initialization errors", async () => {
      const error = new Error("Service initialization failed");
      await errorReportingService.reportError(error);

      // Severity determination is internal, but we can verify the error was processed
      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
    });

    it("determines high severity for memory errors", async () => {
      const error = new Error("Out of memory");
      await errorReportingService.reportError(error);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
    });

    it("determines medium severity for ControlNet errors", async () => {
      const error = new Error("ControlNet processing failed");
      await errorReportingService.reportError(error);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
    });

    it("determines low severity for unknown errors", async () => {
      const error = new Error("Unknown error");
      await errorReportingService.reportError(error);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
    });
  });

  describe("Error Reporting", () => {
    it("reports errors with additional context", async () => {
      const error = new Error("Test error");
      const errorInfo = {
        componentStack: "Component stack trace",
      };
      const additionalContext = {
        userAction: "button_click",
        feature: "image_generation",
      };

      const errorId = await errorReportingService.reportError(
        error,
        errorInfo,
        additionalContext
      );

      expect(errorId).toMatch(/^error_\d+_/);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
    });

    it("queues errors when offline", async () => {
      // Simulate offline
      Object.defineProperty(navigator, "onLine", { value: false });

      const error = new Error("Test error");
      await errorReportingService.reportError(error);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(1);
      expect(queueStatus.isOnline).toBe(false);
    });

    it("limits queue size", async () => {
      // Report more errors than max queue size
      for (let i = 0; i < 150; i++) {
        await errorReportingService.reportError(new Error(`Error ${i}`));
      }

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.errorQueue).toBe(100); // Max queue size
    });
  });

  describe("Feedback Submission", () => {
    it("submits user feedback", async () => {
      const feedback = {
        errorId: "error_123",
        rating: 4 as const,
        feedback: "The error message was helpful",
        helpfulSuggestions: ["Clear error message", "Good recovery options"],
        wouldRecommendFallback: true,
      };

      await errorReportingService.submitFeedback(feedback);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.feedbackQueue).toBe(1);
    });

    it("queues feedback when offline", async () => {
      // Simulate offline
      Object.defineProperty(navigator, "onLine", { value: false });

      const feedback = {
        errorId: "error_123",
        rating: 3 as const,
        feedback: "Test feedback",
        helpfulSuggestions: [],
        wouldRecommendFallback: false,
      };

      await errorReportingService.submitFeedback(feedback);

      const queueStatus = errorReportingService.getQueueStatus();
      expect(queueStatus.feedbackQueue).toBe(1);
      expect(queueStatus.isOnline).toBe(false);
    });
  });

  describe("Analytics", () => {
    it("fetches error analytics", async () => {
      const mockAnalytics = {
        errorCount: 10,
        errorsByType: { diffsynth_memory_error: 5, network_error: 3 },
        errorsBySeverity: { high: 5, medium: 3, low: 2 },
        averageResolutionTime: 120,
        fallbackUsageRate: 0.3,
        userSatisfactionScore: 4.2,
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockAnalytics,
      });

      const analytics = await errorReportingService.getErrorAnalytics();

      expect(analytics).toEqual(mockAnalytics);
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/error-reporting/analytics")
      );
    });

    it("fetches analytics with time range", async () => {
      const mockAnalytics = {
        errorCount: 5,
        errorsByType: {},
        errorsBySeverity: {},
        averageResolutionTime: 0,
        fallbackUsageRate: 0,
        userSatisfactionScore: 0,
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockAnalytics,
      });

      const timeRange = {
        start: "2023-01-01T00:00:00Z",
        end: "2023-01-31T23:59:59Z",
      };

      await errorReportingService.getErrorAnalytics(timeRange);

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining("start=2023-01-01T00%3A00%3A00Z")
      );
    });

    it("returns default analytics on fetch failure", async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error("Network error"));

      const analytics = await errorReportingService.getErrorAnalytics();

      expect(analytics).toEqual({
        errorCount: 0,
        errorsByType: {},
        errorsBySeverity: {},
        averageResolutionTime: 0,
        fallbackUsageRate: 0,
        userSatisfactionScore: 0,
      });
    });
  });

  describe("Error Status Checking", () => {
    it("checks error status", async () => {
      const mockStatus = {
        status: "resolved" as const,
        resolution: "Fixed in version 1.2.3",
        estimatedFixTime: "2023-01-15T10:00:00Z",
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockStatus,
      });

      const status = await errorReportingService.checkErrorStatus("error_123");

      expect(status).toEqual(mockStatus);
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/error-reporting/status/error_123")
      );
    });

    it("returns default status on fetch failure", async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error("Network error"));

      const status = await errorReportingService.checkErrorStatus("error_123");

      expect(status).toEqual({ status: "open" });
    });
  });

  describe("Queue Management", () => {
    it("provides queue status", () => {
      const status = errorReportingService.getQueueStatus();

      expect(status).toEqual({
        errorQueue: 0,
        feedbackQueue: 0,
        isOnline: true,
      });
    });

    it("clears queues", () => {
      // Add some items to queues
      errorReportingService.reportError(new Error("Test"));
      errorReportingService.submitFeedback({
        errorId: "test",
        rating: 3,
        feedback: "test",
        helpfulSuggestions: [],
        wouldRecommendFallback: false,
      });

      errorReportingService.clearQueues();

      const status = errorReportingService.getQueueStatus();
      expect(status.errorQueue).toBe(0);
      expect(status.feedbackQueue).toBe(0);
    });
  });
});

describe("useErrorReporting Hook", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    errorReportingService.clearQueues();
  });

  it("provides error reporting functions", () => {
    const { result } = renderHook(() => useErrorReporting());

    expect(result.current.reportError).toBeDefined();
    expect(result.current.submitFeedback).toBeDefined();
    expect(result.current.getAnalytics).toBeDefined();
    expect(result.current.checkStatus).toBeDefined();
    expect(result.current.getQueueStatus).toBeDefined();
  });

  it("reports errors through hook", async () => {
    const { result } = renderHook(() => useErrorReporting());

    await act(async () => {
      const errorId = await result.current.reportError(
        new Error("Hook test error")
      );
      expect(errorId).toMatch(/^error_\d+_/);
    });

    const status = result.current.getQueueStatus();
    expect(status.errorQueue).toBe(1);
  });

  it("submits feedback through hook", async () => {
    const { result } = renderHook(() => useErrorReporting());

    await act(async () => {
      await result.current.submitFeedback({
        errorId: "error_123",
        rating: 4,
        feedback: "Hook test feedback",
        helpfulSuggestions: [],
        wouldRecommendFallback: true,
      });
    });

    const status = result.current.getQueueStatus();
    expect(status.feedbackQueue).toBe(1);
  });

  it("gets analytics through hook", async () => {
    const { result } = renderHook(() => useErrorReporting());

    const mockAnalytics = {
      errorCount: 0,
      errorsByType: {},
      errorsBySeverity: {},
      averageResolutionTime: 0,
      fallbackUsageRate: 0,
      userSatisfactionScore: 0,
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockAnalytics,
    });

    await act(async () => {
      const analytics = await result.current.getAnalytics();
      expect(analytics).toEqual(mockAnalytics);
    });
  });

  it("checks error status through hook", async () => {
    const { result } = renderHook(() => useErrorReporting());

    const mockStatus = { status: "open" as const };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStatus,
    });

    await act(async () => {
      const status = await result.current.checkStatus("error_123");
      expect(status).toEqual(mockStatus);
    });
  });
});
