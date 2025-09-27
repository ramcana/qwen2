/**
 * Tests for DiffSynth Error Boundary Component
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import DiffSynthErrorBoundary from "../DiffSynthErrorBoundary";

// Mock child component that can throw errors
const ThrowError: React.FC<{ shouldThrow?: boolean; errorType?: string }> = ({
  shouldThrow = false,
  errorType = "generic",
}) => {
  if (shouldThrow) {
    if (errorType === "memory") {
      throw new Error("CUDA out of memory");
    } else if (errorType === "diffsynth") {
      throw new Error("DiffSynth initialization failed");
    } else if (errorType === "controlnet") {
      throw new Error("ControlNet processing failed");
    } else {
      throw new Error("Test error");
    }
  }
  return <div>Child component</div>;
};

// Mock fallback component
const MockFallbackComponent: React.FC = () => (
  <div>Fallback component active</div>
);

describe("DiffSynthErrorBoundary", () => {
  // Suppress console.error for tests
  const originalError = console.error;
  beforeAll(() => {
    console.error = jest.fn();
  });

  afterAll(() => {
    console.error = originalError;
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders children when there is no error", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={false} />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("Child component")).toBeInTheDocument();
  });

  it("catches and displays generic errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("DiffSynth Error")).toBeInTheDocument();
    expect(
      screen.getByText(
        "DiffSynth encountered an issue. Basic features are still available."
      )
    ).toBeInTheDocument();
  });

  it("catches and displays DiffSynth memory errors with specific handling", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("DiffSynth Error")).toBeInTheDocument();
    expect(screen.getByText(/GPU memory is full/)).toBeInTheDocument();
    expect(screen.getByText("Reduce image resolution")).toBeInTheDocument();
  });

  it("catches and displays DiffSynth initialization errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="diffsynth" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("DiffSynth Error")).toBeInTheDocument();
    expect(
      screen.getByText(/DiffSynth service is not available/)
    ).toBeInTheDocument();
  });

  it("catches and displays ControlNet errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="controlnet" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("DiffSynth Error")).toBeInTheDocument();
    expect(
      screen.getByText(/ControlNet processing failed/)
    ).toBeInTheDocument();
  });

  it("calls onError callback when error occurs", () => {
    const onErrorMock = jest.fn();

    render(
      <DiffSynthErrorBoundary onError={onErrorMock}>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    expect(onErrorMock).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );
  });

  it("shows retry button and handles retry", () => {
    const onRetryMock = jest.fn();

    render(
      <DiffSynthErrorBoundary onRetry={onRetryMock}>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    const retryButton = screen.getByText(/Try Again/);
    expect(retryButton).toBeInTheDocument();

    fireEvent.click(retryButton);
    expect(onRetryMock).toHaveBeenCalled();
  });

  it("shows fallback button when fallback is enabled and available", () => {
    render(
      <DiffSynthErrorBoundary
        enableFallback={true}
        fallbackComponent={<MockFallbackComponent />}
      >
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("Use Fallback Mode")).toBeInTheDocument();
  });

  it("activates fallback mode when fallback button is clicked", () => {
    const onFallbackMock = jest.fn();

    render(
      <DiffSynthErrorBoundary
        enableFallback={true}
        fallbackComponent={<MockFallbackComponent />}
        onFallback={onFallbackMock}
      >
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    const fallbackButton = screen.getByText("Use Fallback Mode");
    fireEvent.click(fallbackButton);

    expect(onFallbackMock).toHaveBeenCalled();
    expect(screen.getByText("Fallback Mode Active")).toBeInTheDocument();
    expect(screen.getByText("Fallback component active")).toBeInTheDocument();
  });

  it("shows technical details when enabled", () => {
    render(
      <DiffSynthErrorBoundary showTechnicalDetails={true}>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("Technical Details")).toBeInTheDocument();
  });

  it("toggles technical details visibility", () => {
    render(
      <DiffSynthErrorBoundary showTechnicalDetails={true}>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    const detailsButton = screen.getByText("Technical Details");
    fireEvent.click(detailsButton);

    expect(screen.getByText("Error Message:")).toBeInTheDocument();
    expect(screen.getByText("Test error")).toBeInTheDocument();
  });

  it("limits retry attempts", () => {
    const { rerender } = render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    // Click retry button multiple times
    for (let i = 0; i < 4; i++) {
      const retryButton = screen.queryByText(/Try Again/);
      if (retryButton) {
        fireEvent.click(retryButton);
        // Re-render with error to simulate retry failure
        rerender(
          <DiffSynthErrorBoundary>
            <ThrowError shouldThrow={true} />
          </DiffSynthErrorBoundary>
        );
      }
    }

    // After max retries, button should not be available
    expect(screen.queryByText(/Try Again/)).not.toBeInTheDocument();
  });

  it("shows refresh page button", () => {
    // Mock window.location.reload
    const mockReload = jest.fn();
    Object.defineProperty(window, "location", {
      value: { reload: mockReload },
      writable: true,
    });

    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    const refreshButton = screen.getByText("Refresh Page");
    expect(refreshButton).toBeInTheDocument();

    fireEvent.click(refreshButton);
    expect(mockReload).toHaveBeenCalled();
  });

  it("displays suggested fixes for memory errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("Suggested Solutions:")).toBeInTheDocument();
    expect(screen.getByText("Reduce image resolution")).toBeInTheDocument();
    expect(screen.getByText("Enable CPU offloading")).toBeInTheDocument();
    expect(
      screen.getByText("Close other applications using GPU")
    ).toBeInTheDocument();
  });

  it("displays limitations for memory errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText("Current Limitations:")).toBeInTheDocument();
    expect(
      screen.getByText("May process at lower quality")
    ).toBeInTheDocument();
    expect(screen.getByText("Slower processing speed")).toBeInTheDocument();
  });

  it("shows appropriate severity styling for different error types", () => {
    // Test memory error styling
    const { unmount } = render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    // Memory error should have orange styling (degraded severity)
    const memoryErrorContainer = screen
      .getByText("DiffSynth Error")
      .closest(".border.rounded-lg");
    expect(memoryErrorContainer).toHaveClass("text-orange-600");

    // Clean up first render
    unmount();

    // Test initialization error styling with fresh component
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="diffsynth" />
      </DiffSynthErrorBoundary>
    );

    // The "DiffSynth initialization failed" error should be classified as critical severity
    const initErrorContainer = screen
      .getByText("DiffSynth Error")
      .closest(".border.rounded-lg");
    expect(initErrorContainer).toHaveClass("text-red-600");
  });

  it("resets error state when retry is successful", async () => {
    // Test that the error boundary resets its internal state when retry is clicked
    let shouldThrow = true;

    const TestComponent: React.FC = () => {
      if (shouldThrow) {
        throw new Error("Test error");
      }
      return <div>Child component</div>;
    };

    render(
      <DiffSynthErrorBoundary>
        <TestComponent />
      </DiffSynthErrorBoundary>
    );

    // Error should be displayed initially
    expect(screen.getByText("DiffSynth Error")).toBeInTheDocument();

    // Change the condition so the component won't throw on next render
    shouldThrow = false;

    // Click retry - this should reset the error boundary and re-render children
    const retryButton = screen.getByText(/Try Again/);
    fireEvent.click(retryButton);

    // Should show child component again since the condition is now resolved
    await waitFor(() => {
      expect(screen.getByText("Child component")).toBeInTheDocument();
    });

    // Error UI should no longer be visible
    expect(screen.queryByText("DiffSynth Error")).not.toBeInTheDocument();
  });

  it("shows contact support link", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} />
      </DiffSynthErrorBoundary>
    );

    const supportLink = screen.getByText("contact support");
    expect(supportLink).toBeInTheDocument();
    expect(supportLink).toHaveAttribute("href", "mailto:support@example.com");
  });
});

describe("DiffSynthErrorBoundary Error Classification", () => {
  const originalError = console.error;
  beforeAll(() => {
    console.error = jest.fn();
  });

  afterAll(() => {
    console.error = originalError;
  });

  it("correctly classifies memory errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="memory" />
      </DiffSynthErrorBoundary>
    );

    expect(screen.getByText(/GPU memory is full/)).toBeInTheDocument();
  });

  it("correctly classifies initialization errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="diffsynth" />
      </DiffSynthErrorBoundary>
    );

    expect(
      screen.getByText(/DiffSynth service is not available/)
    ).toBeInTheDocument();
  });

  it("correctly classifies ControlNet errors", () => {
    render(
      <DiffSynthErrorBoundary>
        <ThrowError shouldThrow={true} errorType="controlnet" />
      </DiffSynthErrorBoundary>
    );

    expect(
      screen.getByText(/ControlNet processing failed/)
    ).toBeInTheDocument();
  });
});
