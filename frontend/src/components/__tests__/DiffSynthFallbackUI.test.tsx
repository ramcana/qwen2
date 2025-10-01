/**
 * Tests for DiffSynth Fallback UI Components
 */

import React from "react";
import { render } from "@testing-library/react";
import { screen, fireEvent } from "@testing-library/dom";
import "@testing-library/jest-dom";
import {
  DiffSynthFallbackUI,
  ServiceStatusIndicator,
  FeatureComparison,
  SimplifiedEditPanel,
} from "../DiffSynthFallbackUI";

describe("DiffSynthFallbackUI", () => {
  const defaultProps = {
    fallbackType: "service_unavailable" as const,
    availableFeatures: ["Basic Image Generation", "Simple Text Processing"],
    limitations: ["No advanced editing", "Limited resolution"],
  };

  it("renders service unavailable fallback correctly", () => {
    render(<DiffSynthFallbackUI {...defaultProps} />);

    expect(
      screen.getByText("DiffSynth Service Unavailable")
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Advanced editing features are temporarily unavailable/)
    ).toBeInTheDocument();
    expect(
      screen.getByText("Use Qwen text-to-image generation")
    ).toBeInTheDocument();
  });

  it("renders memory limited fallback correctly", () => {
    render(
      <DiffSynthFallbackUI {...defaultProps} fallbackType="memory_limited" />
    );

    expect(screen.getByText("Memory Optimization Mode")).toBeInTheDocument();
    expect(
      screen.getByText(/Running in memory-optimized mode/)
    ).toBeInTheDocument();
    expect(
      screen.getByText("CPU-based processing enabled")
    ).toBeInTheDocument();
  });

  it("renders feature disabled fallback correctly", () => {
    render(
      <DiffSynthFallbackUI {...defaultProps} fallbackType="feature_disabled" />
    );

    expect(
      screen.getByText("Feature Temporarily Disabled")
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Some advanced features are disabled/)
    ).toBeInTheDocument();
  });

  it("renders processing error fallback correctly", () => {
    render(
      <DiffSynthFallbackUI {...defaultProps} fallbackType="processing_error" />
    );

    expect(screen.getByText("Processing Error Recovery")).toBeInTheDocument();
    expect(
      screen.getByText(/An error occurred during processing/)
    ).toBeInTheDocument();
  });

  it("displays available features", () => {
    render(<DiffSynthFallbackUI {...defaultProps} />);

    expect(screen.getByText("Available Features")).toBeInTheDocument();
    expect(screen.getByText("Basic Image Generation")).toBeInTheDocument();
    expect(screen.getByText("Simple Text Processing")).toBeInTheDocument();
  });

  it("handles feature selection", () => {
    const onFeatureSelect = jest.fn();
    render(
      <DiffSynthFallbackUI
        {...defaultProps}
        onFeatureSelect={onFeatureSelect}
      />
    );

    const featureButton = screen.getByText("Basic Image Generation");
    fireEvent.click(featureButton);

    expect(onFeatureSelect).toHaveBeenCalledWith("Basic Image Generation");
  });

  it("displays limitations", () => {
    render(<DiffSynthFallbackUI {...defaultProps} />);

    expect(screen.getByText("Current Limitations")).toBeInTheDocument();
    expect(screen.getByText("No advanced editing")).toBeInTheDocument();
    expect(screen.getByText("Limited resolution")).toBeInTheDocument();
  });

  it("shows retry button when callback provided", () => {
    const onRetryOriginal = jest.fn();
    render(
      <DiffSynthFallbackUI
        {...defaultProps}
        onRetryOriginal={onRetryOriginal}
      />
    );

    const retryButton = screen.getByText("Try Full Features Again");
    expect(retryButton).toBeInTheDocument();

    fireEvent.click(retryButton);
    expect(onRetryOriginal).toHaveBeenCalled();
  });

  it("highlights selected feature", () => {
    render(<DiffSynthFallbackUI {...defaultProps} />);

    const featureButton = screen.getByText("Basic Image Generation");
    fireEvent.click(featureButton);

    expect(featureButton.closest("button")).toHaveClass(
      "border-blue-500",
      "bg-blue-50"
    );
  });
});

describe("ServiceStatusIndicator", () => {
  const mockServices = [
    {
      name: "DiffSynth Service",
      status: "available" as const,
      description: "Advanced image editing service",
      fallbackAvailable: false,
    },
    {
      name: "ControlNet Service",
      status: "degraded" as const,
      description: "Structural guidance service",
      fallbackAvailable: true,
    },
    {
      name: "EliGen Service",
      status: "unavailable" as const,
      description: "Quality enhancement service",
      fallbackAvailable: true,
    },
  ];

  it("renders service status correctly", () => {
    render(<ServiceStatusIndicator services={mockServices} />);

    expect(screen.getByText("Service Status")).toBeInTheDocument();
    expect(screen.getByText("DiffSynth Service")).toBeInTheDocument();
    expect(screen.getByText("ControlNet Service")).toBeInTheDocument();
    expect(screen.getByText("EliGen Service")).toBeInTheDocument();
  });

  it("shows correct status indicators", () => {
    render(<ServiceStatusIndicator services={mockServices} />);

    // Check that all services are rendered
    expect(screen.getByText("DiffSynth Service")).toBeInTheDocument();
    expect(screen.getByText("ControlNet Service")).toBeInTheDocument();
    expect(screen.getByText("EliGen Service")).toBeInTheDocument();

    // Check that service descriptions are present
    expect(
      screen.getByText("Advanced image editing service")
    ).toBeInTheDocument();
    expect(screen.getByText("Structural guidance service")).toBeInTheDocument();
    expect(screen.getByText("Quality enhancement service")).toBeInTheDocument();
  });

  it("shows fallback availability badges", () => {
    render(<ServiceStatusIndicator services={mockServices} />);

    // Should show fallback badges for degraded and unavailable services
    const fallbackBadges = screen.getAllByText("Fallback Available");
    expect(fallbackBadges).toHaveLength(2);
  });
});

describe("FeatureComparison", () => {
  const originalFeatures = [
    "Advanced Image Editing",
    "ControlNet Integration",
    "Style Transfer",
    "High Resolution Processing",
  ];

  const fallbackFeatures = [
    "Basic Image Generation",
    "Simple Text Processing",
    "Standard Resolution",
  ];

  it("renders feature comparison in original mode", () => {
    render(
      <FeatureComparison
        originalFeatures={originalFeatures}
        fallbackFeatures={fallbackFeatures}
        currentMode="original"
      />
    );

    expect(screen.getByText("Feature Comparison")).toBeInTheDocument();
    expect(screen.getByText("Full Features")).toBeInTheDocument();
    expect(screen.getByText("Fallback Mode")).toBeInTheDocument();

    // Original features should be highlighted
    expect(screen.getByText("Advanced Image Editing")).toHaveClass(
      "text-gray-900"
    );
    expect(screen.getByText("Basic Image Generation")).toHaveClass(
      "text-gray-500"
    );
  });

  it("renders feature comparison in fallback mode", () => {
    render(
      <FeatureComparison
        originalFeatures={originalFeatures}
        fallbackFeatures={fallbackFeatures}
        currentMode="fallback"
      />
    );

    // Fallback features should be highlighted
    expect(screen.getByText("Basic Image Generation")).toHaveClass(
      "text-gray-900"
    );
    expect(screen.getByText("Advanced Image Editing")).toHaveClass(
      "text-gray-500"
    );
  });

  it("shows correct status indicators for current mode", () => {
    const { rerender } = render(
      <FeatureComparison
        originalFeatures={originalFeatures}
        fallbackFeatures={fallbackFeatures}
        currentMode="original"
      />
    );

    // Original mode should have green indicator
    expect(document.querySelector(".bg-green-500")).toBeInTheDocument();

    rerender(
      <FeatureComparison
        originalFeatures={originalFeatures}
        fallbackFeatures={fallbackFeatures}
        currentMode="fallback"
      />
    );

    // Fallback mode should have blue indicator
    expect(document.querySelector(".bg-blue-500")).toBeInTheDocument();
  });
});

describe("SimplifiedEditPanel", () => {
  it("renders simplified edit panel", () => {
    render(<SimplifiedEditPanel />);

    expect(screen.getByText("Simplified Mode")).toBeInTheDocument();
    expect(
      screen.getByText(/Advanced editing features are unavailable/)
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("Describe the image you want to generate...")
    ).toBeInTheDocument();
  });

  it("handles prompt input", () => {
    render(<SimplifiedEditPanel />);

    const textarea = screen.getByPlaceholderText(
      "Describe the image you want to generate..."
    );
    fireEvent.change(textarea, { target: { value: "A beautiful landscape" } });

    expect(textarea).toHaveValue("A beautiful landscape");
  });

  it("enables generate button when prompt is provided", () => {
    render(<SimplifiedEditPanel />);

    const generateButton = screen.getByRole("button", {
      name: /generate image/i,
    });
    const textarea = screen.getByPlaceholderText(
      "Describe the image you want to generate..."
    );

    // Button should be disabled initially
    expect(generateButton).toBeDisabled();

    // Enter prompt
    fireEvent.change(textarea, { target: { value: "A beautiful landscape" } });

    // Button should be enabled
    expect(generateButton).not.toBeDisabled();
  });

  it("shows processing state when generating", () => {
    render(<SimplifiedEditPanel />);

    const textarea = screen.getByPlaceholderText(
      "Describe the image you want to generate..."
    );

    // Enter prompt
    fireEvent.change(textarea, { target: { value: "A beautiful landscape" } });

    // Get button and click it
    const generateButton = screen.getByRole("button", {
      name: /generate image/i,
    });
    fireEvent.click(generateButton);

    // Should show processing state
    expect(screen.getByText("Generating...")).toBeInTheDocument();

    // Get the button again after state change and check if it's disabled
    const processingButton = screen.getByRole("button", {
      name: /generating/i,
    });
    expect(processingButton).toBeDisabled();
  });

  it("displays available features in simplified mode", () => {
    render(<SimplifiedEditPanel />);

    expect(
      screen.getByText("Available in Simplified Mode:")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Basic text-to-image generation")
    ).toBeInTheDocument();
    expect(screen.getByText("Standard resolution output")).toBeInTheDocument();
    expect(screen.getByText("Simple prompt processing")).toBeInTheDocument();
  });
});
