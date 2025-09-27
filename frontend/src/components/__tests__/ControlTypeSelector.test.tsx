import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import ControlTypeSelector, { ControlType } from "../ControlTypeSelector";

describe("ControlTypeSelector", () => {
  const mockOnTypeChange = jest.fn();

  beforeEach(() => {
    mockOnTypeChange.mockClear();
  });

  it("renders all control type options", () => {
    render(
      <ControlTypeSelector currentType="auto" onTypeChange={mockOnTypeChange} />
    );

    expect(screen.getByText("Auto Detect")).toBeInTheDocument();
    expect(screen.getByText("Canny Edge")).toBeInTheDocument();
    expect(screen.getByText("Depth Map")).toBeInTheDocument();
    expect(screen.getByText("Pose Detection")).toBeInTheDocument();
    expect(screen.getByText("Normal Map")).toBeInTheDocument();
    expect(screen.getByText("Segmentation")).toBeInTheDocument();
  });

  it("highlights the current type", () => {
    render(
      <ControlTypeSelector
        currentType="canny"
        onTypeChange={mockOnTypeChange}
      />
    );

    const cannyButton = screen.getByText("Canny Edge").closest("button");
    expect(cannyButton).toHaveClass("border-purple-500", "bg-purple-50");
  });

  it("calls onTypeChange when a different type is clicked", () => {
    render(
      <ControlTypeSelector currentType="auto" onTypeChange={mockOnTypeChange} />
    );

    fireEvent.click(screen.getByText("Depth Map"));
    expect(mockOnTypeChange).toHaveBeenCalledWith("depth");
  });

  it("shows descriptions for each type", () => {
    render(
      <ControlTypeSelector currentType="auto" onTypeChange={mockOnTypeChange} />
    );

    expect(
      screen.getByText("Automatically detect the best control type")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Edge detection for structural control")
    ).toBeInTheDocument();
    expect(
      screen.getByText("3D depth information for spatial control")
    ).toBeInTheDocument();
  });

  it("disables buttons when disabled prop is true", () => {
    render(
      <ControlTypeSelector
        currentType="auto"
        onTypeChange={mockOnTypeChange}
        disabled={true}
      />
    );

    const autoButton = screen.getByText("Auto Detect").closest("button");
    expect(autoButton).toBeDisabled();
  });
});
