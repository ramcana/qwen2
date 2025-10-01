import React from "react";
import { render } from "@testing-library/react";
import { screen, fireEvent } from "@testing-library/dom";
import "@testing-library/jest-dom";
import EditModeSelector, { EditMode } from "../EditModeSelector";

describe("EditModeSelector", () => {
  const mockOnModeChange = jest.fn();

  beforeEach(() => {
    mockOnModeChange.mockClear();
  });

  it("renders all edit mode options", () => {
    render(
      <EditModeSelector currentMode="inpaint" onModeChange={mockOnModeChange} />
    );

    expect(screen.getByText("Inpaint")).toBeInTheDocument();
    expect(screen.getByText("Outpaint")).toBeInTheDocument();
    expect(screen.getByText("Style Transfer")).toBeInTheDocument();
  });

  it("highlights the current mode", () => {
    render(
      <EditModeSelector
        currentMode="outpaint"
        onModeChange={mockOnModeChange}
      />
    );

    const outpaintButton = screen.getByText("Outpaint").closest("button");
    expect(outpaintButton).toHaveClass("border-blue-500", "bg-blue-50");
  });

  it("calls onModeChange when a different mode is clicked", () => {
    render(
      <EditModeSelector currentMode="inpaint" onModeChange={mockOnModeChange} />
    );

    fireEvent.click(screen.getByText("Style Transfer"));
    expect(mockOnModeChange).toHaveBeenCalledWith("style-transfer");
  });

  it("shows descriptions for each mode", () => {
    render(
      <EditModeSelector currentMode="inpaint" onModeChange={mockOnModeChange} />
    );

    expect(
      screen.getByText("Fill masked areas with AI-generated content")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Extend image beyond its current boundaries")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Apply artistic style from reference image")
    ).toBeInTheDocument();
  });

  it("disables buttons when disabled prop is true", () => {
    render(
      <EditModeSelector
        currentMode="inpaint"
        onModeChange={mockOnModeChange}
        disabled={true}
      />
    );

    const inpaintButton = screen.getByText("Inpaint").closest("button");
    expect(inpaintButton).toBeDisabled();
  });
});
