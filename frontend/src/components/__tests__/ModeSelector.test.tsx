import React from "react";
import { render } from "@testing-library/react";
import { screen, fireEvent } from "@testing-library/dom";
import "@testing-library/jest-dom";
import ModeSelector, { WorkflowMode } from "../ModeSelector";

describe("ModeSelector", () => {
  const mockOnModeChange = jest.fn();

  beforeEach(() => {
    mockOnModeChange.mockClear();
  });

  it("renders all mode options", () => {
    render(
      <ModeSelector currentMode="generate" onModeChange={mockOnModeChange} />
    );

    expect(screen.getByText("Generate")).toBeInTheDocument();
    expect(screen.getByText("Edit")).toBeInTheDocument();
    expect(screen.getByText("ControlNet")).toBeInTheDocument();
  });

  it("highlights the current mode", () => {
    render(<ModeSelector currentMode="edit" onModeChange={mockOnModeChange} />);

    const editButton = screen.getByText("Edit").closest("button");
    expect(editButton).toHaveClass("bg-blue-500", "text-white");
  });

  it("calls onModeChange when a different mode is clicked", () => {
    render(
      <ModeSelector currentMode="generate" onModeChange={mockOnModeChange} />
    );

    fireEvent.click(screen.getByText("Edit"));
    expect(mockOnModeChange).toHaveBeenCalledWith("edit");
  });

  it("disables buttons when disabled prop is true", () => {
    render(
      <ModeSelector
        currentMode="generate"
        onModeChange={mockOnModeChange}
        disabled={true}
      />
    );

    const generateButton = screen.getByText("Generate").closest("button");
    expect(generateButton).toBeDisabled();
  });

  it("shows tooltips with mode descriptions", () => {
    render(
      <ModeSelector currentMode="generate" onModeChange={mockOnModeChange} />
    );

    const generateButton = screen.getByText("Generate").closest("button");
    expect(generateButton).toHaveAttribute("title", "Text-to-image generation");
  });
});
