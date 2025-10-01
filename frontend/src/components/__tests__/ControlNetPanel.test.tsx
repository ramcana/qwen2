import React from "react";
import { render } from "@testing-library/react";
import { screen, fireEvent } from "@testing-library/dom";
import "@testing-library/jest-dom";
import ControlNetPanel from "../ControlNetPanel";

// Mock the useWorkspaceState hook
const mockGetCurrentState = jest.fn();
const mockUpdateCurrentState = jest.fn();
const mockResetCurrentState = jest.fn();

jest.mock("../../hooks/useWorkspaceState", () => ({
  useWorkspaceState: () => ({
    getCurrentState: mockGetCurrentState,
    updateCurrentState: mockUpdateCurrentState,
    resetCurrentState: mockResetCurrentState,
  }),
}));

describe("ControlNetPanel", () => {
  const mockOnGenerate = jest.fn();
  const mockOnDetectControl = jest.fn();

  beforeEach(() => {
    mockOnGenerate.mockClear();
    mockOnDetectControl.mockClear();
    mockGetCurrentState.mockReturnValue({
      controlType: "auto",
      controlImage: null,
      prompt: "",
      controlStrength: 1.0,
      steps: 20,
      guidance: 7.5,
    });
    mockUpdateCurrentState.mockClear();
    mockResetCurrentState.mockClear();
  });

  it("renders ControlNet panel with all sections", () => {
    render(
      <ControlNetPanel
        onGenerate={mockOnGenerate}
        onDetectControl={mockOnDetectControl}
      />
    );

    expect(screen.getByText("ControlNet Generation")).toBeInTheDocument();
    expect(screen.getByText("Control Type")).toBeInTheDocument();
    expect(screen.getByText("Control Image")).toBeInTheDocument();
    expect(screen.getByText("Prompt")).toBeInTheDocument();
  });

  it("shows auto detect mode by default", () => {
    render(
      <ControlNetPanel
        onGenerate={mockOnGenerate}
        onDetectControl={mockOnDetectControl}
      />
    );

    expect(screen.getByText("Auto Detect")).toBeInTheDocument();
  });

  it("shows manual detection button when not in auto mode", () => {
    mockGetCurrentState.mockReturnValue({
      controlType: "canny",
      controlImage: new File([""], "test.jpg", { type: "image/jpeg" }),
      prompt: "",
      controlStrength: 1.0,
      steps: 20,
      guidance: 7.5,
    });

    render(
      <ControlNetPanel
        onGenerate={mockOnGenerate}
        onDetectControl={mockOnDetectControl}
      />
    );

    expect(screen.getByText(/Detect canny Features/i)).toBeInTheDocument();
  });

  it("disables generate button when required fields are missing", () => {
    render(
      <ControlNetPanel
        onGenerate={mockOnGenerate}
        onDetectControl={mockOnDetectControl}
      />
    );

    const generateButton = screen.getByRole("button", {
      name: /Generate with ControlNet/i,
    });
    expect(generateButton).toBeDisabled();
  });

  it("shows ControlNet tips", () => {
    render(
      <ControlNetPanel
        onGenerate={mockOnGenerate}
        onDetectControl={mockOnDetectControl}
      />
    );

    expect(screen.getByText("ControlNet Tips")).toBeInTheDocument();
    expect(
      screen.getByText(/ControlNet uses your image to guide/)
    ).toBeInTheDocument();
  });

  it("handles reset functionality", () => {
    // Mock window.confirm
    window.confirm = jest.fn(() => true);

    render(
      <ControlNetPanel
        onGenerate={mockOnGenerate}
        onDetectControl={mockOnDetectControl}
      />
    );

    const resetButton = screen.getByRole("button", { name: /Reset/i });
    fireEvent.click(resetButton);

    expect(mockResetCurrentState).toHaveBeenCalled();
  });
});
