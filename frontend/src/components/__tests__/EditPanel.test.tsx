import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import EditPanel from "../EditPanel";

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

describe("EditPanel", () => {
  const mockOnGenerate = jest.fn();

  beforeEach(() => {
    mockOnGenerate.mockClear();
    mockGetCurrentState.mockReturnValue({
      editMode: "inpaint",
      sourceImage: null,
      maskImage: null,
      styleImage: null,
      prompt: "",
      strength: 0.8,
      steps: 20,
      guidance: 7.5,
    });
    mockUpdateCurrentState.mockClear();
    mockResetCurrentState.mockClear();
  });

  it("renders edit panel with all sections", () => {
    render(<EditPanel onGenerate={mockOnGenerate} />);

    expect(screen.getByText("Image Editing")).toBeInTheDocument();
    expect(screen.getByText("Edit Operation")).toBeInTheDocument();
    expect(screen.getByText("Source Image")).toBeInTheDocument();
    expect(screen.getByText("Prompt")).toBeInTheDocument();
  });

  it("shows inpaint mode by default", () => {
    render(<EditPanel onGenerate={mockOnGenerate} />);

    expect(screen.getByText("Inpaint")).toBeInTheDocument();
    expect(screen.getByText("Mask Image")).toBeInTheDocument();
  });

  it("shows style transfer controls when style transfer mode is selected", () => {
    mockGetCurrentState.mockReturnValue({
      editMode: "style-transfer",
      sourceImage: null,
      maskImage: null,
      styleImage: null,
      prompt: "",
      strength: 0.8,
      steps: 20,
      guidance: 7.5,
    });

    render(<EditPanel onGenerate={mockOnGenerate} />);

    expect(screen.getByText("Style Reference Image")).toBeInTheDocument();
  });

  it("disables generate button when required fields are missing", () => {
    render(<EditPanel onGenerate={mockOnGenerate} />);

    const generateButton = screen.getByRole("button", {
      name: /Start Inpainting/i,
    });
    expect(generateButton).toBeDisabled();
  });

  it("shows appropriate help text for each mode", () => {
    render(<EditPanel onGenerate={mockOnGenerate} />);

    expect(screen.getByText("Inpainting Tips")).toBeInTheDocument();
    expect(
      screen.getByText(/White areas in the mask will be replaced/)
    ).toBeInTheDocument();
  });

  it("handles reset functionality", () => {
    // Mock window.confirm
    window.confirm = jest.fn(() => true);

    render(<EditPanel onGenerate={mockOnGenerate} />);

    const resetButton = screen.getByRole("button", { name: /Reset/i });
    fireEvent.click(resetButton);

    expect(mockResetCurrentState).toHaveBeenCalled();
  });
});
