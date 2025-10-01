import React from "react";
import { render } from "@testing-library/react";
import { screen, fireEvent } from "@testing-library/dom";
import "@testing-library/jest-dom";
import ImageWorkspace from "../ImageWorkspace";
import { ImageData } from "../ComparisonView";
import { ImageVersion } from "../ImageVersioning";

describe("ImageWorkspace", () => {
  const mockCurrentImage: ImageData = {
    id: "current",
    url: "test-image.jpg",
    label: "Test Image",
    timestamp: new Date(),
    metadata: {
      prompt: "Test prompt",
      mode: "generate",
    },
  };

  const mockVersions: ImageVersion[] = [
    {
      id: "v1",
      url: "version1.jpg",
      label: "Version 1",
      timestamp: new Date(Date.now() - 3600000),
      metadata: { prompt: "Version 1 prompt", mode: "generate" },
    },
    {
      id: "v2",
      url: "version2.jpg",
      label: "Version 2",
      timestamp: new Date(Date.now() - 1800000),
      metadata: { prompt: "Version 2 prompt", mode: "edit" },
      isFavorite: true,
    },
  ];

  const mockProps = {
    currentImage: mockCurrentImage,
    versions: mockVersions,
    onVersionSelect: jest.fn(),
    onVersionDelete: jest.fn(),
    onVersionFavorite: jest.fn(),
    onImageDownload: jest.fn(),
    onImageShare: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders image workspace with toolbar", () => {
    render(<ImageWorkspace {...mockProps} />);

    expect(screen.getByText("Image Workspace")).toBeInTheDocument();
    expect(screen.getByText("Compare")).toBeInTheDocument();
    expect(screen.getByText("Versions")).toBeInTheDocument();
  });

  it("shows comparison view by default", () => {
    const propsWithOriginal = {
      ...mockProps,
      originalImage: {
        id: "original",
        url: "original.jpg",
        label: "Original Image",
        timestamp: new Date(Date.now() - 7200000),
        metadata: { prompt: "Original prompt", mode: "generate" },
      },
    };

    render(<ImageWorkspace {...propsWithOriginal} />);

    expect(screen.getByText("Image Comparison")).toBeInTheDocument();
  });

  it("switches to versions view when clicked", () => {
    render(<ImageWorkspace {...mockProps} />);

    fireEvent.click(screen.getByText("Versions"));
    expect(screen.getByText("Version View")).toBeInTheDocument();
  });

  it("shows version history", () => {
    render(<ImageWorkspace {...mockProps} />);

    expect(screen.getByText("Version History")).toBeInTheDocument();
    expect(screen.getByText("Version 1")).toBeInTheDocument();
    expect(screen.getByText("Version 2")).toBeInTheDocument();
  });

  it("calls onVersionSelect when version is clicked", () => {
    render(<ImageWorkspace {...mockProps} />);

    const version1 = screen.getByText("Version 1").closest("div");
    fireEvent.click(version1!);

    expect(mockProps.onVersionSelect).toHaveBeenCalledWith(mockVersions[0]);
  });

  it("shows download and share buttons", () => {
    render(<ImageWorkspace {...mockProps} />);

    expect(screen.getByText("Download")).toBeInTheDocument();
    expect(screen.getByText("Share")).toBeInTheDocument();
  });

  it("handles download action", () => {
    render(<ImageWorkspace {...mockProps} />);

    fireEvent.click(screen.getByText("Download"));
    expect(mockProps.onImageDownload).toHaveBeenCalled();
  });

  it("handles share action", () => {
    render(<ImageWorkspace {...mockProps} />);

    fireEvent.click(screen.getByText("Share"));
    expect(mockProps.onImageShare).toHaveBeenCalledWith(mockCurrentImage);
  });
});
