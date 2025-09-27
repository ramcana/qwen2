/**
 * Simple test to verify test environment
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";

const SimpleComponent: React.FC = () => {
  return <div>Test Component</div>;
};

describe("Simple Test", () => {
  it("renders test component", () => {
    render(<SimpleComponent />);
    expect(screen.getByText("Test Component")).toBeInTheDocument();
  });
});
