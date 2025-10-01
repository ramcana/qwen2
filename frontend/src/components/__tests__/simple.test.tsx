/**
 * Simple test to verify test environment
 */

import React from "react";
import { render } from "@testing-library/react";
import { screen } from "@testing-library/dom";
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
