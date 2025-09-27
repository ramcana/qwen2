// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import "@testing-library/jest-dom";

// Mock Intl for Jest environment
Object.defineProperty(global, "Intl", {
  value: {
    DateTimeFormat: function () {
      return {
        resolvedOptions: () => ({ timeZone: "UTC" }),
      };
    },
  },
});

// Mock fetch for tests
global.fetch = jest.fn();

// Mock window.confirm
Object.defineProperty(window, "confirm", {
  value: jest.fn(() => true),
  writable: true,
});
