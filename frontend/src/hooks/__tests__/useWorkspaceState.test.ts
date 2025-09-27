import { renderHook, act } from "@testing-library/react";
import { useWorkspaceState } from "../useWorkspaceState";

describe("useWorkspaceState", () => {
  it("initializes with generate mode", () => {
    const { result } = renderHook(() => useWorkspaceState());

    expect(result.current.currentMode).toBe("generate");
  });

  it("switches modes correctly", () => {
    const { result } = renderHook(() => useWorkspaceState());

    act(() => {
      result.current.switchMode("edit");
    });

    expect(result.current.currentMode).toBe("edit");
  });

  it("preserves state when switching modes", () => {
    const { result } = renderHook(() => useWorkspaceState());

    // Update generate state
    act(() => {
      result.current.updateCurrentState({ prompt: "test prompt" });
    });

    // Switch to edit mode
    act(() => {
      result.current.switchMode("edit");
    });

    // Switch back to generate mode
    act(() => {
      result.current.switchMode("generate");
    });

    // Check if state is preserved
    const generateState = result.current.getCurrentState();
    expect(generateState.prompt).toBe("test prompt");
  });

  it("updates current state correctly", () => {
    const { result } = renderHook(() => useWorkspaceState());

    act(() => {
      result.current.updateCurrentState({
        prompt: "new prompt",
        steps: 30,
      });
    });

    const state = result.current.getCurrentState();
    expect(state.prompt).toBe("new prompt");
    expect(state.steps).toBe(30);
  });

  it("resets current state to defaults", () => {
    const { result } = renderHook(() => useWorkspaceState());

    // Update state
    act(() => {
      result.current.updateCurrentState({ prompt: "test" });
    });

    // Reset state
    act(() => {
      result.current.resetCurrentState();
    });

    const state = result.current.getCurrentState();
    expect(state.prompt).toBe("");
  });

  it("returns all states correctly", () => {
    const { result } = renderHook(() => useWorkspaceState());

    const allStates = result.current.getAllStates();

    expect(allStates).toHaveProperty("generate");
    expect(allStates).toHaveProperty("edit");
    expect(allStates).toHaveProperty("controlnet");
  });
});
