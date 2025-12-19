"""Console visualization for linear regression training."""

from __future__ import annotations

import time
from dataclasses import dataclass

from models.linear_regression import LinearRegressionGradients, LinearRegressionState


@dataclass
class VisualizationConfig:
    """Configuration for the training visualization."""

    step_delay_s: float = 0.1
    display_every: int = 1
    show_weights: bool = True
    show_gradients: bool = True
    step_through: bool = False
    show_explanations: bool = False


class LinearRegressionConsoleVisualizer:
    """Visualize linear regression training metrics in the console."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()

    def announce(self, learning_rate: float, iterations: int) -> None:
        """Display the setup before training starts."""
        print("\nLinear Regression Training")
        print("-" * 32)
        print(f"Learning rate: {learning_rate}")
        print(f"Iterations:    {iterations}\n")

    def update(
        self,
        step: int,
        loss: float,
        state: LinearRegressionState,
        gradients: LinearRegressionGradients,
    ) -> None:
        """Display the current training step."""
        if step % self.config.display_every != 0:
            return

        pieces = [f"Step {step:>4}", f"Loss: {loss:.6f}"]
        if self.config.show_weights:
            pieces.append(f"Weight: {state.weight:.4f}")
            pieces.append(f"Bias: {state.bias:.4f}")
        if self.config.show_gradients:
            pieces.append(f"dW: {gradients.weight:.4f}")
            pieces.append(f"dB: {gradients.bias:.4f}")
        print(" | ".join(pieces))

        if self.config.show_explanations:
            print(
                "  Explanation: computed gradients, then updated weight and bias "
                "by subtracting learning_rate * gradient."
            )

        if self.config.step_through:
            input("  Press Enter to advance to the next step...")
            return

        time.sleep(self.config.step_delay_s)

    def summarize(self, state: LinearRegressionState) -> None:
        """Display the final parameters."""
        print("\nTraining complete.")
        print(f"Final weight: {state.weight:.4f}")
        print(f"Final bias:   {state.bias:.4f}")
