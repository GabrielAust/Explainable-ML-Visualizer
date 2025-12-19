"""Console visualization for linear regression training."""

from __future__ import annotations

import time
from dataclasses import dataclass

from models.linear_regression import LinearRegressionState


@dataclass
class VisualizationConfig:
    """Configuration for the training visualization."""

    step_delay_s: float = 0.1
    display_every: int = 1


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

    def update(self, step: int, loss: float, state: LinearRegressionState) -> None:
        """Display the current training step."""
        if step % self.config.display_every != 0:
            return

        print(
            "Step {step:>4} | Loss: {loss:.6f} | Weight: {weight:.4f} | Bias: {bias:.4f}".format(
                step=step,
                loss=loss,
                weight=state.weight,
                bias=state.bias,
            )
        )
        time.sleep(self.config.step_delay_s)

    def summarize(self, state: LinearRegressionState) -> None:
        """Display the final parameters."""
        print("\nTraining complete.")
        print(f"Final weight: {state.weight:.4f}")
        print(f"Final bias:   {state.bias:.4f}")
