"""Console visualization for linear regression training."""

from __future__ import annotations

from models.linear_regression import LinearRegressionGradients, LinearRegressionState
from visualizations.common import ConsoleVisualizer, VisualizationConfig


class LinearRegressionConsoleVisualizer:
    """Visualize linear regression training metrics in the console."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.console = ConsoleVisualizer(config)

    def announce(self, learning_rate: float, iterations: int) -> None:
        """Display the setup before training starts."""
        self.console.announce(
            "Linear Regression Training",
            [
                ("Learning rate", f"{learning_rate:.4f}"),
                ("Iterations", str(iterations)),
            ],
        )

    def update(
        self,
        step: int,
        loss: float,
        state: LinearRegressionState,
        gradients: LinearRegressionGradients,
    ) -> None:
        """Display the current training step."""
        self.console.render_step(
            step,
            [("Loss", f"{loss:.6f}")],
            [
                f"Weight = {state.weight:.4f}",
                f"Bias = {state.bias:.4f}",
                f"dW = {gradients.weight:.4f}",
                f"dB = {gradients.bias:.4f}",
            ],
            explanation=(
                "Computed gradients, then updated weight and bias by subtracting "
                "learning_rate * gradient."
            ),
        )

    def summarize(self, state: LinearRegressionState) -> None:
        """Display the final parameters."""
        self.console.summarize(
            [
                "Training complete.",
                f"Final weight: {state.weight:.4f}",
                f"Final bias:   {state.bias:.4f}",
            ]
        )
