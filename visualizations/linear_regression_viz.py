"""Console visualization for linear regression training."""

from __future__ import annotations

from models.linear_regression import LinearRegressionGradients
from utils.state import ModelState
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
        state: ModelState,
        gradients: LinearRegressionGradients,
    ) -> None:
        """Display the current training step."""
        loss = state.loss_history[-1] if state.loss_history else 0.0
        weight = state.parameters.get("weight", 0.0)
        bias = state.parameters.get("bias", 0.0)
        self.console.render_step(
            step,
            [("Loss", f"{loss:.6f}")],
            [
                f"Weight = {weight:.4f}",
                f"Bias = {bias:.4f}",
                f"dW = {gradients.weight:.4f}",
                f"dB = {gradients.bias:.4f}",
            ],
            explanation=(
                "Computed gradients, then updated weight and bias by subtracting "
                "learning_rate * gradient."
            ),
        )

    def summarize(self, state: ModelState) -> None:
        """Display the final parameters."""
        weight = state.parameters.get("weight", 0.0)
        bias = state.parameters.get("bias", 0.0)
        self.console.summarize(
            [
                "Training complete.",
                f"Final weight: {weight:.4f}",
                f"Final bias:   {bias:.4f}",
            ]
        )
