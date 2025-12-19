"""Console visualization for logistic regression training."""

from __future__ import annotations

from models.logistic_regression import (
    LogisticRegressionGradients,
    LogisticRegressionState,
)
from visualizations.common import ConsoleVisualizer, VisualizationConfig


class LogisticRegressionConsoleVisualizer:
    """Visualize logistic regression training metrics in the console."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.console = ConsoleVisualizer(config)

    def announce(self, learning_rate: float, iterations: int, dataset: str) -> None:
        """Display the setup before training starts."""
        self.console.announce(
            "Logistic Regression Training",
            [
                ("Learning rate", f"{learning_rate:.4f}"),
                ("Iterations", str(iterations)),
                ("Dataset", dataset),
            ],
        )

    def update(
        self,
        step: int,
        loss: float,
        state: LogisticRegressionState,
        gradients: LogisticRegressionGradients,
    ) -> None:
        """Display the current training step."""
        self.console.render_step(
            step,
            [("Loss", f"{loss:.6f}")],
            [
                f"Weight X = {state.weight_x:.4f}",
                f"Weight Y = {state.weight_y:.4f}",
                f"Bias = {state.bias:.4f}",
                f"dW_x = {gradients.weight_x:.4f}",
                f"dW_y = {gradients.weight_y:.4f}",
                f"dB = {gradients.bias:.4f}",
            ],
            explanation=(
                "Computed cross-entropy gradients from predicted probabilities and "
                "moved weights/bias opposite the gradient."
            ),
        )

    def summarize(self, state: LogisticRegressionState) -> None:
        """Display the final parameters."""
        self.console.summarize(
            [
                "Training complete.",
                f"Final weight X: {state.weight_x:.4f}",
                f"Final weight Y: {state.weight_y:.4f}",
                f"Final bias:     {state.bias:.4f}",
            ]
        )
