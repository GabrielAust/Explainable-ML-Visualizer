"""Explainable ML Visualizer entry point."""

from __future__ import annotations

from models.linear_regression import LinearRegressionGD, generate_linear_data
from visualizations.linear_regression_viz import (
    LinearRegressionConsoleVisualizer,
    VisualizationConfig,
)


def prompt_float(prompt: str, default: float) -> float:
    """Prompt the user for a float value with a default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid input '{raw}'. Using default {default}.")
        return default


def prompt_int(prompt: str, default: int) -> int:
    """Prompt the user for an int value with a default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid input '{raw}'. Using default {default}.")
        return default


def run_linear_regression_demo() -> None:
    """Run the linear regression training demo."""
    learning_rate = prompt_float("Learning rate", 0.1)
    iterations = prompt_int("Iterations", 30)

    features, targets = generate_linear_data(slope=3.5, intercept=1.2, samples=40, noise=0.05)
    model = LinearRegressionGD(learning_rate=learning_rate, iterations=iterations)

    visualizer = LinearRegressionConsoleVisualizer(
        VisualizationConfig(step_delay_s=0.1, display_every=1)
    )
    visualizer.announce(learning_rate=learning_rate, iterations=iterations)

    final_state = model.fit(features, targets, on_step=visualizer.update)
    visualizer.summarize(final_state)


def main() -> None:
    """Main entry point."""
    run_linear_regression_demo()


if __name__ == "__main__":
    main()
