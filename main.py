"""Explainable ML Visualizer entry point."""

from __future__ import annotations

from data.dataset_generators import (
    generate_blob_data,
    generate_concentric_circles,
    generate_linear_data,
)
from models.linear_regression import LinearRegressionGD
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


def prompt_bool(prompt: str, default: bool) -> bool:
    """Prompt the user for a yes/no value with a default."""
    choices = "y/n"
    raw_default = "y" if default else "n"
    raw = input(f"{prompt} ({choices}) [{raw_default}]: ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    print(f"Invalid choice '{raw}'. Using default {raw_default}.")
    return default


def prompt_choice(prompt: str, options: list[str], default: str) -> str:
    """Prompt the user to select from a list of options."""
    choices = "/".join(options)
    raw = input(f"{prompt} ({choices}) [{default}]: ").strip().lower()
    if not raw:
        return default
    if raw in options:
        return raw
    print(f"Invalid choice '{raw}'. Using default {default}.")
    return default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value within a range."""
    return max(min_value, min(value, max_value))


def display_dataset_preview(points: list[tuple[float, float]], labels: list[int]) -> None:
    """Display a short preview of a classification dataset."""
    print("\nDataset preview (first 5 samples):")
    for point, label in list(zip(points, labels))[:5]:
        print(f"  point={point} label={label}")


def run_linear_regression_demo() -> None:
    """Run the linear regression training demo."""
    learning_rate = prompt_float("Learning rate", 0.1)
    iterations = prompt_int("Iterations", 30)
    dataset_type = prompt_choice("Dataset type", ["linear", "blobs", "circles"], "linear")
    samples = prompt_int("Sample size", 40)
    noise = prompt_float("Noise level", 0.05)
    show_weights = prompt_bool("Show weights/bias", True)
    show_gradients = prompt_bool("Show gradients", True)
    step_through = prompt_bool("Step-through mode", False)

    class_overlap = 0.0
    if dataset_type in {"blobs", "circles"}:
        class_overlap = prompt_float("Class overlap (0-1)", 0.2)
        class_overlap = clamp(class_overlap, 0.0, 1.0)

    if dataset_type == "linear":
        features, targets = generate_linear_data(
            slope=3.5,
            intercept=1.2,
            samples=samples,
            noise=max(0.0, noise),
        )
        model = LinearRegressionGD(learning_rate=learning_rate, iterations=iterations)

        visualizer = LinearRegressionConsoleVisualizer(
            VisualizationConfig(
                step_delay_s=0.0 if step_through else 0.1,
                display_every=1,
                show_weights=show_weights,
                show_gradients=show_gradients,
                step_through=step_through,
                show_explanations=step_through,
            )
        )
        visualizer.announce(learning_rate=learning_rate, iterations=iterations)

        final_state = model.fit(features, targets, on_step=visualizer.update)
        visualizer.summarize(final_state)
        return

    if dataset_type == "blobs":
        points, labels = generate_blob_data(
            samples=samples,
            classes=2,
            class_overlap=class_overlap,
            noise=max(0.0, noise),
        )
        dataset_label = "Blob dataset"
    else:
        points, labels = generate_concentric_circles(
            samples=samples,
            class_overlap=class_overlap,
            noise=max(0.0, noise),
        )
        dataset_label = "Concentric circles dataset"

    print(f"\n{dataset_label} generated with {samples} samples.")
    print(f"Noise level: {noise}")
    print(f"Class overlap: {class_overlap}")
    display_dataset_preview(points, labels)



def main() -> None:
    """Main entry point."""
    run_linear_regression_demo()


if __name__ == "__main__":
    main()
