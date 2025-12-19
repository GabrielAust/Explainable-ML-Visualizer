"""Explainable ML Visualizer entry point."""

from __future__ import annotations

from data.dataset_generators import (
    generate_blob_data,
    generate_concentric_circles,
    generate_linear_data,
)
from models.decision_tree import DecisionTreeClassifier
from models.kmeans import KMeans
from models.linear_regression import LinearRegressionGD
from models.logistic_regression import LogisticRegressionGD
from models.pca import PCA
from visualizations.common import VisualizationConfig
from visualizations.decision_tree_viz import DecisionTreeConsoleVisualizer
from visualizations.kmeans_viz import KMeansConsoleVisualizer
from visualizations.linear_regression_viz import LinearRegressionConsoleVisualizer
from visualizations.logistic_regression_viz import LogisticRegressionConsoleVisualizer
from visualizations.pca_viz import PCAConsoleVisualizer


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


def _prompt_visual_config() -> VisualizationConfig:
    show_state_overlay = prompt_bool("Show state overlay", True)
    step_through = prompt_bool("Step-through mode", False)
    show_explanations = prompt_bool("Show explanations", step_through)
    return VisualizationConfig(
        step_delay_s=0.0 if step_through else 0.1,
        display_every=1,
        step_through=step_through,
        show_explanations=show_explanations,
        show_state_overlay=show_state_overlay,
    )


def _prompt_classification_dataset() -> tuple[list[tuple[float, float]], list[int], str]:
    dataset_type = prompt_choice("Dataset type", ["blobs", "circles"], "blobs")
    samples = prompt_int("Sample size", 80)
    noise = prompt_float("Noise level", 0.05)
    class_overlap = prompt_float("Class overlap (0-1)", 0.2)
    class_overlap = clamp(class_overlap, 0.0, 1.0)

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
    return points, labels, dataset_label


def _prompt_clustering_dataset() -> tuple[list[tuple[float, float]], str]:
    samples = prompt_int("Sample size", 100)
    noise = prompt_float("Noise level", 0.1)
    points, _ = generate_blob_data(
        samples=samples,
        classes=3,
        class_overlap=0.3,
        noise=max(0.0, noise),
    )
    dataset_label = "Blob dataset"
    print(f"\n{dataset_label} generated with {samples} samples.")
    print(f"Noise level: {noise}")
    return points, dataset_label


def run_demo() -> None:
    """Run a selected explainable ML demo."""
    algorithm = prompt_choice(
        "Algorithm",
        [
            "linear_regression",
            "logistic_regression",
            "kmeans",
            "pca",
            "decision_tree",
        ],
        "linear_regression",
    )

    if algorithm == "linear_regression":
        learning_rate = prompt_float("Learning rate", 0.1)
        iterations = prompt_int("Iterations", 30)
        samples = prompt_int("Sample size", 40)
        noise = prompt_float("Noise level", 0.05)
        config = _prompt_visual_config()

        features, targets = generate_linear_data(
            slope=3.5,
            intercept=1.2,
            samples=samples,
            noise=max(0.0, noise),
        )
        model = LinearRegressionGD(learning_rate=learning_rate, iterations=iterations)
        visualizer = LinearRegressionConsoleVisualizer(config)
        visualizer.announce(learning_rate=learning_rate, iterations=iterations)
        final_state = model.fit(features, targets, on_step=visualizer.update)
        visualizer.summarize(final_state)
        return

    if algorithm == "logistic_regression":
        learning_rate = prompt_float("Learning rate", 0.2)
        iterations = prompt_int("Iterations", 40)
        config = _prompt_visual_config()
        points, labels, dataset_label = _prompt_classification_dataset()
        model = LogisticRegressionGD(learning_rate=learning_rate, iterations=iterations)
        visualizer = LogisticRegressionConsoleVisualizer(config)
        visualizer.announce(
            learning_rate=learning_rate,
            iterations=iterations,
            dataset=dataset_label,
        )
        final_state = model.fit(points, labels, on_step=visualizer.update)
        visualizer.summarize(final_state)
        return

    if algorithm == "kmeans":
        clusters = prompt_int("Clusters", 3)
        iterations = prompt_int("Iterations", 8)
        config = _prompt_visual_config()
        points, dataset_label = _prompt_clustering_dataset()
        model = KMeans(clusters=clusters, iterations=iterations)
        visualizer = KMeansConsoleVisualizer(config)
        visualizer.announce(clusters=clusters, iterations=iterations, dataset=dataset_label)
        final_state = model.fit(points, on_step=visualizer.update)
        visualizer.summarize(final_state)
        return

    if algorithm == "pca":
        iterations = prompt_int("Iterations", 12)
        config = _prompt_visual_config()
        points, dataset_label = _prompt_clustering_dataset()
        model = PCA(iterations=iterations)
        visualizer = PCAConsoleVisualizer(config)
        visualizer.announce(iterations=iterations, dataset=dataset_label)
        final_state = model.fit(points, on_step=visualizer.update)
        visualizer.summarize(final_state)
        return

    if algorithm == "decision_tree":
        max_depth = prompt_int("Max depth", 2)
        config = _prompt_visual_config()
        points, labels, dataset_label = _prompt_classification_dataset()
        model = DecisionTreeClassifier(max_depth=max_depth)
        visualizer = DecisionTreeConsoleVisualizer(config)
        visualizer.announce(max_depth=max_depth, dataset=dataset_label)
        root = model.fit(points, labels)
        visualizer.update(1, root)
        visualizer.summarize(root)


def main() -> None:
    """Main entry point."""
    run_demo()


if __name__ == "__main__":
    main()
