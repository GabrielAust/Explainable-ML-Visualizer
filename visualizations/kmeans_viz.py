"""Console visualization for k-means clustering."""

from __future__ import annotations

from models.kmeans import KMeansState
from visualizations.common import ConsoleVisualizer, VisualizationConfig


class KMeansConsoleVisualizer:
    """Visualize k-means clustering updates in the console."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.console = ConsoleVisualizer(config)

    def announce(self, clusters: int, iterations: int, dataset: str) -> None:
        """Display the setup before clustering starts."""
        self.console.announce(
            "K-Means Clustering",
            [
                ("Clusters", str(clusters)),
                ("Iterations", str(iterations)),
                ("Dataset", dataset),
            ],
        )

    def update(self, step: int, inertia: float, state: KMeansState) -> None:
        """Display the current clustering step."""
        centroid_lines = [
            f"Centroid {idx + 1}: ({centroid[0]:.3f}, {centroid[1]:.3f})"
            for idx, centroid in enumerate(state.centroids)
        ]
        self.console.render_step(
            step,
            [("Inertia", f"{inertia:.3f}")],
            centroid_lines,
            explanation=(
                "Assigned points to the nearest centroid, then recomputed centroids "
                "as cluster means."
            ),
        )

    def summarize(self, state: KMeansState) -> None:
        """Display the final centroids."""
        lines = ["Clustering complete."]
        lines.extend(
            [
                f"Final centroid {idx + 1}: ({centroid[0]:.3f}, {centroid[1]:.3f})"
                for idx, centroid in enumerate(state.centroids)
            ]
        )
        self.console.summarize(lines)
