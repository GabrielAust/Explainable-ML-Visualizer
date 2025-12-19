"""Console visualization for PCA."""

from __future__ import annotations

from models.pca import PCAState
from visualizations.common import ConsoleVisualizer, VisualizationConfig


class PCAConsoleVisualizer:
    """Visualize PCA computation steps in the console."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.console = ConsoleVisualizer(config)

    def announce(self, iterations: int, dataset: str) -> None:
        """Display the setup before PCA starts."""
        self.console.announce(
            "PCA Projection",
            [
                ("Iterations", str(iterations)),
                ("Dataset", dataset),
            ],
        )

    def update(self, step: int, state: PCAState) -> None:
        """Display the current PCA step."""
        component = state.components[0]
        explained = state.explained_variance[0]
        self.console.render_step(
            step,
            [("Explained variance", f"{explained:.2%}")],
            [
                f"Mean = ({state.mean[0]:.3f}, {state.mean[1]:.3f})",
                f"Component 1 = ({component[0]:.3f}, {component[1]:.3f})",
            ],
            explanation=(
                "Power iteration refines the first principal component using the "
                "covariance matrix."
            ),
        )

    def summarize(self, state: PCAState) -> None:
        """Display the final PCA components."""
        lines = [
            f"Mean: ({state.mean[0]:.3f}, {state.mean[1]:.3f})",
        ]
        for idx, component in enumerate(state.components, start=1):
            lines.append(
                f"Component {idx}: ({component[0]:.3f}, {component[1]:.3f}) "
                f"| Explained variance: {state.explained_variance[idx - 1]:.2%}"
            )
        self.console.summarize(lines)
