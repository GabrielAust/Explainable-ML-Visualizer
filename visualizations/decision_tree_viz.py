"""Console visualization for decision tree training."""

from __future__ import annotations

from models.decision_tree import DecisionTreeNode
from visualizations.common import ConsoleVisualizer, VisualizationConfig


class DecisionTreeConsoleVisualizer:
    """Visualize decision tree training summary in the console."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.console = ConsoleVisualizer(config)

    def announce(self, max_depth: int, dataset: str) -> None:
        """Display the setup before training starts."""
        self.console.announce(
            "Decision Tree Training",
            [
                ("Max depth", str(max_depth)),
                ("Dataset", dataset),
            ],
        )

    def _node_summary(self, node: DecisionTreeNode, prefix: str = "") -> list[str]:
        lines: list[str] = []
        if node.feature_index is None or node.threshold is None:
            lines.append(
                f"{prefix}Leaf: predict={node.prediction} | samples={node.samples} | "
                f"gini={node.gini:.3f}"
                if node.gini is not None
                else f"{prefix}Leaf: predict={node.prediction}"
            )
            return lines

        lines.append(
            f"{prefix}Split: feature {node.feature_index} <= {node.threshold:.3f} "
            f"| gini={node.gini:.3f} | samples={node.samples}"
        )
        if node.left:
            lines.extend(self._node_summary(node.left, prefix=prefix + "  "))
        if node.right:
            lines.extend(self._node_summary(node.right, prefix=prefix + "  "))
        return lines

    def update(self, step: int, root: DecisionTreeNode) -> None:
        """Display the current tree snapshot."""
        self.console.render_step(
            step,
            [("Nodes", "Tree snapshot")],
            self._node_summary(root),
            explanation="Greedy splits minimize Gini impurity at each depth.",
        )

    def summarize(self, root: DecisionTreeNode) -> None:
        """Display the final tree structure."""
        lines = ["Tree ready. Final structure:"]
        lines.extend(self._node_summary(root))
        self.console.summarize(lines)
