"""Abstract interface for visualizations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class VisualizationBase(ABC):
    """Define a standard lifecycle for visualization modules."""

    @abstractmethod
    def setup(self, metadata: Mapping[str, Any] | None = None) -> None:
        """Prepare the visualization for updates.

        Args:
            metadata: Optional dataset metadata (feature names, label mapping,
                normalization details, or other contextual information needed
                to configure axes and legends).
        """

    @abstractmethod
    def update(
        self,
        state: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Render a new visualization frame based on the model state.

        Args:
            state: Model state for the current iteration. Expected keys include
                model weights, loss values, and predictions (or similarly named
                fields) needed to draw the visualization.
            metadata: Optional dataset metadata (feature names, label mapping,
                input ranges, or other context to interpret the state).
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release any resources used by the visualization."""
