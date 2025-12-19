"""Shared model state definitions for visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class ModelState:
    """Capture model parameters, predictions, and loss history for visualization."""

    parameters: dict[str, float]
    predictions: list[float]
    loss_history: list[float]

    def as_mapping(self) -> Mapping[str, Any]:
        """Return a dictionary-like view of the model state."""
        return {
            **self.parameters,
            "predictions": self.predictions,
            "loss_history": self.loss_history,
        }
