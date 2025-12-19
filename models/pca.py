"""Principal Component Analysis implementation for 2D points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class PCAState:
    """Tracks PCA mean and component vectors."""

    mean: tuple[float, float]
    components: list[tuple[float, float]]
    explained_variance: list[float]


class PCA:
    """Compute PCA components for 2D data using power iteration."""

    def __init__(self, iterations: int = 10) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        self.iterations = iterations
        self.state = PCAState(mean=(0.0, 0.0), components=[], explained_variance=[])

    @staticmethod
    def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
        norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if norm == 0:
            return (0.0, 0.0)
        return (vector[0] / norm, vector[1] / norm)

    def fit(
        self,
        points: Sequence[tuple[float, float]],
        on_step: Callable[[int, PCAState], None] | None = None,
    ) -> PCAState:
        """Fit PCA to 2D points."""
        if not points:
            raise ValueError("points must not be empty.")

        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        centered = [(x - mean_x, y - mean_y) for x, y in points]

        cov_xx = sum(x * x for x, _ in centered) / len(centered)
        cov_xy = sum(x * y for x, y in centered) / len(centered)
        cov_yy = sum(y * y for _, y in centered) / len(centered)

        vector = (1.0, 0.0)
        for step in range(1, self.iterations + 1):
            next_vec = (
                cov_xx * vector[0] + cov_xy * vector[1],
                cov_xy * vector[0] + cov_yy * vector[1],
            )
            vector = self._normalize(next_vec)

            eigenvalue = (
                vector[0] * (cov_xx * vector[0] + cov_xy * vector[1])
                + vector[1] * (cov_xy * vector[0] + cov_yy * vector[1])
            )
            total_variance = cov_xx + cov_yy
            explained = eigenvalue / total_variance if total_variance else 0.0

            self.state = PCAState(
                mean=(mean_x, mean_y),
                components=[vector],
                explained_variance=[explained],
            )

            if on_step:
                on_step(step, self.state)

        perpendicular = self._normalize((-vector[1], vector[0]))
        second_variance = max(0.0, 1.0 - self.state.explained_variance[0])
        self.state = PCAState(
            mean=(mean_x, mean_y),
            components=[vector, perpendicular],
            explained_variance=[self.state.explained_variance[0], second_variance],
        )

        return self.state
