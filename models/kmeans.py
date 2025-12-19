"""K-means clustering model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class KMeansState:
    """Tracks current centroids and assignments."""

    centroids: list[tuple[float, float]]
    assignments: list[int]


class KMeans:
    """Simple k-means clustering for 2D points."""

    def __init__(self, clusters: int = 3, iterations: int = 10, seed: int | None = None) -> None:
        if clusters <= 0:
            raise ValueError("clusters must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        self.clusters = clusters
        self.iterations = iterations
        self.seed = seed
        self.state = KMeansState(centroids=[], assignments=[])

    def _initialize_centroids(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        rng = random.Random(self.seed)
        return list(rng.sample(points, self.clusters))

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def fit(
        self,
        points: Sequence[tuple[float, float]],
        on_step: Callable[[int, float, KMeansState], None] | None = None,
    ) -> KMeansState:
        """Cluster the points using k-means."""
        if not points:
            raise ValueError("points must not be empty.")
        if self.clusters > len(points):
            raise ValueError("clusters must be less than or equal to number of points.")

        centroids = self._initialize_centroids(points)
        assignments = [0] * len(points)

        for step in range(1, self.iterations + 1):
            for idx, point in enumerate(points):
                distances = [self._distance(point, centroid) for centroid in centroids]
                assignments[idx] = distances.index(min(distances))

            new_centroids: list[tuple[float, float]] = []
            for cluster in range(self.clusters):
                cluster_points = [
                    point
                    for point, assignment in zip(points, assignments)
                    if assignment == cluster
                ]
                if cluster_points:
                    avg_x = sum(x for x, _ in cluster_points) / len(cluster_points)
                    avg_y = sum(y for _, y in cluster_points) / len(cluster_points)
                    new_centroids.append((avg_x, avg_y))
                else:
                    new_centroids.append(centroids[cluster])

            centroids = new_centroids
            inertia = sum(
                self._distance(point, centroids[assignment]) ** 2
                for point, assignment in zip(points, assignments)
            )

            self.state = KMeansState(centroids=centroids, assignments=list(assignments))

            if on_step:
                on_step(step, inertia, self.state)

        return self.state
