"""Decision tree classifier for 2D data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DecisionTreeNode:
    """A node in a binary decision tree."""

    feature_index: int | None = None
    threshold: float | None = None
    left: DecisionTreeNode | None = None
    right: DecisionTreeNode | None = None
    prediction: int | None = None
    gini: float | None = None
    samples: int = 0


class DecisionTreeClassifier:
    """Binary decision tree classifier with limited depth."""

    def __init__(self, max_depth: int = 2) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        self.max_depth = max_depth
        self.root: DecisionTreeNode | None = None

    @staticmethod
    def _gini(labels: Sequence[int]) -> float:
        if not labels:
            return 0.0
        proportion = sum(labels) / len(labels)
        return 1.0 - proportion**2 - (1 - proportion) ** 2

    def _best_split(
        self, points: Sequence[tuple[float, float]], labels: Sequence[int]
    ) -> tuple[int, float, float] | None:
        best_feature = None
        best_threshold = None
        best_gini = None
        for feature_index in range(2):
            feature_values = sorted({point[feature_index] for point in points})
            thresholds = [
                (feature_values[idx] + feature_values[idx + 1]) / 2
                for idx in range(len(feature_values) - 1)
            ]
            for threshold in thresholds:
                left_labels = [
                    label
                    for point, label in zip(points, labels)
                    if point[feature_index] <= threshold
                ]
                right_labels = [
                    label
                    for point, label in zip(points, labels)
                    if point[feature_index] > threshold
                ]
                gini = (
                    len(left_labels) * self._gini(left_labels)
                    + len(right_labels) * self._gini(right_labels)
                ) / len(labels)
                if best_gini is None or gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
        if best_feature is None or best_threshold is None or best_gini is None:
            return None
        return best_feature, best_threshold, best_gini

    def _build_tree(
        self,
        points: Sequence[tuple[float, float]],
        labels: Sequence[int],
        depth: int,
    ) -> DecisionTreeNode:
        prediction = 1 if sum(labels) >= len(labels) / 2 else 0
        node = DecisionTreeNode(
            prediction=prediction,
            gini=self._gini(labels),
            samples=len(labels),
        )
        if depth >= self.max_depth or len(set(labels)) == 1:
            return node

        split = self._best_split(points, labels)
        if not split:
            return node

        feature_index, threshold, gini = split
        left_points = [point for point in points if point[feature_index] <= threshold]
        left_labels = [
            label
            for point, label in zip(points, labels)
            if point[feature_index] <= threshold
        ]
        right_points = [point for point in points if point[feature_index] > threshold]
        right_labels = [
            label
            for point, label in zip(points, labels)
            if point[feature_index] > threshold
        ]

        node.feature_index = feature_index
        node.threshold = threshold
        node.gini = gini
        node.left = self._build_tree(left_points, left_labels, depth + 1)
        node.right = self._build_tree(right_points, right_labels, depth + 1)
        return node

    def fit(
        self,
        points: Sequence[tuple[float, float]],
        labels: Sequence[int],
    ) -> DecisionTreeNode:
        """Fit the decision tree to the data."""
        if len(points) != len(labels):
            raise ValueError("points and labels must be the same length.")
        if not points:
            raise ValueError("points must not be empty.")
        self.root = self._build_tree(points, labels, depth=0)
        return self.root

    def predict(self, point: tuple[float, float]) -> int:
        """Predict the label for a point."""
        if not self.root:
            raise ValueError("Decision tree has not been fitted.")

        node = self.root
        while node.feature_index is not None and node.threshold is not None:
            if point[node.feature_index] <= node.threshold:
                node = node.left or node
            else:
                node = node.right or node
        return node.prediction if node.prediction is not None else 0
