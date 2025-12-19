"""Logistic regression model with gradient descent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class LogisticRegressionState:
    """Holds the current parameters of the model."""

    weight_x: float
    weight_y: float
    bias: float


@dataclass
class LogisticRegressionGradients:
    """Tracks gradients for each parameter."""

    weight_x: float
    weight_y: float
    bias: float


class LogisticRegressionGD:
    """Binary logistic regression using batch gradient descent."""

    def __init__(self, learning_rate: float = 0.1, iterations: int = 50) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.state = LogisticRegressionState(weight_x=0.0, weight_y=0.0, bias=0.0)

    def _sigmoid(self, value: float) -> float:
        return 1 / (1 + math.exp(-value))

    def predict_proba(self, points: Sequence[tuple[float, float]]) -> list[float]:
        """Predict probabilities for the given 2D points."""
        return [
            self._sigmoid(self.state.weight_x * x + self.state.weight_y * y + self.state.bias)
            for x, y in points
        ]

    def fit(
        self,
        points: Sequence[tuple[float, float]],
        labels: Sequence[int],
        on_step: Callable[
            [int, float, LogisticRegressionState, LogisticRegressionGradients], None
        ]
        | None = None,
    ) -> LogisticRegressionState:
        """Fit the model using gradient descent."""
        if len(points) != len(labels):
            raise ValueError("points and labels must be the same length.")
        if not points:
            raise ValueError("points must not be empty.")

        sample_count = len(points)
        for step in range(1, self.iterations + 1):
            probabilities = self.predict_proba(points)
            errors = [pred - label for pred, label in zip(probabilities, labels)]
            loss = -sum(
                label * math.log(max(pred, 1e-9))
                + (1 - label) * math.log(max(1 - pred, 1e-9))
                for pred, label in zip(probabilities, labels)
            ) / sample_count

            gradient_wx = sum(error * x for error, (x, _) in zip(errors, points)) / sample_count
            gradient_wy = sum(error * y for error, (_, y) in zip(errors, points)) / sample_count
            gradient_b = sum(errors) / sample_count

            self.state.weight_x -= self.learning_rate * gradient_wx
            self.state.weight_y -= self.learning_rate * gradient_wy
            self.state.bias -= self.learning_rate * gradient_b

            if on_step:
                on_step(
                    step,
                    loss,
                    self.state,
                    LogisticRegressionGradients(
                        weight_x=gradient_wx,
                        weight_y=gradient_wy,
                        bias=gradient_b,
                    ),
                )

        return self.state
