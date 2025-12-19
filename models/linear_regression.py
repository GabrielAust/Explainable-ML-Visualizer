"""Linear regression model implemented with gradient descent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from data.dataset_generators import generate_linear_data as _generate_linear_data


@dataclass
class LinearRegressionState:
    """Holds the current parameters of the model."""

    weight: float
    bias: float


@dataclass
class LinearRegressionGradients:
    """Tracks gradients for each parameter."""

    weight: float
    bias: float


class LinearRegressionGD:
    """Simple linear regression using batch gradient descent."""

    def __init__(self, learning_rate: float = 0.01, iterations: int = 100) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.state = LinearRegressionState(weight=0.0, bias=0.0)

    def predict(self, features: Sequence[float]) -> list[float]:
        """Predict outputs for the given input features."""
        return [self.state.weight * x + self.state.bias for x in features]

    def fit(
        self,
        features: Sequence[float],
        targets: Sequence[float],
        on_step: Callable[
            [int, float, LinearRegressionState, LinearRegressionGradients], None
        ]
        | None = None,
    ) -> LinearRegressionState:
        """Fit the model to features/targets using gradient descent.

        Args:
            features: Input feature values.
            targets: Target values for each feature.
            on_step: Optional callback invoked after each update.
        """
        if len(features) != len(targets):
            raise ValueError("features and targets must be the same length.")
        if not features:
            raise ValueError("features must not be empty.")

        for step in range(1, self.iterations + 1):
            predictions = self.predict(features)
            errors = [pred - target for pred, target in zip(predictions, targets)]
            loss = sum(error**2 for error in errors) / len(errors)

            gradient_w = sum(error * x for error, x in zip(errors, features)) * (2 / len(errors))
            gradient_b = sum(errors) * (2 / len(errors))

            self.state.weight -= self.learning_rate * gradient_w
            self.state.bias -= self.learning_rate * gradient_b

            if on_step:
                on_step(
                    step,
                    loss,
                    self.state,
                    LinearRegressionGradients(weight=gradient_w, bias=gradient_b),
                )

        return self.state


def generate_linear_data(
    slope: float,
    intercept: float,
    samples: int = 50,
    noise: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Generate synthetic linear data for demonstration."""
    return _generate_linear_data(
        slope=slope,
        intercept=intercept,
        samples=samples,
        noise=noise,
    )
