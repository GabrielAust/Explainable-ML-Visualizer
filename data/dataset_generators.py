"""Synthetic dataset generators for the visualizer."""

from __future__ import annotations

import math
import random
from typing import Sequence


def _validate_samples(samples: int) -> None:
    if samples <= 0:
        raise ValueError("samples must be positive.")


def _validate_noise(noise: float) -> None:
    if noise < 0:
        raise ValueError("noise must be non-negative.")


def _validate_overlap(class_overlap: float) -> None:
    if not 0 <= class_overlap <= 1:
        raise ValueError("class_overlap must be between 0 and 1.")


def _allocate_counts(samples: int, classes: int) -> list[int]:
    base = samples // classes
    remainder = samples % classes
    return [base + (1 if index < remainder else 0) for index in range(classes)]


def inject_noise(
    data: Sequence[float] | Sequence[Sequence[float]],
    noise: float,
    seed: int | None = None,
) -> list[float] | list[tuple[float, ...]]:
    """Inject Gaussian noise into scalar or vector data."""
    _validate_noise(noise)
    if not data:
        return []

    rng = random.Random(seed)
    first_item = data[0]

    if isinstance(first_item, (tuple, list)):
        noisy_points: list[tuple[float, ...]] = []
        for point in data:  # type: ignore[assignment]
            noisy_points.append(
                tuple(value + rng.gauss(0, noise) for value in point)
            )
        return noisy_points

    return [value + rng.gauss(0, noise) for value in data]  # type: ignore[return-value]


def generate_linear_data(
    slope: float,
    intercept: float,
    samples: int = 50,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """Generate synthetic linear data for regression demos."""
    _validate_samples(samples)
    _validate_noise(noise)

    features = [i / (samples - 1) if samples > 1 else 0.0 for i in range(samples)]
    targets = [slope * x_val + intercept for x_val in features]

    if noise:
        targets = inject_noise(targets, noise, seed)

    return features, targets


def generate_blob_data(
    samples: int = 100,
    classes: int = 2,
    class_overlap: float = 0.0,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Generate clustered blob data for classification demos."""
    _validate_samples(samples)
    _validate_noise(noise)
    _validate_overlap(class_overlap)
    if classes < 2:
        raise ValueError("classes must be at least 2.")

    rng = random.Random(seed)
    radius = 3.0 * (1 - class_overlap) + 0.5
    centers = [
        (
            radius * math.cos(2 * math.pi * idx / classes),
            radius * math.sin(2 * math.pi * idx / classes),
        )
        for idx in range(classes)
    ]

    points: list[tuple[float, float]] = []
    labels: list[int] = []
    for label, count in enumerate(_allocate_counts(samples, classes)):
        cx, cy = centers[label]
        for _ in range(count):
            points.append((cx, cy))
            labels.append(label)

    if noise:
        points = inject_noise(points, noise, seed=rng.randint(0, 1_000_000))  # type: ignore[assignment]

    return points, labels


def generate_concentric_circles(
    samples: int = 100,
    class_overlap: float = 0.0,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Generate concentric circle data for classification demos."""
    _validate_samples(samples)
    _validate_noise(noise)
    _validate_overlap(class_overlap)

    rng = random.Random(seed)
    inner_radius = 1.0
    gap = 1.0 * (1 - class_overlap)
    outer_radius = inner_radius + gap

    points: list[tuple[float, float]] = []
    labels: list[int] = []
    counts = _allocate_counts(samples, 2)

    for label, count in enumerate(counts):
        radius = inner_radius if label == 0 else outer_radius
        for _ in range(count):
            angle = rng.uniform(0, 2 * math.pi)
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
            labels.append(label)

    if noise:
        points = inject_noise(points, noise, seed=rng.randint(0, 1_000_000))  # type: ignore[assignment]

    return points, labels
