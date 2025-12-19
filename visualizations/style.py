"""Shared plot styling for matplotlib-based visualizations."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ColorPalette:
    """Palette tokens for plot elements."""

    data_points: str = "tab:blue"
    regression_line: str = "tab:red"
    loss_line: str = "tab:purple"
    residuals: str = "tab:gray"
    gradients: str = "tab:green"
    decision_boundary: str = "tab:orange"


@dataclass(frozen=True)
class AxisLabels:
    """Standard axis labels for shared plots."""

    feature: str = "Feature"
    target: str = "Target"
    iteration: str = "Iteration"
    loss: str = "Loss"


@dataclass(frozen=True)
class AxisTitles:
    """Default titles for shared plots."""

    data: str = "Linear Regression Fit"
    loss: str = "Loss Trend"


@dataclass(frozen=True)
class FigureLayout:
    """Figure sizing and layout defaults."""

    figsize: tuple[float, float] = (10, 4)
    tight_layout: bool = True
    control_panel_bottom: float = 0.38
    grid_alpha: float = 0.2


@dataclass(frozen=True)
class PlotStyle:
    """Styling bundle for matplotlib visualizations."""

    palette: ColorPalette = field(default_factory=ColorPalette)
    labels: AxisLabels = field(default_factory=AxisLabels)
    titles: AxisTitles = field(default_factory=AxisTitles)
    layout: FigureLayout = field(default_factory=FigureLayout)


DEFAULT_STYLE = PlotStyle()


def apply_base_style(style: PlotStyle = DEFAULT_STYLE) -> None:
    """Apply shared matplotlib rcParams for consistent visuals."""
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": style.layout.grid_alpha,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    )


def apply_axis_labels(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply standardized labels and title to an axis."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
