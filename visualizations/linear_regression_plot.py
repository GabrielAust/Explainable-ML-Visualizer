"""Matplotlib visualization for linear regression training."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons, Slider
import numpy as np

from models.linear_regression import LinearRegressionGD, generate_linear_data
from utils.state import ModelState
from visualizations.base import VisualizationBase


class LinearRegressionPlotVisualizer(VisualizationBase):
    """Plot scatter data, regression line, and loss trend over time."""

    def __init__(self) -> None:
        self.fig: plt.Figure | None = None
        self.data_ax: plt.Axes | None = None
        self.loss_ax: plt.Axes | None = None
        self._scatter = None
        self._line = None
        self._loss_line = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._loss_history: list[float] = []
        self._residual_collection: LineCollection | None = None
        self._gradient_arrow: FancyArrowPatch | None = None
        self._gradient_text: Text | None = None
        self._decision_boundary: plt.Line2D | None = None
        self._decision_label: Text | None = None
        self._overlay_visibility = {
            "residuals": True,
            "gradients": True,
            "decision_boundary": True,
        }

    def setup(self, metadata: Mapping[str, Any] | None = None) -> None:
        """Prepare the matplotlib figure for updates."""
        if self.fig is not None:
            return

        plt.ion()
        self.fig, (self.data_ax, self.loss_ax) = plt.subplots(1, 2, figsize=(10, 4))
        self.data_ax.set_title("Linear Regression Fit")
        self.data_ax.set_xlabel("Feature")
        self.data_ax.set_ylabel("Target")
        self.loss_ax.set_title("Loss Trend")
        self.loss_ax.set_xlabel("Iteration")
        self.loss_ax.set_ylabel("Loss")

        self._x, self._y = self._extract_xy({}, metadata)
        self._scatter = self.data_ax.scatter(self._x, self._y, color="tab:blue", alpha=0.7)
        self._line, = self.data_ax.plot(self._x, self._y, color="tab:red", linewidth=2)
        self._loss_line, = self.loss_ax.plot([], [], color="tab:purple", linewidth=2)
        self._residual_collection = LineCollection([], colors="tab:gray", linewidths=1, alpha=0.6)
        self.data_ax.add_collection(self._residual_collection)
        self._gradient_arrow = FancyArrowPatch((0, 0), (0, 0), color="tab:green", arrowstyle="->")
        self.data_ax.add_patch(self._gradient_arrow)
        self._gradient_text = self.data_ax.text(
            0.02,
            0.95,
            "",
            transform=self.data_ax.transAxes,
            ha="left",
            va="top",
            color="tab:green",
        )
        self._decision_boundary = self.data_ax.axvline(0, color="tab:orange", linestyle="--", linewidth=1.5)
        self._decision_label = self.data_ax.text(
            0.02,
            0.9,
            "",
            transform=self.data_ax.transAxes,
            ha="left",
            va="top",
            color="tab:orange",
        )

        self.data_ax.relim()
        self.data_ax.autoscale_view()
        self.loss_ax.set_xlim(0, 1)
        self.loss_ax.set_ylim(0, 1)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(
        self,
        state: Mapping[str, Any] | ModelState,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Render a new frame using the provided model state."""
        if self.fig is None or self.data_ax is None or self.loss_ax is None:
            self.setup(metadata)

        normalized_state = self._normalize_state(state)
        x, y = self._extract_xy(normalized_state, metadata)
        self._x, self._y = x, y

        if self._scatter is not None:
            self._scatter.set_offsets(np.column_stack([x, y]))

        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        line_y = self._compute_regression_line(normalized_state, line_x)
        if self._line is not None:
            self._line.set_data(line_x, line_y)

        self._update_loss_history(normalized_state)
        if self._loss_line is not None:
            iterations = np.arange(len(self._loss_history))
            self._loss_line.set_data(iterations, self._loss_history)

        self._update_overlays(normalized_state, metadata)
        self.data_ax.relim()
        self.data_ax.autoscale_view()
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        plt.pause(0.001)

    def teardown(self) -> None:
        """Close the matplotlib figure."""
        if self.fig is None:
            return
        plt.close(self.fig)
        self.fig = None
        self.data_ax = None
        self.loss_ax = None
        self._scatter = None
        self._line = None
        self._loss_line = None
        self._residual_collection = None
        self._gradient_arrow = None
        self._gradient_text = None
        self._decision_boundary = None
        self._decision_label = None

    def set_overlay_visibility(self, overlay: str, visible: bool) -> None:
        if overlay not in self._overlay_visibility:
            return
        self._overlay_visibility[overlay] = visible
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _extract_xy(
        self,
        state: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self._coerce_vector(state.get("x") or state.get("features"))
        y = self._coerce_vector(state.get("y") or state.get("targets"))

        if x is None or y is None:
            meta = metadata or {}
            x = x or self._coerce_vector(meta.get("x") or meta.get("features"))
            y = y or self._coerce_vector(meta.get("y") or meta.get("targets"))

        if x is None or y is None:
            x = np.linspace(0, 10, 50)
            rng = np.random.default_rng(42)
            y = 2.5 * x + 1.0 + rng.normal(scale=2.0, size=x.shape)

        return x, y

    def _compute_regression_line(self, state: Mapping[str, Any], line_x: np.ndarray) -> np.ndarray:
        predictions = state.get("predictions")
        if predictions is not None and len(predictions) == len(line_x):
            return np.asarray(predictions, dtype=float)

        weight = state.get("weight")
        bias = state.get("bias")
        if weight is not None and bias is not None:
            return np.asarray(weight, dtype=float) * line_x + np.asarray(bias, dtype=float)

        return np.interp(line_x, self._x, self._y)

    def _update_loss_history(self, state: Mapping[str, Any]) -> None:
        history = state.get("loss_history")
        if history is not None:
            self._loss_history = [float(value) for value in history]
            return

        loss = state.get("loss")
        if loss is not None:
            self._loss_history.append(float(loss))

        if not self._loss_history:
            self._loss_history = [1.0, 0.8, 0.6, 0.55]

    def _update_overlays(
        self,
        state: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        if self.data_ax is None:
            return
        x = self._x
        y = self._y
        if x is None or y is None:
            return
        predictions = self._compute_predictions(state, x)
        self._update_residuals(x, y, predictions)
        self._update_gradient_annotation(x, predictions, metadata)
        self._update_decision_boundary(x, y, state)

    def _update_residuals(self, x: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> None:
        if self._residual_collection is None:
            return
        show = self._overlay_visibility.get("residuals", False)
        if not show:
            self._residual_collection.set_visible(False)
            return
        segments = [
            [(float(xi), float(yi)), (float(xi), float(pred))]
            for xi, yi, pred in zip(x, y, predictions)
        ]
        self._residual_collection.set_segments(segments)
        self._residual_collection.set_visible(True)

    def _update_gradient_annotation(
        self,
        x: np.ndarray,
        predictions: np.ndarray,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        if self._gradient_arrow is None or self._gradient_text is None:
            return
        show = self._overlay_visibility.get("gradients", False)
        gradients = (metadata or {}).get("gradients")
        if not show or not gradients:
            self._gradient_arrow.set_visible(False)
            self._gradient_text.set_text("")
            return
        gradient_w = float(gradients.get("weight", 0.0))
        gradient_b = float(gradients.get("bias", 0.0))
        mean_x = float(np.mean(x))
        mean_pred = float(np.interp(mean_x, x, predictions))
        delta_pred = -(gradient_w * mean_x + gradient_b)
        y_min, y_max = self.data_ax.get_ylim()
        scale = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
        arrow_len = np.clip(delta_pred, -scale, scale)
        start = (mean_x, mean_pred)
        end = (mean_x, mean_pred + arrow_len)
        self._gradient_arrow.set_positions(start, end)
        self._gradient_arrow.set_visible(True)
        self._gradient_text.set_text(
            f"Gradient: dW={gradient_w:.3f}, dB={gradient_b:.3f}"
        )

    def _update_decision_boundary(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: Mapping[str, Any],
    ) -> None:
        if self._decision_boundary is None or self._decision_label is None:
            return
        show = self._overlay_visibility.get("decision_boundary", False)
        weight = state.get("weight")
        bias = state.get("bias")
        if not show or weight is None or bias is None or weight == 0:
            self._decision_boundary.set_visible(False)
            self._decision_label.set_text("")
            return
        target_level = float(np.mean(y))
        boundary_x = (target_level - float(bias)) / float(weight)
        self._decision_boundary.set_xdata([boundary_x, boundary_x])
        self._decision_boundary.set_visible(True)
        self._decision_label.set_text("Decision boundary (mean target)")

    def _compute_predictions(
        self,
        state: Mapping[str, Any],
        x: np.ndarray,
    ) -> np.ndarray:
        predictions = state.get("predictions")
        if predictions is not None and len(predictions) == len(x):
            return np.asarray(predictions, dtype=float)
        return self._compute_regression_line(state, x)

    @staticmethod
    def _normalize_state(state: Mapping[str, Any] | ModelState) -> Mapping[str, Any]:
        if isinstance(state, ModelState):
            return state.as_mapping()
        return state

    @staticmethod
    def _coerce_vector(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float).squeeze()
        if array.ndim != 1:
            array = array.reshape(-1)
        if array.size == 0:
            return None
        return array


class LinearRegressionInteractiveTrainer:
    """Interactive trainer that adds UI controls to the linear regression plot."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        iterations_per_step: int = 1,
        max_iterations: int = 100,
        slope: float = 2.5,
        intercept: float = 1.0,
        noise: float = 2.0,
        samples: int = 50,
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations_per_step = iterations_per_step
        self.max_iterations = max_iterations
        self._paused = False
        self._current_step = 0
        self._last_state: ModelState | None = None

        features, targets = generate_linear_data(
            slope=slope,
            intercept=intercept,
            samples=samples,
            noise=noise,
        )
        self.features = np.asarray(features, dtype=float)
        self.targets = np.asarray(targets, dtype=float)

        self.model = LinearRegressionGD(learning_rate=learning_rate, iterations=max_iterations)
        self.visualizer = LinearRegressionPlotVisualizer()
        self._learning_rate_slider: Slider | None = None
        self._iterations_slider: Slider | None = None
        self._pause_button: Button | None = None
        self._overlay_checks: CheckButtons | None = None
        self._timer = None

    def show(self) -> None:
        """Render the interactive plot with controls."""
        self.visualizer.setup(metadata={"features": self.features, "targets": self.targets})
        self._build_controls()
        self._render_current_state()
        self._start_timer()
        plt.show()

    def _build_controls(self) -> None:
        if self.visualizer.fig is None:
            return

        self.visualizer.fig.subplots_adjust(bottom=0.38)

        lr_ax = self.visualizer.fig.add_axes([0.15, 0.2, 0.7, 0.03])
        iterations_ax = self.visualizer.fig.add_axes([0.15, 0.14, 0.7, 0.03])
        button_ax = self.visualizer.fig.add_axes([0.82, 0.03, 0.15, 0.06])
        overlay_ax = self.visualizer.fig.add_axes([0.15, 0.03, 0.3, 0.1])

        self._learning_rate_slider = Slider(
            lr_ax,
            "Learning rate",
            valmin=0.001,
            valmax=1.0,
            valinit=self.learning_rate,
            valstep=0.001,
        )
        self._iterations_slider = Slider(
            iterations_ax,
            "Iterations / step",
            valmin=1,
            valmax=20,
            valinit=self.iterations_per_step,
            valstep=1,
        )
        self._pause_button = Button(button_ax, "Pause")
        self._overlay_checks = CheckButtons(
            overlay_ax,
            ["Residuals", "Gradients", "Decision boundary"],
            [True, True, True],
        )

        self._learning_rate_slider.on_changed(self._on_learning_rate_change)
        self._iterations_slider.on_changed(self._on_iterations_change)
        self._pause_button.on_clicked(self._on_pause_toggle)
        self._overlay_checks.on_clicked(self._on_overlay_toggle)

    def _start_timer(self) -> None:
        if self.visualizer.fig is None:
            return
        self._timer = self.visualizer.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._on_timer_tick)
        self._timer.start()

    def _on_learning_rate_change(self, value: float) -> None:
        self.learning_rate = float(value)
        self.model.learning_rate = self.learning_rate
        self._render_current_state()

    def _on_iterations_change(self, value: float) -> None:
        self.iterations_per_step = int(value)

    def _on_pause_toggle(self, _event: Any) -> None:
        self._paused = not self._paused
        if self._pause_button is not None:
            label = "Resume" if self._paused else "Pause"
            self._pause_button.label.set_text(label)
        self._render_current_state()

    def _on_timer_tick(self) -> None:
        if self._paused or self._current_step >= self.max_iterations:
            return

        steps_to_run = min(self.iterations_per_step, self.max_iterations - self._current_step)
        for _ in range(steps_to_run):
            self._last_state, gradients = self.model.step(self.features, self.targets)
            self._current_step += 1

        if self._last_state is not None:
            self._render_state(self._last_state, {"weight": gradients.weight, "bias": gradients.bias})

    def _render_current_state(self) -> None:
        if self._last_state is None:
            predictions = self.model.predict(self.features)
            errors = [pred - target for pred, target in zip(predictions, self.targets)]
            loss = sum(error**2 for error in errors) / len(errors)
            if not self.model.loss_history:
                self.model.loss_history = [loss]
            gradients = self._compute_gradients(predictions)
            self._last_state = ModelState(
                parameters={
                    "weight": self.model.state.weight,
                    "bias": self.model.state.bias,
                },
                predictions=predictions,
                loss_history=self.model.loss_history.copy(),
            )
        else:
            predictions = self._last_state.predictions
            gradients = self._compute_gradients(predictions)
        self._render_state(self._last_state, gradients)

    def _render_state(self, state: ModelState, gradients: dict[str, float]) -> None:
        self.visualizer.update(
            state,
            metadata={
                "features": self.features,
                "targets": self.targets,
                "gradients": gradients,
            },
        )

    def _on_overlay_toggle(self, _label: str) -> None:
        if self._overlay_checks is None:
            return
        labels = ["Residuals", "Gradients", "Decision boundary"]
        states = self._overlay_checks.get_status()
        mapping = {
            "Residuals": "residuals",
            "Gradients": "gradients",
            "Decision boundary": "decision_boundary",
        }
        for label, visible in zip(labels, states):
            overlay = mapping[label]
            self.visualizer.set_overlay_visibility(overlay, visible)
        self._render_current_state()

    def _compute_gradients(self, predictions: np.ndarray | list[float]) -> dict[str, float]:
        errors = [pred - target for pred, target in zip(predictions, self.targets)]
        gradient_w = sum(error * x for error, x in zip(errors, self.features)) * (2 / len(errors))
        gradient_b = sum(errors) * (2 / len(errors))
        return {"weight": float(gradient_w), "bias": float(gradient_b)}
