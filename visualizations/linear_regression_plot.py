"""Matplotlib visualization for linear regression training."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
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

        self.visualizer.fig.subplots_adjust(bottom=0.3)

        lr_ax = self.visualizer.fig.add_axes([0.15, 0.2, 0.7, 0.03])
        iterations_ax = self.visualizer.fig.add_axes([0.15, 0.14, 0.7, 0.03])
        button_ax = self.visualizer.fig.add_axes([0.82, 0.03, 0.15, 0.06])

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

        self._learning_rate_slider.on_changed(self._on_learning_rate_change)
        self._iterations_slider.on_changed(self._on_iterations_change)
        self._pause_button.on_clicked(self._on_pause_toggle)

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
            self._last_state, _ = self.model.step(self.features, self.targets)
            self._current_step += 1

        if self._last_state is not None:
            self._render_state(self._last_state)

    def _render_current_state(self) -> None:
        if self._last_state is None:
            predictions = self.model.predict(self.features)
            errors = [pred - target for pred, target in zip(predictions, self.targets)]
            loss = sum(error**2 for error in errors) / len(errors)
            if not self.model.loss_history:
                self.model.loss_history = [loss]
            self._last_state = ModelState(
                parameters={
                    "weight": self.model.state.weight,
                    "bias": self.model.state.bias,
                },
                predictions=predictions,
                loss_history=self.model.loss_history.copy(),
            )
        self._render_state(self._last_state)

    def _render_state(self, state: ModelState) -> None:
        self.visualizer.update(
            state,
            metadata={"features": self.features, "targets": self.targets},
        )
