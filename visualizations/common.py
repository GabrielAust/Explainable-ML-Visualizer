"""Shared console visualization utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass
class VisualizationConfig:
    """Configuration shared across visualizations."""

    step_delay_s: float = 0.1
    display_every: int = 1
    min_render_interval_s: float = 0.05
    step_through: bool = False
    show_explanations: bool = False
    show_state_overlay: bool = True
    export_dir: str | None = None
    export_png: bool = True
    export_gif: bool = False


class VisualizationExporter:
    """Export console visualization steps as images or GIFs."""

    def __init__(
        self,
        output_dir: str,
        save_png: bool,
        save_gif: bool,
        frame_duration_s: float,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_png = save_png
        self.save_gif = save_gif
        self.frame_duration_ms = max(int(frame_duration_s * 1000), 50)
        self.frames: list[Image.Image] = []

    def render_frame(self, lines: Iterable[str], name: str) -> None:
        """Render a text-based frame and persist it."""
        text_lines = list(lines) or [""]
        font = ImageFont.load_default()
        line_height = font.getbbox("Ag")[3] + 4
        max_width = max(int(font.getlength(line)) for line in text_lines)
        width = max_width + 20
        height = line_height * len(text_lines) + 20
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        y_offset = 10
        for line in text_lines:
            draw.text((10, y_offset), line, fill="black", font=font)
            y_offset += line_height

        if self.save_png:
            image.save(self.output_dir / f"{name}.png")
        if self.save_gif:
            self.frames.append(image)

    def finalize(self, gif_name: str = "visualization.gif") -> None:
        """Write the collected frames into a GIF."""
        if not self.save_gif or not self.frames:
            return
        gif_path = self.output_dir / gif_name
        self.frames[0].save(
            gif_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.frame_duration_ms,
            loop=0,
        )


class ConsoleVisualizer:
    """Reusable console visualization renderer."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()
        self.exporter: VisualizationExporter | None = None
        self._last_render_time: float | None = None
        self._pending_update: tuple[
            int,
            Iterable[tuple[str, str]],
            Iterable[str],
            str | None,
        ] | None = None
        if self.config.export_dir:
            self.exporter = VisualizationExporter(
                self.config.export_dir,
                self.config.export_png,
                self.config.export_gif,
                self.config.step_delay_s or 0.1,
            )

    def announce(self, title: str, control_items: Iterable[tuple[str, str]]) -> None:
        """Display the visualization title and a control panel."""
        print(f"\n{title}")
        print("-" * len(title))
        print("Control Panel")
        print("-------------")
        for label, value in control_items:
            print(f"{label:<18} {value}")
        print()

        if self.exporter:
            lines = [
                title,
                "-" * len(title),
                "Control Panel",
                "-------------",
                *[f"{label:<18} {value}" for label, value in control_items],
            ]
            self.exporter.render_frame(lines, "frame_0000")

    def render_step(
        self,
        step: int,
        metrics: Iterable[tuple[str, str]],
        state_lines: Iterable[str],
        explanation: str | None = None,
    ) -> None:
        """Render a single step update."""
        if step % self.config.display_every != 0:
            return
        if self.config.step_through:
            self._render_step(step, metrics, state_lines, explanation)
            self._last_render_time = time.monotonic()
            return

        min_interval = max(0.0, self.config.min_render_interval_s)
        now = time.monotonic()
        if self._last_render_time is not None and now - self._last_render_time < min_interval:
            self._pending_update = (step, metrics, state_lines, explanation)
            return

        self._pending_update = None
        self._render_step(step, metrics, state_lines, explanation)
        self._last_render_time = now

    def _render_step(
        self,
        step: int,
        metrics: Iterable[tuple[str, str]],
        state_lines: Iterable[str],
        explanation: str | None = None,
        *,
        apply_delay: bool = True,
        allow_step_through: bool = True,
    ) -> None:
        """Render a step payload to the console and optional exporter."""

        metric_parts = [f"Step {step:>4}"] + [f"{name}: {value}" for name, value in metrics]
        print(" | ".join(metric_parts))

        export_lines = [" | ".join(metric_parts)]

        if self.config.show_state_overlay:
            print("  State Overlay:")
            export_lines.append("State Overlay:")
            for line in state_lines:
                print(f"    {line}")
                export_lines.append(f"  {line}")

        if explanation and self.config.show_explanations:
            print(f"  Explanation: {explanation}")
            export_lines.append(f"Explanation: {explanation}")

        if self.exporter:
            self.exporter.render_frame(export_lines, f"frame_{step:04d}")

        if self.config.step_through and allow_step_through:
            input("  Press Enter to advance to the next step...")
            return
        if apply_delay and self.config.step_delay_s > 0:
            time.sleep(self.config.step_delay_s)

    def summarize(self, summary_lines: Iterable[str]) -> None:
        """Display the final summary."""
        if self._pending_update is not None:
            step, metrics, state_lines, explanation = self._pending_update
            self._render_step(
                step,
                metrics,
                state_lines,
                explanation,
                apply_delay=False,
                allow_step_through=False,
            )
            self._pending_update = None
        print("\nSummary")
        print("-------")
        lines = list(summary_lines)
        for line in lines:
            print(line)

        if self.exporter:
            export_lines = ["Summary", "-------", *lines]
            self.exporter.render_frame(export_lines, "summary")
            self.exporter.finalize()
