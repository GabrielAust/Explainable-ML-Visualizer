"""Shared console visualization utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class VisualizationConfig:
    """Configuration shared across visualizations."""

    step_delay_s: float = 0.1
    display_every: int = 1
    step_through: bool = False
    show_explanations: bool = False
    show_state_overlay: bool = True


class ConsoleVisualizer:
    """Reusable console visualization renderer."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()

    def announce(self, title: str, control_items: Iterable[tuple[str, str]]) -> None:
        """Display the visualization title and a control panel."""
        print(f"\n{title}")
        print("-" * len(title))
        print("Control Panel")
        print("-------------")
        for label, value in control_items:
            print(f"{label:<18} {value}")
        print()

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

        metric_parts = [f"Step {step:>4}"] + [f"{name}: {value}" for name, value in metrics]
        print(" | ".join(metric_parts))

        if self.config.show_state_overlay:
            print("  State Overlay:")
            for line in state_lines:
                print(f"    {line}")

        if explanation and self.config.show_explanations:
            print(f"  Explanation: {explanation}")

        if self.config.step_through:
            input("  Press Enter to advance to the next step...")
            return

        time.sleep(self.config.step_delay_s)

    def summarize(self, summary_lines: Iterable[str]) -> None:
        """Display the final summary."""
        print("\nSummary")
        print("-------")
        for line in summary_lines:
            print(line)
