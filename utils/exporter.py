"""Utilities for exporting matplotlib visualizations as PNGs or GIFs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image


class VisualizationFrameExporter:
    """Capture matplotlib figures as PNGs or compile them into GIFs."""

    def __init__(
        self,
        output_dir: str = "exports",
        png_prefix: str = "frame",
        gif_name: str = "visualization.gif",
        frame_duration_s: float = 0.1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.png_prefix = png_prefix
        self.gif_name = gif_name
        self.frame_duration_ms = max(int(frame_duration_s * 1000), 50)
        self._png_index = 0
        self._frames: list[Image.Image] = []

    def save_png(self, figure, filename: Optional[str] = None) -> Path:
        """Save the current matplotlib figure as a PNG."""
        if filename is None:
            filename = f"{self.png_prefix}_{self._png_index:04d}.png"
            self._png_index += 1
        path = self.output_dir / filename
        figure.savefig(path, format="png", dpi=figure.dpi, bbox_inches="tight")
        return path

    def capture_frame(self, figure) -> None:
        """Capture the current matplotlib figure into the GIF frame list."""
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=figure.dpi, bbox_inches="tight")
        buffer.seek(0)
        image = Image.open(buffer).convert("RGBA")
        self._frames.append(image)

    def finalize_gif(self, gif_name: Optional[str] = None) -> Path | None:
        """Write captured frames to a GIF and return the file path."""
        if not self._frames:
            return None
        gif_path = self.output_dir / (gif_name or self.gif_name)
        self._frames[0].save(
            gif_path,
            save_all=True,
            append_images=self._frames[1:],
            duration=self.frame_duration_ms,
            loop=0,
        )
        return gif_path

    def reset_gif(self) -> None:
        """Clear captured frames for a fresh recording."""
        self._frames = []
