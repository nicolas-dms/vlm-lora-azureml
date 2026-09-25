"""Render a 30-minute, 4-sensor window as a single ~384x384 PNG.

Why 384 px: SmolVLM tiles large images and every tile costs visual tokens, which on a
T4 translates directly into memory. The image budget - not the LoRA weights - is what
makes or breaks this training run.

Why fixed y-limits: the labelling rules are absolute (e.g. "temperature >= 40 degC").
If every chart were autoscaled, the same visual height would mean a different value
from one image to the next and the task would be unlearnable. Fixed axes make the
level itself a visual feature.

Why the last value is printed in each subplot title, and why tick labels are removed:
at 384 px a fontsize-4 tick label is unreadable, so it costs rasterisation time and
teaches the model nothing - text rendering measured at ~4x the cost of everything else
in the figure. Gridlines stay (they are the visual scale reference, and the y-limits
are identical on every image), the numeric value moves into the title. The task becomes
"read the chart AND the number, then reason" - still genuinely multimodal, and learnable
in 2 epochs on a 256M model. The goal is a chain that runs, not a state-of-the-art detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: AML compute has no display
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_PX = 384
DPI = 96

# Global limits computed on the full snapshot, padded so nothing ever clips.
AXES = [
    ("temperature_c", "Temp", "degC", (15.0, 55.0), "tab:red"),
    ("humidity_pct", "Humidity", "%", (30.0, 80.0), "tab:blue"),
    ("pressure_bar", "Pressure", "bar", (3.5, 7.0), "tab:green"),
    ("vibration_mm_s", "Vibration", "mm/s", (0.0, 8.0), "tab:purple"),
]


# Building a matplotlib figure is far more expensive than drawing into one, and we
# render hundreds of identically-shaped charts. The figure is created once and then
# only its y-data and titles are updated.
_CANVAS: dict | None = None


def _get_canvas(n_points: int) -> dict:
    global _CANVAS
    if _CANVAS is not None and _CANVAS["n_points"] == n_points:
        return _CANVAS

    if _CANVAS is not None:
        plt.close(_CANVAS["fig"])

    size_in = IMAGE_PX / DPI
    fig, axes = plt.subplots(2, 2, figsize=(size_in, size_in), dpi=DPI)

    lines = []
    flat_axes = list(axes.ravel())
    zeros = [0.0] * n_points
    for ax, (_column, _label, _unit, ylim, color) in zip(flat_axes, AXES):
        (line,) = ax.plot(range(n_points), zeros, color=color, linewidth=1.0)
        lines.append(line)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, n_points - 1)
        # Tick positions kept so the grid has something to draw on, labels dropped.
        ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
        ax.set_xticks(np.linspace(0, n_points - 1, 4))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(length=0)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_title(" ", fontsize=6, pad=2)

    suptitle = fig.suptitle(" ", fontsize=7, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.4)

    _CANVAS = {"fig": fig, "axes": flat_axes, "lines": lines, "suptitle": suptitle, "n_points": n_points}
    return _CANVAS


def render_window(window: Any, device_id: str, out_path: str | Path) -> Path:
    """Render one window (a pandas DataFrame) to a PNG. Returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = _get_canvas(len(window))

    for ax, line, (column, label, unit, _ylim, _color) in zip(canvas["axes"], canvas["lines"], AXES):
        series = window[column].to_numpy(dtype=float)
        line.set_ydata(series)
        ax.set_title(f"{label} {series[-1]:.1f}{unit}", fontsize=6, pad=2)

    canvas["suptitle"].set_text(f"{device_id} - last 30 min")
    canvas["fig"].savefig(out_path, format="png", dpi=DPI)

    return out_path
