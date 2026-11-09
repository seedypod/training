"""Pod crops exactly as core's pod-state analyser reads them.

Mirrors `cropWindow` and `cropInput` in core's src/vision/podState.ts. Any
change here must land there too, and the parity fixture (scripts/make_parity_fixture.py)
must be regenerated: the model has to see in training what it sees on the unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    """A box in pixels (may be fractional), like core's `Rect`."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ModelInput:
    width: int = 64
    height: int = 64
    channels: int = 3
    scale: float = 255.0
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.25, 0.25, 0.25)

    def header(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "layout": "HWC",
            "scale": self.scale,
            "mean": list(self.mean),
            "std": list(self.std),
        }


def crop_window(rect: Rect, width: int, height: int) -> tuple[int, int, int, int]:
    """Integer window (x0, y0, x1, y1): floor of the top-left, ceil of the bottom-right, clamped, at least 1 px."""
    x0 = min(width - 1, max(0, math.floor(rect.x)))
    y0 = min(height - 1, max(0, math.floor(rect.y)))
    x1 = min(width, math.ceil(rect.x + rect.w))
    y1 = min(height, math.ceil(rect.y + rect.h))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def box_to_rect(box: dict, width: int, height: int) -> Rect:
    """A box in fractions of the frame (core's podRegions) to pixels, as core's toPixels does."""
    return Rect(box["x"] * width, box["y"] * height, box["w"] * width, box["h"] * height)


def _span(o: int, n: int, length: int, start: int, limit: int) -> tuple[int, int]:
    a = min(limit - 1, math.floor(start + (o * length) / n))
    return a, min(limit, max(a + 1, math.floor(start + ((o + 1) * length) / n)))


def crop_input(rgb: np.ndarray, rect: Rect, inp: ModelInput) -> np.ndarray:
    """One pod's crop as the model's normalised HWC float32 input.

    `rgb` is H x W x C (C >= inp.channels) uint8. Output cell (oy, ox) averages the
    source pixels of its share of the window, the same arithmetic as core.
    """
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = crop_window(rect, width, height)
    sw, sh = x1 - x0, y1 - y0
    ys = [_span(o, inp.height, sh, y0, y1) for o in range(inp.height)]
    xs = [_span(o, inp.width, sw, x0, x1) for o in range(inp.width)]
    src = rgb[..., : inp.channels].astype(np.float64)
    # Summed-area table: each cell's sum in O(1), so a big crop costs no more than a small one.
    sat = np.zeros((height + 1, width + 1, inp.channels), dtype=np.float64)
    sat[1:, 1:] = src.cumsum(0).cumsum(1)
    ya = np.array([a for a, _ in ys])
    yb = np.array([b for _, b in ys])
    xa = np.array([a for a, _ in xs])
    xb = np.array([b for _, b in xs])
    total = sat[yb][:, xb] - sat[ya][:, xb] - sat[yb][:, xa] + sat[ya][:, xa]
    count = ((yb - ya)[:, None] * (xb - xa)[None, :])[..., None]
    mean = np.array(inp.mean, dtype=np.float64)
    std = np.array(inp.std, dtype=np.float64)
    return ((total / count / inp.scale - mean) / std).astype(np.float32)
