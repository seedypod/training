"""Synthetic plants painted onto real empty-pod crops, for classes the unit has no photos of yet.

The bootstrap model (`bootstrap-2026-11`) was trained before any pod on the unit had sprouted, so
every real crop is `empty`. To give the other classes something to learn, leaves are drawn as
rotated ellipses radiating from the cup (the middle of the crop) over a real empty crop, so the
background, lighting and camera are real and only the plant is invented:

| class | leaves | leaf length (share of crop side) | colour |
|---|---|---|---|
| sprouted | 1-3 | 0.04-0.10 | green |
| growing | 4-10 | 0.12-0.30 | green |
| ready | 14-30 | 0.25-0.50 | green |
| unhealthy | 3-12 | 0.10-0.35 | yellow to brown, some with green |

This teaches the model "how much plant, and what colour", the same signal the rule-based analysers
read, with the cup's surroundings ignored. It is a baseline to beat with real labels, not a model of
what real seedlings look like.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SPEC = {
    "sprouted": (1, 3, 0.04, 0.10),
    "growing": (4, 10, 0.12, 0.30),
    "ready": (14, 30, 0.25, 0.50),
    "unhealthy": (3, 12, 0.10, 0.35),
}


def _leaf(cx: float, cy: float, angle: float, length: float, width: float, steps: int = 16) -> list[tuple[float, float]]:
    """A leaf: an ellipse from the stem point outward along `angle`."""
    pts = []
    ca, sa = math.cos(angle), math.sin(angle)
    for i in range(steps):
        t = 2 * math.pi * i / steps
        u = (math.cos(t) + 1) / 2 * length
        v = math.sin(t) * width / 2
        pts.append((cx + u * ca - v * sa, cy + u * sa + v * ca))
    return pts


def _colour(rng: np.random.Generator, unhealthy: bool, light: float) -> tuple[int, int, int, int]:
    if unhealthy and rng.random() < 0.8:
        h, s, v = rng.uniform(25, 58) / 360, rng.uniform(0.35, 0.85), rng.uniform(0.35, 0.9)
    else:
        h, s, v = rng.uniform(80, 145) / 360, rng.uniform(0.4, 0.9), rng.uniform(0.3, 0.85)
    r, g, b = colorsys.hsv_to_rgb(h, s, min(1.0, v * light))
    return int(r * 255), int(g * 255), int(b * 255), 255


def paint(window: np.ndarray, label: str, rng: np.random.Generator) -> np.ndarray:
    """A copy of `window` (H x W x 3 uint8) with a synthetic plant of class `label`; `empty` returns it as is."""
    if label == "empty":
        return window
    lo, hi, lmin, lmax = SPEC[label]
    h, w = window.shape[:2]
    side = min(h, w)
    # Leaves pick up the scene's light: a dim crop gets darker leaves.
    light = float(np.clip(window.mean() / 128, 0.35, 1.3))
    base = Image.fromarray(window).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = w / 2 + rng.uniform(-0.12, 0.12) * w
    cy = h / 2 + rng.uniform(-0.12, 0.12) * h
    for _ in range(int(rng.integers(lo, hi + 1))):
        length = rng.uniform(lmin, lmax) * side
        width = length * rng.uniform(0.3, 0.6)
        angle = rng.uniform(0, 2 * math.pi)
        # Larger plants spread from several stems around the cup.
        sx = cx + rng.normal(0, 0.04 * side) * (2 if label == "ready" else 1)
        sy = cy + rng.normal(0, 0.04 * side) * (2 if label == "ready" else 1)
        colour = _colour(rng, label == "unhealthy", light)
        draw.polygon(_leaf(sx, sy, angle, length, width), fill=colour)
        # A vein or a brown spot now and then, so a leaf isn't one flat colour.
        if rng.random() < 0.4:
            dark = tuple(int(c * 0.7) for c in colour[:3]) + (255,)
            draw.line([(sx, sy), (sx + math.cos(angle) * length * 0.8, sy + math.sin(angle) * length * 0.8)], fill=dark,
                      width=max(1, int(width * 0.08)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(0.5, side / 200)))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def jitter(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exposure, white balance and mirror changes, as the same pod sees across a day."""
    out = window.astype(np.float32) * rng.uniform(0.75, 1.25) * rng.uniform(0.92, 1.08, size=3)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.clip(out, 0, 255).astype(np.uint8)


def leaf_fraction(window: np.ndarray) -> float:
    """Share of pixels core's isLeafColour (src/vision/greenArea.ts) calls leaf. No blob assignment, so a
    rough stand-in for green-threshold v2 when scoring synthetic crops, which have no rule-based reading."""
    rgb = window.astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    span = mx - rgb.min(-1)
    ok = (g >= mx) & (mx >= 70) & (span >= 0.35 * mx)
    hue = np.where(span > 0, (b - r) / np.maximum(span, 1), 0)
    ok &= (hue >= -0.75) & (hue <= 5 / 6)
    green = (2 * g - r - b > 20) & (g > 40)
    return float((ok & green).mean())
