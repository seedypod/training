"""Pod boxes, for exports that carry none.

Exports from core with labels (#184) include `regions.json` and per-label boxes;
use those. Older exports have neither, so this is a copy of core's measured
defaults (src/vision/regions.ts, REFERENCE_CUPS at 2026-11-09). Fractions of the frame.
"""

from __future__ import annotations

import math

REF_WIDTH, REF_HEIGHT = 1280, 960
SPAN = 2.6
MIN_SIDE = 0.01

REFERENCE_CUPS = {
    "upper": [
        (150, 195, 48), (420, 205, 75), (1010, 275, 75),
        (110, 385, 40), (283, 448, 42), (712, 530, 62),
        (228, 550, 42), (472, 655, 50), (855, 745, 50),
        (192, 642, 32), (378, 740, 38), (660, 830, 40),
        (305, 760, 35), (505, 850, 35), (765, 920, 35),
    ],
    "lower": [
        (215, 445, 75), (748, 360, 68), (1048, 330, 42),
        (490, 650, 50), (858, 555, 42), (1068, 490, 30),
        (365, 855, 45), (705, 760, 45), (950, 655, 38),
        (530, 910, 30), (785, 815, 30), (968, 722, 25),
        (430, 945, 30), (670, 925, 30), (865, 840, 28),
    ],
}


def _round4(n: float) -> float:
    # Math.round in JavaScript rounds .5 up; Python's round() rounds half to even.
    return math.floor(n * 10_000 + 0.5) / 10_000


def clamp_region(x: float, y: float, w: float, h: float) -> dict:
    w = _round4(min(1, max(MIN_SIDE, w)))
    h = _round4(min(1, max(MIN_SIDE, h)))
    return {"x": _round4(min(1 - w, max(0, x))), "y": _round4(min(1 - h, max(0, y))), "w": w, "h": h}


def camera_slots(camera: str) -> list[int]:
    first = 1 if camera == "upper" else 16
    return list(range(first, first + 15))


def default_regions(camera: str) -> dict[int, dict]:
    out = {}
    for slot, (x, y, r) in zip(camera_slots(camera), REFERENCE_CUPS[camera], strict=True):
        side = r * SPAN
        out[slot] = clamp_region((x - side / 2) / REF_WIDTH, (y - side / 2) / REF_HEIGHT, side / REF_WIDTH,
                                 side / REF_HEIGHT)
    return out


def parse_regions(doc: object) -> dict[str, dict[int, dict]]:
    """`regions.json` ({upper: {"1": box}}) or GET /api/vision/regions ([{camera, regions: [{slot, ...}]}])."""
    out: dict[str, dict[int, dict]] = {}
    if isinstance(doc, list):
        for cam in doc:
            out[cam["camera"]] = {r["slot"]: {k: r[k] for k in ("x", "y", "w", "h")} for r in cam["regions"]}
    elif isinstance(doc, dict):
        for camera, boxes in doc.items():
            out[camera] = {int(slot): box for slot, box in boxes.items()}
    return out
