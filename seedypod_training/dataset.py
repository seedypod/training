"""Read a dataset export from core (`GET /api/datasets/export`, ADR 0019).

Layout (format `seedypod-dataset` v1, under one root folder in the tar):

- `manifest.json`, `captures.jsonl`, `analyses.jsonl`, `pods.json`, `grow_cycles.json`, `images/<captureId>.jpg`
- with pod labels (core #184): `labels.jsonl`, one row per labelled pod
  `{captureId, ts, camera, slot, label, labelledAt, image, width, height, region, box}`, where `region` is
  the integer pixel window in that still and `box` the same box in fractions of the frame; and
  `regions.json`, `{upper: {"1": {x, y, w, h}}, lower: {...}}` in fractions, for unlabelled captures.

Older exports have neither file: every pod is then unlabelled and cropped on core's measured default boxes.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import ModelInput, Rect, box_to_rect, crop_input, crop_window
from .regions import camera_slots, default_regions, parse_regions

RULE_ANALYZERS = ("green-threshold", "exg-otsu")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract(tar_path: Path, into: Path) -> Path:
    """Unpack an export tar (safely: no paths outside `into`) and return its root folder."""
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(into, filter="data")
    roots = [p for p in into.iterdir() if (p / "manifest.json").exists()]
    if len(roots) != 1:
        raise ValueError(f"expected one export folder in {tar_path}, found {len(roots)}")
    return roots[0]


@dataclass
class Export:
    root: Path
    manifest: dict
    captures: list[dict]
    pods: list[dict]
    grow_cycles: list[dict]
    labels: list[dict]
    regions: dict[str, dict[int, dict]]
    # (captureId, slot) -> {analyzer: green_fraction from its newest version}
    cover: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path, regions_override: dict | None = None) -> Export:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        name = manifest.get("format", {}).get("name")
        if name != "seedypod-dataset":
            raise ValueError(f"{root} is not a seedypod dataset export (format {name!r})")
        regions = {c: default_regions(c) for c in ("upper", "lower")}
        # Boxes a file leaves out keep the measured default, as in core.
        for doc in ([json.loads((root / "regions.json").read_text())] if (root / "regions.json").exists() else []) + (
                [regions_override] if regions_override else []):
            for camera, boxes in parse_regions(doc).items():
                regions.setdefault(camera, {}).update(boxes)
        cover: dict[tuple[str, int], dict[str, float]] = {}
        versions: dict[tuple[str, int, str], int] = {}
        for r in _jsonl(root / "analyses.jsonl"):
            if r.get("metric") != "green_fraction" or r.get("analyzer") not in RULE_ANALYZERS:
                continue
            key = (r["captureId"], r["slot"], r["analyzer"])
            if r["version"] >= versions.get(key, 0):
                versions[key] = r["version"]
                cover.setdefault((r["captureId"], r["slot"]), {})[r["analyzer"]] = r["value"]
        pods_path, cycles_path = root / "pods.json", root / "grow_cycles.json"
        return cls(
            root=root,
            manifest=manifest,
            captures=_jsonl(root / "captures.jsonl"),
            pods=json.loads(pods_path.read_text()) if pods_path.exists() else [],
            grow_cycles=json.loads(cycles_path.read_text()) if cycles_path.exists() else [],
            labels=_jsonl(root / "labels.jsonl"),
            regions=regions,
            cover=cover,
        )

    def image(self, capture: dict) -> np.ndarray | None:
        rel = capture.get("image")
        if not rel or not (self.root / rel).exists():
            return None
        with Image.open(self.root / rel) as im:
            return np.asarray(im.convert("RGB"))

    def label_rect(self, row: dict, width: int, height: int) -> Rect:
        """A label row's crop: its pixel `region`, else its `box`, else the camera's box for the slot."""
        if row.get("region"):
            r = row["region"]
            return Rect(r["x"], r["y"], r["w"], r["h"])
        box = row.get("box") or self.regions[row["camera"]][row["slot"]]
        return box_to_rect(box, width, height)


@dataclass
class PodCrop:
    capture_id: str
    ts: str
    camera: str
    slot: int
    rect: Rect
    # Mean brightness of the whole still, 0-1: a dark still (lights off) says nothing about the pods.
    brightness: float
    cover: dict[str, float]
    label: str | None
    window: np.ndarray  # the source pixels, H x W x 3 uint8, for augmentation

    def input(self, inp: ModelInput) -> np.ndarray:
        h, w = self.window.shape[:2]
        return crop_input(self.window, Rect(0, 0, w, h), inp)


def iter_crops(export: Export) -> Iterator[PodCrop]:
    """Every pod of every still with an image, oldest first, with its human label when it has one."""
    labels = {(r["captureId"], r["slot"]): r for r in export.labels}
    for cap in sorted(export.captures, key=lambda c: c["ts"]):
        rgb = export.image(cap)
        if rgb is None:
            continue
        height, width = rgb.shape[:2]
        brightness = float(rgb.mean() / 255)
        for slot in camera_slots(cap["camera"]):
            row = labels.get((cap["id"], slot))
            if row:
                rect = export.label_rect(row, width, height)
            else:
                rect = box_to_rect(export.regions[cap["camera"]][slot], width, height)
            x0, y0, x1, y1 = crop_window(rect, width, height)
            yield PodCrop(cap["id"], cap["ts"], cap["camera"], slot, rect, brightness,
                          export.cover.get((cap["id"], slot), {}), row["label"] if row else None,
                          rgb[y0:y1, x0:x1].copy())
