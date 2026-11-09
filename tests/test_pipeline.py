"""Synthetic checks of the pipeline: no unit, no photos of anyone's home."""

from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seedypod_training.dataset import Export, extract, iter_crops  # noqa: E402
from seedypod_training.evaluate import rule_class, scores  # noqa: E402
from seedypod_training.model import CLASSES, PodStateNet, from_file, logits_hwc, to_file  # noqa: E402
from seedypod_training.preprocess import ModelInput, Rect, crop_input, crop_window  # noqa: E402
from seedypod_training.regions import default_regions, parse_regions  # noqa: E402
from seedypod_training.synth import leaf_fraction, paint  # noqa: E402
from seedypod_training.weak_labels import weak_label  # noqa: E402


class Preprocess(unittest.TestCase):
    def test_window_matches_core(self):
        self.assertEqual(crop_window(Rect(10.4, 7.6, 70.3, 66.2), 160, 120), (10, 7, 81, 74))
        self.assertEqual(crop_window(Rect(150, 110, 30, 30), 160, 120), (150, 110, 160, 120))
        self.assertEqual(crop_window(Rect(200, 5, 0, 0), 160, 120), (159, 5, 160, 6))

    def test_cells_average_and_normalise(self):
        img = np.full((4, 4, 3), 51, dtype=np.uint8)
        out = crop_input(img, Rect(0, 0, 4, 4), ModelInput(2, 2, 3, 255, (0.2, 0, 0.1), (1, 0.5, 0.1)))
        np.testing.assert_allclose(out.reshape(-1, 3), [[0, 0.4, 1]] * 4, atol=1e-6)

    def test_matches_a_loop_over_cells(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
        inp = ModelInput(8, 8)
        rect = Rect(3.3, 2.9, 41.2, 30.4)
        got = crop_input(img, rect, inp)
        x0, y0, x1, y1 = crop_window(rect, 53, 37)

        def span(o, n, length, start, limit):
            a = min(limit - 1, math.floor(start + (o * length) / n))
            return a, min(limit, max(a + 1, math.floor(start + ((o + 1) * length) / n)))

        for oy in range(8):
            ya, yb = span(oy, 8, y1 - y0, y0, y1)
            for ox in range(8):
                xa, xb = span(ox, 8, x1 - x0, x0, x1)
                want = (img[ya:yb, xa:xb].reshape(-1, 3).mean(0) / 255 - 0.5) / 0.25
                np.testing.assert_allclose(got[oy, ox], want, atol=1e-5)


class WeightsFile(unittest.TestCase):
    def test_round_trip_and_layout(self):
        torch.manual_seed(0)
        model = PodStateNet()
        doc = to_file(model, ModelInput(), version=3, revision="t")
        self.assertEqual(doc["tensors"]["convs.0.weight"]["shape"], [16, 3, 3, 3])
        self.assertEqual([layer["type"] for layer in doc["layers"]],
                         ["conv3x3", "maxpool2", "conv3x3", "maxpool2", "conv3x3", "gap", "dense"])
        back, inp = from_file(json.loads(json.dumps(doc)))
        x = np.random.default_rng(0).standard_normal((2, 64, 64, 3)).astype(np.float32)
        np.testing.assert_allclose(logits_hwc(back, x), logits_hwc(model, x), atol=1e-6)
        self.assertEqual(inp, ModelInput())
        self.assertLess(len(json.dumps(doc)), 500_000)

    def test_hwc_weight_layout_matches_a_hand_convolution(self):
        torch.manual_seed(2)
        model = PodStateNet((4,), classes=2)
        doc = to_file(model, ModelInput(), version=1, revision="t", classes=("a", "b"))
        t = doc["tensors"]["convs.0.weight"]
        w = np.frombuffer(base64.b64decode(t["data"]), dtype="<f4").reshape(t["shape"])  # [out][ky][kx][in]
        x = np.random.default_rng(3).standard_normal((5, 5, 3)).astype(np.float32)
        conv = model.convs[0]
        ref = conv(torch.from_numpy(x.transpose(2, 0, 1))[None])[0].detach().numpy()
        k, cy, cx = 1, 2, 2
        hand = conv.bias[k].item() + sum(
            float(x[cy + dy, cx + dx] @ w[k, dy + 1, dx + 1]) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        self.assertAlmostEqual(hand, float(ref[k, cy, cx]), places=4)


def _export(root: Path, labels: list[dict] | None = None) -> Path:
    """A tiny export: one bright and one dark still from the upper camera."""
    folder = root / "seedypod-test-20261101-20261102"
    (folder / "images").mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"format": {"name": "seedypod-dataset", "version": 1}}))
    caps = [{"id": "upper-a", "camera": "upper", "ts": "2026-11-01T12:00:00.000Z", "image": "images/upper-a.jpg"},
            {"id": "upper-b", "camera": "upper", "ts": "2026-11-01T23:00:00.000Z", "image": "images/upper-b.jpg"},
            {"id": "upper-c", "camera": "upper", "ts": "2026-11-02T01:00:00.000Z", "image": "images/upper-c.jpg"},
            {"id": "upper-d", "camera": "upper", "ts": "2026-11-02T02:00:00.000Z", "image": None}]
    (folder / "captures.jsonl").write_text("\n".join(json.dumps(c) for c in caps) + "\n")
    Image.new("RGB", (320, 240), (200, 200, 190)).save(folder / "images/upper-a.jpg")
    Image.new("RGB", (320, 240), (5, 5, 5)).save(folder / "images/upper-b.jpg")
    Image.new("RGB", (320, 240), (190, 195, 185)).save(folder / "images/upper-c.jpg")
    rows = [{"captureId": "upper-a", "slot": s, "analyzer": a, "version": 2, "metric": "green_fraction",
             "value": 0.5 if s == 2 else 0.0} for s in range(1, 16) for a in ("green-threshold", "exg-otsu")]
    rows += [{**r, "captureId": "upper-c"} for r in rows]
    rows.append({"captureId": "upper-a", "slot": 1, "analyzer": "exg-otsu", "version": 1, "metric": "green_fraction", "value": 0.9})
    (folder / "analyses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (folder / "pods.json").write_text(json.dumps([{"slot": 3, "plantedAt": "2026-10-01"}]))
    (folder / "grow_cycles.json").write_text("[]")
    if labels is not None:
        (folder / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")
        (folder / "regions.json").write_text(json.dumps({"upper": {"1": {"x": 0, "y": 0, "w": 0.5, "h": 0.5}}}))
    return folder


class Dataset(unittest.TestCase):
    def test_reads_an_export_and_weak_labels_it(self):
        with tempfile.TemporaryDirectory() as d:
            ex = Export.load(_export(Path(d)))
            self.assertEqual(ex.cover[("upper-a", 1)], {"green-threshold": 0.0, "exg-otsu": 0.0})
            crops = list(iter_crops(ex))
            self.assertEqual(len(crops), 45)
            labels = {(c.capture_id, c.slot): weak_label(c, {3: {"plantedAt": "2026-10-01"}}.get(c.slot)) for c in crops}
            self.assertEqual(labels[("upper-a", 1)].label, "empty")
            self.assertIsNone(labels[("upper-a", 2)].label)  # cover on a pod nobody planted
            self.assertIsNone(labels[("upper-a", 3)].label)  # planted a month ago, no cover
            self.assertEqual(labels[("upper-b", 1)].reason, "dark still")

    def test_human_labels_and_their_boxes_win(self):
        label = {"captureId": "upper-a", "ts": "2026-11-01T12:00:00.000Z", "camera": "upper", "slot": 4, "label": "unhealthy",
                 "region": {"x": 10, "y": 20, "w": 30, "h": 40}, "box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}
        boxed = {**label, "slot": 5, "label": "ready", "region": None}
        with tempfile.TemporaryDirectory() as d:
            ex = Export.load(_export(Path(d), [label, boxed]))
            crops = {(c.capture_id, c.slot): c for c in iter_crops(ex)}
            self.assertEqual(crops[("upper-a", 4)].window.shape, (40, 30, 3))
            self.assertEqual(crops[("upper-a", 5)].window.shape, (24, 32, 3))
            self.assertEqual(crops[("upper-a", 1)].window.shape, (120, 160, 3))  # regions.json
            self.assertEqual(weak_label(crops[("upper-a", 4)], None).label, "unhealthy")

    def test_unpacks_a_tar(self):
        with tempfile.TemporaryDirectory() as d:
            folder = _export(Path(d) / "src")
            tar = Path(d) / "export.tar"
            with tarfile.open(tar, "w") as tf:
                tf.add(folder, arcname=folder.name)
            self.assertEqual(extract(tar, Path(d) / "out").name, folder.name)

    def test_regions_from_either_shape(self):
        api = [{"camera": "lower", "regions": [{"slot": 16, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "moved": True}]}]
        self.assertEqual(parse_regions(api), {"lower": {16: {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}})
        self.assertEqual(default_regions("upper")[1], {"x": 0.0684, "y": 0.1381, "w": 0.0975, "h": 0.13})


class Synthetic(unittest.TestCase):
    def test_more_plant_for_later_classes(self):
        rng = np.random.default_rng(0)
        window = np.full((120, 120, 3), (170, 160, 150), dtype=np.uint8)
        frac = {c: np.mean([leaf_fraction(paint(window, c, rng)) for _ in range(20)]) for c in CLASSES}
        self.assertEqual(frac["empty"], 0)
        self.assertLess(frac["sprouted"], frac["growing"])
        self.assertLess(frac["growing"], frac["ready"])
        self.assertLess(frac["unhealthy"], frac["growing"])


class Evaluate(unittest.TestCase):
    def test_scores_and_rules(self):
        self.assertEqual([rule_class(v) for v in (None, 0.0, 0.02, 0.2, 0.6)], [None, "empty", "sprouted", "growing", "ready"])
        s = scores(["empty", "empty", "ready"], ["empty", None, "growing"])
        self.assertEqual((s["n"], s["abstained"], s["accuracy"]), (2, 1, 0.5))
        self.assertEqual(s["recall"], {"empty": 1.0, "ready": 0.0})


class EndToEnd(unittest.TestCase):
    def test_trains_on_a_synthetic_export(self):
        with tempfile.TemporaryDirectory() as d:
            folder = _export(Path(d))
            out = Path(d) / "run"
            subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--export", str(folder), "--out", str(out),
                            "--epochs", "1", "--holdout", "0.5", "--revision", "smoke"], check=True, capture_output=True)
            doc = json.loads((out / "pod-state.json").read_text())
            self.assertEqual((doc["revision"], doc["classes"]), ("smoke", list(CLASSES)))
            self.assertIn("baseline to beat", doc["notes"])
            model, inp = from_file(doc)
            self.assertEqual(logits_hwc(model, np.zeros((1, 64, 64, 3), np.float32)).shape, (1, 5))

    def test_parity_fixture_script(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "parity.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/make_parity_fixture.py"), str(out)], check=True, capture_output=True)
            doc = json.loads(out.read_text())
            self.assertEqual(len(doc["cases"]), 4)
            model, inp = from_file(doc["model"])
            img = np.frombuffer(base64.b64decode(doc["image"]["rgba"]), np.uint8).reshape(120, 160, 4)
            r = doc["cases"][0]["rect"]
            x = crop_input(img, Rect(r["x"], r["y"], r["w"], r["h"]), inp)
            np.testing.assert_allclose(logits_hwc(model, x[None])[0], doc["cases"][0]["logits"], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
