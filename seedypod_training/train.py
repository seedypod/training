"""Train the pod-state model from a core dataset export and write the weights file core ships."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import Export, PodCrop, extract, iter_crops
from .evaluate import rule_class, scores
from .model import CLASSES, PodStateNet, logits_hwc, to_file, write_file
from .preprocess import ModelInput, Rect, crop_input
from .synth import jitter, leaf_fraction, paint
from .weak_labels import DARK, weak_label

# A class with fewer real examples than this is filled in with synthetic plants.
MIN_REAL = 30


def fetch(pod: str, days: int, out: Path) -> Path:
    """Download a dataset export with photos from a unit. Read-only: one GET."""
    host = pod if pod.startswith("http") else f"http://{pod}"
    to = datetime.now(UTC)
    q = urllib.parse.urlencode({"from": (to - timedelta(days=days)).isoformat(), "to": to.isoformat(), "images": "1"})
    out.mkdir(parents=True, exist_ok=True)
    tar = out / "export.tar"
    with urllib.request.urlopen(f"{host}/api/datasets/export?{q}", timeout=1800) as r, tar.open("wb") as f:  # noqa: S310
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return tar


def load(path: Path, work: Path, regions: Path | None) -> Export:
    root = extract(path, work / "export") if path.suffix == ".tar" else path
    return Export.load(root, json.loads(regions.read_text()) if regions else None)


def label_all(export: Export) -> tuple[list[tuple[PodCrop, str]], Counter]:
    pods = {p["slot"]: p for p in export.pods}
    out, reasons = [], Counter()
    for crop in iter_crops(export):
        w = weak_label(crop, pods.get(crop.slot))
        reasons[w.reason] += 1
        if w.label:
            out.append((crop, w.label))
    return out, reasons


def split(labelled: list[tuple[PodCrop, str]], holdout: float) -> tuple[list, list]:
    """Hold out the newest stills, so validation is later in time than anything trained on."""
    ids = sorted({c.capture_id for c, _ in labelled}, key=lambda i: min(c.ts for c, _ in labelled if c.capture_id == i))
    cut = set(ids[max(1, int(len(ids) * (1 - holdout))):])
    return [x for x in labelled if x[0].capture_id not in cut], [x for x in labelled if x[0].capture_id in cut]


def normalisation(train: list[tuple[PodCrop, str]], inp: ModelInput) -> ModelInput:
    raw = ModelInput(inp.width, inp.height, inp.channels, inp.scale, (0.0,) * inp.channels, (1.0,) * inp.channels)
    x = np.stack([c.input(raw) for c, _ in train[:2000]])
    mean = tuple(round(float(v), 4) for v in x.mean(axis=(0, 1, 2)))
    std = tuple(round(max(float(v), 1e-3), 4) for v in x.std(axis=(0, 1, 2)))
    return ModelInput(inp.width, inp.height, inp.channels, inp.scale, mean, std)


def samples(rows: list[tuple[PodCrop, str]], synthetic: list[str], rng: np.random.Generator, inp: ModelInput,
            augment: bool) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Real crops, plus one synthetic plant of each `synthetic` class on every real empty crop."""
    xs, ys, leaf = [], [], []
    for crop, label in rows:
        variants = [(crop.window, label)]
        if label == "empty":
            variants += [(paint(crop.window, s, rng), s) for s in synthetic]
        for window, y in variants:
            w = jitter(window, rng) if augment else window
            h, wd = w.shape[:2]
            xs.append(crop_input(w, Rect(0, 0, wd, h), inp))
            ys.append(CLASSES.index(y))
            leaf.append(leaf_fraction(w))
    return np.stack(xs), np.array(ys), leaf


def train(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    work = args.out
    work.mkdir(parents=True, exist_ok=True)
    source = fetch(args.pod, args.days, work) if args.pod else args.export
    export = load(source, work, args.regions)
    labelled, reasons = label_all(export)
    real = Counter(lbl for _, lbl in labelled)
    synthetic = [c for c in CLASSES if c != "empty" and real[c] < MIN_REAL] if args.synthetic else []
    print(f"crops labelled: {dict(real)}; abstained/skipped: {dict(reasons)}; synthetic classes: {synthetic}")
    if real["empty"] == 0 and synthetic:
        raise SystemExit("no empty crops to paint synthetic plants on")
    tr, va = split(labelled, args.holdout)
    inp = normalisation(tr, ModelInput())

    model = PodStateNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(args.epochs):
        x, y, _ = samples(tr, synthetic, rng, inp, augment=True)
        # Balance the classes: real empties outnumber everything else on a young grow.
        weights = 1.0 / np.bincount(y, minlength=len(CLASSES))[y]
        order = rng.choice(len(y), size=len(y), p=weights / weights.sum())
        model.train()
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = torch.from_numpy(np.ascontiguousarray(x[idx].transpose(0, 3, 1, 2)))
            yb = torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / len(order):.4f} ({time.time() - t0:.0f} s)")

    report = evaluate(model, va, synthetic, inp, args.seed)
    report["data"] = {
        "captures": len({c.capture_id for c, _ in labelled}),
        "window": [min(c.ts for c, _ in labelled), max(c.ts for c, _ in labelled)] if labelled else None,
        "labels": {"real": dict(real), "skipped": dict(reasons)},
        "synthetic": synthetic,
        "holdout": args.holdout,
    }
    notes = args.notes or ("Trained on weak labels (planting metadata and rule-based cover) with synthetic plants for "
                           f"{', '.join(synthetic)}; a baseline to beat, not a validated classifier." if synthetic else None)
    doc = to_file(model, inp, version=args.version, revision=args.revision, trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
                  notes=notes, training=report, min_brightness=DARK)
    size = write_file(doc, work / "pod-state.json")
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {work / 'pod-state.json'} ({size} bytes)")
    return report


def evaluate(model: PodStateNet, va: list[tuple[PodCrop, str]], synthetic: list[str], inp: ModelInput, seed: int) -> dict:
    """Held-out scores for the model and for the rule-based analysers, on real crops and on synthetic ones."""
    out: dict = {}
    if va:
        x = np.stack([c.input(inp) for c, _ in va])
        pred = [CLASSES[i] for i in logits_hwc(model, x).argmax(1)]
        truth = [lbl for _, lbl in va]
        rules = [rule_class(max(c.cover.values()) if c.cover else None) for c, _ in va]
        out["real"] = {"model": scores(truth, pred), "rules": scores(truth, rules),
                       "note": "Weak labels come from the rules' cover, so the rules agree with them by construction."}
    if synthetic and va:
        rng = np.random.default_rng(seed + 1)
        x, y, leaf = samples(va, synthetic, rng, inp, augment=False)
        truth = [CLASSES[i] for i in y]
        pred = [CLASSES[i] for i in logits_hwc(model, x).argmax(1)]
        out["synthetic"] = {"model": scores(truth, pred), "rules": scores(truth, [rule_class(v) for v in leaf]),
                            "note": "Rules here are leaf-colour fraction without blob assignment, a stand-in for green-threshold v2."}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train core's pod-state classifier from a dataset export.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--export", type=Path, help="An export tar or its unpacked folder")
    src.add_argument("--pod", help="Unit address (host[:port]); downloads an export with photos (read-only)")
    ap.add_argument("--days", type=int, default=30, help="With --pod: how many days to export")
    ap.add_argument("--regions", type=Path, help="Pod boxes (regions.json or GET /api/vision/regions) for exports without them")
    ap.add_argument("--out", type=Path, default=Path("data/run"), help="Writes pod-state.json and report.json here")
    ap.add_argument("--version", type=int, default=1, help="Analyser version core stores results under")
    ap.add_argument("--revision", default="local", help="Name of this model, e.g. bootstrap-2026-11")
    ap.add_argument("--notes", help="One line for the Analysis screen; defaults to a weak-label disclaimer when synthetic")
    ap.add_argument("--no-synthetic", dest="synthetic", action="store_false", help="Train only on real labelled crops")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args(argv))
