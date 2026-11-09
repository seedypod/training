"""Score the model against the labels and against core's rule-based analysers on the same crops."""

from __future__ import annotations

import numpy as np

from .model import CLASSES
from .weak_labels import EMPTY_COVER, READY_COVER, SPROUTED_COVER


def rule_class(cover: float | None) -> str | None:
    """What leaf cover alone says: the rule-based analysers' reading mapped onto the classes they can tell
    apart. They can't see `unhealthy`, and `ready` here ignores planting age."""
    if cover is None:
        return None
    if cover <= EMPTY_COVER:
        return "empty"
    if cover < SPROUTED_COVER:
        return "sprouted"
    return "ready" if cover >= READY_COVER else "growing"


def scores(truth: list[str], predicted: list[str | None]) -> dict:
    """Accuracy, macro F1 over the classes present in `truth`, per-class recall and a confusion matrix."""
    pairs = [(t, p) for t, p in zip(truth, predicted, strict=True) if p is not None]
    n = len(pairs)
    present = [c for c in CLASSES if c in truth]
    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for t, p in pairs:
        confusion[t][p] += 1
    recall, f1 = {}, []
    for c in present:
        tp = confusion[c][c]
        actual = sum(confusion[c].values())
        said = sum(confusion[t][c] for t in CLASSES)
        r = tp / actual if actual else 0.0
        p = tp / said if said else 0.0
        recall[c] = round(r, 4)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "n": n,
        "abstained": len(truth) - n,
        "accuracy": round(sum(t == p for t, p in pairs) / n, 4) if n else None,
        "macroF1": round(float(np.mean(f1)), 4) if f1 else None,
        "recall": recall,
        "confusion": {t: {p: v for p, v in row.items() if v} for t, row in confusion.items() if any(row.values())},
    }
