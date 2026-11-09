"""Weak labels: a pod's likely state from what core already knows, for when nobody has labelled it.

Two signals, both imperfect:

- **Planting metadata** (`pods.json`): whether the pod was planted by the still's date, and how many
  days before. `pods.json` is the pod's state at export time, so a pod replanted inside the window
  has its earlier history wrong; keep windows short or label by hand.
- **Rule-based leaf cover** (`analyses.jsonl`, green-threshold and exg-otsu v2, which read 2 % or
  less on an empty pod after core #132/#145).

The rules give `empty`, `sprouted`, `growing` and `ready` and abstain when the two signals disagree.
They never give `unhealthy`: nothing core records says so. A human label always wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .dataset import PodCrop

# Below this brightness (mean of the whole still, 0-1) the grow light was off: no label.
DARK = 0.12
# Rule cover at or below this is an empty cup: v2 analysers read 0.5 % or less on most empty pods (core #145).
EMPTY_COVER = 0.005
SPROUTED_COVER = 0.08
READY_COVER = 0.35
GERMINATION_DAYS = 5
HARVEST_DAYS = 45


@dataclass(frozen=True)
class WeakLabel:
    label: str | None
    reason: str


def _day(ts: str) -> date:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()


def cover_of(crop: PodCrop) -> float | None:
    """The larger of the rule-based analysers' readings, or None when neither scored this pod."""
    return max(crop.cover.values()) if crop.cover else None


def weak_label(crop: PodCrop, pod: dict | None, harvest_days: int = HARVEST_DAYS) -> WeakLabel:
    if crop.label:
        return WeakLabel(crop.label, "human")
    if crop.brightness < DARK:
        return WeakLabel(None, "dark still")
    cover = cover_of(crop)
    if cover is None:
        return WeakLabel(None, "no rule-based reading")
    planted = (pod or {}).get("plantedAt")
    days = (_day(crop.ts) - date.fromisoformat(planted)).days if planted else None
    if days is None or days < 0:
        return WeakLabel("empty", "not planted, no cover") if cover <= EMPTY_COVER else WeakLabel(None, "cover on an unplanted pod")
    if days < GERMINATION_DAYS:
        return WeakLabel("empty", "seeded, not yet due to sprout") if cover <= EMPTY_COVER else WeakLabel(None, "cover before sprouting is due")
    if cover <= EMPTY_COVER:
        return WeakLabel(None, "planted, no cover: failed or too small to see")
    if cover < SPROUTED_COVER:
        return WeakLabel("sprouted", "small cover")
    if days >= harvest_days and cover >= READY_COVER:
        return WeakLabel("ready", "past harvest days with high cover")
    return WeakLabel("growing", "cover between sprout and ready")
