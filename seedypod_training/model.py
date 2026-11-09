"""The pod-state network and its weights file.

The architecture is what core's plain-TypeScript forward pass (src/vision/cnn.ts)
can run: 3x3 convolutions (stride 1, padding 1) each followed by ReLU, 2x2 max
pools, a global average pool, then dense layers. The weights file
(`seedypod-cnn`, format version 1) is JSON: a header that describes the input,
the layers and the classes, and each tensor as little-endian float32 in base64.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .preprocess import ModelInput

FORMAT = "seedypod-cnn"
FORMAT_VERSION = 1
CLASSES = ("empty", "sprouted", "growing", "ready", "unhealthy")


class PodStateNet(nn.Module):
    """64x64x3 crop -> conv3x3 16 -> pool -> conv3x3 32 -> pool -> conv3x3 64 -> GAP -> dense 5.

    About 24k parameters, ~100 KB as float32; 753 ms per pod on a Pi Zero W in plain JavaScript.
    """

    def __init__(self, channels: Sequence[int] = (16, 32, 64), classes: int = len(CLASSES), in_channels: int = 3):
        super().__init__()
        self.widths = list(channels)
        self.convs = nn.ModuleList()
        c = in_channels
        for k in channels:
            self.convs.append(nn.Conv2d(c, k, 3, padding=1))
            c = k
        self.fc = nn.Linear(c, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is N x C x H x W (PyTorch's layout); returns logits."""
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(x))
            if i < last:
                x = nn.functional.max_pool2d(x, 2)
        return self.fc(x.mean(dim=(2, 3)))

    def layers(self) -> list[dict]:
        """The header's layer list, naming the tensors in the file."""
        out: list[dict] = []
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            out.append({"type": "conv3x3", "in": conv.in_channels, "out": conv.out_channels,
                        "weight": f"convs.{i}.weight", "bias": f"convs.{i}.bias"})
            if i < last:
                out.append({"type": "maxpool2"})
        out.append({"type": "gap"})
        out.append({"type": "dense", "in": self.fc.in_features, "out": self.fc.out_features,
                    "weight": "fc.weight", "bias": "fc.bias"})
        return out


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _from_b64(t: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(t["data"]), dtype="<f4").reshape(t["shape"])


def to_file(model: PodStateNet, inp: ModelInput, *, version: int, revision: str, classes: Sequence[str] = CLASSES,
            analyzer: str = "pod-state", trained_at: str | None = None, notes: str | None = None,
            training: dict | None = None, min_brightness: float | None = None) -> dict:
    """The weights file as a dict. `min_brightness`: core skips stills darker than this (mean of all channels, 0-1),
    which the model never saw in training. Conv weights go from PyTorch's [out][in][ky][kx] to the file's [out][ky][kx][in]."""
    tensors = {}
    for name, p in model.state_dict().items():
        a = p.detach().cpu().numpy().astype(np.float32)
        if name.startswith("convs.") and name.endswith(".weight"):
            a = a.transpose(0, 2, 3, 1)
        tensors[name] = {"shape": list(a.shape), "dtype": "float32", "data": _b64(a)}
    out = {
        "format": FORMAT,
        "formatVersion": FORMAT_VERSION,
        "analyzer": analyzer,
        "version": version,
        "revision": revision,
        "classes": list(classes),
        "input": inp.header(),
        "layers": model.layers(),
        "tensors": tensors,
    }
    if min_brightness is not None:
        out["minBrightness"] = min_brightness
    if trained_at:
        out["trainedAt"] = trained_at
    if notes:
        out["notes"] = notes
    if training:
        out["training"] = training
    return out


def from_file(doc: dict) -> tuple[PodStateNet, ModelInput]:
    """Rebuild the network from a weights file (the reverse of `to_file`)."""
    if doc.get("format") != FORMAT or doc.get("formatVersion") != FORMAT_VERSION:
        raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file")
    widths = [layer["out"] for layer in doc["layers"] if layer["type"] == "conv3x3"]
    i = doc["input"]
    model = PodStateNet(widths, classes=len(doc["classes"]), in_channels=i["channels"])
    state = {}
    for name, t in doc["tensors"].items():
        a = _from_b64(t)
        if name.startswith("convs.") and name.endswith(".weight"):
            a = a.transpose(0, 3, 1, 2)
        state[name] = torch.from_numpy(a.copy())
    model.load_state_dict(state)
    inp = ModelInput(i["width"], i["height"], i["channels"], i["scale"], tuple(i["mean"]), tuple(i["std"]))
    return model, inp


def write_file(doc: dict, path: Path) -> int:
    """Write compact JSON; returns its size in bytes."""
    text = json.dumps(doc, separators=(",", ":"))
    path.write_text(text + "\n")
    return len(text) + 1


def logits_hwc(model: PodStateNet, crops: np.ndarray) -> np.ndarray:
    """Logits for N x H x W x C crops (the layout core uses)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(crops.transpose(0, 3, 1, 2)))
        return model(x).numpy()
