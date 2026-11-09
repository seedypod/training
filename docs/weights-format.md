# Weights file: `seedypod-cnn` v1

The file core reads from `models/pod-state.json` (`src/vision/cnn.ts` parses it). One JSON
object; `seedypod_training/model.py` writes it with `to_file`.

| Field | Meaning |
|---|---|
| `format`, `formatVersion` | `"seedypod-cnn"`, `1` |
| `analyzer` | The analyser it is for: `pod-state` |
| `version` | The analyser version core stores results under. A new model needs a new one. |
| `revision` | A name for this training run, e.g. `bootstrap-2026-11` |
| `classes` | For `pod-state`, exactly `empty, sprouted, growing, ready, unhealthy`, in that order |
| `input` | `width`, `height`, `channels`, `layout` (`HWC`), `scale` (255), per-channel `mean` and `std`. Input = (pixel / scale − mean) / std |
| `layers` | In order: `conv3x3` (`in`, `out`, `weight`, `bias`; stride 1, zero padding 1, then ReLU), `maxpool2` (2×2, stride 2, odd edge dropped), `gap` (mean over height and width), `dense` (`in`, `out`, `weight`, `bias`; the last one gives the logits). A `dense` must follow `gap` or another `dense`. |
| `tensors` | By name: `{shape, dtype: "float32", data}`, `data` being little-endian float32 in base64. A conv weight is `[out][ky][kx][in]` (PyTorch's `[out][in][ky][kx]` transposed); a dense weight is `[out][in]`, as PyTorch stores it. |
| `minBrightness` | Optional. Core skips a still whose mean R, G, B (0–1) is below it: the light was off, and training skipped such stills too. |
| `trainedAt`, `notes`, `training` | Optional. `notes` shows on core's Analysis screen; `training` holds the run's label counts and held-out scores. |

Core refuses a file that breaks any of this and lists the analyser as off, with the reason.

## Pod crops

Both sides crop a pod the same way (`preprocess.py` here, `src/vision/podState.ts` in core):

1. The pod's box in pixels becomes an integer window: floor of the top-left, ceil of the
   bottom-right, clamped to the frame, at least one pixel. The rule-based analysers read the
   same window.
2. Output cell `(oy, ox)` of the `height × width` input averages source rows
   `floor(y0 + oy·h/height)` up to `floor(y0 + (oy+1)·h/height)` (exclusive, at least one row),
   and likewise for columns, where `h` is the window's height.
3. Each channel is normalised with `scale`, `mean` and `std`.

Core decodes JPEGs with `jpeg-js` and training with Pillow (libjpeg-turbo); they can differ
by a level or two per pixel, which the averaging mostly absorbs.
