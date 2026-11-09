# Training

`seedypod/training`: off-device training for the models seedypod runs on the unit.
The first is **pod-state**, a small CNN that names each pod `empty`, `sprouted`,
`growing`, `ready` or `unhealthy` from its crop of a camera still.

The unit, a Pi Zero W, has no runtime for ONNX, so core runs the model in plain
TypeScript from a weights file ([ADR 0019](https://github.com/seedypod/core/blob/dev/docs/adr/0019-pluggable-capture-analysis.md),
2026-11-09 amendment). This repository fits the model in PyTorch and writes that file.
Core's [Capture analysis](https://github.com/seedypod/core/blob/dev/docs/ANALYSIS.md#the-learned-pod-state-analyser)
guide covers the on-device side.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --locked
```

`uv.lock` pins Python 3.11, PyTorch (CPU wheels on Linux), NumPy and Pillow.

## Train

```sh
uv run --locked python scripts/train.py --pod seedypod.local --days 30 --version 2 --revision my-model-2026-12
```

`--pod` downloads a dataset export with photos from the unit (`GET /api/datasets/export`,
read-only; it is a Pi Zero, so it takes a minute). `--export` takes an export you already
downloaded (the tar or its unpacked folder). Everything lands in `--out` (default
`data/run/`, ignored by Git):

- `pod-state.json`: the weights file core reads.
- `report.json`: label counts, held-out scores for the model and for the rule-based
  analysers, and the confusion matrices.

## Put it on a unit

Upload it as a model revision (core #186). It needs the unit's result ingest token
(**Settings → Off-device storage and analysis**) and the file's SHA-256:

```sh
curl -X POST "http://seedypod.local/api/models?sha256=$(shasum -a 256 data/run/pod-state.json | cut -d' ' -f1)&enable=1" \
  -H "Authorization: Bearer $SEEDYPOD_INGEST_TOKEN" --data-binary @data/run/pod-state.json
```

An agent can do the same with core's MCP `upload_model`. An enabled revision runs on every
capture as `pod-state-<revision>`, beside the shipped model and the rule-based analysers on
the Analysis screen. `set_model` (or `PATCH /api/models/{id}`) enables, disables or makes a
revision primary; `reset` goes back to the shipped model and `rollback` undoes the newest
change. Uploading writes to the unit, so ask its owner first.

To ship a model with every unit instead, replace `models/pod-state.json` in a core PR, with
a new `--version` so its results sit beside the previous model's.

## Labels

Labels come from the export, in this order:

1. **Human labels** (core #184): `labels.jsonl`, one row per pod a grower tagged, with the
   pixel crop (`region`) and box (`box`) core used.
2. **Weak labels** (`seedypod_training/weak_labels.py`) for everything else: whether the pod
   was planted by the still's date and how long before (`pods.json`), and the rule-based
   analysers' leaf cover (`analyses.jsonl`). A pod nobody planted that reads 0.5 % cover or
   less is `empty`; a planted one with cover is `sprouted`, `growing` or, past its harvest
   days, `ready`. The rules abstain when the signals disagree, skip stills taken with the
   light off, and never say `unhealthy`.
3. **Synthetic plants** (`seedypod_training/synth.py`) for any class with fewer than 30 real
   examples: leaves drawn onto real empty-pod crops, growing in number and size from
   `sprouted` to `ready`, yellow-brown for `unhealthy`. `--no-synthetic` turns this off.

### The shipped bootstrap model

Core ships `bootstrap-2026-11`, trained on 2026-11-09 from the reference unit's first
2.5 days of photos, before anyone had labelled a pod and before any pod had sprouted. So
every real crop was weakly labelled `empty`, and the other four classes are synthetic.
It is a baseline to beat, not a validated classifier. Its held-out scores:

| Held-out set | Crops | Model accuracy | Rule-based accuracy |
|---|---|---|---|
| Real stills (weakly labelled, all `empty`) | 252 | 99.6 % | 100 % |
| Synthetic plants on those stills (all five classes) | 1,260 | 88.9 % (macro F1 0.89) | 48.6 % (macro F1 0.44) |

Per class on the synthetic set the model recalls `empty` 99.6 %, `sprouted` 73 %,
`growing` 94 %, `ready` 100 % and `unhealthy` 78 %. It was trained for 24 epochs
(`--epochs 24 --revision bootstrap-2026-11`). Of 1,530 pod crops from 102 stills, 1,230 from
83 lit stills were weakly labelled `empty` (953 unplanted, 277 seeded too recently to sprout)
and 300 skipped (285 from stills taken with the light off, 15 with cover where none was
expected). The newest 20 % of those stills were held out.

Real held-out crops are the newest 20 % of stills. The rule-based analysers score 100 % on
them because they made those labels. Synthetic scores only say the model learned the
painted plants; they say nothing about real seedlings. The photos themselves are not in this
repository and never will be.

## Parity with core

`scripts/make_parity_fixture.py` writes the fixture core's parity test reads: a seeded
(untrained) network, a synthetic still, and PyTorch's preprocessed inputs, logits and
probabilities for four pod boxes. Core checks its TypeScript forward pass against it
within 1e-4, so its CI needs no Python:

```sh
uv run --locked python scripts/make_parity_fixture.py ../core/src/vision/tests/parity/pod-state-parity.json
```

Regenerate it whenever `preprocess.py` or `model.py` changes, together with the matching
change in core's `src/vision/podState.ts` or `src/vision/cnn.ts`.

## Checks

```sh
uv run --locked ruff check .
uv run --locked python -m unittest discover -s tests -v
```

The tests use synthetic exports only, including a one-epoch end-to-end training run.
