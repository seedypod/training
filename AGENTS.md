# Training guidance

This repository trains seedypod's on-device models off the unit. Core
([seedypod/core](https://github.com/seedypod/core)) runs them in plain TypeScript
(ADR 0019, 2026-11-09 amendment); this repository fits them and writes the weights file.

## Layout

- `seedypod_training/`: the package. `preprocess.py` (pod crops), `model.py` (network and
  weights file), `dataset.py` (core's dataset export), `weak_labels.py`, `synth.py`,
  `evaluate.py`, `train.py`.
- `scripts/train.py`: the training CLI. `scripts/make_parity_fixture.py`: the fixture core's
  parity test reads.
- `tests/`: synthetic checks only. CI runs `ruff check .` and the unit tests.
- `data/`: ignored. Exports, photos, runs and weights files stay here, never in Git.

## Invariants

- **Crops match core exactly.** `preprocess.py` mirrors `cropWindow` and `cropInput` in core's
  `src/vision/podState.ts`, and `model.py`'s layers mirror `src/vision/cnn.ts`. Change both
  sides together, regenerate the parity fixture into core
  (`src/vision/tests/parity/pod-state-parity.json`) and run core's tests.
- **Class order is fixed:** `empty, sprouted, growing, ready, unhealthy`. Core stores the class
  index, so a new model can't reorder or rename them.
- **A new model gets a new `--version`.** Core keeps every analyser version's results side by
  side; reusing a version mixes two models' outputs.
- **Never commit or publish a unit's photos**, exports, or a run folder. Only aggregate numbers
  (counts, scores) belong in a weights file's `training` block or in a PR.
- **Say what labels a model saw.** A model trained on weak or synthetic labels says so in its
  `notes`, and its scores are reported as such: the rules agree with weak labels by
  construction, and synthetic scores say nothing about real seedlings.

## Working with the unit

The unit serves a LAN API (port 80 or 3000). Reading is fine: `GET /api/datasets/export`,
`GET /api/datasets/summary`, `GET /api/vision/regions`. It is a Pi Zero; fetch one export at a
time and don't poll. Ask before any write, including uploading or enabling a model.

Use conventional commit subjects.
