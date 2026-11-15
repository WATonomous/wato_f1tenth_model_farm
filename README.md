# wato_f1tenth_model_farm

The WATonomous F1TENTH model farm: perception data for the car, from recorded bags to labelled
datasets and trained models. Everything reads and writes data in the
[Data Standard](docs/data-standard.md); read it before writing extraction, labelling or training code.

It is a plain Python 3.12 package in a conda environment: no Docker and no `watod`. Python 3.12 is
the Python of ROS 2 Jazzy, so code from here can be installed into the car stack unchanged.

## Quick start

You need [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) (or any conda).

```bash
./setup_env.sh                 # creates the model_farm env (Python 3.12) and installs this repo into it
conda activate model_farm
farm validate docs/examples    # check data against the Data Standard
python -B -m unittest discover -s tests -t .   # CPU tests
```

`./setup_env.sh` is safe to re-run; it updates the env. `--name NAME` picks another env name.

## What is here

| Path | What |
| --- | --- |
| `docs/data-standard.md` | The Data Standard (version 1.0): every format, ID and rule. Changes need lead approval and a version bump |
| `docs/examples/` | Small examples of each format |
| `src/` | The `farm` package: `standard/` (the Data Standard as a library and validator), `cli.py` (the `farm` command), `extractor/` and `autolabel/` (not built yet) |
| `tests/` | CPU tests (`python -B -m unittest discover -s tests -t .`) |
| `data/` | `DATA_DIR`: recordings, datasets, predictions. Git-ignored |
| `AGENTS.md`, `agents/` | Instructions for AI coding agents (and a good map for people too) |

## Data

All data lives under `DATA_DIR`: `data/` by default, or the `DATA_DIR` environment variable to
put it on another disk.

```text
data/
├── recordings/<session_id>/      rosbag2 MCAP, unchanged
├── datasets/<session_id>/        rgb/ depth/ meta/ labels/ masks/ + session.json, tracks.json
└── predictions/<run_id>/         run.json + <session_id>/detections.jsonl, masks/, probs/
```

Never commit data, bags, models or checkpoints. Never move, rename or edit dataset files by hand;
use the library, and check a session before using it:

```bash
farm validate data/datasets/<session_id>     # one session
farm validate                                     # everything under DATA_DIR
```

## Using the library

```python
from farm.standard import ids, images, labels, provenance
from farm.standard.layout import SessionDir

s = SessionDir.for_id("20261006_e7-hallway_r03")
rgb = images.read_rgb(s.file("rgb", "20261006_e7-hallway_r03_000412"))   # (H, W, 3) uint8, RGB order
box, truncated = labels.Box.from_xyxy_px(-42, 250, 96, 318, width=640, height=480)
record = provenance.apply(provenance.empty_record(), "generated", "sam3-autolabel@0.1.0")
```

Each module's docstring has more examples. Writes are atomic (temporary file, then rename).

## Status

| Part | State |
| --- | --- |
| Data Standard library and validator | Built |
| WATcloud job tool (GPU jobs without hogging the cluster) | Planned |
| Auto-labelling (SAM3 pre-labels, review in CVAT) | Planned |
| Bag extraction, training | Owned by team members |
