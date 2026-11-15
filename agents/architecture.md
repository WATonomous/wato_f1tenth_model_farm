# Architecture

## Layout

```text
docs/data-standard.md     the Data Standard, version 1.0 (the contract)
docs/examples/            small examples of every format (images still to be added; see its README)
src/                      the `farm` package (installed editable by setup_env.sh)
  cli.py                  the `farm` command
  standard/               farm.standard: the Data Standard as a library (below)
  extractor/              farm.extractor: bag extraction (owned by a member, not built yet)
  autolabel/              farm.autolabel: SAM3 pre-labels for CVAT (planned)
data/                     DATA_DIR (git-ignored): recordings/, datasets/, predictions/
tests/                    CPU tests; helpers.py builds small valid sessions
environment.yml           conda env: Python 3.12 + pip only
pyproject.toml            the package and its dependencies (base: numpy, Pillow)
setup_env.sh              create or update the env, install the package
```

`src/` is the `farm` package itself (`package-dir` in `pyproject.toml`), so `src/standard/`
imports as `farm.standard`; add each new folder under `src/` to `packages` there. Code never
goes in `data/`. `DATA_DIR` (environment variable) moves the data to another disk.

## `farm.standard`

| Module | Standard section | What |
| --- | --- | --- |
| `ids` | 1, 2, 11 | Session, sample, run, tool and member IDs: regexes, make, split |
| `layout` | 3 | `data_dir()`, `SessionDir` (paths of a session's files, samples in timestamp order) |
| `jsonio` | 1 | JSON in the standard's format; atomic writes; UTC dates |
| `images` | 5, 9.1, 11.3 | Read/write RGB, depth, mask and probability PNGs; read PNG headers without decoding |
| `labels` | 8.3 | `Box`, clipping a pixel box (returns `truncated`), parse/format/check label files |
| `provenance` | 10 | Status records: `apply` (allowed transitions only), `check_record`, `generator_may_write` |
| `schema` | 6, 7, 8.4 | Field checks for `session.json`, meta records, `tracks.json`; `new_meta`, `new_tracks` |
| `predictions` | 11 | `run.json` and `detections.jsonl`: build, write, check |
| `validate` | 12 | `validate_session`, `validate_prediction_run`, `validate_path` → `Report` |
| `issues` | | `Issue` and `Report` (errors and warnings) |

Errors break a section 12 rule or a format the standard defines; data with errors is not used.
Warnings are things the standard asks for in prose that section 12 does not list (six decimals,
a final newline, unknown fields).

## Not here yet

- **WATcloud job tool**: GPU jobs with presets, a per-user budget and an idle watchdog.
- **Auto-labelling**: SAM3 pre-labels and a SigLIP2 check, reviewed in CVAT, written back with
  provenance.
- **Bag extraction and training**: owned by team members, built by them.

## Car stack interface

The car stack (`wato_f1tenth`) records bags with `./watod bag record` (its
`agents/workflows/deploy-to-car.md` and `ml-data.md`). Bags are copied unchanged to
`data/recordings/<session_id>/`. Models go back to the car through a perception node in that
repo; this package's Python 3.12 lets it share inference code.
