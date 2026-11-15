# AGENTS.md

Instructions for AI coding agents working in this repo (Claude Code, Codex, Cursor and others).
Plain Markdown only, so any tool can read it. If your tool does not load this file
automatically, read it before doing anything else. Details live in `agents/`.

## What this repo is

The WATonomous F1TENTH model farm: data and models for the car's perception. It is a Python 3.12
package (`farm`: the `src/` folder, with the data in `data/`) in the `model_farm` conda
environment. The car stack's `./watod tools` clones it into that repo's `tools/` folder, but it
is a separate repo with its own history. No Docker, no `watod`.

The [Data Standard](docs/data-standard.md) is the contract for every file this repo reads or
writes. `farm.standard` implements it; nothing writes data around it.

## Hard rules

Reasons are in `agents/rules.md`.

1. **The Data Standard is the contract.** Read `docs/data-standard.md` before writing code that
   touches data, and write data only through `farm.standard`. Changing the standard needs lead
   approval and a version bump; if the code and the standard disagree, say so, do not pick one.
2. **Never commit data, bags, models or checkpoints.** `data/` (`DATA_DIR`) is git-ignored.
3. **Never move, rename or edit dataset files by hand**, and never renumber sample IDs. A
   generator only overwrites annotations whose status is `generated` (section 10.3).
4. **Never overwrite outputs.** Write to a temporary file and rename (`farm.standard.jsonio`),
   and give new outputs new names. Record what produced them (tool ID, inputs) next to them.
5. **Ask before anything that loads a GPU**: model inference, auto-labelling, training,
   and any WATcloud job. CPU tests, `--help` and dry runs are fine.
6. **Python 3.12 only, dependencies only in `pyproject.toml` / `environment.yml`.** No ad-hoc
   `pip install` into the env; add the dependency and re-run `./setup_env.sh`.
7. **Run the CPU tests after every code change:** `python -B -m unittest discover -s tests -t .`
   Add tests for new behaviour. Report anything you could not test as untested.
8. **Keep to the task.** Bag extraction and training recipes are owned by team members: do not
   add code for them unless asked.
9. **Never credit an agent in git**: no `Co-Authored-By` trailers or "Generated with" lines in
   commits or PRs. Commits use the human's own git identity.

## Index

| File | Read it when |
| --- | --- |
| `docs/data-standard.md` | Any code that reads or writes data |
| `agents/architecture.md` | You need the layout, the package's modules or where something goes |
| `agents/rules.md` | You want the reason behind a hard rule |
| `agents/workflows/data-standard.md` | You use the library, validate data, or need to change the standard |

## Quick reference

```bash
./setup_env.sh && conda activate model_farm     # create or update the env
farm validate <session | run | DATA_DIR>        # check data against the standard
python -B -m unittest discover -s tests -t .    # CPU tests
```

## How to work here

- Read before you write: the standard's section for the data you touch, then the module.
- Make the smallest change that does the job, and match the surrounding style: Python 3.12,
  `from __future__ import annotations`, `pathlib`, f-strings, about 110-character lines, a module
  docstring with a usage example, functions that fail loudly instead of guessing.
- Update the docs (`README.md`, `agents/`) in the same change as the code.
- When you finish, list the files you changed and any command a human still needs to run.

## Maintaining these docs

Keep this file under 150 lines and each `agents/` file under about 150 lines. Add detail to a
file in `agents/` and link it from the index.
