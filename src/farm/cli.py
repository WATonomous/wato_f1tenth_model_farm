"""The `farm` command.

    farm validate src/data/datasets/20261006_e7-hallway_r03      one session
    farm validate src/data                                       every session and prediction run
    farm validate --json --quiet src/data/predictions/<run_id>   machine-readable, errors only

Exit status: 0 when nothing has errors, 1 when something does, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .standard import validate as validate_mod
from .standard.layout import data_dir


def _validate(args: argparse.Namespace) -> int:
    paths = args.paths or [str(data_dir())]
    reports = []
    for path in paths:
        found = validate_mod.validate_path(path, check_mask_values=not args.no_mask_values)
        if not found:
            print(f"{path}: no session or prediction run found", file=sys.stderr)
            return 2
        reports += found
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        for r in reports:
            shown = r.errors if args.quiet else r.issues
            status = "ok" if r.ok else "FAILED"
            print(f"{status}: {r.target} ({len(r.errors)} errors, {len(r.warnings)} warnings)")
            for issue in shown:
                print(f"  {issue}")
    return 0 if all(r.ok for r in reports) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="farm", description="WATonomous F1TENTH model farm")
    parser.add_argument("--version", action="version", version=f"farm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check sessions and prediction runs against the Data Standard")
    v.add_argument("paths", nargs="*", help="a session, a prediction run, datasets/, predictions/ or DATA_DIR "
                                            "(default: DATA_DIR)")
    v.add_argument("--json", action="store_true", help="print the reports as JSON")
    v.add_argument("--quiet", action="store_true", help="show errors only, not warnings")
    v.add_argument("--no-mask-values", action="store_true", help="skip decoding masks to check their values")
    v.set_defaults(func=_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
