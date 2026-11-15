#!/usr/bin/env bash
# Create (or update) the model farm's conda environment and install this repository into it.
#
#   ./setup_env.sh                  env named model_farm, base install
#   ./setup_env.sh --name NAME      another env name
#   ./setup_env.sh --extras a,b     also install optional groups from pyproject.toml
#
# Safe to re-run: an existing env is updated, not recreated. Works the same on a laptop and on a
# WATcloud node (where conda is not installed system-wide: install Miniconda in your home first).
# On WATcloud, run it inside a CPU job, not on a login node (login nodes allow 1 core and 2 GB).
set -euo pipefail

ENV_NAME=model_farm
EXTRAS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) ENV_NAME=${2:?--name needs a value}; shift 2 ;;
    --extras) EXTRAS=${2:?--extras needs a value}; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done

REPO_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
command -v conda >/dev/null || { echo "conda not found: install Miniconda first" >&2; exit 1; }

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "== updating conda env '$ENV_NAME'"
  conda env update -n "$ENV_NAME" -f "$REPO_DIR/environment.yml"
else
  echo "== creating conda env '$ENV_NAME' (Python 3.12)"
  conda env create -n "$ENV_NAME" -f "$REPO_DIR/environment.yml"
fi

run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

target="$REPO_DIR"
[[ -n "$EXTRAS" ]] && target="$REPO_DIR[$EXTRAS]"
echo "== installing $target (editable)"
run python -m pip install --upgrade pip
run python -m pip install -e "$target"

echo "== checking"
run python -c "import sys, farm; assert sys.version_info[:2] == (3, 12), sys.version; \
print('ok: python', sys.version.split()[0], 'farm', farm.__version__)"

echo
echo "Done. Use it with: conda activate $ENV_NAME"
