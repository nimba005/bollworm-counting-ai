#!/usr/bin/env bash
# One-time environment setup inside WSL (CPU-only machine).
# Creates the venv in the Linux filesystem (much faster than /mnt/c).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-$HOME/.venvs/bollworm}"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 "$VENV"
# CPU-only torch (~200 MB instead of ~2.5 GB of CUDA libs -- disk C: is nearly full)
uv pip install --python "$VENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VENV/bin/python" -r requirements.txt
uv pip install --python "$VENV/bin/python" -e .

mkdir -p data/raw outputs
echo
echo "Done. Activate with:  source $VENV/bin/activate"
