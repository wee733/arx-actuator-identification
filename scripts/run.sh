#!/usr/bin/env bash
# Run the arx-sysid CLI from a checkout without installing it.
# ARX_SYSID_PYTHON selects an interpreter that already has Newton/Warp/MuJoCo/PyTorch.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ARX_SYSID_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -m arx_sysid.cli "$@"
