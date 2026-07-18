#!/bin/zsh
set -euo pipefail

SCRIPT="$HOME/Downloads/23_DARCA_V24_CLONED_STATE_INTERVENTION.py"
CORE="$HOME/Downloads/21_darca_v24_intrinsic_strict_core.py"
OUTDIR="$HOME/Desktop/DARCA_V24_CLONED_STATE_INTERVENTION"

mkdir -p "$OUTDIR"
python3 -u "$SCRIPT" \
  --core "$CORE" \
  --outdir "$OUTDIR" \
  --seeds 32 \
  --workers 8 \
  2>&1 | tee "$OUTDIR/run.log"
