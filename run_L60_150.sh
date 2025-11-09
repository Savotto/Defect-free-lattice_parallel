#!/usr/bin/env bash
# Wrapper script for L=60..150 scaling sweep
# This runs the complete algorithm analysis for lattice sizes 60 to 150 (step size 10)

set -euo pipefail

PY=$(command -v python || echo python3)
# Parameters
SIZES="60,70,80,90,100,110,120,130,140,150"
OCC=0.75
LOSS=0.0
TRIALS=20
SEED=42
OUTDIR="N_scaling_075_L60-150"
STRATEGY=center
MAXCYCLES=6
TARGET_FILL=1.0

mkdir -p "$OUTDIR"

echo "Running comparison_N.py: sizes=$SIZES trials=$TRIALS seed=$SEED -> output=$OUTDIR"
"$PY" benchmarks/comparison_N.py \
  --initial-sizes "$SIZES" \
  --occupation $OCC \
  --loss $LOSS \
  --trials $TRIALS \
  --seed $SEED \
  --output $OUTDIR \
  --strategy $STRATEGY \
  --max-cycles $MAXCYCLES \
  --target-fill $TARGET_FILL
