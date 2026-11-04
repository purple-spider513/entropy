#!/bin/bash
# Time svr_general_residual_unimol_predict.py (model: svr_general_residual_run_all_conformer_symmetry)
# on each hybrid_test_*k.csv, one fresh process per file, fresh output dir (no conformer cache reuse).
# Files run one after another on one GPU.
#
# Run from the repository root:  bash eval_500k/run_speed_test_all.sh
# Needs the unimol conda env, the downloaded all-features checkpoint, and the input files
# unimol_residual/hybrid_data/hybrid_test_<n>.csv (not included in the repository).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/unimol_residual"
OUT=${OUT:-$ROOT/eval_500k_rerun}   # new folder, so the included results are not overwritten
GPU=${GPU:-0}                       # CUDA device index or GPU UUID
mkdir -p "$OUT"
: > "$OUT/timing.txt"
for n in 10k 50k 100k 200k 500k; do
  rm -rf "$OUT/eval_$n"
  t0=$(date +%s.%N)
  conda run --no-capture-output -n unimol python svr_general_residual_unimol_predict.py \
    --model svr_general_residual_run_all_conformer_symmetry \
    --input hybrid_data/hybrid_test_$n.csv --gpu "$GPU" -o "$OUT/eval_$n" > "$OUT/predict_$n.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "$n rc=$rc seconds=$(echo "$t1 - $t0" | bc)" | tee -a "$OUT/timing.txt"
  [ $rc -ne 0 ] && { echo failed > "$OUT/FAILED"; exit 1; }
done
echo done > "$OUT/DONE"
