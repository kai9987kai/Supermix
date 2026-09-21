#!/usr/bin/env bash
# v93 preflight: the production command at 40 steps, two evals, two growth
# events, one probe, the end-of-run ablations and receipt. Exists so the
# first failure of any new path costs half an hour, not the first hours of
# the real run. Split fractions are shrunk so the tier scoring at the end is
# minutes, not an hour; the real run uses its own.
set -u
cd "$(dirname "$0")/../.."
source output/v93_neurogenesis/v93_args.sh
OUT=output/v93_preflight
mkdir -p "$OUT"
echo "=== $(date -Is) preflight start" >> "$OUT/preflight.log"
python source/train_supervised.py --max_restarts 0 -- \
  --run_name v93_preflight --output_dir "$OUT" \
  --steps 40 --lr 0.0003 --new_param_lr_mult 10 \
  --eval_every 20 --accuracy_every 40 --grow_every 20 \
  --dev_fraction 0.003 --test_fraction 0.002 --tier3_row_fraction 0.002 --ablation_rows 800 \
  "${V93_MODEL[@]}" "${V93_DATA[@]}" > "$OUT/train.log" 2>&1
echo "=== $(date -Is) preflight end exit $?" >> "$OUT/preflight.log"
