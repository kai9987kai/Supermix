#!/usr/bin/env bash
# v93: two fly hemispheres bonded to the trunk, with neurogenesis. One
# serial chain (this box runs one heavy job at a time):
#
#   1. the part of v91's pre-registered PRIMARY readout that never ran
#      (arms C and B on the 630 novel problems, McNemar pairs; ~1 h)
#   2. v93 training, 8,000 steps warm-started from v89, supervised
#      (train_supervised.py restarts a crashed leg from its partial checkpoint)
#   3. v93's own readout: the v89-comparable benchmark (--task_set v89, the
#      fingerprint 3b99a446... pairs with v89 and v91 C), the new families
#      (--task_set new), McNemar against v89 and against v91 C
#   4. the deferred v91 E1 battery (gate-off evals, dev-row bootstraps); it
#      rewrites hashes_at_chain_end.txt, whose original is kept as
#      hashes_at_chain_end.original_20260919.txt
#
# Design, contracts and the pre-registered readout:
# docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md. Log: output/v93_neurogenesis/chain.log
set -u
cd "$(dirname "$0")/../.."
source output/v93_neurogenesis/v93_args.sh
OUT=output/v93_neurogenesis
LOG=$OUT/chain.log
step() { echo "=== $(date -Is) $*" >> "$LOG"; }
step "chain start"
sha256sum source/mimomix_core.py source/neurogenesis.py source/train_mimomix_generalisation.py \
  source/train_mimomix_talk.py source/mimomix_text.py source/malecns_connectome.py \
  source/eval_problem_solving.py source/build_v93_corpus.py source/build_v93_train_mix.py \
  datasets/v93_malecns/malecns_hemispheres_384.npz datasets/v93/v93_train_mix.jsonl \
  output/v89_corpus/v89_corpus.pt > "$OUT/hashes_at_start.txt"

# 1. v91 primary readout, arms C and B (skips anything already there).
bash output/v91_malecns/run_remaining_primary.sh
step "v91 remaining primary done"

# 2. v93 training. Budget (preflight, docs/V93_...md): 8,000 steps x 6.6 s +
# 8 dev evals (5,441 rows, ~5 min) + 8 probes (205 problems, ~11 min) +
# growth events (~3 s) + end-of-run tiers and ablations ~= 18 h. Peak LR 3e-4 (twice v91's continuation rate: the new
# families must be learned, not maintained), grafted parameters x10 = 3e-3
# (v91's core rate). Selection on the 205-problem probe every 1,000 steps.
step "start v93_neurogenesis"
python source/train_supervised.py --max_restarts 4 -- \
  --run_name v93_neurogenesis --output_dir "$OUT" \
  --steps 8000 --lr 0.0003 --new_param_lr_mult 10 \
  --eval_every 1000 --accuracy_every 1000 --grow_every 1000 \
  --dev_fraction 0.003 --test_fraction 0.01 --ablation_rows 4000 \
  "${V93_MODEL[@]}" "${V93_DATA[@]}" > "$OUT/train.log" 2>&1
step "end v93_neurogenesis exit $?"

# 3. v93 readout.
CKPT=$OUT/v93_neurogenesis.pt
if [ -f "$CKPT" ]; then
  python source/eval_problem_solving.py --checkpoint "$CKPT" --task_set v89 --novel 630 --seen 0 \
    --seed 65 --max_new_tokens 96 --dump_replies "$OUT/v93_v89tasks_replies.jsonl" \
    --output "$OUT/v93_v89tasks_replies.run.json" >> "$LOG" 2>&1
  step "evaluated v93 on the v89 task set rc $?"
  python source/eval_problem_solving.py --checkpoint "$CKPT" --task_set new --novel 231 --seen 0 \
    --seed 65 --max_new_tokens 96 --dump_replies "$OUT/v93_newtasks_replies.jsonl" \
    --output "$OUT/v93_newtasks_replies.run.json" >> "$LOG" 2>&1
  step "evaluated v93 on the new task set rc $?"
  for base in v89:output/v87_measurements/v89_replies.jsonl C:output/v91_malecns/C_replies.jsonl A:output/v91_malecns/A_replies.jsonl; do
    name=${base%%:*}; file=${base#*:}
    [ -f "$file" ] || { step "skip compare vs $name: no $file"; continue; }
    python source/compare_problem_transcripts.py --baseline "$file" \
      --candidate "$OUT/v93_v89tasks_replies.jsonl" --output "$OUT/cmp_v93_vs_${name}.json" >> "$LOG" 2>&1
    step "compared v93 vs $name rc $?"
  done
else
  step "no final checkpoint at $CKPT; readout skipped"
fi

# 4. The deferred v91 battery (gate-off evals, dev-row bootstraps).
bash output/v91_malecns/run_analysis.sh
step "chain done"
