#!/usr/bin/env bash
# Repairs the v91 primary readout without touching the running chain.
#
# run_remaining_primary.sh (chain step 1) scored arms C and B over the
# default task list, which since v93 is 41 tasks, not the thirty that pair
# with v89's dump (fingerprint 3b99a446...). The chain could not be stopped,
# so this watcher: (1) waits for that step to end and moves B's mis-scoped
# files aside (C's were moved by hand at 13:10); (2) then patches
# run_remaining_primary.sh for any future use; (3) waits for the chain to
# finish -- step 4, run_analysis.sh, re-scores C and B with --task_set v89
# because their replies files are gone -- and pairs v93 with the corrected
# C and B, which chain step 3 will have skipped for lack of the files.
# Log: output/v93_neurogenesis/watch_fix.log
set -u
cd "$(dirname "$0")/../.."
V91=output/v91_malecns
OUT=output/v93_neurogenesis
LOG=$OUT/watch_fix.log
step() { echo "=== $(date -Is) $*" >> "$LOG"; }
step "watcher start"

until grep -q "remaining primary battery done" "$V91/analysis.log" 2>/dev/null; do sleep 60; done
for n in C B; do
  [ -f "$V91/${n}_replies.jsonl" ] && mv "$V91/${n}_replies.jsonl" "$V91/${n}_replies.41tasks_misscoped.jsonl"
  [ -f "$V91/${n}_replies.run.json" ] && mv "$V91/${n}_replies.run.json" "$V91/${n}_replies.41tasks_misscoped.run.json"
done
# any pairing written by the mis-scoped step (compare refuses mismatched cohorts, so none is expected)
find "$V91" -maxdepth 1 -name 'cmp_*.json' -newer "$OUT/hashes_at_start.txt" -exec mv {} {}.misscoped \; 2>/dev/null
step "mis-scoped C/B files set aside"

if ! grep -q "task_set v89" "$V91/run_remaining_primary.sh"; then
  sed -i 's/eval_problem_solving.py --checkpoint "\$ckpt" --novel 630/eval_problem_solving.py --checkpoint "$ckpt" --task_set v89 --novel 630/' "$V91/run_remaining_primary.sh"
  step "run_remaining_primary.sh patched with --task_set v89"
fi

until grep -q "chain done" "$OUT/chain.log" 2>/dev/null; do sleep 300; done
step "chain done seen; pairing v93 with the corrected C and B"
for n in C B; do
  [ -f "$V91/${n}_replies.jsonl" ] && [ -f "$OUT/v93_v89tasks_replies.jsonl" ] || { step "skip v93 vs $n: missing file"; continue; }
  [ -f "$OUT/cmp_v93_vs_${n}.json" ] && continue
  python source/compare_problem_transcripts.py --baseline "$V91/${n}_replies.jsonl" \
    --candidate "$OUT/v93_v89tasks_replies.jsonl" --output "$OUT/cmp_v93_vs_${n}.json" >> "$LOG" 2>&1
  step "compared v93 vs $n rc $?"
done
step "watcher done"
