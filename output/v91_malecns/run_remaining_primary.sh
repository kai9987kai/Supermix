#!/usr/bin/env bash
# The part of the v91 post-hoc battery that never ran. run_analysis.sh died
# on 2026-09-19 after scoring arm A (analysis.log ends at 19:53 with A's
# receipt and no "evaluated A rc" line). This script runs only the
# pre-registered PRIMARY readout -- arms C and B on the 630 novel problems and
# the McNemar pairs -- and deliberately does not re-run run_analysis.sh,
# which would overwrite hashes_at_chain_end.txt (the D0 hash record; a copy
# is kept as hashes_at_chain_end.original_20260919.txt) with post-v93
# source hashes. The gate-off evals and the dev-row bootstrap (E1) are left
# for run_analysis.sh after v93 finishes; they are existence-guarded there.
#
# Serial with everything else on this box: never run beside training.
set -u
cd "$(dirname "$0")/../.."
OUT=output/v91_malecns
LOG=$OUT/analysis.log
step() { echo "=== $(date -Is) $*" >> "$LOG"; }
step "remaining primary battery start (run_remaining_primary.sh)"

A=output/v91_cns_connectome/v91_cns_connectome.pt
B=output/v91_cns_rewired/v91_cns_rewired.pt
C=output/v91_control/v91_control.pt

evaluate() {
  local name="$1" ckpt="$2"
  [ -f "$OUT/${name}_replies.jsonl" ] && { step "skip $name: replies exist"; return 0; }
  [ -f "$ckpt" ] || { step "skip $name: no $ckpt"; return 0; }
  python source/eval_problem_solving.py --checkpoint "$ckpt" --task_set v89 --novel 630 --seen 0 --seed 65 \
    --max_new_tokens 96 --dump_replies "$OUT/${name}_replies.jsonl" \
    --output "$OUT/${name}_replies.run.json" >> "$LOG" 2>&1
  step "evaluated $name rc $?"
}
evaluate C "$C"
evaluate B "$B"

compare() {  # compare BASE CANDIDATE
  local base="$1" cand="$2"
  [ -f "$OUT/${base}_replies.jsonl" ] && [ -f "$OUT/${cand}_replies.jsonl" ] || return 0
  [ -f "$OUT/cmp_${cand}_vs_${base}.json" ] && return 0
  python source/compare_problem_transcripts.py --baseline "$OUT/${base}_replies.jsonl" \
    --candidate "$OUT/${cand}_replies.jsonl" --output "$OUT/cmp_${cand}_vs_${base}.json" >> "$LOG" 2>&1
  step "compared $cand vs $base rc $?"
}
compare C A
compare B A
compare C B
compare v89 A
compare v89 B
compare v89 C
step "remaining primary battery done"
