#!/usr/bin/env bash
# v91 post-hoc battery (Amendment 1, "E1"). Waits for the training chain to
# finish, then runs every measurement serially -- this box cannot run two heavy
# jobs at once. Judge arms by their files, not chain.log's exit codes (which
# report `date`'s status; see docs/V91_MALECNS_CONNECTOME.md, D0).
set -u
cd "$(dirname "$0")/../.."
OUT=output/v91_malecns
LOG=$OUT/analysis.log

until grep -q "chain done" "$OUT/chain.log" 2>/dev/null; do sleep 300; done
echo "=== $(date -Is) analysis start" >> "$LOG"
sha256sum source/*.py datasets/v91_malecns/malecns_modules_512.npz > "$OUT/hashes_at_chain_end.txt"

A=output/v91_cns_connectome/v91_cns_connectome.pt
B=output/v91_cns_rewired/v91_cns_rewired.pt
C=output/v91_control/v91_control.pt
V89=output/v89_corpus/v89_corpus.pt
for f in "$A" "$B" "$C"; do [ -f "$f" ] || echo "MISSING $f" >> "$LOG"; done

step() { echo "=== $(date -Is) $*" >> "$LOG"; }

[ -f "$A" ] && python source/v91_analysis.py gate_off_checkpoint --checkpoint "$A" --output "$OUT/A_off.pt" >> "$LOG" 2>&1
[ -f "$B" ] && python source/v91_analysis.py gate_off_checkpoint --checkpoint "$B" --output "$OUT/B_off.pt" >> "$LOG" 2>&1
python source/v91_analysis.py weights --checkpoint "$A" "$B" --output "$OUT/weights.json" >> "$LOG" 2>&1
step "weights rc $?"

# Exact match, 630 novel problems, the settings of v89's dump (seed 65,
# max_new_tokens 96, fingerprint 3b99a446cd533be9bc5f8ae57d1310b4).
cp -n output/v87_measurements/v89_replies.jsonl "$OUT/v89_replies.jsonl"
evaluate() {
  local name="$1" ckpt="$2"
  [ -f "$OUT/${name}_replies.jsonl" ] && return 0
  [ -f "$ckpt" ] || { step "skip $name: no $ckpt"; return 0; }
  # --task_set v89 (added 2026-09-20): since v93 registered eleven more
  # tasks, the default task list is 41 names and --novel 630 no longer draws
  # 21 per task over the thirty that pair with v89's dump. Arm C was scored
  # that way once (C_replies.41tasks_misscoped.*) and read 0.691 with 34
  # abstentions on tasks v91 never saw; the v89 set reproduces fingerprint
  # 3b99a446cd533be9bc5f8ae57d1310b4.
  python source/eval_problem_solving.py --checkpoint "$ckpt" --task_set v89 --novel 630 --seen 0 --seed 65 \
    --max_new_tokens 96 --dump_replies "$OUT/${name}_replies.jsonl" \
    --output "$OUT/${name}_replies.run.json" >> "$LOG" 2>&1
  step "evaluated $name rc $?"
}
evaluate A "$A"
evaluate C "$C"
evaluate B "$B"
evaluate A_off "$OUT/A_off.pt"
evaluate B_off "$OUT/B_off.pt"

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
compare A_off A
compare B_off B
compare v89 A
compare v89 B
compare v89 C

# Per-row dev loss, 11,230 paired rows; on / gate-off / mean-ablation.
python source/v91_analysis.py dev --output "$OUT/dev_rows.json" --spec \
  "v89=$V89:on" "C=$C:on" "A=$A:on" "A_off=$A:off" "A_mean=$A:mean" \
  "B=$B:on" "B_off=$B:off" "B_mean=$B:mean" >> "$LOG" 2>&1
step "dev rows rc $?"
python source/v91_analysis.py compare --rows "$OUT/dev_rows.rows.npz" --output "$OUT/dev_paired.json" \
  --pair C,A B,A C,B A_off,A A_mean,A B_off,B B_mean,B v89,C v89,A v89,B >> "$LOG" 2>&1
step "dev paired rc $?"
echo "=== $(date -Is) analysis done" >> "$LOG"
