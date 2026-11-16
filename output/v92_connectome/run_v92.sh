#!/usr/bin/env bash
# v92 temporal-connectome rig. Waits for the v91 post-hoc battery to finish
# (this box runs one heavy job at a time), then: cache block-2 states once,
# run the pre-registered 16-run plan (resumable: finished runs are skipped),
# and write the report. Log: output/v92_connectome/v92.log
set -u
cd "$(dirname "$0")/../.."
OUT=output/v92_connectome
LOG=$OUT/v92.log
mkdir -p "$OUT"

until grep -q "analysis done" output/v91_malecns/analysis.log 2>/dev/null; do sleep 300; done
echo "=== $(date -Is) v92 start" >> "$LOG"
sha256sum source/v92_temporal_connectome.py source/mimomix_core.py source/malecns_connectome.py \
  datasets/v92_connectome/nulls_512.npz datasets/v91_malecns/malecns_modules_512.npz > "$OUT/hashes_at_start.txt"

[ -f datasets/v92_connectome/nulls_512.npz ] || python source/v92_temporal_connectome.py nulls >> "$LOG" 2>&1
if [ ! -f "$OUT/cache/meta.json" ]; then
  python source/v92_temporal_connectome.py prepare >> "$LOG" 2>&1
  echo "=== $(date -Is) cache rc $?" >> "$LOG"
fi
python source/v92_temporal_connectome.py run >> "$LOG" 2>&1
echo "=== $(date -Is) runs rc $?" >> "$LOG"
python source/v92_temporal_connectome.py report > "$OUT/report.txt" 2>> "$LOG"
echo "=== $(date -Is) v92 done" >> "$LOG"
