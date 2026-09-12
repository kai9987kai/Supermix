# v85 measurements

Every number in [`docs/V85_MEASURABLE_ARCHITECTURE.md`](../../docs/V85_MEASURABLE_ARCHITECTURE.md)
comes from one of these, and each script is here so the number can be reproduced
rather than trusted.

| file | question it answers |
|---|---|
| `v80_measurements.json` | What does the generation cap cost? Does the recursive thinking core do anything? Do greedy and speculative decoding agree? |
| `generator_era_comparison.json` | How much of v80's per-task score is the model, and how much is the generator version the benchmark ran against? |
| `flag_smoke.json` | Does each new v85 flag actually train, and what does it weigh? |
| `paired_timing.json` | What does each flag cost per step, measured against this machine's drift? |
| `natural_phrasing.json` | What does the prompt normaliser buy on naturally-typed questions? |
| `passk_headroom.json` | Is solver-verified rejection sampling worth a training run? |
| `v80_paired_baseline_n630.json` | The number every future run must beat: v80 on the current generators, n=630, cap 96, fingerprint recorded. |
| `v74_vs_v80_paired.json` | Is v80's regression against v74 real, or an artifact of the two never being scored on the same problems? |
| `batch_size_sweep.json` | Is a smaller batch free on this machine, and which batch size buys the most exposure per hour? |

## Reproducing

```bash
python output/v85_measurements/measure_v80.py --per_task 3
python output/v85_measurements/era_compare.py --n 12
python output/v85_measurements/flag_smoke.py --steps 40
python output/v85_measurements/paired_timing.py --rounds 6 --steps 10
python output/v85_measurements/natural_phrasing.py
python output/v85_measurements/passk_headroom.py --per_task 3 --k 6
python source/eval_problem_solving.py --checkpoint output/v80_omni/v80_omni.pt     --novel 630 --seed 65 --max_new_tokens 96     --output output/v85_measurements/v80_paired_baseline_n630.json
python output/v85_measurements/v74_vs_v80_paired.py --per_task 30 --cap 96
python output/v85_measurements/batch_ab.py
```

`reference_logits.py` and `perf_ab.py` are the two oracles used to check that
v85 did not change default behaviour: one compares forward-pass output against a
snapshot taken before any edit, the other A/Bs step cost between two source
trees in a single process.

## Two rules these scripts encode

**Timings are only valid back to back.** The same benchmark read 2.045, then
11.037, then 2.136 s/step on this machine with nothing changed. `paired_timing.py`
therefore interleaves each arm with a freshly built baseline and reports a ratio,
and refuses to quote a number when the round-to-round spread exceeds the effect.

**Sample sizes are small and the intervals say so.** n = 3 per task is ±37
points, n = 12 is ±25, n = 30 is ±17, n = 630 is ±4. Every receipt carries its
own non-claims block.
