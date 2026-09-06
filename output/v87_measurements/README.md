# v87 measurements

Every number in [`docs/V87_WHERE_THE_POINTS_WENT.md`](../../docs/V87_WHERE_THE_POINTS_WENT.md)
comes from a file here. All of them were taken on `output/v86_corpus/v86_corpus.pt`,
the checkpoint that scores 0.779, on 5 September 2026.

Each script writes its own `.json` beside it and can be rerun on its own.

## The measurements

| file | question | answer |
|---|---|---|
| `division_dose_response.py` | Within one task, format and wording fixed, does accuracy fall as the operands grow? | Yes, and it is the **quotient**: 0.725 / 0.375 / 0.125 for 1 / 2 / 3 digits. The divisor is worth 0.725 against 0.750. |
| `significant_digits_sweep.py` | At a fixed three-digit width, does it matter how many digits had to be worked out? | Yes: 0.525 for `200`, 0.275 for `250`, 0.075 for `174`. This is what decided the rewrite was worth making. |
| `arity_scan.py` | How many written steps does each format ever use? | 16 of 22 tasks use exactly one step count across all 40,000 of their rows. |
| `subdivision_probe.py` | Can v86 do the sub-divisions a decomposed format would ask for? | Out of distribution, no — and it rewrites the prompt's numbers to fit its trained range. See the caveat below. |
| `coverage_audit.json` | Does any task ask about a value its corpus never teaches? | `percent` did: 12% and 15%, in a third of its benchmark problems and none of its 40,000 corpus rows. Fixed; nothing else. |
| `v86_step_audit.json` | Where does each wrong reply first go wrong? | Every failing task at a written step, except `average` and `percent` — the only two whose format performs an operation it does not write. |
| `build_v87_corpus.py` | — | Assembles `datasets/v87/v87_combined.jsonl`. Not a measurement. |

## Reading `subdivision_probe.json`

This one is easy to over-read, and I did before checking. It puts each
place-value sub-division of a hard `power` problem to the model on its own, and
scores 0.123 — from which it looks as though splitting the step cannot help.

It cannot support that. The sub-problems use two-digit divisors in the
`arithmetic` prompt shape, and `division` trains only on divisors 2-9, so every
one of them is out of distribution. What the replies show is the model
rewriting the dividend to something it has seen — `700 / 7` answered as
`70 / 7 = 10, total 10` — and occasionally landing the right answer above
working that does not support it (`420 / 7` answered as `320 / 7 = 60, total 60`).

That is an out-of-distribution collapse and not a property of the scratchpad.
**In distribution, 2 of 491 correct replies (0.4%) stand above a false step**
(`v86_step_audit.json`). The probe is kept because it is where the fabrication
behaviour was first seen, not because it measures the rewrite.

## Regenerating

`v86_replies.jsonl` is the full 630-reply transcript the audit reads:

```bash
python source/eval_problem_solving.py --checkpoint output/v86_corpus/v86_corpus.pt --novel 630 --seen 0 --seed 65 --dump_replies output/v87_measurements/v86_replies.jsonl
```

```bash
python source/step_audit.py --replies output/v87_measurements/v86_replies.jsonl
```

Add `--task average` to print every wrong reply for one task with its first
false step marked.
