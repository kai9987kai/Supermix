# v87 training preparation: prompt meaning and verified working

Status checked 2026-09-05. Full candidate and control data are prepared and their
integrity preflights pass. **Training for these control/combined bundles has not
started. No model was promoted, and the running v86/v80 server was not replaced.**

A separate `v87_corpus` supervised job was observed running with 21,500 steps,
`datasets/v87/v87_combined.jsonl`, and 100-problem accuracy probes. It was launched
outside this work, uses a different recipe, and was left untouched. Do not run
these prepared experiments concurrently with it or claim its results belong to
the matched bundles documented here.

This document covers the controlled local language-model experiment. Concurrent
Nexus v88-v90, code-corpus, percentage-coverage, and natural-phrasing work in this
checkout is separate; these bundles do not incorporate those new data sources.

## Why this experiment

The recorded v86 paired development evaluation at `output/v85_measurements/v86_paired_n630.json`
scores 491/630 (0.7794), against v80's 0.6302 on the same original 21 tasks.
However, average is 1/30 and two-step is 9/30, versus v80's two-step 19/30.
The previously completed v86 run took about 20.9 hours on this machine. A new
18,000-step run with larger accuracy probes may take longer; no speedup is promised.

The research rationale is in [V87_RESEARCH_NOTES.md](V87_RESEARCH_NOTES.md).
Compact equation targets are a plausible small-model intervention, not a proven
Supermix improvement. Recent scratchpad work distinguishes a correct written
trace from evidence that the model internally uses that trace. Our verifier
checks supported text equations only, and makes no causal/internal-reasoning claim.
[Equation-target study](https://arxiv.org/abs/2409.12393),
[June 2026 scratchpad study](https://arxiv.org/abs/2606.29522).

## Prepared artifacts

All generated artifacts are local and ignored by Git:

| Bundle | Rows | Revised rows | Use |
|---|---:|---:|---|
| `output/v87_preparation_20260905/control` | 976,108 | 0 | Source-identical control |
| `output/v87_preparation_20260905/combined` | 976,108 | 80,000 | Compact working plus varied prompts |
| `output/v87_preparation_20260905/combined_rehearsal` | 1,472 | 128 | Earlier preparation smoke only |

The full control is byte-identical to `datasets/v86/v86_combined.jsonl`.
Both full bundles retain 927,572 training rows, 9,823 development rows, and 38,713
test rows, in the same original-source membership and order. The new split is
not v86's historical content-dependent split: train a fresh matched control.
Response-novelty tiers are reclassified honestly for each transformed arm.
Vocabulary and initialization can differ when wording changes; this is a data
recipe comparison, not an isolated architecture or fixed-tokenizer ablation.

Each full bundle contains `train.jsonl`, `frozen_split.json`, `evaluation.json`,
and `manifest.json`. The manifest records data hashes, partition identity,
critical source-file hashes, explicit trainer arguments, and non-promotion status.

| Bound artifact | SHA256 |
|---|---|
| Original/control corpus | `09db9c39282fd19ef5378cc888fa74c3ff7f95871742cbfa987240567d366d81` |
| Combined corpus | `cf8dc6ad4e1d26b0292e19afc98a1d21ed753cb2dd8623c1eca24d3cf4d94830` |
| Shared evaluation | `5294b3be335690041131afb861f3558e586a24bd6367ecaa648d8c0ef9773d8c` |
| Shared partition identity | `623069ae2f6d15e12247f415a8db0e37e01aec54a982523675db068df9d09f25` |

The combined arm changes only 40,000 averages and 40,000 two-step rows. It writes
explicit binary additions for average and exact fraction division before the
second operation, with deterministic paraphrase selection independent of the
operand RNG. Maximum revised turn lengths are 104 and 58 tokens respectively,
inside the 128-token training context. All revised worked answers were checked.
Other rows retain their original bytes, values, order, and task exposure.

## Evaluation contract

The shared frozen set has 100 average groups and 100 two-step groups, totaling
1,200 prompts. Each problem has a canonical prompt and three reserved phrasings;
each two-step group includes both addition and subtraction. Average permutations
and both operation contrasts are excluded against 65,432 supported semantic
groups parsed from the entire source, not just a rehearsal sample.

This exclusion is grammar-bounded, not a claim to detect arbitrary paraphrases
inside legacy prose. The 96,108 inherited unlabelled rows have not received a new
privacy, licensing, or provenance clearance. Do not upload these bundles or treat
an integrity preflight as permission to publish or expand their use.

`eval_prompt_robustness.py` measures raw greedy responses without prompt
normalisation: final-answer correctness, whole-group correctness, supported
process correctness, first incorrect step, token-cap hits, unknown input tokens,
and native-context overruns. Comparisons reject mismatched settings, code
fingerprints, item order, or altered scores. Bootstrap resampling uses semantic
groups rather than treating correlated paraphrases as independent evidence.
This evaluation does not replace the original 21-task non-regression check.

The v86 smoke receipt is `output/v87_preparation_20260905/v86_prompt_smoke.json`:
36 prompts, 3 correct, 0/6 all-variant groups correct, using a 96-token cap.
Average was 0/12; two-step 3/24 (3/6 canonical and 0/18 paraphrased).
This tiny subset has `complete=false` and no confidence interval or promotion
authority. Do not use it to claim population accuracy or tune a candidate.

## Running the next experiment

From the repository root, these commands validate only; they do not train:

```powershell
python source/run_v87_training.py --bundle output/v87_preparation_20260905/control
python source/run_v87_training.py --bundle output/v87_preparation_20260905/combined
```

After reviewing the retained data's permitted use and compute budget, adding
`--train` explicitly launches the supervised trainer. It refuses rehearsal data,
changed hashes, and an existing output directory. The recipe is from scratch,
18,000 steps, seed 87, v86-shaped 256-wide/four-layer architecture, 128-token
turn-aligned packing, and a fixed original-21-task 420-problem development probe
every 3,000 steps with 112 generated tokens. New coding tasks cannot silently
enlarge the selection exam. The frozen 1,200-prompt set is never used to select
training checkpoints.

Run control before combined, then evaluate both completed candidates on the full
frozen set without `--limit_groups`. Use the same decode cap and implementation.
`--baseline PATH` adds a paired comparison to the second evaluation receipt.
The preparer also supports `average`, `two_step`, and `paraphrases` arms to
separate contributions before attributing a combined gain to one intervention.
Use a new output directory for every preparation; files are never overwritten.

If source code or the original corpus changes, preflight intentionally refuses
the existing bundle. Reprepare both arms together and compare their partition,
evaluation, and source-code identities. Do not manually refresh frozen hashes.

## Runtime and recovery changes

- Prompt normalisation now requires a whole supported request. Compound requests,
  negation, extra constraints, and unsupported physics prose reach the model intact.
- Chat and its development benchmark default to 96 tokens, expose actual token
  counts/termination, validate malformed controls, and identify benchmark settings.
  For families covered by the whole-request parser, answer badges are withheld
  when the parser declines a compound or constrained prompt. Other checker
  families retain their existing bounded-pattern limitations.
- Accuracy selection requires a fresh measurement of the current weights.
  The receipt binds the complete ordered development probe and per-task counts.
- Recovery files preserve matching selection-best weights, selection history,
  sampler/global RNG state, optimizer/scheduler state, and AMP state where used.
  Selection-best inference files are separate from recovery files. Legacy files
  lacking complete state may warm-start but cannot pretend to be exact resumes.
- A four-step CPU test with a simulated interruption matched uninterrupted final
  weights exactly. This is a recovery test, not a full model-training result.

Validation included 607 focused tests plus the three subsequently added preparation
checks (the final preparation suite passed 29 tests), focused Ruff/compile checks,
runtime model-variant parity, and `git diff --check`. Existing touched suites are
in CI; run the new `test_v87_training_preparation.py` and `test_v87_frozen_split.py`
explicitly as well. Their CI listing should be added with the new files when this
work is committed; the release guard rejects workflow paths absent from Git.
No full-repository pass or validation of concurrent Nexus work is claimed.

The final chat/prompt suite passed 111 tests after the badge correction. A
temporary real-model server and Playwright confirmed default 96-token controls,
streaming, token-limit display, and a NOT CHECKED badge for the intact constrained
prompt. Desktop/mobile captures are under `output/playwright/v87-chat-*.png`.
The existing server on port 8080 still runs its earlier loaded code; these source
changes require an intentional restart/redeployment to become active there.

Implementation readiness is not model capability. A completed matched control,
candidate results, per-task non-regression, group/process gains, and independent
review remain required before any model-selection or promotion decision.
