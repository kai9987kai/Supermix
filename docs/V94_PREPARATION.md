# v94 preparation: useful growth with retention checks

Prepared 2026-09-20 while v93 work continues. **No v94 training has started.**
This bundle adds standalone research tools; it does not alter the v93 model,
trainer, corpus, checkpoints, active pointer or packaged runtime.

## Decision

Test whether **mature, utility-guided capacity adaptation** improves learning
and preserves old skills within a bounded active budget. Keep the existing
two-hemisphere architecture for the first comparison. A new communication
architecture and a new growth policy in the same run would obscure causality.

The research basis and eight primary sources are in
[V94_RESEARCH_REVIEW.md](V94_RESEARCH_REVIEW.md). The most direct recent inputs
are the May/June 2026 growth-stability study (newborn gradient starvation),
April/May 2026 Expert Upcycling (gradient utility and fixed-top-k expansion),
and August 2026 elastic-growth study (compact turnover). Their reported
results are not Supermix results; in particular, the elasticity study uses
permuted image tasks rather than language modeling.

Three paths were considered: more architectural capacity, learned
cross-hemisphere communication, and better selection/integration of existing
capacity. Start with the last: v93 already introduces several architectural
changes, and the current evaluation needs to measure them correctly before
another component is added. Cross-hemisphere gates and signed-gradient edge
proposals remain explicit follow-up experiments.

## Implemented preparation tools

| File | Available now | Boundary |
|---|---|---|
| `source/v94_growth_policy.py` | Deterministic proposals, EMA utility, maturity protection, original-unit protection, active/event budgets, family witness checks, exact JSON state resume | No tensor mutation, utility collector, optimizer edit, or rollback |
| `source/v94_connectome_audit.py` | Complete gate-off, commissure/side/grown-component ablations; exact restoration of gates, buffers and MiMoMix telemetry; paired group-bootstrap loss analysis | Requires an offline eval model, no concurrent inference or training |
| `source/prepare_v94.py` | Hash-bound grouped data memberships, source/checkpoint fingerprints, controlled experiment plan and readiness report | Never loads a checkpoint as code, launches training, or promotes a model |

All files are opt-in and have no import path from the active trainer.
`runtime_python` does not need a mirror of preparation-only tools. Integrating
new runtime model behavior later will require explicit source/package parity.

### Growth policy contract

Use one `GrowthPolicy` per population: all connectome modules, or one MoE
layer. `capacity`, `min_alive` and `max_alive` must match the actual population;
for experts, `min_alive` must be at least the model's top-k. Capacity slots
remain allocated. The policy does not claim to reduce allocated memory.

Supply a `UnitObservation` for every slot. Utilities must be finite,
nonnegative and comparable within that population. They are **measured
contribution estimates**, not automatically the existing activation/load
statistics. A first proposed probe is normalized gradient energy on training
calibration; the normalization and cadence must be fixed in preflight. No
collector is installed yet. Different utility scales require different
thresholds; defaults are testable policy settings, not tuned v94 values.

The policy requires the `growth_calibration` role and its frozen membership
SHA256. It rejects a changed cohort, duplicate/out-of-order observations,
backdated births after initialization, and a changed training horizon after
resume. A reused slot must be observed dead before rebirth. An adapter that
prunes and reuses a slot between observations must expose those transitions.
Original-unit provenance cannot be rewritten to make an original prunable.

Newborns must mature before pruning or becoming parents; low utility must
persist across observations. Each parent can receive at most one proposed
child per event. Births stop with less than one maturity window remaining.
This bookkeeping does not prove that newborns receive useful gradients.

Proposals are advisory and may depend on *all* listed prunes being accepted
to respect the active budget. Future integration must apply the entire event
transactionally or replan from actual state. State records observed histories,
not accepted edit counts. Global accepted-edit and lifetime parent-duplication
budgets, scarce-family protection, and edge/tap budgets remain integration work.

`witness_acceptance` pairs rows by ID and rejects a loss increase beyond the
configured nats-per-token budget in **any** witnessed family. An aggregate
improvement cannot hide a family regression. Calibration/witness feedback is
adaptive training feedback, never a held-out promotion result. Role and hash
validation are API contracts; the adapter must supply the actual manifest rows.

### Correct interpretation of ablations

The existing v91 evaluator's `off` mode closes only `core.gate`. v93 has
additional write gates and `thinking_gate`. The new `all_off` context closes
every one and restores them on success or exception. A tiny-model regression
test confirms that the old partial closure leaves an observable contribution.

`commissure_off`, `left_off`, `right_off`, and `grown_components_off` use
independent settings, not leftover switches from a previous mode. Statistics
collection is disabled during the audit and restored afterward, including
nonpersistent router statistics and plain telemetry attributes.

`grown_components_off` is **deletion sensitivity**. Mitosis halves the parent
and child's outgoing/readout weights. Deleting the child does not reunite the
parent, undo optimization, or reconstruct a no-growth run. That causal claim
requires the matched no-growth training arm.

For loss analysis, pass `RowLoss(row_id, family, loss_sum, token_count,
group_id)` records to `compare_losses`. Positive delta means higher loss
after deletion. Identical IDs, groups, families and token counts are required;
file order may differ. Provide semantic group IDs for paraphrases so the
bootstrap does not pretend they are independent. Without group IDs, each
row is treated as independent. One group produces no confidence interval.
Family intervals are descriptive and are not multiplicity-adjusted tests.

## Frozen data and experiment protocol

Corpus JSONL preparation rows must contain `row_id`, `semantic_id`, `family`,
`prompt`, and `response`. The generator must assign the same semantic ID to
all paraphrases of the same problem. IDs are global across files. This tool
does not solve semantic-equivalence detection; normalized prompt equality is
an additional conservative overlap check. It does not reject common answers
alone, since independent arithmetic problems can correctly share an answer.

`freeze_cohorts` makes family-stratified, semantic-group-disjoint cohorts:
gradient training, training calibration for growth, selection dev, and an
externally supplied final evaluation. Default calibration and dev shares are
10% each by group, rounded up with at least one group per family. At least
three independent groups per family are required. Membership and full row
content receive SHA256 hashes. Input ordering does not change the result.

Do not relabel dev data as calibration after seeing results. Freeze new
training-calibration membership before collecting utility. Preserve historical
v89/v93 benchmarks for retention comparisons, but describe repeatedly used
benchmarks as such. A claim about unseen tasks needs a separately frozen
external evaluation cohort with no shared semantic groups.

Run the initial experiment only after a completed parent is verified:

| Arm | Treatment | Primary contrast |
|---|---|---|
| A | Fixed-topology continuation of v93 | Additional-training baseline |
| B | v93 growth heuristics, using training calibration | Growth package versus A |
| C | v94 maturity, utility and family witness guards | Policy package versus B |

All arms use the identical parent/tokenizer, initial capacity/top-k, ordered
training rows, cumulative supervised tokens per family, mixture schedule,
optimizer/scheduler, diagnostic cadence and paired seeds 94/95/96. B moves
feedback off dev, so it is not an exact rerun of v93. C changes several policy
rules; a win justifies single-factor follow-ups, not attribution to one rule.

Hold edge/tap heuristics and quotas constant between B and C for the initial
module/expert-policy comparison. Register old/new mixture ramps in all arms;
compare abrupt versus gradual schedules separately if needed. Report measured
wall-clock including utility/witness passes, throughput and peak memory beside
token-matched results. A later compute-matched run is a distinct comparison,
not a retrospective relabeling of equal-step experiments.

Report old-family retention, new-family generation accuracy, paired regressions,
per-family loss, newborn survival/gradient/update norms and all branch ablations.
Evaluate generation with the same decoding/token caps. Keep raw row receipts.
Final evaluation must not feed growth or checkpoint selection. Freeze numerical
retention and compute tolerances before those runs; there is no automatic
promotion gate or established performance gain in this preparation bundle.

## Commands and handoff

From the Supermix root, a lightweight plan with no model/corpus reads:

```powershell
python source/prepare_v94.py --output output/v94_preparation_20260920
```

Once the completed v93 parent and independently grouped inputs are available,
use a **new** directory (existing frozen bundles cannot be overwritten):

```powershell
python source/prepare_v94.py --output output/v94_bound_inputs --parent-checkpoint PATH_TO_COMPLETED_PARENT --corpus PATH_TO_GROUPED_TRAIN_JSONL --evaluation PATH_TO_GROUPED_FINAL_JSONL
```

The bundle writes `readiness.json`, `experiment_plan.json`, and, when inputs
are supplied, `cohort_manifest.json`. Even a bound bundle remains
`launch_enabled: false`: hashing a checkpoint is not proof of completed
training, integration correctness, or improvement.

Next integration sequence:

1. Verify v93's completed receipt, final/selected checkpoint, exact tokenizer,
   source snapshot, growth log, generation results and matched baselines.
2. Freeze the new data manifest and utility definition. Measure probe cost.
3. Add an explicit opt-in training adapter on a separate candidate. Snapshot
   affected weights, optimizer moments, controller state and RNG for each
   transaction; restore all on rejection, including exceptions/nonfinite loss.
   Register cadence and accepted-event limits; ensure exact resume.
4. Run tiny transaction/resume tests, parent-load/generation tests, then a
   capped timing preflight before choosing the matched training token budget.
5. Run A/B/C only when resources are available. Compare held-out receipts
   before considering runtime integration or promotion.

The v93 checkout changed while this preparation was being written: its corpus
builder and controller arrived after the initial read-only audit. Source
fingerprints in readiness are observations, not proof of another run's source.
The final local process check belongs in the handoff report, not a permanent
claim that v93 is or is not training.

## Validation recorded for this preparation

On 2026-09-20, with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`:

```powershell
python -m pytest -q test_v94_growth_policy.py test_v94_connectome_audit.py test_prepare_v94.py test_v93_core.py
python -m py_compile source/v94_growth_policy.py source/v94_connectome_audit.py source/prepare_v94.py
git diff --check
```

Result: **83 passed** in 16.25 seconds, with one existing v93 tensor-to-scalar
warning. Compilation and diff whitespace checks passed. These are behavioral
and compatibility tests, not training or benchmark evidence. The focused tests
include accepted-birth resume, fixed horizon, original-unit provenance,
exception-safe ablation restoration, grouped pairing, content-bound membership
and rejection of train/evaluation overlap. No full repository test run was
performed for these isolated additions.
