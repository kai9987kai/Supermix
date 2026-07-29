# v52 unified merge log

July 27, 2026.

This records the consolidation of three working trees into one line, so that a
later reader can tell what was merged, what was deliberately rejected, and why.

## Trees examined

| Tree | Commit | State | Verdict |
| --- | --- | --- | --- |
| `Supermix` | `5fbea442` (2026-07-25) | 66 uncommitted files | merge target |
| `Supermix_27` | `5fbea442` (2026-07-25) | clean | **contributes nothing** |
| `Supermix_52` | `f04896dc` (2026-06-11) | 68 uncommitted files | donor |

`Supermix_27` is a clean checkout of the same commit as the merge target. A
content-hash comparison of all 372 of its text files found one file not present
in the target, `.claude/settings.local.json`, which is local editor state. Every
other difference was the target being ahead. Nothing was taken from it, and
nothing was lost by not taking anything.

`Supermix_52` sits on an older base but carried a generation of model work that
was never merged forward. That work is the substance of this merge.

## What the two trees each had that the other did not

The donor and the target had genuinely diverged rather than one simply being
behind:

- the target's `CognitiveLeapUltraExpertHead` had `_kl`,
  `_normalize_prediction_class_indices`, and `_topk_js_divergence` — the v51
  prediction-stability verifier — which the donor lacked;
- the donor's had `_route_core_updates` — sparse top-k core execution — which
  the target lacked, plus an entire `CognitiveLeapV52ExpertHead` subclass.

Both survive in the merged head.

## A: ported verbatim

- `source/materialize_v52_from_v50.py`
- `source/benchmark_v521_metacognitive_controller.py`
- `launch_v52_unified_chat.ps1`
- `docs/V52_UNIFIED_ARCHITECTURE.md`
- `test_v52_materialize.py`
- `artifacts/v52_initialization/` (v50 donor checkpoint, initialized v52
  checkpoint, and manifest)

The materialize test passes against the checked-in manifest, which means the
merged head produces a **bit-identical state dict** to the donor's. That is the
strongest available evidence that the model port is faithful.

## B: ported with adaptation

- `CognitiveLeapV52ExpertHead` and `ChampionNetCognitiveLeapV52Expert` into
  `source/model_variants.py`. The donor's `forward` accepted only the v51
  arguments that existed when it was written. The merged version accepts and
  forwards the full current set — `prediction_stability_top_k`,
  `prediction_stability_margin`, `prediction_output_transform`,
  `prediction_class_indices`, `prediction_stability_rank_depth` — so a v52 model
  keeps the checkpoint-bound verifier exit it inherits instead of silently
  losing it. Verified by
  `test_v51_prediction_stability_controls_are_forwarded`.
- `_route_core_updates`, the router load-balance and z-loss regularizers, and
  the `last_active_cores` / `last_router_load_balance` / `last_router_z_loss`
  telemetry into the shared ultra head. The new buffers are non-persistent so
  existing v51 checkpoints load unchanged.
- Registry wiring: `SUPPORTED_MODEL_SIZES`, `build_model`,
  `load_weights_for_model`, and `detect_model_size_from_state_dict`. The v52
  probe is deliberately placed **before** the ultra probe, because a v52
  checkpoint also carries the `core_router` and `cross_attn` tensors it
  inherits.

One latent defect in the donor was fixed rather than carried over: `_aux_loss`
was only ever assigned inside `forward`, so reading it before a forward pass
raised `AttributeError`. It is now initialized in `__init__`.

## C: reimplemented against the current tree

Nothing. The donor's ideas either ported directly or were already present.

## D: deliberately not ported

- **`runtime_python/model_variants_shared.py`** (donor-only, 8,905 lines). The
  target already solves the same problem with
  `source/sync_runtime_model_variants.py`, which mechanically generates a
  standalone `runtime_python/model_variants.py` and is enforced by
  `test_runtime_model_variants_parity.py`. Adding a second shared module would
  create two sources of truth.
- **`test_v52_interaction_planner.py`** (donor-only). It asserts a nested
  `diagnostic["appraisal"]` shape from before the target's Plan-Evaluate work.
  The target's planner is a strict superset — it still has
  `validate_then_help`, `reflective_support`, `needs_validation`, the crisis
  path, and the sycophancy guards — but exposes them through a flatter, richer
  diagnostics contract (`emotion_cue`, `distress`, `context_distress`,
  `affect_trend`, `continuity_applied`, `factuality_risk`, …). The capability is
  kept; only the outdated assertion shape was dropped.
  `test_interaction_planner.py` covers the current contract.
- **The donor's in-ranker interaction-plan fallback.** In the donor,
  `rank_response_candidates` built its own plan when the caller passed none. The
  target made that opt-in on purpose and pins it with
  `test_none_plan_preserves_bit_exact_legacy_scores_and_order`. Porting the
  fallback would break a deliberate contract.

  A related suggestion — that
  `benchmark_all_models_common.py` and `qwen_supermix_pipeline.py` should build
  explicit plans at their `pick_response` call sites — was also rejected. Both
  are measurement paths that run with empty history: the first is a benchmark
  generator and the second mines preference pairs for training. Injecting an
  interaction plan there would change benchmark numbers and training data rather
  than improve a user-facing response.

## Conversation State v1

Added during this consolidation, not inherited from either tree.

`source/conversation_state.py` derives a bounded, deterministic view of the
whole session. Both prior understanding layers are per-turn: `analyze_prompt`
and `plan_interaction` each see only the last four turns, so a constraint stated
earlier is gone and an unanswered clarifying question can be asked again.

The layer tracks durable user commitments with supersession when the user
changes their mind, questions the assistant asked and whether a later user turn
answered them, topic threads and topic shift, what has already been delivered,
requests the next reply did not engage with, and stated contradictions.

It is wired in three places and is advisory everywhere:

1. `rank_response_candidates` takes an optional `conversation_state` and adds a
   term bounded to ±0.12. The parameter defaults to `None` and the block is
   skipped entirely in that case, so the single-turn path stays bit-exact and
   `test_interaction_ranking_regression.py` still passes unchanged.
2. `pick_response` suppresses a clarifying question when a clarification loop is
   already detected — asking a third time is worse than answering imperfectly.
3. `chat_app`, `source/chat_web_app.py`, and `runtime_python/chat_web_app.py`
   build the state per turn and report privacy-safe diagnostics. The web engines
   accept `conversation_enabled=False` for controlled evaluation, matching the
   existing `interaction_enabled` and `grounding_enabled` switches.

Diagnostics carry counts, category labels, and flags only — never turn text.

## Repository mechanics touched

- `source/generate_studio_runtime_manifest.py` now registers
  `conversation_state.py` in `RUNTIME_MODULES` and `CONTRACT_CONSTANTS`.
  Without this the packaged desktop build would ship a `chat_pipeline` whose
  import of `conversation_state` fails.
- `runtime_python/chat_pipeline.py` was reconstructed with a three-way merge
  rather than copied from `source/`. The two files are **not** byte mirrors: the
  runtime copy hoists `SPLIT_ROLE_RE`, `ACTION_REQUEST_RE`,
  `CONTEXT_V3_IMPERATIVE_RE`, `COREF_RE`, and `CONTINUE_RE` to module level as a
  performance optimization. That optimization is preserved.
- `runtime_python/chat_web_app.py` is likewise not a byte mirror, but
  `test_champion_web_knowledge.py` asserts its web-knowledge method is
  **AST-equivalent** to the source copy, so the conversation wiring was written
  identically in both rather than defensively in one.

## Verification

Full suite: 1,052 passed, 2 skipped, 0 failed.

New tests: `test_conversation_state.py` (13) and
`test_cognitive_leap_v52_expert.py` (12), plus the ported
`test_v52_materialize.py` (2).

---

# Second pass: closing the control surface

The first pass merged the v52 model. A follow-up differential audit of all
seven remaining `Supermix_52` subsystems against this tree surfaced 14
candidates. Their disposition is below so nothing is silently dropped.

## Defects the first pass introduced

| # | Finding | Disposition |
|---|---|---|
| 1 | `launch_v52_unified_chat.ps1` passed `--core_top_k 2` to a parser that did not define it, so the ported launcher aborted with argparse exit code 2 | **Fixed.** Flag added to both apps; `test_v52_runtime_controls.py` now parses the launcher and asserts the target app accepts every flag it passes |
| 2 | `docs/V52_UNIFIED_ARCHITECTURE.md` asserted that installer upgrades delete the prior bundled-model directory; no `[InstallDelete]` rule existed | **Fixed.** Rule added for `{app}\_internal\bundled_models`, matching the PyInstaller `--add-data` target. Contract version moved to `2026.07.27` |
| 3 | The v51 dense forward became 10% slower: the router z-loss stacked and reduced raw gate logits on every forward, including inference | **Fixed.** Computed only under `self.training`, matching `last_ponder_cost`. Back to parity, output bit-identical |

## Capabilities merged but unreachable

| # | Finding | Disposition |
|---|---|---|
| 4 | No CLI flag or web setting reached the sparse router or the quality/continue verifier, leaving merged model code unusable | **Fixed.** Four controls added to `chat_app.py` and `chat_web_app.py` and threaded through the compute helpers, the sweep, and progressive auto compute |
| 5 | `collect_runtime_compute_metrics` dropped the seven telemetry attributes the v52 head emits | **Fixed.** All seven mapped; they stay `None` on pre-v52 variants |
| 6 | The standalone-runtime test never exercised `cognitive_leap_v52_expert`, the whole point of merging it into the snapshot | **Fixed.** `test_runtime_model_variants_parity.py` now builds and runs v52, with its controls, in an isolated directory |

## Pre-existing defects found in passing

| # | Finding | Disposition |
|---|---|---|
| 7 | `Engine.chat` snapshots session history, runs inference outside the lock, then appends. Concurrent same-session requests lost turns and corrupted the derived conversation state | **Fixed.** Per-session turn serialization via a bounded lock table |
| 8 | The heads publish telemetry as attributes on themselves, so concurrent forwards reported each other's metrics | **Fixed.** An `inference_lock` covers every forward in `chat` and `compute_sweep`; placement is asserted structurally, not by grep |
| 9 | Both Qwen PyInstaller specs carried absolute paths into a *different* sibling checkout (`Supermix_27`), so building from the spec resolved datas and the icon outside this repository | **Fixed.** Made checkout-relative. The Studio spec was already guarded for portability; the Qwen specs now are too |
| 10 | Generated desktop branding stamped the stale footer `Supermix_27` | **Fixed.** Now `Supermix`. The donor's own value (`Supermix_52`) was equally wrong for this tree |

## Documentation gaps

| # | Finding | Disposition |
|---|---|---|
| 11 | `ARCHITECTURE.md`'s classifier-head table stopped at v26 | **Fixed.** v50/v51/v52 rows added with the v52 status caveats |
| 12 | `RESEARCH_UPGRADES.md` had no entry for the runtime control surface | **Fixed.** Entry added, including the measured costs and the three performance findings |
| 13 | `runtime_python/MODEL_CARD.md` described only the v27 checkpoint and never mentioned what the shipped snapshot can build | **Fixed.** Variant table, v52 limits, and the compute controls documented |

## Not ported

| # | Finding | Disposition |
|---|---|---|
| 14 | A web UI toggle and status readout for verifier escalation | **Deferred, deliberately.** The backend settings, CLI flags, and API surface all accept the controls, so nothing is unreachable. A prominent UI switch for a head that is still randomly initialised would invite reading its output as meaningful. Worth adding once the verifier and appraisal heads have labelled supervision and a held-out evaluation |

## Measurements

Taken on CPU, single thread, median of 12–15 runs. This machine's numbers, not a
general claim.

- v51 ultra dense forward: 26.914 ms before the merge, 26.293 ms after, output
  bit-identical.
- Legacy ranking path: bit-exact and unchanged when no conversation state is
  passed.
- Conversation-state ranking overhead: +7.6% on a 60-candidate pool, reduced
  from +18.9% by preparing the conversation view once per turn instead of once
  per candidate.
- v52 forward: 41.802 ms vs 32.108 ms for v51 ultra, +176,668 parameters. Opt-in
  variant only; v51 checkpoints never build the v52 branches.
- Sparse `core_top_k=2`: **1.63x slower** than its own dense path at this batch
  size. Per-core masking costs more than the cores it skips, confirming the
  donor document's warning. This is why dense remains the default.

## Verification

Full suite: 1,068 passed, 2 skipped, 0 failed, 0 warnings.

New tests: `test_v52_runtime_controls.py` (6) and
`test_chat_web_app_concurrency.py` (7), plus extensions to
`test_runtime_model_variants_parity.py`, `test_studio_runtime_manifest.py`, and
`test_conversation_state.py`.

Both entrypoints were run end to end with the launcher's exact argument list.
