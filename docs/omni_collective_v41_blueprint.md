# Omni Collective V41 Blueprint

This is the first concrete preparation pass for `omni_collective_v41`, built from the finished local `v8` artifact and a small set of recent primary-source training ideas.

## Why V41 Exists

`v8` ended at `112,025,979` parameters and improved multimodal fusion, but its weakest metric was still `response_accuracy = 0.0935` in stage 2. The sample outputs also show three clear problems:

- human communication is still too brittle and templated
- grounded routing and model/tool choice still fail on specialist prompts
- coding/problem-solving is better than earlier omni lines, but not strong enough to dominate the local stack

So `v41` should not just be `v8 + more rows`. It should become:

- smarter on hard reasoning and coding prompts
- more creative without becoming less grounded
- more natural and useful when talking to humans
- better at choosing the right route, specialist, or tool before answering

## Target Envelope

- target params: around `126M`
- keep the model compact enough for local deployment
- raise stage-2 response quality first, not just overall score
- challenge `v40_benchmax` on reasoning while keeping the broader multimodal specialist edge of the omni line

## Architecture Plan

The parameter increase should be moderate:

- `text_hidden`: `272 -> 288`
- `fusion_hidden`: `1088 -> 1152`
- `expert_count`: `10 -> 12`
- `expert_hidden`: `1792 -> 1920`
- `deliberation_passes`: `10 -> 12`

New `v41` heads:

- `latent_plan_slots`
  Distill short hidden plan states from explicit planning traces, then keep the plans internal at inference.
- `multi_token_prediction_aux`
  Predict short future spans to improve long-range coherence and information density.
- `communication_polish_head`
  Separate answer correctness from presentation quality so the model can solve first, then rewrite more clearly.
- `uncertainty_anchor_head`
  Reward transparent uncertainty and missing-evidence statements instead of invented specifics.

Two more explicit `v41` ideas should now be treated as first-class:

- `reflection memory shards`
  Keep compact bad-attempt -> reflection -> repaired-answer traces for coding and reasoning tasks.
- `reasoning budget labels`
  Supervise when a task deserves a short, medium, or deep solve instead of globally increasing thinking depth.

## New Training Ideas

These are the main `v41` innovations, grounded in recent work but adapted to the Supermix local-first setup:

1. `Latent Plan Distillation`
   Train on explicit micro-plans and route choices, but compress them into hidden plan slots before deployment.

2. `Reasoning Budget Curriculum`
   Train short, medium, and long budget targets so `v41` learns when to think longer instead of always answering at one fixed depth.

3. `Disagreement Harvesting`
   Mine prompts where `v8`, `v40_benchmax`, and the Qwen teachers disagree. Turn those into compare-and-justify rows instead of taking a single teacher answer as truth.

4. `Code Critique -> Repair`
   Add bug reports, failing tests, stack traces, and patch-review examples where the model must diagnose, propose, and refine a repair.

5. `Two-Pass Human Communication`
   First solve the problem. Then run a short communication-polish rewrite trained to improve clarity, brevity, empathy, and structure without changing the facts.

6. `Creativity Rescue Pairs`
   Use pairs of dull-correct answers and vivid-but-still-grounded answers so creativity is learned as controlled improvement rather than free hallucination.

7. `Route-Then-Answer Supervision`
   Train on hidden or explicit route-selection labels before the final answer. This should improve Auto mode, collective mode, and specialist handoff decisions.

8. `Reflection Memory`
   Keep concise post-mortems for failed code, route, and reasoning attempts, then train on the repaired response.

9. `Reasoning Budget Curriculum`
   Train the same task under short, medium, and deep answer budgets so the model learns when extra deliberation is actually worth it.

## Data Mix Changes

The data should become denser and more varied, not just larger:

- `high_density_reasoning_core`
  A compact, hard reasoning set with fewer repetitive examples and higher per-row value.
- `code_critique_repair`
  Traceback, failing-test, patch, and critique loops for coding and debugging.
- `reflection_memory_rows`
  Failure -> short reflection -> improved retry traces.
- `human_communication_polish`
  Rewrites, comparisons, explanations, de-escalation, uncertainty framing, and direct-but-clear human answers.
- `teacher_disagreement_hardcases`
  Prompts where strong local teachers disagree on answer, route, or framing.
- `grounded_uncertainty_rows`
  Current-fact and low-evidence prompts that explicitly train “what I know / what I don’t know / what I need”.
- `creative_constraint_pairs`
  Creativity with fidelity, style control, and factual discipline.
- `route_then_answer_supervision`
  Tool/model choice before answer.
- `reasoning_budget_curriculum`
  Same task solved under short, medium, and deep answer budgets.
- `specialist_dense_topups`
  Materials, protein, 3D, vision, and native-image deltas that keep the omni stack broad.

## Training Recipe

Recommended `v41` phases:

1. `Stage 0: latent-plan bootstrap`
   Distill short explicit plan traces, route labels, and critique stubs into hidden plan slots.

2. `Stage 1: teacher-heavy SFT`
   Mix `v8`, `v40_benchmax`, `qwen_v28`, `qwen_v30`, and `v7` with harder coding/problem-solving weighting and denser reasoning rows.

3. `Stage 2: self-play preference loop`
   Generate chosen vs rejected pairs using self-play and teacher disagreement mining.

4. `Stage 3: code refine`
   Train on critique -> repair traces and keep only objectively improved fixes.

5. `Stage 4: communication polish`
   Finish with human-facing clarity, compare/explain prompts, and grounded uncertainty.

6. `Stage 5: self-correction finish`
   Run deterministic self-correction on weak drafts with fair prompts and keep only improved outputs.

## Runtime and Reliability Defaults

`v8` was trained on CPU with no AMP, no compile, and no accumulation. `v41` should not repeat that if avoidable.

Recommended defaults:

- prefer `cuda` or `dml`, not CPU
- `amp=auto`
- `compile_model=true`
- `grad_accum_steps=4`
- `ema_decay=0.9993`
- `warmup_ratio=0.06`
- checkpoint every `100` optimizer steps
- always keep rolling stage checkpoints and stage-complete snapshots
- preview-benchmark stage-2 checkpoints during the run

## Success Gates

Minimum goals:

- stage-2 `response_accuracy >= 0.145`
- stage-2 score `>= 0.455`
- common benchmark exact `>= 0.25`
- must beat `omni_collective_v8_preview` and `omni_collective_v7`
- stretch goal: approach or beat `v40_benchmax` on benchmark-heavy reasoning while remaining better on specialist and conversational tasks

Promotion protocol:

- benchmark at least `3` preview checkpoints during stage 2
- do not promote a final model that regresses vision or domain accuracy by more than `0.02` vs `v8`
- require a clean win on the new communication, code-repair, and grounded-uncertainty eval packs before promotion

## Primary-Source Inspirations

These papers are being used as inspirations, not copied recipes:

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Better & Faster Large Language Models via Multi-token Prediction: https://arxiv.org/abs/2404.19737
- s1: Simple test-time scaling: https://arxiv.org/abs/2501.19393
- LIMO: Less Is More for Reasoning: https://arxiv.org/abs/2502.03387
- RefineCoder: https://arxiv.org/abs/2502.09183
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Play Fine-Tuning (SPIN): https://arxiv.org/abs/2401.01335
- Self-Refine: https://arxiv.org/abs/2303.17651
- Large Language Models have Intrinsic Self-Correction Ability: https://arxiv.org/abs/2406.15673
- RLAIF: https://arxiv.org/abs/2309.00267

## Immediate Next Steps

- add a `v41` prep script that emits a machine-readable blueprint from the latest `v8` summary
- build new dataset builders for disagreement mining, communication polish pairs, and route-then-answer supervision
- extend the omni model code with latent-plan, communication-polish, and uncertainty-anchor heads
- keep the `v8`-style resume and preview export path from the first training run onward
