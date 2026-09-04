# Supermix

This is the working monorepo for the current Supermix / ChampionNet / Omni Collective line.

This repository combines:

- local-first chat and multimodal runtime code
- experimental training and continuation pipelines
- desktop EXE and installer packaging
- benchmark tooling and graph generation
- published-model export helpers
- bundled datasets and generated research artifacts

It is intentionally a mixed workspace, not a minimal source-only model repo.

## Supermix v85 — making the next run measurable, and the ruler that moved

V85 is the **training line** (v58–v81); v82–v84 above are the parallel NexusMind
evidence line. Nothing in v85 was trained and no accuracy number moved because of
it. It is a release about instruments, and it found two things.

**The accuracy probe could not see the task it was scoring.** The mid-run probe
capped generation at 64 tokens while every `arithmetic_series` reply is 81–84
tokens long, so `--select_on accuracy` would have read 0.00 on that task whatever
the model learned. The offline benchmark's 40-token default truncated more still:
re-scoring v80 at 40 tokens gives 18 truncated replies against 0 at 96, and
`wave_speed` alone moves 0.000 → 0.667 on the cap. The cap is now
`--probe_max_new_tokens` (default 112), the offline default is 96, and the trainer
prints a per-task budget warning — or refuses to start under `--strict`.

**The benchmark had been asking easier questions.** Commit `c7041897` rewrote the
three zero-scoring tasks, and because the benchmark draws from the same
generators, it rewrote the exam too. On **unchanged v80 weights**:

| task | `a5bd5bf2` (what v80 trained on) | `c7041897` (current) |
|---|---|---|
| `arithmetic_series` | 0.000 | **0.500** |
| `kinetic_energy` | 0.167 | **0.833** |
| `combination` | 0.000 | 0.000 |
| `force` (control) | 0.917 | 1.000 |

So a v80-versus-v85 comparison is only paired if both are scored against the same
generator version. `eval_problem_solving.py` now records the seed, task list,
generation cap and a generator fingerprint so pairing can be checked rather than
assumed. `combination` reads 0.000 in every era and is the one that genuinely
needs the retrain.

**v80's regression against v74 is real; the cause is probably exposure.** The two had never
been scored on the same problems: before v85 one RNG was shared across tasks in
turn, and v74 was scored over 10 registered tasks against v80's 21. Re-scored on
one identical problem list, v74 reads **0.874** and v80 **0.659**, a −0.215 gap.
That is the measurement; it must not be differenced against the published −0.272,
which came from two runs on different problems at an unrecorded cap.

Both corpora hold exactly 40,000 rows per arithmetic task, but v80's added twelve
science tasks while the step budget stayed at 18,000, so each arithmetic task fell
from 8.06% to 4.39% of the corpus and v80 saw each one **54% as often**. That is
arithmetic over corpus composition, and it is the leading explanation — but v80
also added twelve tasks to a 15M-parameter model, and capacity interference
degrades incumbent tasks at *constant* exposure, which no sampling weight would
repair. The free test that would separate them (bimodal versus uniform damage)
comes out **skewed, not bimodal** (Sarle 0.285), so at nine tasks it does not
decide. Both are probably running.

Two throughput results bear on the fix. Removing the MTP heads (`--n_mtp_layers 0`)
is measured at 0.550× step time; keeping one depth did not resolve (0.845, spread
0.303) and must not be quoted at 45%. And `--batch_size 32` gives **20% more
sequences per hour** than v80's 16 — the opposite of a recommendation to shrink
the batch, which measurement showed costs 12% *fewer* sequences per hour. No run
has yet tested whether restoring exposure restores the score.

**The memorisation control was measuring a corpus the model never trained on.**
`eval_problem_solving.py --corpus` was hard-coded to a v62 dataset, so every
receipt since v62 whose run used a different corpus published a meaningless
`memorisation_gap` — v80's read −0.27, i.e. worse on its "own" rows than on
unseen ones. The flag now has no default, checkpoints record the corpus they
trained on, and a mismatch withholds the gap and names both files.

**The prompt normaliser learned physics.** It covered only arithmetic, so two of
v80's five natural-phrasing failures were physics questions it passed straight
through. With eight science rules added, the same checkpoint on eighteen
hand-typed questions goes from **10/18 to 14/18**, four fixed and none broken.
Of the four still wrong, three were rewritten correctly and the model still
missed them and the fourth was already in the trained form, so what remains is
the capability gap the benchmark already reports rather than a phrasing gap.

V85 also fixes four confirmed architecture bugs (the MLA rotary embedding was not
a rotation, MLA broke speculative decoding, Mixture-of-Depths decoded differently
than it trained, and weight init overwrote the thinking core's deliberate zeros),
exposes 29 previously unreachable `MiMoMixConfig` fields on the trainer (15 of 51
were reachable before, 52 of 64 now), and
adds a dozen cited techniques behind default-off flags. All sixteen flags are
verified to train without NaN at the v80 shape before anyone spends a run on one.
The recursive thinking core is measured inert on a problem-solving checkpoint for
the first time: identical 42/63 at 1, 2, 3 and 6 cycles.

One cost result is worth acting on: the multi-token prediction heads are 7.7% of
the parameters and **45% of step time** (0.550x measured with a paired,
drift-controlled A/B), for a draft path whose acceptance length is 2.5 of 3.

**This machine has been emulating x86 all along.** The hardware is ARM64; the
Python and PyTorch are not. The interpreter's PE machine word is `0x8664`,
`sysconfig.get_platform()` is `win-amd64`, and torch links Intel oneAPI MKL "for
Intel 64 architecture applications" with AVX2, while `platform.machine()` returns
`ARM64`. Every FLOP in an 11-hour run is binary-translated by Windows-on-Arm
Prism. So the earlier "bf16 is 25-60x slower because this box is ARM64" was the
wrong diagnosis: it is an x86 bf16 fallback under emulation, and the Snapdragon X
core has native bf16 this stack never reaches. Native `win_arm64` wheels exist and
are now the largest unexplored variable on the machine.

See [`docs/V85_MEASURABLE_ARCHITECTURE.md`](docs/V85_MEASURABLE_ARCHITECTURE.md)
for every measurement, the research map with citations, and what is not claimed,
and [`docs/V86_PLAN.md`](docs/V86_PLAN.md) for the arms to run next with their
acceptance criteria stated in advance. Receipts and reproduction scripts are in
`output/v85_measurements/`.

## Supermix v84 — Autonomous Epistemic & Multimodal Reasoning Frontier

V84 pushes forward the synthesis of Xiaomi MiMo's multi-token speculative reasoning with AI-Dem-Lab's advanced quantum, cellular, and cognitive kinematic physics:
- **Speculative Tree Search with Step-Level PRM (`nexus_engine.py`)**: Multi-token speculative drafting evaluated by Process Reward Modeling (PRM), tracking Shannon entropy transitions and dynamically backtracking when branches diverge or violate verification invariants. Exposes `POST /v1/speculative-tree`.
- **Quantum Density Matrix & Decoherence Channels (`nexus_engine.py`)**: Formulates full 2-qubit bipartite Werner states $\rho(p)$, computing Von Neumann entropy $S(\rho)$, purity $\gamma$, and concurrence $\mathcal{C}(\rho)$ under depolarizing and phase-damping quantum noise channels. Exposes `POST /v1/quantum/state`.
- **Wolfram Rule 110 Glider & Soliton Logic Engine (`nexus_engine.py`)**: Simulates soliton dynamics and glider collisions ($A, B, C, E, F$) on the 14-cell periodic ether background of Rule 110, implementing logical gate analogs (annihilation NOT, deflection AND) in the 1D computational universe. Exposes `POST /v1/wolfram/gliders`.
- **Dynamic 5D Cognitive Trajectory Tracking (`nexus_engine.py`)**: Kinematic analysis of multi-step thought flow across 5 cognitive archetype basins (*Logos*, *Mythos*, *Ethos*, *Telos*, *Pathos*), calculating step velocities, turning curvatures, path length, and dispersion entropy. Exposes `POST /v1/resonance/trajectory`.
- **NexusMind Studio v84 Single-Page Interface (`nexus_studio.html`)**: Interactive tabs and HTML5 Canvas visualizers for Quantum Density heatmap & eigenvalue spectrum, Rule 110 Glider spacetime diagram, 5D Cognitive Trajectory vector radar, and Speculative Tree branching graph.
- **Epistemic Invariant Guarantee**: All new engines operate strictly under the `analysis_only` boundary (`answer_authority: false`). See [`docs/V84_AUTONOMOUS_EPISTEMIC_MULTIMODAL_FRONTIER.md`](docs/V84_AUTONOMOUS_EPISTEMIC_MULTIMODAL_FRONTIER.md).

## Supermix v83 — Unified Hybrid Frontier (Xiaomi MiMo + AI-Dem-Lab + Supermix)


V83 unifies Xiaomi MiMo's advanced 2026 architecture (sparse MoE, hybrid sliding window + global attention, multi-token prediction heads, and multimodal token projection) with AI-Dem-Lab's deep research sandboxes (Quantum Bell CHSH locality test, Wolfram computational universe complexity analyzer, 5D semantic resonance cognitive archetype mapping, Compare Bench with continuous auto-looping) into Supermix's verified evidence-first runtime.

Key advancements in v83:
- **Multimodal Token Projection (`mimomix_core.py`)**: `MultimodalProjectionHead` pre-normalizes via `RMSNorm` and projects continuous visual/audio feature embeddings into transformer sequence representations.
- **Quantum Bell Sandbox (`nexus_engine.py`)**: Simulates the CHSH Bell inequality test, evaluating analytical quantum correlations violating local hidden variable limits ($S = 2\sqrt{2} \approx 2.8284 > 2.0$) against classical Monte Carlo bounds ($S \le 2.0$). Exposes `POST /v1/quantum/bell`.
- **Wolfram Complexity Analyzer (`nexus_engine.py`)**: Quantifies ECA dynamics across Rules 0-255 using Langton's $\lambda$, active site density evolution, and spatial Shannon entropy, mapping to Wolfram Classes 1-4.
- **Semantic Resonance Radar (`nexus_engine.py`)**: Maps queries into 5 cognitive archetype basins (`logos`, `mythos`, `ethos`, `telos`, `pathos`) with Dirichlet smoothing and pentagonal simplex projection. Exposes `POST /v1/resonance`.
- **Compare Bench & Auto-Loop Engine (`nexus_engine.py`)**: Side-by-side mode/prompt execution with character 3-gram Jensen-Shannon Divergence (JSD), semantic distance, differential latency ($\Delta\%$), and continuous auto-looping. Exposes `POST /v1/compare`.
- **NexusMind Studio v83 Single-Page Interface (`nexus_studio.html`)**: Interactive tabs and HTML5 Canvas visualizers for Compare Bench, Quantum Bell CHSH, and Semantic Resonance radar.
- **Epistemic Invariant Preserved**: All exploratory sandboxes operate strictly under the `analysis_only` boundary; only `grounding_runtime.finalize_grounded_response` with valid SHA-256 nonces holds `answer_authority: true`. See [`docs/V83_NEXUS_MIMO_DEMLAB_HYBRID_FRONTIER.md`](docs/V83_NEXUS_MIMO_DEMLAB_HYBRID_FRONTIER.md).

## Supermix v82 — calibrated verify-or-defer lab


V82 adds a research-grounded, shadow-only reliability lab. A frozen 128-case
arithmetic/adversarial cohort is evaluated across a precommitted `{1, 2, 4,
8}` policy grid with exact one-sided Clopper–Pearson bounds and Bonferroni
correction. Hash-bound receipts record the benchmark, runtime sources, policy
matrix, coverage, and risk interval; explicit assumptions and authority bits
remain false. The `/v1/risk-control` endpoints and Studio Verify-or-Defer tab
expose this evidence for inspection, but never change routing, activation,
promotion, or answer authority.

The public adaptive mode is correspondingly honest: authored Q/RSI signals are
reported as shadow recommendations, while only an explicit request or fixed
safe default controls the applied ACT cap. Observed cycles and exit reasons are
reported from the forward telemetry, and optional attention/MoD/MLA mechanisms
are not claimed active unless observed. See
[`docs/NEXUS_CALIBRATED_VERIFY_OR_DEFER.md`](docs/NEXUS_CALIBRATED_VERIFY_OR_DEFER.md).

V82.1 hardens the presentation boundary across Chat, Solver, Scientific, Think,
SSE, and `/v1/verify`: authoritative answers require a valid 16-128 character
ASCII nonce, every proof capsule requires a passing independent witness, and
the bounded in-memory/SQLite freshness ledgers fail closed at capacity without
evicting live entries. `/health` exposes these policy flags for deployment
inspection. The full repository suite remains the release gate; no model,
adapter, active pointer, or promotion receipt is changed by this work.

The next retrieval-facing layer is additive and shadow-only: the source-locked
temporal evidence ledger records immutable server snapshots, opened spans,
typed sentence provenance, freshness, conflicts, and explicit revisions without
turning retrieved text into authority. Snapshot admission requires an immutable,
ledger-bound server-fetch receipt. Its v2 claim manifest binds every generated
sentence before attribution, rejects caller-asserted checker flags, and accepts
inference verification only through an executed, allowlisted, ledger-bound
checker receipt. Incomplete or unverified coverage defers, while SQLite trigger,
foreign-key, content, turn, claim, output, and receipt integrity failures surface
as degraded health. It remains local shadow telemetry—not a signed transparency
log or answer-authority path. See
[`docs/NEXUS_SOURCE_LOCKED_EVIDENCE_LEDGER.md`](docs/NEXUS_SOURCE_LOCKED_EVIDENCE_LEDGER.md).

## Supermix v80 — experimental hybrid diagnostics

V80 keeps three research lineages behind the same evidence boundary:

- **MiMo architecture probe:** sparse MoE, hybrid local/global attention,
  attention sinks, MTP heads, and bounded latent cycles are implemented as
  newly initialized diagnostics. No Nexus text checkpoint is loaded, and no
  cache-reduction or decoding-speed claim is made for this path.
- **AI-Dem-Lab diagnostics:** explicit software sources (`crypto`, `seeded`,
  `os_csprng_transform`, and `chaotic`), a synthetic numeric-sequence RSI
  probe, a disconnected tabular Q-learning experiment, and deterministic
  five-role swarm scaffolding. These are not live reasoning-quality signals,
  quantum hardware, calibrated routing, or verified answers.
- **Supermix grounding core:** fresh closed-world arithmetic and allowlisted
  science recomputation remains the only answer-authority path. V78.2 also
  binds every displayed numeric span to the request, exact output, verifier
  receipt, answer surface, and browser nonce, then requires a second live
  verification before the Studio reveals it. Checksums are still not
  signatures or factual proof.

See [the bounded v80 architecture notes](docs/V80_UNIFIED_HYBRID_FRONTIER.md).

## Supermix v78.2 — proof-carrying conversation

V78.2 extends the v78.1 answer-admission boundary into the presentation layer.
Every Nexus result remains exactly one of `answered`, `analysis_only`, or
`abstained`, but authoritative results now carry a closed-schema
`nexus-selective-answer-v2` receipt and a
`nexus-proof-carrying-number-v2` capsule. Every authoritative capsule must carry
a passing algorithmically independent witness; today those witnesses cover
scoped arithmetic and allowlisted science, while unsupported grounded families
defer instead of receiving server-only authority.

Only a fresh call to `grounding_runtime.finalize_grounded_response` may return
`answered`, and only for accepted exact arithmetic or an allowlisted scientific
scenario. Every numeric span in the public output must be classified as a
derived answer, an input echo, a verified unit literal, or a verified derivation
literal. An unrelated appended number, Unicode numeric confusable, stale
runtime/schema, unknown field, changed query, changed display, changed output,
changed answer surface, or changed request nonce fails closed.

The Studio generates a 128-bit request nonce and withholds an answer until
`POST /v1/verify` repeats the strict grounding pass and exactly matches the
entire expected capsule. A valid 16-128 character ASCII nonce is mandatory for
every public exact answer and verification; missing or malformed nonces fail
closed before verification. Accepted nonces are consumed in a bounded
process-local freshness ledger to reject duplicate verification. Think
also supports an SSE `stream=true` transport whose ordered chunks end with the
same proof capsule and still require `/v1/verify`. Exact math and allowlisted
science can therefore be answered directly in Chat, Solver, and Think;
open-world persona chat remains
`analysis_only`. Concurrent requests use per-surface sequence tokens so a stale
response cannot overwrite a newer turn. A verified view is reconstructed from
capsule-bound result fields in a fixed wrapper, so free-form candidate prose,
reasoning steps, and telemetry do not inherit the verified badge.

For multi-worker or restart-spanning freshness, launch with
`--verification-nonce-db path/to/nonces.sqlite` (or inject
`verification_nonce_db` into `NexusApiService`). The opt-in SQLite ledger is
WAL-backed and stores only nonce digests and timestamps; the default remains
process-local and no prompts or answers are written. Both backends preserve all
unexpired entries and reject new verification when full, so capacity pressure
cannot silently make an accepted nonce reusable.

The contract deliberately publishes `confidence=null`. A successful check is
labeled `deterministic_assurance_not_probability`: it is same-implementation
recomputation, not empirical calibration, algorithmic independence, a digital
signature, or open-world factual certainty. All tool, permission, safety,
memory, routing, activation, and promotion authority bits remain false.

The other recognizable experimental surfaces remain bounded: fast/deep are
newly initialized telemetry probes that abstain; agent executes no tools;
SCAMPER/TRIZ, swarm, and graph outputs are analysis-only; and unverified output
cannot update the public Q-policy. See
[`docs/NEXUS_PROOF_CARRYING_CONVERSATION.md`](docs/NEXUS_PROOF_CARRYING_CONVERSATION.md)
for the renderer protocol, threat cases, research mapping, and non-claims, and
[`docs/NEXUS_EVIDENCE_FIRST_SELECTIVE_ANSWERING.md`](docs/NEXUS_EVIDENCE_FIRST_SELECTIVE_ANSWERING.md)
for the underlying admission boundary.

```bash
# Focused Nexus evidence, risk-control, and compatibility suite
python -m pytest -q test_nexus_proof.py test_nexus_epistemics.py test_nexus_engine.py \
  test_nexus_api.py test_nexus_swarm.py test_nexus_got.py \
  test_nexus_ideation.py test_nexus_chat.py test_nexus_solver.py \
  test_nexus_studio_contract.py test_nexus_hybrid_advancements.py \
  test_nexus_adaptive.py test_nexus_risk_control.py \
  test_nexus_independent_checker.py test_nexus_nonce_ledger.py \
  test_nexus_evidence_ledger.py \
  test_nexus_compare.py test_nexus_quantum_bell.py \
  test_nexus_resonance.py test_nexus_studio_v83.py \
  test_nexus_studio_v84.py test_nexus_v84_innovations.py \
  test_mimomix_multimodal.py \
  test_mimomix_differential.py test_mimomix_mla.py test_mimomix_mod.py

# Start interactive CLI terminal
python source/nexus_cli.py

# Launch the experimental REST API server
python source/nexus_api.py --port 8000
# Then open http://127.0.0.1:8000/studio
```

## Supermix v72 — NexusMind: Unified Hybrid Thinking Architecture (new, additive)

> Historical architecture description. The v78.2 evidence contract above
> supersedes any “production-grade,” answer-confidence, context-window,
> verification, or live-telemetry implication in this section. Swarm and GoT
> defaults are deterministic analysis scaffolds; the neural core is not a
> checkpoint-backed text generator.

V72 originally assembled three experimental research lineages behind one interface:

* **Xiaomi MiMo Frontier Architecture** (MiMo-V2-Flash & MiMo-V2.5-Pro): Hybrid Sliding-Window Attention (SWA:GA 5:1 ratio) reducing KV-cache footprint by ~6.2x, learnable attention sinks per head, auxiliary-loss-free sparse MoE load balancing with router z-loss, Multi-Token Prediction (MTP) self-speculative draft decoding, decoupled dual-base RoPE (YaRN/NTK context scaling up to 1M tokens), and unified Flash vs Pro routing.
* **Supermix Cognition Stack** (v51–v71): Weight-tied recursive latent ACT refinement with ponder cost halting, supervised quality & continue verifier, cross-budget ordered top-k agreement with exact output reuse, v56 latent state machine with row-stochastic log-space transition matrices, v70 multi-domain sparse expert routing, and v71 deterministic closed-world scientific scenario solver with exact rational SI arithmetic and cryptographic answer receipts.
* **AI-Dem-Lab Systems**: 5-Agent Cognitive Swarm (Generator, Critic, Skeptic, Archivist, Anomaly Hunter) with discrete replicator dynamics, Graph-of-Thoughts (GoT) multi-branch speculative search with prune-and-merge graph topology, closed-loop Tabular Q-Learning budget adaptation (`BudgetPolicyLearner`), and the complete Dem-Lab statistical telemetry battery (Shannon/min-entropy, chi-square uniformity p-values, runs & monobit tests, CHSH Bell inequality validation, and RSI momentum meters).

### Historical v72 command examples

```bash
# Run the original four-file v72 subset (not the current focused suite)
python -m pytest test_nexus_swarm.py test_nexus_got.py test_nexus_engine.py test_nexus_api.py -v

# Start interactive CLI terminal
python source/nexus_cli.py

# Launch the experimental local REST API server
python source/nexus_api.py --port 8000
```

Open [`web_static/nexus_studio.html`](web_static/nexus_studio.html) through the
live API host for the experimental interface. If the backend is unavailable it
shows an explicit unavailable state and generates no local substitute result.

See [`docs/V72_NEXUSMIND_UNIFIED_ARCHITECTURE.md`](docs/V72_NEXUSMIND_UNIFIED_ARCHITECTURE.md) for complete mathematical specifications, formulas, and cryptographic receipt schemas.

## Supermix v71 - Verified Scientific Scenarios (new, additive)

V71 extends the shipped deterministic runtime with a strict scientific-plan
boundary. It currently supports six closed-world targets: final velocity and
displacement under explicitly constant acceleration, plus pressure, volume,
temperature, and amount under the explicitly stated ideal-gas model.

The parser accepts one target and every required labelled quantity, binds each
input to a hashed prompt span, normalises units into SI with exact rational
arithmetic, and executes only a versioned local formula registry. It never
evaluates model-generated code. Registry integrity, plan integrity, input
bindings, dimensions, domains, and substitution must all pass before the result
can replace a candidate answer. The answer and its receipt remain
model-conditional and carry no route, tool, permission, safety, compute, or
open-world authority.

V71 also closes two prediction trust gaps. `predicted` is now understood as a
prediction cue, `same success probability` no longer imports an unrelated prior
turn, syntactically plausible forecasts need a relevant basis and explicit
limit, and a verified empirical rate can replace a conflicting candidate only
with the canonical `not a guarantee / not calibrated` qualification. It still
does not become a fact about the next trial.

```bash
python -m pytest -q test_science_plan.py test_reasoning_engine.py \
  test_grounding_runtime.py test_interaction_planner.py test_prompt_understanding.py
python source/generate_studio_runtime_manifest.py --check
```

The source and compatibility runtime contract is `71.0.0`. Existing local
Windows binaries were not rebuilt by this source upgrade; see
[`docs/V71_VERIFIED_SCIENTIFIC_SCENARIOS.md`](docs/V71_VERIFIED_SCIENTIFIC_SCENARIOS.md)
for supported grammar, receipts, fail-closed cases, and non-claims.

## Supermix v77/v78 — a desktop build, and a tracker that could not see the training (new, additive)

**v77: the chat interface as a Windows application**, with v74 inside it —
1,735 MB (torch dominates), 62 MB exe, 33 MB bundled model. The Flask server is
unmodified, so the prompt normaliser and the independent answer check behave
exactly as they do on the web. `build_chat_desktop_installer.ps1` compiles a
single setup.exe with Inno Setup 6, and falls back to a zip plus a PowerShell
installer (per-user, shortcuts, Add/Remove Programs, real uninstaller) when
Inno Setup is absent. Verified by installing and running, not just building.

The first build **succeeded and produced an application that died on launch** —
`torch.distributions` was in the spec's `excludes` and torch's own `__init__`
imports it. PyInstaller reported success, the windowed build showed no error,
and it was only findable by redirecting stderr to a file. A test now parses the
spec and rejects any exclude containing a dot.

**v78: `training_monitor_gui.py` cannot see these runs.** The 5,185-line Tkinter
tracker parses the LoRA pipeline's `[train] step=... loss=...`; the
generalisation trainer emits `step 12000/18000  train 0.0839 ... acc 0.70`.
Against the v74 log: **13 step lines, 0 matched**. Every run since v58 has been
invisible to it, which is why v74 was tracked by hand.

`source/training_tracker.py` is the headless replacement, built around the
error that hand-tracking kept making — quoting the fastest recent interval as
the rate. Within one v74 run the observed rate ranged **1.98–5.73 s/step**. So
it uses the *median* of non-probe intervals, prices accuracy probes separately,
reports the ETA as a **range**, and states aloud when recent pace disagrees
with the run average. It refuses to estimate at all from a log that has stopped
moving:

```
v74  [stalled]
  step   11,500 / 18,000 (63.9%)
  eta    unknown - no new step for 14h21m
  note   recent pace is 79% slower than the run average (4.92 vs 2.76 s/step)
```

That `note` is the paging degradation immediately before the segfault — the
warning was in the log all along, and reading by eye did not catch it.

Details: [`V77_DESKTOP_BUILD.md`](V77_DESKTOP_BUILD.md),
[`V78_TRAINING_TRACKER.md`](V78_TRAINING_TRACKER.md).

## Supermix v74/v76 — ten task types, a recovered crash, and the prompt-format gap (new, additive)

**v74 is the best problem-solving model in this repo: 0.894 (447/500) against
v73's 0.756, z=5.74**, on a benchmark carrying *ten* task types where v73 was
measured on five.

| task | v74 | v73 | verdict |
|---|---|---|---|
| division | **1.00** | — | new |
| multiplication | **1.00** | — | new |
| sequence | **0.98** | — | new |
| two_step | **0.98** | — | new |
| word_problem | 0.96 | 0.99 | tie |
| algebra_one_step | 0.89 | 0.94 | tie |
| arithmetic | 0.89 | 0.99 | **v73 better (z=−2.81)** |
| percent | 0.75 | 0.70 | tie |
| average | **0.59** | 0.16 | **v74 better (z=5.54)** |

**Read the headline carefully.** The 0.894 is flattered by four new tasks v74
finds easy. Like for like on the five shared tasks it is 0.818 vs 0.756 at
z=1.99 — over the line by a hair, driven almost entirely by `average`. And
`arithmetic` genuinely regressed, 0.99 → 0.89.

**The run segfaulted at step 11,500 of 18,000**, the second long run to do so
(v64 was the first, both at a checkpoint boundary on a 15.6 GB box). Three
changes make that cheap rather than fatal: checkpoint writes are now **atomic**
(they were not — a fault mid-write would have destroyed the only recovery
point), `--start_step` rejoins the *same* OneCycle curve instead of re-warming,
and `train_supervised.py` restarts automatically while refusing to loop on a
failure that makes no progress. The resumed leg finished in one leg, exit 0.

Judging the run at the crash would have been wrong: at 64% trained, three
shared tasks looked badly damaged and two of the three were merely unfinished.

**v76: the benchmark said 0.894 and the chat said nonsense.** Both true. The
benchmark generates prompts in the corpus format (`What is 47 x 6?`); people
type `what is 47 times 6`, which answered 242. Probing isolated it to the
operator token and the presence of a lead-in — capitalisation and punctuation
do not matter, and a bare `47 x 6` is read as *algebra*.
`source/prompt_normaliser.py` maps the way people write onto the trained form:
**7/7 typed naturally, from 0/5.** It is presentation, not capability — it
computes nothing and never alters a number, and questions the model gets wrong
in the trained format stay wrong. Every rewrite is shown in the reply
(`asked as: ...`); `--no-normalise` disables it.

Details: [`V75_CRASH_RECOVERY.md`](V75_CRASH_RECOVERY.md),
[`V76_PROMPT_FORMAT.md`](V76_PROMPT_FORMAT.md).

## Supermix v73 — accuracy in the training loop, and a tie (new, additive)

**The process change is the durable part.** Every run in this line cost 12–17
hours and reported whether it worked only afterwards, because the loop tracked
dev loss — a metric that has now failed in both directions (v71: better loss, 28
points less accuracy; v72: worse loss, worse accuracy).

`--accuracy_every N` measures exact-match accuracy on **freshly generated**
problems mid-run, and `--select_on accuracy` picks the checkpoint that answers
correctly rather than the one that fits the corpus. Problems never come from
training, so a memorised answer scores zero. It tracked this run honestly:
0.15 → 0.50 → 0.60 across steps 8k–14k.

**The model is a tie.** v73 applied v71's decomposition at v72's sequence length,
decomposing only the averages that fit 128 so nothing is dropped:

| task | v70 | v73 | delta | ±95% CI |
| --- | --- | --- | --- | --- |
| `arithmetic` | 91.7% | **100%** | +8.3 | ±0.0 |
| `word_problem` | 100% | **100%** | 0.0 | ±0.0 |
| `percent` | 58.3% | **62.5%** | +4.2 | ±19.4 |
| `average` | **33.3%** | 25.0% | −8.3 | ±17.3 |
| **overall** | 75.0% | **75.8%** | **+0.8** | **±7.7** |

**Settled at n=500** (100 problems per task, generation only): v73 is
**significantly better at arithmetic** — 99/100 against 91/100, z=2.64 — and
`word_problem` points the same way at z=1.94. But the **overall difference stays
insignificant** (73.2% vs 75.6%, z=0.87), because `average` regressed 24.0% →
16.0% and cancelled gains on the other four tasks. Decomposition did **not** help
`average` at seq 128, reversing v71's seq-160 result; `average` remains the worst
task in both models by a factor of three.

**At n=120 the margin was 91 correct against 90 — one problem.** Every
per-task difference except `arithmetic` sits inside its own confidence interval.
The bar was "beat 75.0% with `average` above 33.3%"; v73 matched the first and
missed the second. Calling this an improvement would be the over-claiming this
line has already corrected twice.

**Two flaws in the process change, found by using it.** The 20-problem probe
carries ±10 points — fine for aborting a run, inadequate for selecting within
one (it read 0.15 where a 60-problem check read 0.467). And under
`--select_on accuracy` the partial checkpoint writes only on accuracy
improvement, so this run went **6.3 hours without one**, weakening the v63 crash
protection. Neither is fixed yet.

Details: [`docs/V73_ACCURACY_IN_THE_LOOP.md`](docs/V73_ACCURACY_IN_THE_LOOP.md).

## Supermix v72 — sequence length caused v71's collapse, not the scratchpad (new, additive)

v71 decomposed two task formats, got worse at everything, and I attributed it to
scratchpad length. **That attribution was wrong.** v72 is v70's exact corpus at
v71's sequence length — one variable — and it reproduces most of the collapse.

| task | v70 (seq 128) | v72 (seq 160) | v71 (seq 160 + decomposed) |
| --- | --- | --- | --- |
| `algebra_one_step` | **91.7%** | 62.5% | 33.3% |
| `arithmetic` | **91.7%** | 58.3% | 62.5% |
| `average` | **33.3%** | 4.2% | 16.7% |
| `word_problem` | **100%** | 66.7% | 58.3% |
| **overall** | **75.0%** | 50.8% | 46.7% |

Of the 28.3-point fall from v70 to v71: **24.2 points (86%) is sequence length
alone**, 4.1 is the formats. And at matched length the decomposition **helped**
the task it targeted — `average` 4.2% → **16.7%**, four times better, exactly as
v68's rule predicted. v71 read as a failure only because the length change buried
it; I compared across two variables and drew a conclusion anyway.

**Padding dilution is ruled out** — turn-aligned packing at 160 gives *more*
supervised tokens per block (39.3 vs 32.4), not fewer. The cause is untested;
sliding-window attention (`sliding_window=64`) and padding-position count are the
two candidates.

**The practical rule: use the shortest sequence length that fits your turns.** A
25% length increase cost a quarter of this model's task accuracy, so sequence
length is a tuned hyperparameter here, not a safe upper bound.

**v70 remains the model to use, at 75.0%.** Details:
[`docs/V72_SEQUENCE_LENGTH.md`](docs/V72_SEQUENCE_LENGTH.md).

## Supermix v71 — a scratchpad can be too long, and lower loss proved it (new, additive)

V70's two worst tasks still had one-shot steps, so v71 decomposed them per v68's
rule. **It made the model worse at everything, while fitting its corpus better.**

| task | v70 | v71 |
| --- | --- | --- |
| `arithmetic` | **91.7%** | 62.5% |
| `algebra_one_step` | **91.7%** | 33.3% |
| `word_problem` | **100%** | 58.3% |
| `average` | **33.3%** | 16.7% |
| **overall** | **75.0%** | 46.7% |
| final dev loss | 0.0783 | **0.0613** |

**The lower loss is the finding.** v64 showed loss preferring recitation, measured
against a recall proxy. This shows loss preferring *worse task performance*
against an objective correctness metric, where a wrong answer is simply wrong.
Decomposed text is formulaic and so cheap per token; accuracy needs **every step
in the chain** right.

Operations per answer on `average` went 0 → 5.0. Treating success as `p ** n`,
per-operation reliability **improved** (0.844 → 0.856) while total accuracy
**halved** — the decomposition worked and the answer still got worse, because
there were twice as many steps to survive.

**This conclusion was later overturned by v72**, which showed 86% of the
regression was the sequence-length change rather than chain length. See the v72
section above.

Unattributed: `arithmetic`, `algebra` and `word_problem` have byte-identical
formats in both corpora and all regressed, which chain length cannot explain.
Capacity and the sequence-length change (128 → 160) are both candidates; no
matched arm was run.

**v70 remains the model to use.** Details:
[`docs/V71_SCRATCHPAD_LENGTH.md`](docs/V71_SCRATCHPAD_LENGTH.md).

## Supermix v70 — sparse experts beat both specialists (new, additive)

V69 paid 25 points of maths to unify chat and arithmetic. V61 had already found
extra capacity useless — but on a **single-domain** corpus, while also measuring
that MoE routing genuinely specialises (3.15x costlier to destroy at 32 experts
than 8). The prediction, stated before this run: sparse experts should pay when
there are two domains to specialise *into*.

| task | v68 (maths only) | v69 (8 experts) | **v70 (32 experts)** |
| --- | --- | --- | --- |
| `word_problem` | **100%** | 41.7% | **100%** |
| `arithmetic` | **91.7%** | 41.7% | **91.7%** |
| `algebra_one_step` | 79.2% | 50.0% | **91.7%** |
| `average` | 0% | 4.2% | **33.3%** |
| `percent` | 54.2% | **62.5%** | 58.3% |
| **overall** | 65.0% | 40.0% | **75.0%** |
| holds a chat register | no | yes | **yes** |

**v70 beats the maths specialist by 10 points while also holding a conversation.**
`average` — 0% in v68, the task whose scratchpad lists results without
decomposing each step — reaches 33.3%, with every partial sum correct:
`sum: 40 then 100 then 120 then 200, total 200, divide by 4, total 50.0`.

The capacity was nearly free: **1.87x total parameters for +0.7% active compute
and +14% wall-clock**, because `top_k=2` leaves most of it dark per token.

Chat is unchanged and still **recitation** (`largely_recalled, 1.00`) next to
arithmetic that is **`mostly_novel, 0.00`**. Sparse experts bought computation,
not composition.

Attribution is limited: experts, maths volume (200k→300k) and steps (14k→16k) all
changed together, and no matched 8-expert arm was run. This also does not refute
v61 — capacity is plausibly useless without something to specialise into, and
valuable with it.

**`output/v70_moe/v70_moe.pt` supersedes v68 and v69.**

Details and non-claims: [`docs/V70_SPARSE_EXPERTS.md`](docs/V70_SPARSE_EXPERTS.md).

## Supermix v69 — one model does both, and is worse at each than the specialist (new, additive)

V69 combined the dialogue corpus that made v60 fluent with the scratchpad
arithmetic that took v68 to 65%, using every fix from this session. **The
unification worked; "better in most ways" did not.**

| | v68 (maths) | v60 (chat) | v69 (unified) |
| --- | --- | --- | --- |
| maths, overall | **65.0%** | — | 40.0% |
| arithmetic | **91.7%** | — | 41.7% |
| word_problem | **100%** | — | 41.7% |
| percent | 54.2% | — | **62.5%** |
| holds a chat register | no | yes | **yes** |

It answers both kinds of prompt in one model — `why is my script failing` →
*"Check the traceback first…"*, and `617 + 288` → `600 + 200 = 800, 17 + 88 =
105, total 905` — and its chat is indistinguishable from v60's. But the capacity
split flagged before the run is exactly what happened, and cost 25 aggregate
points. Both ingredients were individually learnable and it still cost that, so
the constraint is **capacity, not the unlearnability of one ingredient**.

**The honest headline is the recitation result.** Scored against its own corpus:

| prompt | verdict |
| --- | --- |
| hello | largely_recalled, **1.00** |
| why is my script failing | largely_recalled, **1.00** |
| `617 + 288` | **mostly_novel, 0.00** |

The fluent half is recall; the computed half is not — the clearest demonstration
yet of what the recall meter was built for. It applies equally to v60, whose
replies come from the same corpus: **v60's apparent fluency was always
recitation**, and v69 makes that visible by sitting it next to arithmetic that
isn't.

Use `v68_average_fix` for maths, `v60_control_2000` or v69 for chat, and v69 only
when both are needed in one process.

Details and non-claims: [`docs/V69_UNIFIED.md`](docs/V69_UNIFIED.md).

## Supermix v68 — 65% overall, and a scratchpad only helps where it decomposes (new, additive)

| task | v66 | v67 | v68 |
| --- | --- | --- | --- |
| `word_problem` | 0% *(absent)* | 41.7% | **100%** |
| `arithmetic` | 55% | 41.7% | **91.7%** |
| `algebra_one_step` | 0% *(absent)* | 66.7% | **79.2%** |
| `percent` | 65% | 62.5% | 54.2% |
| `average` | 0% | 0% | **0%** |
| **overall** | 24% | 42.5% | **65.0%** |
| median relative error | 0.179 | 0.004 | **0.000** |

The typical answer is now exactly right. **Two predictions I made before the run
were wrong**: `average` did not move off 0%, and `arithmetic` did not stay near
41.7% — it went to 91.7%. I also called this a single-variable experiment and it
was not; lengthening the average rows shifted packing and the token distribution
for every task, so the arithmetic gain is real but **unattributed**.

**Why `average` still fails is now exact.** The coverage fix worked — six-number
problems get six terms, `divide by 6`, and `327 / 6 = 54.166667` computed exactly.
Counting and division are right. The running sum is not: `107 + 5 = 155`.

Compare the two formats:

```
arithmetic:  900 + 700 = 1600, 87 + 2 = 89, total 1689   <- each sum decomposed
average:     sum: 72 then 107 then 155 then ...          <- each sum one-shot
```

**A scratchpad helps only where it decomposes the operation.** Listing
intermediate *results* is not showing intermediate *work*. `average` is the
control that proves it: same model, same corpus, same training — one task whose
steps are decomposed scores 91.7%, one whose steps are not scores 0%. A model
doing 3-digit addition at 91.7% fails at 2-digit running sums purely because of
how the steps are written.

Details and non-claims:
[`docs/V68_AVERAGE_AND_THE_LIMIT_OF_A_SCRATCHPAD.md`](docs/V68_AVERAGE_AND_THE_LIMIT_OF_A_SCRATCHPAD.md).

## Supermix v67 — the scratchpad format transfers (new, additive)

V66 left two task types at 0% because its corpus did not contain them. V67 adds
them, same benchmark, same scorer, 120 novel problems:

| task | v66 | v67 |
| --- | --- | --- |
| `algebra_one_step` | 0% *(absent)* | **66.7%** |
| `word_problem` | 0% *(absent)* | **41.7%** |
| `percent` | 65% | 62.5% |
| `arithmetic` | 55% | 41.7% |
| `average` | 0% | **0%** |
| **overall** | **24%** | **42.5%** |
| median relative error | 0.179 | **0.004** |
| within 10% of truth | 44% | **85%** |

**The format transfers**: two untrained task types went to 66.7% and 41.7% just
by writing their working the same way. Median relative error fell **45x** — the
model is now typically within 0.4% of the true answer.

`arithmetic` regressed 13 points. Six task types share the same 300,000 rows and
12,000 steps that four had, so each got less data and less capacity — the same
breadth-versus-depth trade v62 paid on prose.

**I was wrong about `average`.** I predicted v66's zero came from the corpus
rounding to four decimals against a same-sized tolerance. That fix went in and
average is *still* 0%. Reading the working shows two real causes, one of them
mine: `_scratchpad_average` emits 4–5 values while the benchmark tests 4–6, so
every six-number problem is out of distribution and the model truncates to five
and divides by five. The rest drift on the running sum — a six-term chain has
five places to slip, against one for a place-value split, which is why the
arithmetically simplest task is structurally the hardest here.

Details and non-claims: [`docs/V67_SCRATCHPAD_COVERAGE.md`](docs/V67_SCRATCHPAD_COVERAGE.md).

## Supermix v66 — showing the working took exact arithmetic from 0% to 55% (new, additive)

V65 got the model answering at the right magnitude with the wrong digits, which
is what a model produces when it guesses an answer in one step instead of
computing it. V66 puts the intermediate work in the target.

Same benchmark, same novel problems, same scorer:

| task | v65 | v66 |
| --- | --- | --- |
| arithmetic (3-digit + and -) | 0% | **55%** |
| percent | 15% | **65%** |
| average | 0% | 0% |
| **combined, tasks both were trained on** | **5%** | **40%** |
| median relative error | 0.250 | **0.179** |
| within 10% of truth | 23% | **44%** |

`algebra_one_step` and `word_problem` score 0% and are **excluded** above because
v66's corpus does not contain them; including them gives 24%, an honest number
for the wrong question.

The scratchpad is a two-step place-value split, chosen because it is *always*
valid — `504 - 309` becomes `500 - 300 = 200, 4 - 9 = -5, total 195`, where a
column method would need borrow handling. The corpus was verified, not trusted:
20,000 rows independently recomputed, **0 incorrect answers**, every
addition/subtraction row checked for `high + low == total`.

The failures are procedural, which is the interesting part:

| prompt | reply | truth |
| --- | --- | --- |
| 987 + 702 | `900 + 700 = 1600, 87 + 2 = 90, total 1690` | 1689 |
| What is 12% of 1049? | `1 percent of 1049 = 10.49, times 10, total 104.9` | 125.88 |

The hundreds column is exactly right and `87 + 2` is off by one; one percent of
1049 is computed **exactly** and then multiplied by the wrong number. These are
slips inside a method being followed — a different failure from v65's, where
there was no method to slip in. The full structure appeared by **step 500 of
12,000**.

Details and non-claims: [`docs/V66_SCRATCHPAD_ARITHMETIC.md`](docs/V66_SCRATCHPAD_ARITHMETIC.md).

## Supermix v65 — arithmetic was unlearnable in principle (new, additive)

`mimomix_text.TOKEN_PATTERN` matches `\s*\d+`, so **`498` was a single opaque
token**. On a 240,000-row arithmetic corpus, **8,588 of 9,058 distinct tokens
(94.8%) were numbers**. Answering `498 - 419` therefore required a memorised
lookup from token(498) x token(419) to token(79), with no way to see the digits.
Arithmetic was not unlearned — it was **unrepresentable**.

`source/eval_problem_solving.py` proved it, and is the first metric here that
**recitation cannot fake**: a remembered answer to a fresh problem is just wrong.
It scores *seen* problems (verbatim from training) against *novel* ones (operands
generated at evaluation time). v64 scored **1.7% on both, gap 0.0000** — a zero
gap means it had not even memorised, because there was nothing memorisable.

`DIGIT_TOKEN_PATTERN` splits digit runs. Roundtrip is preserved and the setting
travels with the checkpoint, so a model cannot be reloaded under the other one
and silently re-segment every number. Default is off; every prior result stands.

| | whole numbers | digit tokens |
| --- | --- | --- |
| vocabulary | 16,390 | **874** |
| coverage | 0.9964 | **1.0000** |
| parameters | 9,257,385 | **3,132,585** |
| seconds/step | ~4.3 | **0.8** |

The 18x smaller softmax made training ~5x faster, so the fix pays for itself.
Measured over 200 problems:

| | v64 | v65 |
| --- | --- | --- |
| replies containing no number | **50 / 200** | **0 / 200** |
| novel exact accuracy | 1.0% | **4.0%** |
| novel median relative error | 0.621 | **0.250** |
| novel within 10% of truth | 9.0% | **23.0%** |

Exact accuracy is still near useless. The real change is that the model went from
emitting no number a quarter of the time to **always answering, in the right
format, at roughly the right magnitude** — `51.5` where the truth is `51.333`,
and `.25` decimals on four-number averages. That is approximate arithmetic, not
noise. The near-zero memorisation gap in both models confirms it is **computing
badly rather than reciting**.

Exact match alone could not have shown this (1% vs 4% looks like nothing), which
is why the benchmark also reports relative error.

Details and non-claims: [`docs/V65_DIGIT_TOKENS_AND_ACCURACY.md`](docs/V65_DIGIT_TOKENS_AND_ACCURACY.md).

## Supermix v64 — held-out loss is anti-correlated with generation quality (new, additive)

V64 gave the model a larger vocabulary (**32,774** against 16,390) and a corpus
led by real prose instead of generated dialogue. Its most useful result was not
either of those.

**Training it further made it worse at the thing it is for, while the loss
improved.** Same run, same corpus, same split; the only difference is 4,500 more
steps. Ten prompts, scored against the training corpus by the recall meter:

| | step 5,500 | step ~10,000 |
| --- | --- | --- |
| dev loss | 1.0762 | **0.9910** |
| mean verbatim rate | **0.14** | 0.76 |
| replies judged novel | **5 of 8** | 1 of 6 |
| degenerate replies | 2 of 10 | 4 of 10 |

Lower loss bought **5.4x more recitation** and twice the degeneracy. V63 showed
the same shape without a way to quantify it; the meter makes it a number, and it
replicates.

The cause is structural: verbatim reproduction of training text is the
**lowest-loss possible behaviour**, so perplexity does not merely fail to detect
recitation, it prefers it. **Do not select checkpoints on held-out loss in this
regime** — every promotion gate here that reads dev loss alone inherits the flaw,
including the one that chose step 10,000 over the better step 5,500.

The corpus work did land: 41,871 word types (3.7x v63), 187,351 distinct
sentences from 33% fewer rows, sentence repetition halved to 2.1x, and the most
common fragment down from 17.6% of rows to 4.4%. On real prose the ladder finally
measures something hard — unseen-sentence cost **+1.3801 nats**, ratio **2.750x**,
against +0.0955 for v60. That refines v60's finding rather than overturning it:
the gap measures **how compositional the corpus is**, not how capable the model is.

**Serve `output/v64_meaning/v64_meaning.partial.pt`** (step 5,500) rather than the
final checkpoint. It exists only because `--checkpoint_every_improvement` was
added in v63 — the run segfaulted at that step, and without mid-run checkpointing
there would be no v64 at all.

Details and non-claims: [`docs/V64_LOSS_VERSUS_QUALITY.md`](docs/V64_LOSS_VERSUS_QUALITY.md).

## The recall meter — telling a written reply from a remembered one (new, additive)

V63 ended on an uncomfortable fact: this line's most fluent output was not
composition. *"The moment hung in the air like a held breath"* reads like
writing, and appears verbatim in **51,022 of 289,169** training rows. A chat
interface that prints that sentence unannotated is presenting recall as
generation, which is the most misleading thing it could do.

`source/recall_index.py` indexes every 8-word window of a corpus and scores each
reply against it. The chat app takes `--corpus` and shows a badge per reply:

```bash
python source/mimomix_talk_web_app.py --checkpoint output/v63_aligned/v63_aligned.pt \
  --corpus datasets/v63/v63_coherent.jsonl --port 8764
```

| reply | verdict | verbatim | longest run |
| --- | --- | --- | --- |
| "…with vivid storytelling] The moment hung in the air…" | **RECALLED** | 77% | **27 words** |
| "Let me work through this step by step. Answer: 25.75…" | part recalled | 5% | 8 words |
| the memory analogy | **RECALLED** | 74% | **39 words** |

The two replies that *sound* impressive are recitation, one of them a 39-word
unbroken quote; the only reply that is genuinely the model's own work is the
wrong arithmetic.

Two numbers are reported rather than one, because an average hides the
difference: a scattered 70% of common phrases is ordinary language use, while a
single 39-word run is recitation. Only assistant text is indexed -- indexing
prompts would score a reply as recalled for echoing the user -- and without
`--corpus` the field is `null`, meaning *not checked*, which is deliberately not
the same as "checked and found novel". Hash collisions would overstate recall
rather than understate it; nothing here rounds in the flattering direction.

## Supermix v63 — a six-version training bug, and the corpus ceiling (new, additive)

V62's replies ignored the prompt. The diagnosis looked like undertraining; it was
not. `build_training_tensors` concatenates every turn into one stream and cuts it
on a fixed stride, blind to turn boundaries, so a block can start mid-reply with
its prompt in the *previous* block. Measured on the v63 corpus:

| | stream packing (default) | turn-aligned |
| --- | --- | --- |
| **supervised tokens with no prompt in their block** | **56.0%** | **0.0%** |
| blocks with no turn start at all | 879 of 21,673 | 0 |

**Over half of every gradient step taught the model to continue a reply without
having seen the question.** It stayed invisible for six versions because on a
templated corpus the modal reply is usually the right one — v57 through v60
trained this way and looked fine. `--turn_aligned_packing` fixes it (default off,
so every prior result reproduces).

It worked, and it was not enough. The run reached the best numbers in the repo —
dev **0.1230**, tier perplexities 1.0765 / 1.0712 / 1.2205 — and register
conditioning visibly improved: storytelling prompts get storytelling, arithmetic
prompts get numeric formats. Content did not follow, because the corpus is the
ceiling. Its most fluent output is recall, not composition:

| fragment | rows containing it, of 289,169 |
| --- | --- |
| `The moment hung in the air like a held breath` | **51,022 (17.6%)** |
| `It was the kind of moment that divides life into 'before' and 'after.'` | 10,261 (3.5%) |

800,231 sentence instances from 179,880 distinct sentences — 4.4 repeats each —
with generator scaffolding (`[strategic-set2]`, `(real-world-set2 genre variant)`)
left in the text. **Perplexity 1.14 is the right score for memorising a corpus
that is 17.6% one sentence.** Generation at step 3,500 (dev 0.2030) was *more*
coherent than at 12,000 (dev 0.1230): driving loss down drove the model deeper
into the templates.

Three further defects fixed here, each found by testing rather than assumed:
optimizer state is now saved and restored (v62's resume cost ~1,500 steps
re-warming); `--checkpoint_every_improvement` means a kill no longer loses
everything; and `OneCycleLR` crashed on any run of ≤10 steps, which is why this
trainer never had a cheap smoke test — a 2-step run now takes 2 seconds.

Details and non-claims:
[`docs/V63_TURN_ALIGNED_PACKING.md`](docs/V63_TURN_ALIGNED_PACKING.md).

## Supermix v62 — breadth bought, coherence not (new, additive)

V61 showed the system is data-limited, so v62 replaced the corpus and held the
model identical: **239,063 rows and 40,810 word types** across 11 capped domains
(logic, creativity, conversation, writing, maths, scripture, vocabulary, science,
coding), against v60's 96,227 rows and 5,280 types.

**What was bought.** Vocabulary reachability — the fraction of a domain's words
the model can represent at all, and a hard ceiling on what it can say, since an
out-of-vocabulary word can never be generated:

| domain | v60_control_2000 | v62 |
| --- | --- | --- |
| scripture | 0.6657 | **0.9641** |
| writing | 0.7175 | **0.9654** |
| logic | 0.8721 | **0.9987** |
| maths | 0.8920 | **0.9374** |
| **coding** | **1.0000** | **0.6900** |

Nine domains up; **coding regressed hard**, the direct cost of capping it at 380
rows for balance.

**What was not bought.** A 6,000-step continuation ran **17.5 hours** to reach
0.71 epochs. Dev loss fell 0.8919 → 0.7531 and every domain improved — and the
model still answers "hello" and "tell me a story" with the same worked-solution
template. It has learned the *format* of arithmetic (a bare number) and not the
function: `17 + 25` → `31`.

**The finding.** What it learns tracks whether the text was generated:

| domain | word types | perplexity @8k |
| --- | --- | --- |
| creativity | 1,215 | **1.26** |
| logic | 3,471 | **1.27** |
| conversation | 3,806 | **1.39** |
| scripture | 7,912 | 12.67 |
| writing | 25,889 | **18.65** |
| vocabulary | 8,112 | **19.09** |

The three domains it masters are the three assembled from generator templates;
the ones it fails contain real human text. **At this scale the model learns
templates, not language** — which reframes v57's headline 1.27 as memorising 192
sentences, since v62 reproduces 1.26 on the templated portion while scoring 18.65
on the literary portion beside it.

Also fixed here: an 8-character minimum response length inherited from the chat
loader was **silently deleting 73.5% of arithmetic rows** (`"79"` and `"9/14"`
are answers, not truncations), and `--init_from` now allows a continuation
without discarding finished compute, verifying vocabulary identity so a mismatch
raises instead of training a corrupted embedding.

**`v60_control_2000` remains the checkpoint to serve** — narrow but fluent. v62
is broader and coherent nowhere.

Details and non-claims: [`docs/V62_MULTIDOMAIN.md`](docs/V62_MULTIDOMAIN.md).

## Supermix v61 — 3.18× the parameters, and what they did not buy (new, additive)

V61 scaled the mechanism v59 identified as load-bearing: 8 routed experts to 32,
4 layers to 6, **4,988,073 parameters to 15,883,701** — for only 1.26× active
compute, since `top_k=2` leaves most of the new capacity dark on any given token.

On held-out loss it bought **nothing measurable**. All three runs share a corpus,
split seed and tokenizer, so the losses are comparable by construction:

| run | total params | steps | tier1 | tier2 | tier3 | dev |
| --- | --- | --- | --- | --- | --- | --- |
| `v60_diverse` | 4,988,073 | 1,000 | 0.2260 | 0.2180 | 0.3135 | 0.2852 |
| `v61_scaled` | **15,883,701** | 2,000 | 0.1856 | 0.2016 | 0.2873 | 0.2531 |
| `v60_control_2000` | 4,988,073 | 2,000 | 0.1870 | 0.2058 | 0.2860 | **0.2512** |

Every v61-vs-control difference is 0.0013–0.0042 nats **with inconsistent sign** —
the signature v59 identified as noise, at the same magnitude, now arguing against
the conclusion this version was built to reach. The whole v60 → v61 gain came from
**doubling the schedule**, not from tripling the parameters; at matched 1,000
steps the bigger model was *worse* (dev 0.3170 vs 0.2852).

The capacity is not inert, though — it is used, and just does not help. Audited on
both 2,000-step checkpoints, destroying learned routing costs **3.15× more** at 32
experts than at 8 (+0.15298 vs +0.04854 nats). The experts specialised and predict
the same tokens equally well. **Specialisation is not capability**: this corpus at
this budget is limited by schedule and data, not capacity.

The generalisation gap also closes completely with the longer schedule:

| checkpoint | corpus | steps | unseen − seen |
| --- | --- | --- | --- |
| `v58_full` | 292 word types | 1,000 | **+0.2309** |
| `v60_diverse` | 10,538 types | 1,000 | +0.0043 |
| `v60_control_2000` | 10,538 types | 2,000 | **+0.0002** |
| `v61_scaled` | 10,538 types | 2,000 | −0.0008 |

At 2,000 steps a model predicts sentences it has never seen exactly as well as the
ones beside them in the same response (perplexity ratio 1.000).

**Practical conclusion: train the small model longer rather than making it bigger.**
`v60_control_2000` matches a model 3.18× its size on every tier at 1.26× less
active compute, and is the checkpoint to prefer.

```bash
python source/compare_generalisation_runs.py output/v60_diverse output/v61_scaled output/v60_control_2000
python source/mechanism_causality.py --checkpoint output/v61_scaled/v61_scaled.pt --database artifacts/qwen_supermix_enhanced_v29_full_20260320_190817/prepared_train_pairs.jsonl
```

Details and the explicit non-claims:
[`docs/V61_SPARSE_CAPACITY.md`](docs/V61_SPARSE_CAPACITY.md).

## Supermix v60 — the ladder on a corpus with real language (new, additive)

V58's gate list opens with an item it could not retire: *"a corpus with measured
diversity beyond 292 word types."* The corpus was already on disk — 44 MB of
`artifacts/qwen_supermix_enhanced_v29_full_20260320_190817/prepared_train_pairs.jsonl`,
read by no trainer, already in the `(user, assistant)` shape the pipeline consumes:

| | `llm_chat.db` (v57/v58) | v29 pipeline corpus (v60) |
| --- | --- | --- |
| word types | **292** | **5,280** |
| distinct assistant sentences | **192** | **39,235** |
| trained vocabulary | 582 | **10,538** (coverage 1.0000) |

Retiring the gate changed the headline. V58's sharpest result scored withheld
sentences' own tokens against seen ones *inside the same rows*, reported
**+0.2309 nats**, and concluded the model is "markedly worse" at sentences it
never saw. Run under one command on both checkpoints:

| token set | v58_full (292 types) | v60_diverse (10,538 types) |
| --- | --- | --- |
| inside a **seen** sentence | 0.1880 | 0.2945 |
| inside a **withheld** sentence | 0.4188 | 0.2988 |
| **unseen − seen** | **+0.2309** | **+0.0043** |
| perplexity ratio | 1.260× | **1.004×** |

**53× smaller on real language.** Withholding a sentence from a 192-sentence
corpus removes something only memorisation can supply; withholding one from
37,958 removes nothing the model cannot rebuild from the rest of the language.
Row- and token-level measurements now disagree in direction — tier-3 rows cost
+0.0955 nats while their withheld sentences cost +0.0043 — so the difficulty is
spread across the row, not concentrated where v58's method assumed.

The largest confound is stated rather than buried: v58 withheld **15.6%** of its
sentence inventory (30 of 192), v60 only **1.7%** (640 of 37,958). Matching that
fraction is the experiment that would separate "diversity" from "how much
language was removed", and it has not been run.

Getting there required fixing the split machinery, which **raised** on real text
(`283 tier-2 row(s) contain a sentence absent from training` — 59% of tier 2).
The remedy is the one `verify_split` named in its own error message. It moves
**0 rows** on v58's corpus and reproduces its published tier sizes
(1,883 / 468 / 2,438) exactly.

```bash
python source/train_mimomix_generalisation.py --steps 1000 --run_name v60_diverse --output_dir output/v60_diverse --corpus_jsonl artifacts/qwen_supermix_enhanced_v29_full_20260320_190817/prepared_train_pairs.jsonl
python source/eval_mimomix_unseen_sentences.py --run_dir output/v58_full --run_dir output/v60_diverse
```

Details, the non-monotonic ladder, and the explicit non-claims:
[`docs/V60_DIVERSE_CORPUS_LADDER.md`](docs/V60_DIVERSE_CORPUS_LADDER.md).

## Supermix v59 — the mechanism causality audit (new, additive)

V58 ablated the recursive thinking core by training two arms, reported tier
deltas between 0.0006 and 0.007 nats, and said the result sat below a noise floor
that *"has not itself been quantified"*. V59 measures the mechanism instead of
the floor, and the mechanism turns out to be **inert**:

| | v58_full checkpoint |
| --- | --- |
| Δ held-out loss, thinking core on vs. off | **+8.84e-08 nats** |
| held-out predictions changed | **0 of 12,192** |
| smallest delta v58 reported as a finding | 5.9e-04 nats |
| ratio | **6,673×** |

V58's two arms differ in baseline loss by 9.5e-04 nats — roughly **10,700×** the
entire causal contribution of the mechanism they differ by. The arms were
functionally the same model, so those deltas measured run-to-run variance.
V58's conclusion stands; its stated reason does not, and the caveat is retired.

The whole core is gated by one scalar, `RecursiveThinkingCore.residual_scale`
(`refined = flat + scale * residual_mixture`), initialised to exactly `0.0` and
still only `6.410e-04` after v58's 1,000 steps. The obvious hypothesis — that a
zero start starves the core of gradient, since the gate multiplies its own
gradient too — is **wrong, and v59 tested it rather than asserting it.** Two
matched 400-step arms differing only in that initial value:

| | gate init 0.0 | gate init 0.1 |
| --- | --- | --- |
| final `residual_scale` | **−2.082e-02** | +4.596e-02 |
| decisions changed | **35 / 12,192** | 8 / 12,192 |
| verdict | **active** | **active** |

From exactly zero the gate reached 32× the magnitude v58 managed in 1,000 steps,
and the core came out causally *active*; warm-starting made it **less** active,
not more. So the initialisation is not the cause and the knob is not a fix. Why
v58's particular configuration held the gate near zero is **open work**.

`source/mechanism_causality.py` intervenes on one mechanism at a time in a fixed
checkpoint and re-scores the same tokens, so everything but the mechanism is held
bit-identical — which a retraining ablation cannot do:

```bash
python source/mechanism_causality.py --checkpoint output/v58_full/v58_full.pt --output output/v59_causality/v58_full_causality.json
```

| mechanism | Δ nats | decisions changed | verdict |
| --- | --- | --- | --- |
| `moe_routing_inverted` | +5.756e-02 | 1,056 / 12,192 | active |
| `moe_routing_random` | +4.160e-02 | 905 / 12,192 | active |
| `moe_shared_expert` | +4.460e-03 | 595 / 12,192 | active |
| `thinking_core` | +8.841e-08 | 0 / 12,192 | **inert** |
| `mtp_main_path_leak` | +0.000e+00 | 0 / 12,192 | inert (expected) |

**Sparse-MoE routing is the load-bearing mechanism of the v53 stack** — 470,000×
the thinking core's contribution. A mechanism counts as active only if it changes
an argmax *or* clears 5.9e-04 nats; the numerical floor alone (3.68e-09) is tight
enough to call dead mechanisms live, and decisions alone call live mechanisms
dead on a degenerate model. The routing interventions run through a reimplemented
MoE forward that must reproduce the baseline bit-exactly first, and a test
sabotages that rebuild by 5% to prove the check can fail.

`MiMoMixConfig.thinking_residual_init` (default `0.0`, so every pre-v59
checkpoint is reproduced exactly) exposes the gate's starting value as
`--thinking_residual_init`. It is the instrument that produced the negative
result above, **not** an improvement — on this evidence there is no reason to
set it above zero.

V59 also applies the same question to the test suite, and finds the same shape of
answer: **30 functions across 20 `test_*.py` files are never collected by pytest**
(the default `python_functions = test*` does not match `smoke_test_*`), and
**17 of those files run zero assertions**. The worst case is 8 dark functions
inside `test_runtime_compute_controls.py` — a file CI runs and reports green on.
`source/dark_test_audit.py --check` pins the count as a baseline and fails on any
rise, so the debt can fall but not grow.

Details, the two supporting findings about the never-trained verifier head, and
the explicit non-claims:
[`docs/V59_MECHANISM_CAUSALITY.md`](docs/V59_MECHANISM_CAUSALITY.md).

## Supermix v58 — the generalisation ladder (new, additive)

V57 reported held-out perplexity **1.27** and said, in the same document, that
the number *"measures fit to a template distribution, not generalisation to
unseen language"*. The caveat was correct and unquantified. V58 quantifies it.

Measured first, because it is the reason v58 exists: under the v57 row split,
**1,875 of 2,400 validation responses (78.1%) appear verbatim in training**, and
the corpus's 37,543 distinct responses are compositions of just **192 distinct
sentences**. Splitting by row cannot produce an unseen response in any meaningful
sense.

So v58 withholds whole **sentences** from training and scores three tiers
separately, on the identical v57 architecture (3,076,521 parameters, vocabulary
582), 1,000 steps:

| tier | what it measures | rows | perplexity |
| --- | --- | --- | --- |
| `tier1_seen_response` | template recall — the response string was in training | 1,883 | **1.2622** |
| `tier2_unseen_response` | sentence recombination — novel response, every sentence seen | 468 | **1.2891** |
| `tier3_unseen_sentence` | unseen-sentence composition — a sentence never seen | 2,438 | **1.3723** |

The ladder is **monotonic** — each step costs what its name predicts, so the
split measures what it claims.

That tier-3 number is diluted, though: only one sentence of a tier-3 response is
withheld, so most of its tokens are familiar. Scoring the two token sets
separately **inside the same rows** — same prompts, same packing, same forward
passes — removes the dilution:

| token set (same 2,438 rows) | tokens | perplexity |
| --- | --- | --- |
| inside a **seen** sentence | 27,064 | **1.2068** |
| inside a **withheld** sentence | 32,855 | **1.5202** |

**+0.231 nats — 2.8× the diluted gap.** The model can emit every word of every
withheld sentence (coverage 1.0000) and is still markedly worse at them than at
the sentences beside them in the same response. V57's caveat was right, and the
effect is larger than its row-split could show.

V58 also runs the **thinking-core ablation** v57 listed as never run — a matched
pair differing only in `use_thinking_core`, checked for matching steps, seeds and
withheld sentences before it will report. Every tier delta lands between 0.0006
and 0.007 nats and **the sign flips across tiers**, so on this corpus the
recursive thinking core has **no measurable effect on text quality** — a bound on
this task, not a verdict on the mechanism. The second arm doubles as an
independent replication: it puts the withheld-sentence penalty at +0.2238 nats
against the first arm's +0.2309.

Every tier has response vocabulary coverage **1.0000**, so tier 3 is a
composition test, not a vocabulary test. Checkpoint selection reads a separate
**dev** split and the tiers are scored once, after selection — v57 selected the
best validation loss and then reported that same validation set.

```bash
python source/train_mimomix_generalisation.py --steps 1000 --arm full
python source/train_mimomix_generalisation.py --steps 1000 --arm ablation
python source/eval_mimomix_unseen_sentences.py --run_dir output/v58_full
python source/train_mimomix_generalisation.py --compare output/v58_full output/v58_ablation
```

`verify_split` re-derives every property the tier names promise and raises if one
fails, inside the trainer rather than in a test; three tests corrupt a split and
require it to raise, because a verifier that never rejects is no verifier.

Details, the sharper token-level number, and the explicit non-claims:
[`docs/V58_GENERALISATION_LADDER.md`](docs/V58_GENERALISATION_LADDER.md).

## Supermix v57 — the talking MiMoMix (new, additive)

V53 built a complete decoder-only language model and never trained it on
language — its own document says the backends are randomly initialised and their
text is noise by design. V57 supplies the missing half: the **same**
`mimomix_core.MiMoMixModel`, unchanged, trained on real dialogue until it
generates text, plus a tokenizer, a corpus loader, a checkpoint format, and a
chat interface.

```bash
python source/train_mimomix_talk.py --steps 3000 --run_name v57_talk_v2 --output_dir output/v57_talk_v2
python source/mimomix_talk_web_app.py --checkpoint output/v57_talk_v2/v57_talk_v2.pt
```

Every reply is generated token by token by the model — no parser, no templates.
This is the first line in this repo where the model itself produces the words.

| measurement | value |
| --- | --- |
| parameters | 3,076,521 (1,292,829 active/token) |
| vocabulary | 582 types, held-out coverage 1.0000 |
| validation loss | 0.2351 (0.339 bits/token), perplexity 1.27 |
| MTP acceptance length | 2.429 — 7 trunk forwards vs 17 greedy, output identical |
| routing | entropy 1.000, 0 starved experts |
| serving | 55–75 tokens/second on CPU |

> **you:** hello
> **model:** Hello. Tell me what you need and I will do my best to help.
>
> **you:** can you help me with tests
> **model:** Understood. Yes. I can add unit tests and integration tests based on your code.

**What it is:** a small domain-specific chat model. It speaks fluently in one
register — coding-assistant small talk — and nothing else. **What it is not:**
knowledgeable. The only local corpus is `databases/llm_chat.db`: 120,000
templated turns, 21M characters, and a measured **292 distinct word types**. A
word outside that vocabulary can never be generated, and the interface warns you
when your own message contains one. Held-out perplexity measures fit to a
template distribution, not generalisation to language, because only 37,543 of the
120,000 responses are distinct — the receipt names the metric that way.

Greedy replies run through **MTP self-speculative decoding**, which is provably
token-identical to plain greedy decoding and only changes the cost; the interface
reports the measured acceptance length. A sampling mode is offered separately and
is explicitly labelled as *not* identical to greedy.

Details and the full list of non-claims:
[`docs/V57_TALKING_MIMOMIX.md`](docs/V57_TALKING_MIMOMIX.md).

## Supermix v56 — Latent State Reasoner (new, additive)

V56 reuses the v53 MiMoMix components verbatim — hybrid SWA/global attention with
learnable sinks, sparse MoE with the auxiliary-loss-free bias rule, and the
recursive thinking core with its ACT halting and supervised verifier — and adds
an explicit **latent state machine** between the trunk and the answer. The state
is a distribution over latent states and each operator slot emits a
row-stochastic transition matrix, composed in log space. It is the first line in
this repository to train a MiMoMix-derived model to a checkpoint, persist it, and
serve it.

On the v51 `chained_modular_arithmetic` benchmark, evaluated on the **identical**
untouched held-out set (`make_chained_task(1000, seed=52)`):

| model | params | training data | held-out accuracy |
| --- | --- | --- | --- |
| majority-class constant | — | — | 0.1430 |
| v51 `CognitiveLeapUltraExpert` (recorded best) | 2,245,715 | 12,000 × 4 epochs | 0.1710 |
| **v56 reasoner, matched protocol** | **808,626** | same 12,000 × 4 epochs | **0.2410** |
| v56 reasoner, first curriculum | 808,626 | 160,000 (same generator support) | 0.9220 |
| **v56 reasoner, current best** | **808,626** | 160,000 (same generator support) | **0.9740** |

A paired gate over 20 fresh cohorts × 2,000 samples (40,000 paired samples, no
seed reused from any existing v51 gate) puts the current checkpoint at
**0.9762** against the baseline's **0.1718** — 20 seed wins to 0, sign test
p = 1.9e-6, McNemar 32,288 candidate-only versus 112 baseline-only.

Two measured fixes took 0.9220 to 0.9740 at an identical training budget, each
worth about +2.6 points: spreading the curriculum's identity operations over
**random slots** rather than a trailing prefix (the prefix starved the last slot,
and 83% of the earlier model's errors first diverged at exactly that step), and
enabling the **operator-entropy prior** so transition rows are pushed toward the
deterministic functions they are supposed to represent. Removing positional
embeddings for full equivariance was tried and **hurt** (0.8210), as did a
strictly slot-local operator (0.6920).

The matched row changes only the model: identical examples, identical seed,
identical epoch budget, 2.8× fewer parameters. The curriculum row changes the
training recipe as well and is reported separately for that reason.

Two measurements make the numbers readable. The **majority-class floor is
0.1430**, so the previous line beat a constant predictor by 2.8 points. And the
Bayes-optimal accuracy for any predictor that sees all four operations but not
the start digit is **0.4105** — so an accuracy above that is the first evidence a
model on this task is composing the whole chain rather than reading its tail.

Details, the full Bayes-by-information-subset table, and the explicit non-claims:
[`docs/V56_LATENT_STATE_REASONER.md`](docs/V56_LATENT_STATE_REASONER.md).

```bash
python source/benchmark_mimomix_reasoner.py --protocol matched
python source/benchmark_mimomix_reasoner.py --protocol curriculum --enforce_gates
python source/run_v56_promotion_gate.py --candidate output/v56b_randslots_entropy/v56b_randslots_entropy.pt --baseline output/benchmark_v51_cognitive_leap_ultra_latest/cognitive_leap_ultra_v51_trained.pth
python source/mimomix_reasoner_web_app.py --checkpoint output/v56b_randslots_entropy/v56b_randslots_entropy.pt
```

The current-best checkpoint is `output/v56b_randslots_entropy/` — the run with
random identity slots and the operator-entropy prior. `output/v56_curriculum/` is
the **first** curriculum run, kept for the ablation table; it scores 0.9220 and
gates at 0.9329, so pointing these commands at it reproduces the superseded row
rather than the one quoted above.

The web interface binds `127.0.0.1:8156`, refuses to start without a
`supermix-v56-reasoner-checkpoint-v1` file, and displays the model's own
reasoning: the latent state after every operation and the learned transition
matrix per operation, read from the forward pass rather than illustrated. It also
serves `web_static/mimomix_lab.html` at `/lab`, where the observatory's new
**Live model** panel reads real telemetry from the loaded checkpoint — every
other panel on that page remains a slider-driven simulation and now says so.

`/chat` adds a conversational surface. **The model has no language ability** — no
tokenizer, input is a fixed 128-dim vector — so a deterministic parser reads the
arithmetic out of a sentence and the model only ever does the arithmetic. The
page says exactly that in a banner. Answers are graded against the generator's
own rule, never by the model; chains beyond four operations run the model
repeatedly on its own answer and report the call count.

```bash
python source/benchmark_reasoner_chat.py --checkpoint output/v56b_randslots_entropy/v56b_randslots_entropy.pt
```

| operations | parse rate | model given parse | end to end | model calls | mean latency |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.000 | 1.0000 | 1.0000 | 1.0 | 27 ms |
| 4 | 1.000 | 0.9800 | 0.9800 | 1.0 | 27 ms |
| 8 | 1.000 | 0.9800 | 0.9800 | 2.0 | 52 ms |
| 16 | 1.000 | 0.9400 | 0.9400 | 4.0 | 106 ms |

Parser and model are reported separately because only one of them is learned. All
three prompt-injection payloads answer identically to the benign input.

V56 is `source/`-only by design and adds nothing to the packaged runtime, the
Studio manifest, or `MODEL_SPECS`.

## Supermix v55 — Authority-Bound Memory and Verified Answer Receipts (new, additive)

V55 hardens the boundary between remembered text, verified computation, and
runtime authority. Conversation Memory v3 binds every newly admitted item to an
immutable origin, source turn, allowed-use class, truth status, lifecycle, and
content digest. Only direct-user name and answer-detail preferences may enter
the shared personalization prompt. Project notes and factual claims remain
inspectable, attributed, and unverified; they cannot become evidence, authorize
tools, select routes, change compute, grant permission, override safety, or gain
authority from retrieval relevance. Assistant, tool, consultant, malformed,
legacy-unbound, quoted, fenced, encoded, and tampered rows fail closed, and old
assistant exemplars are no longer injected into prompts.

The grounding layer now emits `supermix-verified-answer-receipt-v1` on every
source, compatibility, Qwen, and Studio surface. A receipt reports only
allowlisted diagnostics: problem class, method, verification/independence,
conflict and selection state, abstention, and model-conditional epistemic
limits. It contains no prompt, answer, expression, proof step, or evidence text,
and every routing, compute, interaction, permission, and promotion authority bit
is false.

Studio exposes no-store, loopback-only, session-scoped memory inspection and
exact-ID lifecycle controls. Revocation is terminal through the review UI;
quarantined rows alone may be restored. The release manifest now enforces
the recursive local-import closure of every Studio and compatibility entry point, binds both new
contracts, and CI lints release surfaces plus the previously omitted packaged
web/concurrency suites. These are runtime, safety, observability, and release-
integrity changes: no checkpoint or adapter was trained, promoted, activated, or
packaged by v55. See
[`docs/V55_MEMORY_AUTHORITY_AND_ANSWER_RECEIPTS.md`](docs/V55_MEMORY_AUTHORITY_AND_ANSWER_RECEIPTS.md).

## Supermix v54 — Verified Probabilistic Scenarios (new, additive)

V54 advances the shipped deterministic runtime with exact, bounded probability
scenarios. Given one fully specified finite Bernoulli model, the source and
compatibility runtimes can compute `exactly`, `at least`, or `at most` event
probabilities without sampling or floating-point approximation. The accepted
grammar requires 1–200 independent, constant-probability trials (or IID fair
coin tosses), an exact probability written as a fraction, decimal, or percent,
and one complete event question.

Each accepted result is computed from exact binomial masses and independently
rebuilt by repeated Bernoulli convolution. The result receives answer-replacement
authority only when both paths agree, all masses are non-negative and sum to one,
the complementary event closes exactly, every bounded solver has been considered,
and the grounding boundary successfully reparses the original request. Dependent
trials, changing or unknown probabilities, sampling without replacement,
ambiguous or superseded requests, high-stakes predictions, and open-world
forecasts abstain.

Inspect the capability directly:

```bash
python source/reasoning_cli.py --query "Assuming 5 independent Bernoulli trials with fixed success probability of 1/2, what is the probability of exactly 3 successes?" --steps
```

V54 is an additive runtime and release-contract upgrade. It does not replace the
v52 model line, the v53 MiMoMix research stack, Deliberate Reasoning v3, or Qwen
Promotion Evidence v4, and it does not claim a newly trained or promoted model.
The full grammar, verification contract, packaging surfaces, and non-claims are
in [`docs/V54_VERIFIED_PROBABILISTIC_SCENARIOS.md`](docs/V54_VERIFIED_PROBABILISTIC_SCENARIOS.md).

## MiMoMix v53 (new, additive)

`source/mimomix_*.py` is a new self-contained line that fuses the current Xiaomi
MiMo structural techniques, the Supermix v51/v52 verified-recursion cognition,
and the AI-Dem-Lab research-sandbox concepts into one stack:

- **hybrid attention** — local sliding-window layers interleaved with global
  layers at a configurable ratio (MiMo-V2-Flash uses 5:1, V2.5-Pro 6:1, both with
  a 128-token window), plus a learnable per-head attention sink. Only the global
  layers keep an unbounded KV cache, which is where the cache saving comes from
- **auxiliary-loss-free sparse MoE** — experts are *selected* by score + bias but
  *weighted* by score alone, so the balancer can never distort the forward value;
  plus router z-loss, shared always-on experts, and dense lower layers
- **multi-token prediction** reused at inference as a self-speculative draft.
  Greedy output is bit-identical to plain autoregressive decoding, proven by
  `assert_greedy_equivalence` across seeds, batches, RoPE policies and layouts
- **progressive RoPE context extension** with explicit `none` / NTK-aware / YaRN
  policies rather than one implicit default
- **the v52 recursive thinking core** — weight-tied refinement, ACT halting with
  a ponder cost, trainable temperature calibration, and the supervised
  quality/continue verifier
- **a progressive thinking controller** — fast/deep/agent routing, difficulty and
  epistemic-risk floors, and a budget ladder whose early exit needs the verifier
  to stand down *and* confidence/entropy targets met *and* cross-budget ordered
  top-k agreement. The accepted output is the probe the model actually produced,
  never a blend
- **the Dem-Lab observatory** — entropy and randomness batteries with exact
  chi-square p-values, novelty/stability meters, semantic resonance, routing
  attribution, robust anomaly detection, replicator dynamics over controller
  policies, and Q-learning feedback that proposes a starting budget
- **MOPD post-training** — group-relative domain RL, then multi-teacher
  on-policy distillation with a dense per-token teacher signal

Measured end to end on a 1.2M-parameter model trained for 250 CPU steps on a
synthetic periodic task: **6.00x smaller KV cache** than all-global attention at
1M tokens, **3.917 MTP acceptance length** (74.5% fewer trunk forwards) with
output bit-identical to greedy decoding, **0.970 normalised routing entropy**
with no starved experts, and **+25% cycle reduction at 100% top-1 and ordered
top-3 decision fidelity**. Reproduce with:

```bash
python source/benchmark_mimomix.py --steps 250 --enforce-gates
```

The later correctness hardening did not rerun that 250-step benchmark. It makes
speculative decoding EOS- and output-budget-safe, handles finished batch rows,
fixes post-prefill acceptance accounting, aligns distillation targets to causal
next-token prediction, and keeps top-k teacher reverse-KL finite. The current CI
slice runs all six MiMoMix suites (`187` tests).

Run the routing demo, the browser observatory, and the tests:

```bash
python source/mimomix_api.py --example
```

```text
web_static/mimomix_lab.html
```

```bash
python -m pytest test_mimomix_core.py test_mimomix_decoding.py test_mimomix_controller.py test_mimomix_observatory.py test_mimomix_distill.py test_mimomix_api.py
```

Design, the research each mechanism comes from, and an explicit list of what this
does **not** prove are in
[`docs/V53_MIMOMIX_ARCHITECTURE.md`](docs/V53_MIMOMIX_ARCHITECTURE.md). The
default backends are randomly initialised: they produce well-formed responses and
honest telemetry, and their text is noise until real weights are trained. v53 is
additive — no existing v52 module, checkpoint, manifest, or gate changed.

## Current status

As of August 20, 2026 this tree has a **v71 source/runtime product contract**,
the additive v56-v70 research line, and v54 as the latest locally built Windows
artifact. The v52 unified model line, v53 MiMoMix stack, v54 Verified
Probabilistic Scenarios, v55 memory-authority boundary, and v71 Verified
Scientific Scenarios now live in one tree. See
[`docs/V52_UNIFIED_ARCHITECTURE.md`](docs/V52_UNIFIED_ARCHITECTURE.md) for the
model merge contract, [`docs/V53_MIMOMIX_ARCHITECTURE.md`](docs/V53_MIMOMIX_ARCHITECTURE.md)
for the self-contained research stack, and
[`docs/V54_VERIFIED_PROBABILISTIC_SCENARIOS.md`](docs/V54_VERIFIED_PROBABILISTIC_SCENARIOS.md)
for the exact-probability runtime contract, plus
[`docs/V55_MEMORY_AUTHORITY_AND_ANSWER_RECEIPTS.md`](docs/V55_MEMORY_AUTHORITY_AND_ANSWER_RECEIPTS.md)
for the memory trust boundary, and
[`docs/V71_VERIFIED_SCIENTIFIC_SCENARIOS.md`](docs/V71_VERIFIED_SCIENTIFIC_SCENARIOS.md)
for the current scientific-plan runtime contract.

- `source/` is the active Supermix Studio runtime and packaging tree
- the curated desktop build selects `11` core model artifacts and leaves expansion to the model store
- the route control plane includes durable lifecycle evidence, Policy Lab diagnostics,
  bounded-exposure rehearsal, and a fail-closed stateful experiment protocol preflight
- a shared Plan-Evaluate v4 interaction layer builds one bounded intent, appraisal,
  risk, and response contract per turn; it adds anti-sycophancy-aware candidate
  ranking, high-precision response guards, task-specific maths, science, causal,
  prediction, and conversation strategies, and compact diagnostics
- a shared Prompt Understanding v3 layer separates instructions from quoted or
  code data, recovers bounded cue typos, tracks turn references, detects
  conflicting constraints, extracts polarity-aware reasoning facets, and creates
  privacy-safe prompt diagnostics
- the shared Deliberate Reasoning v5 layer plus v54 Verified Probabilistic
  Scenarios and v71 Verified Scientific Scenarios solve word-stated problems
  across twenty-three solver families,
  including bounded geometry, exact finite Bernoulli events, empirical prediction,
  physics, ordered quantity transitions, and strict positive-Horn logical
  entailment; only a supported result whose bounded checks pass and whose
  applicable solvers agree may replace a response
- a shared Conversation State v2 layer accumulates across the whole session rather
  than a four-turn window: durable user commitments with supersession, questions the
  assistant asked and whether they were answered, topic threads, cross-conversation
  repetition, and stated contradictions; explicitly per-turn requests such as
  "this time" are not promoted into standing commitments
- a Conversation Directive v2 layer routes that state onto the generative surfaces
  (the Qwen web app and the Studio Qwen backend), which previously read none of it:
  it renders a bounded, sanitised, prompt-control-filtered contract as user-level
  history, keeps the current request last, and selects a generation preset from a
  standing style preference only when the caller did not provide one; a missed
  request is resurfaced only after the user explicitly asks for repair
- Qwen Web defaults to an **Auto** preset that omits generation overrides so the
  conversation preset can apply (with a balanced fallback); one striped session
  lock covers history snapshot, routing, generation, and append, while stale client
  history may hydrate only an empty server session. The server retains 80 messages,
  derives state from 40, and sends at most 12 to the model
- persistent Studio memory uses `supermix-conversation-memory-v3` with the
  `supermix-memory-authority-firewall-v1` policy: direct-user provenance and
  exact allowed uses are content-bound, relevance cannot elevate authority,
  assistant/tool/consultant text cannot be laundered into memory, legacy and
  tampered rows default-deny, assistant exemplars stay out of prompts, and users
  can inspect or change one memory lifecycle by exact ID; remote Studio chat
  callers are server-forced to memory-off
- grounding emits prompt- and answer-free Verified Answer Receipts across the
  terminal, packaged web, Qwen, and Studio surfaces; they expose verification
  and model-conditional limits but carry no routing, compute, permission,
  interaction, or promotion authority
- both Qwen EXE build scripts and the tracked `SupermixQwenDesktop*.spec` files
  bundle `conversation_state.py` and `conversation_directive.py`
- the `cognitive_leap_v52_expert` model variant adds a supervised quality/continue
  verifier, bounded emotion/intent/strategy appraisal heads, trainable temperature
  calibration, and optional sparse top-k recurrent-core execution, while still
  forwarding the v51 prediction-stability controls it inherits
- v51 local inference supports progressive accepted-probe reuse plus a post-head,
  allowed-label-scoped decision verifier that checks the ordered top-3 boundary before
  an adaptive early exit
- `runtime_python/` remains a legacy compatibility snapshot for the smaller chat runtime;
  it is not the source of truth for the multimodel Studio route control plane
- the checked Studio manifest enforces the recursive local-import closure of all
  source and compatibility release entry points, with only explicit justified
  training-only exclusions
- the source/runtime Windows installer contract version is `71.0.0`; existing
  local binaries remain older until a separately verified release build

## What is in this repo

- `source/`
  - active development workspace
  - training scripts, model definitions, dataset builders, benchmark runners, desktop packaging helpers
- `runtime_python/`
  - legacy, self-contained compatibility runtime
  - model-variant parity is generated from `source/model_variants.py`, but Studio-only
    routing features intentionally ship through the active `source/` desktop build
- `datasets/`
  - conversation, coding, reasoning, science, and related local training inputs
- `output/`
  - generated artifacts, benchmark graphs, summaries, logs, Hugging Face upload folders
- `installer/`
  - Inno Setup definitions for the desktop app
- `dist/`
  - built EXEs and installer outputs
- `web_static/`
  - lightweight browser-only metadata bundle

## Main capabilities

- multimodel desktop app with model selector, Auto routing, collective mode, and agent mode
- local chat, image-prompt, math, science-image, and omni-fusion model families
- native-image experimental checkpoints
- training pipelines for frontier, omni, lite, and specialist model lines
- benchmark sweeps across common text benchmarks
- release-gated v51 adaptive compute with source/package parity and frozen-prompt
  response-fidelity checks
- v55 authority-bound conversation memory, exact-ID review, privacy-safe verified
  answer receipts, recursive package dependency closure, and release-surface linting
- v54 exact finite Bernoulli scenarios with source/package parity, independent
  convolution verification, and a second grounding-boundary applicability gate
- Plan-Evaluate interaction intelligence across the source and compatibility
  runtimes, Studio routes, Qwen, and the static browser copies
- export and publishing workflows for GitHub releases and Hugging Face model/dataset repos

## Quick start

### Run the packaged runtime

```bash
python runtime_python/chat_web_app.py
```

Windows launchers:

```bat
runtime_python\launch_chat_web_supermix.bat
runtime_python\launch_chat_terminal_supermix.bat
```

### Run the active source app

```bash
python source/chat_web_app.py
```

### Run the desktop multimodel app from source

```bash
python source/supermix_multimodel_desktop_app.py
```

### Inspect the route experiment control plane

```bash
python source/route_policy_study_cli.py --example
python source/route_policy_protocol_cli.py --example
```

Both commands are prompt-free, non-executing design tools. They do not assign a
route, write evidence, estimate policy value, or enable promotion.

### Inspect the deliberate reasoning engine

```bash
python source/reasoning_cli.py --example
python source/reasoning_cli.py --query "Convert 5 km to miles" --steps
python source/reasoning_cli.py --query "A tank starts with 120 liters. Then 25% is removed. Then 15000 milliliters are added. What is the final volume?" --steps
```

This shows, for each request, which solver applied, how the answer was checked,
whether that check is independent, and whether the result would be allowed to
replace a retrieved response. It computes and audits only.

### Run the browser-only static bundle

Open:

```text
web_static/index.html
```

## Historical desktop release

The repository still documents this older published artifact for provenance; it
is not evidence that the current development tree has been released:

- Release page:
  - `https://github.com/kai9987kai/Supermix_29/releases/tag/studio-desktop-20260329-omni-v4-allmodels`
- Installer:
  - `https://github.com/kai9987kai/Supermix_29/releases/download/studio-desktop-20260329-omni-v4-allmodels/SupermixStudioDesktopSetup.exe`
- EXE:
  - `https://github.com/kai9987kai/Supermix_29/releases/download/studio-desktop-20260329-omni-v4-allmodels/SupermixStudioDesktop.exe`

Local build outputs:

- `dist/SupermixStudioDesktop/SupermixStudioDesktop.exe`
- `dist/SupermixStudioDesktop/SupermixRouteStudy.exe`
- `dist/SupermixStudioDesktop/SupermixRouteShadow.exe`
- `dist/installer/SupermixStudioDesktopSetup.exe`
- `dist/installer/SupermixStudioDesktopReleaseSHA256.txt`

## Model families in the workspace

The repo contains code and artifacts for several model lines:

- Qwen adapter line
  - `v28`
  - `v30`
- Champion / frontier line
  - `v31`
  - `v32`
  - `v33`
  - `v34`
  - `v35`
  - `v39`
- native-image line
  - `v36`
  - `v37`
  - `v38`
- omni-collective line
  - `v1`
  - `v2`
  - `v3`
  - `v4`
- specialist lines
  - `math_equation_micro_v1`
  - `science_image_recognition_micro_v1`

## Latest finished omni model

The latest finished omni checkpoint in this repo is `omni_collective_v4`.

Key details from [`output/supermix_omni_collective_v4_frontier_20260329/omni_collective_v4_frontier_summary.json`](output/supermix_omni_collective_v4_frontier_20260329/omni_collective_v4_frontier_summary.json):

- parameter count: `19,032,281`
- stage-1 rows: `8,589`
- stage-2 rows: `9,447`
- final stage-2 weighted validation score: `0.5176`
- final stage-2 validation:
  - intent: `0.8195`
  - response: `0.1402`
  - vision: `0.9020`
  - domain: `0.7765`

Local packaged artifact:

- [`output/supermix_omni_collective_v4_frontier_20260329.zip`](output/supermix_omni_collective_v4_frontier_20260329.zip)

## Hugging Face models

Public model repos already published from this workspace:

- `Kai9987kai/supermix-v33-frontier`
- `Kai9987kai/supermix-omni-collective-v1`
- `Kai9987kai/supermix-v38-native-image-xlite-fp16`
- `Kai9987kai/supermix-v39-frontier-reasoning-plus`
- `Kai9987kai/supermix-omni-collective-v2-frontier`
- `Kai9987kai/supermix-math-equation-micro-v1`
- `Kai9987kai/supermix-omni-collective-v4-frontier`

## Studio Model Store

The live Studio store is published at
[`Kai9987kai/supermix-model-zoo`](https://huggingface.co/datasets/Kai9987kai/supermix-model-zoo).
Its `supermix-model-store-manifest-v2` catalog binds every package to an exact
byte size and SHA-256 digest. Studio verifies both values and a safe ZIP layout
before atomically making a download selectable; incomplete, unhashed, corrupt,
encrypted, path-traversing, symlinked, or case-colliding archives fail closed.

The 2026-08-12 catalog contains `36` packages, including these three locally
reproducible Cognitive Leap additions:

- `cognitive_leap_v50` — manual-only archived conversational checkpoint
- `cognitive_leap_ultra_v51_demo` — manual-only bounded arithmetic/runtime demo
- `cognitive_leap_ultra_v51_1_balanced_blend30` — manual-only, unpromoted
  experimental checkpoint

The v51.1 candidate improved fresh-cohort synthetic mod-10 accuracy from
`15.2000%` to `15.4975%` (`+0.2975` percentage points), reduced mean loss from
`2.404259` to `2.360715`, and improved all eight coarse operation families.
It nevertheless passed only `15/20` non-regressing evaluation seeds against a
predeclared `16/20` requirement. Its promotion receipt therefore records
`passed: false`; Studio does not automatically install, auto-route, default to,
or implicitly consult it. Explicit installation and selection remain available
for bounded research. This evidence is intentionally bounded and is not a broad
chat or general-reasoning improvement claim.

## Hugging Face datasets

Public dataset repos already published from this workspace:

- `Kai9987kai/supermix-conversation-datasets`
- `Kai9987kai/supermix-science-vision-dataset`

## Benchmarks

The current local multibench comparison bundle is:

- [`output/pdf/benchmark_local_all_models_multibench_20260329.pdf`](output/pdf/benchmark_local_all_models_multibench_20260329.pdf)
- [`output/benchmark_local_all_models_multibench_20260329.json`](output/benchmark_local_all_models_multibench_20260329.json)
- [`output/benchmark_local_all_models_multibench_20260329.csv`](output/benchmark_local_all_models_multibench_20260329.csv)

The current graph covers `20` benchmarked local model entries and keeps specialist-only models labeled separately when the common text suite is not the right evaluation fit.

Representative current common-benchmark leaders from the local graph JSON:

- `v33_final`: `0.1867`
- `v39_final`: `0.1800`
- `omni_collective_v1`: `0.1633`
- `v34_final`: `0.1600`
- `v36_native`: `0.1533`
- `v35_final`: `0.1533`
- `omni_collective_v4`: `0.0900`

## Training entry points

Representative training and continuation scripts:

- `source/train_omni_collective_v2.py`
- `source/train_omni_collective_v3.py`
- `source/train_omni_collective_v4.py`
- `source/train_omni_collective_v5.py`
- `source/train_math_equation_model.py`
- `source/train_image_recognition_model.py`
- `source/build_reasoning_benchmix_v39.py`
- `source/benchmark_all_models_common.py`

If you want the active experimental path, start in `source/`.

## Desktop build entry points

Primary desktop build helpers:

- `source/build_supermix_studio_desktop_exe.ps1`
- `source/build_supermix_studio_desktop_installer.ps1`
- `SupermixStudioDesktop.spec`
- `installer/SupermixStudioDesktop.iss`

The model bundle manifest is generated from the selected local model store at
packaging time as `output/supermix_studio_bundled_models_manifest.json`; it is a
build artifact, not a checked source-of-truth file.

The deterministic Studio code-and-contract manifest is:

- [`source/studio_runtime_manifest.json`](source/studio_runtime_manifest.json)

Verify it before packaging:

```bash
python source/generate_studio_runtime_manifest.py --check
```

When a release host intentionally has no curated model ZIP/benchmark bundle,
build a base-model runtime package without fabricating those inputs:

```powershell
powershell -ExecutionPolicy Bypass -File source\build_supermix_studio_desktop_exe.ps1 `
  -RuntimeOnly -SkipDependencyInstall
powershell -ExecutionPolicy Bypass -File source\build_supermix_studio_desktop_installer.ps1 `
  -Version 71.0.0
```

The resulting bundle manifest records
`runtime_only_base_model_plus_model_store` and zero curated ZIPs; the local base
model and remote model-store support remain available.

The desktop build also produces `SupermixRouteStudy.exe`, a console entrypoint
for exporting and auditing fail-closed protocol drafts and portable
multi-stratum review bundles. A review bundle embeds the complete canonical,
prompt-free source plans and the closed builder options so its protocol can be
reconstructed exactly:

```powershell
SupermixRouteStudy.exe --example-bundle --output route-review.json
SupermixRouteStudy.exe --audit-bundle route-review.json --compact
```

`full_source_bound_reconstruction` proves internal semantic conformance to the
checked v1 builder. It is deliberately not a signature, trusted timestamp,
causal validation, protocol seal, route assignment, or activation approval.
The browser Studio can collect compatible support strata in ephemeral client
memory, build/download the same bundle, and re-import it for verification; it
never pools unweighted strata into a policy-value estimate. The installer
includes the Studio and both route consoles. The checked manifest binds the
active entrypoints, routing schema versions, module hashes, reconstruction
capability, and non-activation package guards.

The build also produces `SupermixRouteShadow.exe`, an explicit local console for
the separate shadow-only whole-policy commitment/reveal registry. It accepts
only a fully reconstructable review bundle and freezes exactly two 50/50 arms:
`incumbent_source_policy`, bound to the complete source-policy cohort, and
`candidate_target_policy`, bound to the complete target-policy class. A
prompt-specific route plan's `eligible_actions` remain support-stratum evidence;
they are never reinterpreted as campaign treatment arms.
These arms bind declarative policy-class manifests, not executable code or a
runtime code digest. Each cluster input must be the exact canonical
`session-hash-v1` value: 64 lowercase hexadecimal characters, matching
`source.route_policy_ledger.hash_session_identity(...)`. Raw identifiers and
alternate spellings (including uppercase, whitespace, or digest prefixes) fail
closed instead of being normalized. Membership in an external cluster map
remains an external prerequisite for any future experiment.

The mutating workflow is CLI-only:

```powershell
$seal = SupermixRouteShadow.exe seal --registry route-policy-shadow-registry.sqlite3 --bundle route-review.json --seed-output route-shadow-seed.private.json --compact | ConvertFrom-Json
$campaign = $seal.public_package.campaign_seal.seal.campaign_id
SupermixRouteShadow.exe commit --registry route-policy-shadow-registry.sqlite3 --campaign $campaign --seed-input route-shadow-seed.private.json --cluster-input cluster.private.json
SupermixRouteShadow.exe close --registry route-policy-shadow-registry.sqlite3 --campaign $campaign
SupermixRouteShadow.exe reveal --registry route-policy-shadow-registry.sqlite3 --campaign $campaign --seed-input route-shadow-seed.private.json
SupermixRouteShadow.exe verify --registry route-policy-shadow-registry.sqlite3 --campaign $campaign
SupermixRouteShadow.exe status --registry route-policy-shadow-registry.sqlite3 --campaign $campaign
```

`cluster.private.json` must contain exactly
`{"cluster_identifier":"<canonical-lowercase-session-hash>"}`. The CLI derives
and stores a study-scoped pseudonym, not that session hash or the underlying raw
identifier. Keep both the cluster input and the
separately created seed capsule private; pseudonymity is not anonymity, and
post-reveal unlinkability is not guaranteed. POSIX capsules are created as
`0600`. On Windows, the CLI installs a protected, non-inheriting DACL containing
only the current-user full-control ACE, verifies it before writing seed bytes,
re-verifies it after `fsync`, and rejects later reads if that boundary changes.
ACL setup or verification failure deletes a newly created capsule and fails the
command closed. Before closure the registry stores only an opaque assignment
commitment. After closure, `reveal` opens the seed and `verify` reconstructs each
frozen assignment. The browser exposes only a GET
status view of the canonical Studio registry; it has no seal, commit, close,
reveal, or verify mutation endpoint. CLI `status` also opens SQLite in read-only
mode and audits required tables, append-only/state-transition triggers,
high-volume campaign indexes, their exact schema-definition fingerprint, every
stored artifact, the event chain, and the one-to-one event/evidence inventory.
Reveal processing distinguishes matched assignments from mismatches; a fully
processed campaign is not reported as verification-complete unless the final
whole-campaign audit also passes.

The Studio web server binds to `127.0.0.1` by default. Remote exposure requires
an explicit `--host` override and its own authentication/network controls. The
read-only status endpoint sends `Cache-Control: no-store`; the runtime reuses an
audited in-process snapshot only while the durable SQLite database and non-empty
WAL signature remain unchanged.

This registry is isolated from `route-policy-ledger.sqlite3`. Its artifacts are
explicitly non-ledger-eligible and never execute a route, record an executed
propensity, run inference, estimate policy value, certify a causal design,
activate a policy, or promote one automatically. Its SQLite triggers and event
hash chain detect ordinary local mutation; they are not a signature, trusted
timestamp, external witness, inclusion/consistency proof, or transparency
service. Randomized executed-ledger rows additionally require the closed
`route-execution-assignment-v1:<sha256>` namespace, so a bare shadow commitment
digest is rejected instead of being silently relabelled as execution evidence.
This is type separation, not proof against a hostile local caller fabricating an
executed envelope.

## Research and experiment notes

The backward-compatible `supermix-interaction-plan-v1` schema is implemented by
the Plan-Evaluate v3 runtime. It uses observable request and recent-turn cues to
select a task-specific response strategy and contract before generation, then
reranks or audits the response against that contract. Its ranking contribution
is bounded, and automatic rewrites are limited to high-precision crisis,
urgent-medical, explicit unearned-agreement, and explicit dismissive-language
cases. Missing empathy, unsupported certainty, topical continuity, and lexical
relevance remain audit-only signals. Interaction diagnostics omit the raw
prompt.

The plan's compute advice is `shadow_advisory_only`: it cannot change the
reasoning budget or bypass the checkpoint-bound prediction verifier. Controlled
Studio evaluations can preserve a response unmodified by this layer with
`settings.interaction_intelligence=false`; direct Champion and Qwen engine
calls use `interaction_enabled=False`. Design rationale and research boundaries
are recorded in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#july-2026-plan-evaluate-interaction-intelligence-v1).

## Prompt Understanding v2

`prompt_understanding.py` creates one deterministic, JSON-safe prompt profile
from the raw turn and bounded recent context. It recognizes multiple requested
acts, negation and instruction polarity, output constraints, hard conflicts,
follow-up references, evidence/freshness needs, and immediate personal-safety
cues. Quoted text, code, URLs, and paths are masked before intent matching, and
typo recovery is restricted to a small cue vocabulary instead of rewriting the
user's content. V2 adds polarity-aware mathematical, scientific, predictive,
causal, investigative, conversational, and multi-part facets. Multiline quoted
payloads remain data, generic words such as `project` and `variable` are not
enough to activate forecast or science contracts, and forbidden objective spans
cannot recreate obligations downstream. The raw prompt is never replaced, and
diagnostics omit prompt text and extracted literals.

The profile is shared across planning, retrieval, grounding, and response
constraint auditing. Missing required references or irreconcilable hard
constraints produce one targeted clarification; profile signals cannot enable
tools, expand permissions, change compute, or override the existing safety
path. Any consumer that uses the profile for routing remains bound by its
existing model-eligibility and permission checks.

The design is motivated by ambiguity clarification, realistic typo robustness,
composed constraint following, and multi-turn dialogue research, including
[ClarifyMT-Bench](https://arxiv.org/abs/2512.21120),
[MulTypo](https://arxiv.org/abs/2510.09536),
[ComplexBench](https://arxiv.org/abs/2407.03978),
[Multi-IF](https://arxiv.org/abs/2410.15553), and
[StructFlowBench](https://arxiv.org/abs/2502.14494), with
[RECAP](https://arxiv.org/abs/2509.04472) and
[SAGE](https://arxiv.org/abs/2511.08798) informing contextual rewriting and
ask-versus-act decisions. These sources motivate the design; they do not
validate Supermix. This upgrade improves runtime logic, pipeline integration,
tests, and a verifier-gated curriculum. Existing model weights were not
retrained, so it is not evidence of a smarter trained checkpoint. Full design
and evaluation boundaries are in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#august-2026-epistemic-conversation-and-deliberate-reasoning-v2).

## Grounded problem solving and verified training

Normal chat now enables a bounded grounding layer alongside the existing
interaction planner:

- explicit arithmetic such as `Calculate (7 * 9) + 5.` is evaluated with a
  no-eval exact rational solver;
- local evidence is assigned stable `S1` identifiers and audited for coverage,
  conflict, and fabricated citations;
- Champion Web can use the same optional `llm_chat.db` and persistent
  `chat_memory.db` paths as Champion Terminal;
- Qwen keeps evidence separate from conversation history and treats retrieved
  text as untrusted data; and
- Studio shows grounding status and source cards without giving the grounding
  layer any model-routing or adaptive-exit authority.

Raw fidelity evaluations can pass `grounding_enabled=False` to direct Champion
or Qwen engine calls, or `settings.grounding_intelligence=false` to Studio.

## Epistemic Conversation and Deliberate Reasoning v3

`reasoning_engine.py` extends solving from literal arithmetic expressions to
problems stated in words. It is deterministic, dependency-free, uses no `eval`
and no network, and computes every answer with exact rational arithmetic, so
`10% of 0.1` is `1/100` rather than a float approximation.

Twenty-two solver families are covered: percentages and percent change, ordered
percent chains such as a discount followed by tax, unit conversion across
length, mass, volume, time, data, area, speed, and temperature, linear
equations, speed-distance-time, combined work rates, proportions, sequences,
statistics, gcd/lcm/primality/factorization, combinatorics, date differences
and offsets, simple and compound interest, and sum-and-difference problems,
plus narrow verified grammars for geometry, finite probability, formula-based
physics, model-conditional empirical prediction, and ordered quantity-state
plans that mix percentage and fixed changes across compatible units. V3 adds a
strict positive-Horn grammar such as `Facts: a, b. Rules: a & b -> c; c -> d.
Query: d.` for bounded multi-hop entailment. It rejects natural-language rules,
negation, disjunction, quantifiers, malformed sections, and oversized theories.
`Not entailed` means only that the query does not follow from the stated theory;
it is never presented as a real-world falsehood. V54 adds exact finite Bernoulli
events as the twenty-second family while retaining the v3 logic, geometry,
physics, prediction, and ordered-state behavior unchanged.

Prompt Understanding and Plan-Evaluate now carry task-specific reasoning
facets instead of relying on a generic request to "think step by step". Numeric
work must show a value or formula plus a genuine recomputation or dimensional
check. Scientific investigations require observation/evidence and a
hypothesis-or-test structure. Forecasts require assumptions plus a probability,
range, scenario, or explicit abstention; causal answers require a mechanism and
alternatives or limitations. These are bounded response contracts, not permission
to change routes or runtime compute.

The empirical predictor is deliberately narrow. It applies only when the user
explicitly assumes independent trials with a constant success probability, uses
the observed success fraction as a plug-in estimate, and labels the result
model-conditional, uncalibrated, and not a guarantee. Open-world forecasts are
left to grounded evidence and calibrated language. Neither the empirical
estimate nor an open-world forecast receives deterministic answer-replacement
authority.

Verification is a precondition for authority rather than a report:

- each solver publishes how its answer was checked and whether that check is
  genuinely independent of the path that produced it; algebraic inversion and
  complement checks are labelled as consistency checks, not independent proof
  of formula applicability;
- a linear equation is re-checked by a second substitution evaluator written
  against a different strategy than the symbolic collector that solved it;
- a unit conversion must pass both an exact round trip and a magnitude
  direction test, because a round trip alone cancels an inverted factor;
- a sequence rule must hold for every supplied term, not just the last pair;
- Horn conclusions must agree between forward closure and an exhaustive finite
  Boolean-model oracle; cycles, missing conjuncts, and unseeded rules are covered;
- every authority decision exhausts the bounded solver registry, and any
  disagreement withdraws override authority.

Compute is bounded. A deterministic complexity score still reports the
recommended inspection tier, but an answer cannot gain override authority from
an early exit: all twenty-two cheap applicability checks are exhausted first.
Solver count, literal digit length, list and sequence sizes, factorial and
combination sizes, date deltas, and result bit width are all capped.

A computed answer replaces a retrieved response at exactly one point,
`finalize_grounded_response`, and only when the problem is solved, its
verification passed, and no solver disagreed. Explicit arithmetic keeps its
existing dedicated path and takes precedence, including the bounded suffixes
`Explain your reasoning`, `Show your work`, and `Verify the result`; unrelated
prose, multiple expressions, quoted expressions, and code remain ineligible.
The strict-evidence override
outranks both. If the request asks for working, the recorded steps are
included. The layer has no routing, compute, or adaptive-exit authority, and
its diagnostics carry class, method, verification, consensus, and budget only —
never the prompt, the extracted numbers, or the answer.

The same `settings.grounding_intelligence=false` and `grounding_enabled=False`
switches disable it for raw fidelity evaluation. Design rationale, the papers
that motivate it, and the evaluation boundary are in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#august-2026-formal-deliberation-v3-and-oracle-grounded-promotion-v4).
This upgrade changes runtime logic, verifier-grounded training/evaluation data,
and tests; no model weights were retrained, so it is not evidence of a smarter
trained checkpoint.

Build a deterministic verifier-grounded Qwen curriculum:

```powershell
python source/build_verifiable_reasoning_curriculum.py `
  --output-dir output/verifiable_reasoning_curriculum_v1 `
  --train-rows 2000 `
  --eval-rows 400
```

Build the mixed General Intelligence v3 curriculum (including held-out
compositional quantity transitions and model-checked logical entailment):

```powershell
python source/build_general_intelligence_curriculum.py `
  --output-dir output/general_intelligence_curriculum_v3 `
  --seed 6201 `
  --train-rows 1200 `
  --eval-rows 150
```

The generated train/evaluation templates are disjoint and every included answer
passes `supermix-verifier-v2`. The logical family additionally separates atom
vocabularies, graph topologies, and surface markers across train/evaluation,
stores a canonical task IR, and recomputes the target with exhaustive finite-
model semantics. The Qwen pipeline revalidates tagged teacher caches and reports
verified accuracy by problem family. This adds a training and promotion path; it
does not claim that model weights improved until an adapter is trained and passes
the fixed held-out gate. Design details
and research boundaries are in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#july-2026-grounded-problem-solving-and-verifier-grounded-training-v1).

## Promotion Evidence v4

Qwen training outputs remain inert candidates. A candidate can become the
implicit Studio adapter only after the v4 evaluator and promotion gate bind the
adapter and base-model revision to the selected evaluation set, curriculum,
evaluator code, detailed base/tuned samples, sample comparison, and recomputed
paired-evidence digest. The gate replays the deterministic verifier from trusted
`eval_pairs.jsonl` metadata; correctness flags and identities reported by the
sample files are not trusted. It also reconstructs the exact family-balanced
selection from the bound 150-row curriculum, requires complete unique example
identities, and derives the comparison artifact again from the base and tuned
sample rows.

Promotion is conjunctive. The default aggregate floors require at least 20
verified samples, a 5-point verified-accuracy gain, 20% tuned accuracy, no
problem-family regression, an eval-loss ratio no greater than 1.05, token-F1
delta of at least -0.02, and a generation-cap rate no greater than 5%. V4 retains
the requirement for an exact one-sided paired McNemar result at `p <= 0.05`, no
more than a 2% paired regression rate, at least five template clusters, and a strictly
positive lower 95% bound from a deterministic template-cluster bootstrap. The
bootstrap uses exactly 5,000 resamples and seed 5203 for production, so repeated
validation of the same evidence is byte-stable and seed-searching cannot create
a production receipt. The versioned production policy pins the curriculum
schema and holdout digest, selection/decode protocol, bootstrap configuration,
and minimum statistical floors. Custom or weaker protocols can write an
inspectable research gate only; they cannot write the adapter pointer or a
promotion manifest. Missing, changed, path-escaping, duplicated, incomplete, or
internally inconsistent evidence also fails closed.

The archived v1-holdout 30-item candidate showed six verified wins, one
regression, and 23 ties. Its point gain was promising, but exact McNemar was
`p = 0.0625`, its 3.33% regression rate exceeded the 2% ceiling, and the
template-cluster interval crossed zero, so it remains unpromoted. Curriculum v3
adds a seventeenth, formally model-checked logical family, making the pinned
family-balanced production selection 34 items. Older receipts cannot be reused
for the new holdout or verifier.

Studio performs a second local trust check before loading an implicit Qwen
adapter. It accepts a content-valid v4 promotion or a narrowly recognized legacy
artifact, rejects candidate namespaces and invalid/revoked receipts before model
loading, and exposes the activation kind, adapter hash, receipt schemas, and
base-revision status as `adapter_attestation`. Receipt-free legacy compatibility
requires an exact historical adapter/config hash pair; a matching filename or
prefix alone grants no trust. A promoted adapter must resolve to the receipt's
exact Hugging Face repository and immutable snapshot revision. Unproven packaged
model copies fail closed, and Studio re-hashes the adapter configuration and
weights immediately before and after loading to detect a concurrent swap. This
is local content/provenance attestation, not a signature, trusted timestamp, or
external witness. Full rationale and limitations are recorded in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#august-2026-formal-deliberation-v3-and-oracle-grounded-promotion-v4).

The accepted v51 decision-fidelity configuration is checkpoint- and
workload-calibrated: ordered top-3 persistence, a `0.0005` minimum adjacent
probability gap through the top-3/outside boundary, and exact fallback to the
trained three-cycle reference output whenever an earlier decision is not
certified. Legacy latent, entropy, and ACT signals cannot bypass this post-head
verifier.

The clean CPU release gate at commit `81c4dbe7` evaluated 4,096 held-out
requests in each of two modes (isolated verifier and exact release-runtime
defaults):

- zero top-1, ordered top-3, top-3-set, or exact-output disagreements in both
  modes, with zero per-seed accuracy deltas;
- 3,941 certified cycle-2 exits and 155 exact cycle-3 reference fallbacks,
  for 2.0378 mean cycles and a 32.07% cycle reduction from fixed cycle 3;
- positive counterbalanced latency results in both modes: 4.76% weighted / 3.71%
  median-per-seed reduction for release runtime and 7.23% / 5.57% for the
  isolated verifier;
- the frozen 16-prompt source/package response gate had zero response,
  top-five-order, runtime-contract, or packaged-behavior mismatches; and
- progressive accepted-probe auto compute matched the legacy controller exactly
  on 256 requests while reducing forward evaluations by 31.25% and weighted
  latency by 30.09%.

```powershell
python source/run_v51_prediction_stability_gate.py --device cpu --torch-num-threads 8 --torch-interop-threads 1 --strict-determinism --samples-per-seed 512 --enforce-gates
python source/run_v51_chat_response_fidelity_gate.py --device cpu --torch-num-threads 8 --torch-interop-threads 1 --strict-determinism --enforce-gates
python source/benchmark_progressive_auto_compute.py --device cpu --torch-num-threads 8 --torch-interop-threads 1 --strict-determinism --enforce-gates
```

The stricter gate rejected the earlier `0.0001` candidate: release mode showed
5 top-1, 18 ordered-top-3, and 10 top-3-set disagreements; isolated mode showed
2, 6, and 4. That failure led to the common exit guard, ordered-rank state, the
calibrated margin, bounded disagreement evidence, and reference-budget fallback.
Top-k Jensen-Shannon divergence remains shadow telemetry only and cannot
authorize an exit. These are deterministic results for this checkpoint,
synthetic task, seeds, and frozen prompt matrix—not a universal reasoning or
chat-quality guarantee.

This repo is a living experiment workspace. It contains finished artifacts, release-ready packaging, and in-progress work at the same time.

That means you will see mixed generations such as:

- `v28`
- `v30`
- `v33`
- `v34`
- `v35`
- `v36`
- `v37`
- `v38`
- `v39`
- `omni_collective_v1` through `omni_collective_v5`

That is expected.

## Recommended starting points

If you want to:

- run a packaged local system
  - use `runtime_python/`
- work on the active multimodel app
  - use `source/supermix_multimodel_web_app.py`
  - use `source/supermix_multimodel_desktop_app.py`
- work on training
  - start in `source/`
- inspect the current benchmark outputs
  - use `output/benchmark_local_all_models_multibench_20260329.*`
- build a Windows installer
  - use the PowerShell build scripts in `source/` plus `installer/`

## Platform notes

- Windows is the main desktop packaging target
- the repo includes PyInstaller specs, PowerShell build scripts, and Inno Setup definitions
- some training flows were designed around cloud GPU workflows, but the repo also supports local CPU experimentation

## Security note

Do not commit or publish browser-session dumps, cookies, temporary automation state, or live access tokens.

Relevant policy docs:

- `SECURITY.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`

## License

See `LICENSE`.
