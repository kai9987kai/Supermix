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

## Current status

As of July 27, 2026 this tree is the **v52 unified line**: the v51 adaptive-compute
and understanding work and the previously unmerged v52 model generation now live in
one tree. See [`docs/V52_UNIFIED_ARCHITECTURE.md`](docs/V52_UNIFIED_ARCHITECTURE.md)
for the merge contract and [`docs/V52_MERGE_LOG.md`](docs/V52_MERGE_LOG.md) for what
was taken from where.

- `source/` is the active Supermix Studio runtime and packaging tree
- the curated desktop build selects `11` core model artifacts and leaves expansion to the model store
- the route control plane includes durable lifecycle evidence, Policy Lab diagnostics,
  bounded-exposure rehearsal, and a fail-closed stateful experiment protocol preflight
- a shared Plan-Evaluate interaction layer builds one bounded intent, appraisal,
  risk, and response contract per turn; it adds anti-sycophancy-aware candidate
  ranking, high-precision response guards, and compact diagnostics
- a shared Prompt Understanding v1 layer separates instructions from quoted or
  code data, recovers bounded cue typos, tracks turn references, detects
  conflicting constraints, and creates privacy-safe prompt diagnostics
- a shared Deliberate Reasoning v1 layer solves word-stated problems across
  fifteen solver families with exact rational arithmetic, and only lets an
  answer replace a response after an independent verification passes with no
  disagreement between solvers
- a shared Conversation State v1 layer accumulates across the whole session rather
  than a four-turn window: durable user commitments with supersession, questions the
  assistant asked and whether they were answered, topic threads, cross-conversation
  repetition, and stated contradictions
- the `cognitive_leap_v52_expert` model variant adds a supervised quality/continue
  verifier, bounded emotion/intent/strategy appraisal heads, trainable temperature
  calibration, and optional sparse top-k recurrent-core execution, while still
  forwarding the v51 prediction-stability controls it inherits
- v51 local inference supports progressive accepted-probe reuse plus a post-head,
  allowed-label-scoped decision verifier that checks the ordered top-3 boundary before
  an adaptive early exit
- `runtime_python/` remains a legacy compatibility snapshot for the smaller chat runtime;
  it is not the source of truth for the multimodel Studio route control plane
- the Windows installer contract version is `2026.07.27`

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

The `supermix-interaction-plan-v1` layer uses observable request and recent-turn
cues to select a response strategy and contract before generation, then
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

## Prompt Understanding v1

`prompt_understanding.py` creates one deterministic, JSON-safe prompt profile
from the raw turn and bounded recent context. It recognizes multiple requested
acts, negation and instruction polarity, output constraints, hard conflicts,
follow-up references, evidence/freshness needs, and immediate personal-safety
cues. Quoted text, code, URLs, and paths are masked before intent matching, and
typo recovery is restricted to a small cue vocabulary instead of rewriting the
user's content. The raw prompt is never replaced, and diagnostics omit prompt
text and extracted literals.

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
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#july-2026-prompt-understanding-v1).

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

## Deliberate Reasoning v1

`reasoning_engine.py` extends solving from literal arithmetic expressions to
problems stated in words. It is deterministic, dependency-free, uses no `eval`
and no network, and computes every answer with exact rational arithmetic, so
`10% of 0.1` is `1/100` rather than a float approximation.

Fifteen solver families are covered: percentages and percent change, ordered
percent chains such as a discount followed by tax, unit conversion across
length, mass, volume, time, data, area, speed, and temperature, linear
equations, speed-distance-time, combined work rates, proportions, sequences,
statistics, gcd/lcm/primality/factorization, combinatorics, date differences
and offsets, simple and compound interest, and sum-and-difference problems.

Verification is a precondition for authority rather than a report:

- each solver publishes how its answer was checked and whether that check is
  independent of the path that produced it;
- a linear equation is re-checked by a second substitution evaluator written
  against a different strategy than the symbolic collector that solved it;
- a unit conversion must pass both an exact round trip and a magnitude
  direction test, because a round trip alone cancels an inverted factor;
- a sequence rule must hold for every supplied term, not just the last pair;
- `deep` tier runs every applicable solver and any disagreement withdraws
  override authority.

Compute is adaptive and bounded. A deterministic complexity score selects
`fast`, which stops at the first self-verified path, or `deep`, which explores
all paths and requires agreement. Solver count, literal digit length, list and
sequence sizes, factorial and combination sizes, date deltas, and result bit
width are all capped.

A computed answer replaces a retrieved response at exactly one point,
`finalize_grounded_response`, and only when the problem is solved, its
verification passed, and no solver disagreed. Explicit arithmetic keeps its
existing dedicated path and takes precedence; the strict-evidence override
outranks both. If the request asks for working, the recorded steps are
included. The layer has no routing, compute, or adaptive-exit authority, and
its diagnostics carry class, method, verification, consensus, and budget only —
never the prompt, the extracted numbers, or the answer.

The same `settings.grounding_intelligence=false` and `grounding_enabled=False`
switches disable it for raw fidelity evaluation. Design rationale, the papers
that motivate it, and the evaluation boundary are in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#july-2026-deliberate-reasoning-v1).
This upgrade changes runtime logic and tests only; no model weights were
retrained, so it is not evidence of a smarter trained checkpoint.

Build a deterministic verifier-grounded Qwen curriculum:

```powershell
python source/build_verifiable_reasoning_curriculum.py `
  --output-dir output/verifiable_reasoning_curriculum_v1 `
  --train-rows 2000 `
  --eval-rows 400
```

The generated train/evaluation templates are disjoint and every included answer
passes a safe deterministic verifier. The Qwen pipeline revalidates tagged
teacher caches and reports verified accuracy by problem family. This adds a
training and promotion path; it does not claim that model weights improved
until an adapter is trained and passes the fixed held-out gate. Design details
and research boundaries are in
[`source/RESEARCH_UPGRADES.md`](source/RESEARCH_UPGRADES.md#july-2026-grounded-problem-solving-and-verifier-grounded-training-v1).

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
