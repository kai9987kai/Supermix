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

As of July 18, 2026:

- `source/` is the active Supermix Studio runtime and packaging tree
- the curated desktop build selects `11` core model artifacts and leaves expansion to the model store
- the route control plane includes durable lifecycle evidence, Policy Lab diagnostics,
  bounded-exposure rehearsal, and a fail-closed stateful experiment protocol preflight
- `runtime_python/` remains a legacy compatibility snapshot for the smaller chat runtime;
  it is not the source of truth for the multimodel Studio route control plane
- the Windows installer contract version is `2026.07.18`

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
runtime code digest. Cluster identifiers are treated as opaque private inputs;
v1 does not validate membership in an external cluster map or canonicalize
aliases. Both checks remain external prerequisites for any future experiment.

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
`{"cluster_identifier":"..."}`. The CLI derives and stores a study-scoped
pseudonym, not that raw identifier. Keep both the cluster input and the
separately created seed capsule private; pseudonymity is not anonymity, and
post-reveal unlinkability is not guaranteed. Before closure the registry stores
only an opaque assignment commitment. After closure, `reveal` opens the seed and
`verify` reconstructs each frozen assignment. The browser exposes only a GET
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

The v51 stability benchmark supports a research-derived, shadow-only
distribution-drift diagnostic:

```powershell
python source/benchmark_v51_prediction_stability.py --distribution_top_k 5
```

It reports consecutive-prefix top-k Jensen-Shannon divergence with a retained
`other` mass bucket. The metric is telemetry only and does not authorize an
early exit. A 480-request CPU pilot retained 96/96 agreement for the current
patience-2/tolerance-0.005 setting with 28.8% fewer cycles; the release gate is a
larger fresh-seed confirmation, not that pilot result.

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
