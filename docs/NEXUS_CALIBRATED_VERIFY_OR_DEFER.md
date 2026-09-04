# Nexus v82: calibrated verify-or-defer lab

Status: implemented as a source-tree, shadow-only evaluation surface. It does
not promote a model, change a runtime pointer, update Q-learning, or grant
answer authority.

## Why this slice

Recent selective-prediction work makes the missing distinction explicit:
reliability claims require a frozen labelled cohort, a precommitted policy
family, finite-sample intervals, and a declared sampling assumption. SAFER
uses exact Clopper--Pearson control and abstains when a cap cannot meet its
target; [SCoRE](https://arxiv.org/abs/2603.24704) and
[CovCal](https://arxiv.org/abs/2605.28365) likewise make coverage conditional
on calibration and exchangeability. See [SAFER](https://arxiv.org/abs/2510.10193)
for the exact sample-then-filter construction. Nexus v82 turns that research
direction into an inspectable protocol without pretending that a synthetic
regression pack is live-traffic evidence.

## Frozen protocol

`source/nexus_risk_control.py` defines:

- a 128-case benchmark: 96 exact integer cases and 32 adversarial cases that
  should abstain;
- four immutable candidate policies with budgets `{1, 2, 4, 8}`;
- a Bonferroni grid regime and a disjoint `dev_then_cal` regime;
- one-sided exact Clopper--Pearson upper bounds on binary accepted-answer
  errors;
- complete policy-by-example matrices, canonical content hashes, and explicit
  exchangeability/label-independence assumptions;
- authority flags that are permanently false for runtime, routes, activation,
  promotion, and answers.

`GET /v1/risk-control` exposes only protocol metadata. `POST
/v1/risk-control/audit` runs the frozen cohort through the existing strict
grounding gate and returns a conditional receipt. `POST
/v1/risk-control/evaluate` accepts a complete caller-supplied matrix for
offline study. None of these endpoints writes state.

The receipt binds the exact hashes of the grounding, proof, independent
checker, nonce ledger, engine, API, and risk-control source files. These are
integrity checks, not signatures or authentication. Any code, verifier,
decoder, prompt protocol, freshness backend, or benchmark change requires a
new binding and a new evaluation.

## Independent scientific witness

V82.1 extends the arithmetic cross-check to every allowlisted science-plan
formula: constant-acceleration final velocity/displacement and the four ideal-
gas targets. The second implementation reparses the submitted prompt, converts
supported units to SI, evaluates exact rational formulas, and checks the
displayed value plus unit. Unsupported or mixed prompts fail closed; this is a
bounded implementation witness, not a semantic proof or production model
promotion signal.

## Renderer freshness

A valid 16-128 character ASCII browser nonce is required for every authoritative
answer and `/v1/verify` call, then consumed after successful verification and
held for 15 minutes in a bounded ledger. The default ledger is process-local;
deployments that need cross-worker or restart coverage can pass
`NexusApiService(verification_nonce_db=...)` or the
`--verification-nonce-db` CLI flag to use SQLite WAL with short serialized
transactions. Only SHA-256 nonce digests and timestamps are stored; prompts,
answers, and capsules are never persisted. Missing, malformed, and replayed
nonces are rejected before a second grounding pass. When all bounded slots hold
live nonces, verification fails closed without evicting them.
This follows nonce-freshness patterns such as [ACME Replay-Nonce in RFC
8555](https://datatracker.ietf.org/doc/html/rfc8555), but the ledger is not
authentication or authorization. `/health` reports whether the process is
using the default process-local backend or the opt-in durable SQLite backend,
without exposing the configured filesystem path.

## Adaptive compute truthfulness

The public `adaptive` mode is now an ACT telemetry probe. Its authored Q/RSI
policy emits `shadow_recommended_cycles`, while runtime applies only an
explicit caller budget or a fixed safe default. Telemetry reports applied cap,
observed cycles, and exit reason separately. Differential attention, Mixture
of Depths, and MLA are reported from observed instantiated telemetry; they are
not inferred from a request and remain disabled by default in the untrained
probe. `answer_authority=false` and `policy_calibrated=false` are invariant.

This aligns with input-adaptive compute research such as
[Learning How Hard to Think](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html)
and [Token Signature](https://proceedings.mlr.press/v267/liu25ci.html): those
results motivate a future predictor, but do not make authored heuristics or
random weights evidence of savings or quality.

## Promotion gate (not yet satisfied)

The lab is not a production governor. Before any policy could influence live
routing, the project still needs frozen held-out data from the intended traffic
family, documented exchangeability or an alternative validity argument,
matched-compute fixed-budget baselines, risk and coverage intervals,
out-of-distribution checks, latency/cost measurements, family-level
non-regression gates, and an independently reviewed promotion receipt. Until
then every result is verify-or-defer evidence: useful for comparison, never an
answer probability or activation signal.
