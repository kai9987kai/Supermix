# NexusMind evidence-first selective answering

Status: v2 implemented in the source-tree experimental Nexus surfaces on
2026-08-27. This change does not promote, activate, package, or alter a trained
model.

V82 adds a separate shadow-only calibrated verify-or-defer lab. It evaluates a
frozen arithmetic/adversarial cohort with exact Clopper--Pearson bounds and a
precommitted policy grid, but its exchangeability and label-independence
assumptions remain explicitly unestablished. It cannot select a live route or
grant answer authority; see
[`NEXUS_CALIBRATED_VERIFY_OR_DEFER.md`](NEXUS_CALIBRATED_VERIFY_OR_DEFER.md).

## Outcome

Nexus distinguishes a verified closed-world answer from a useful but
non-authoritative artifact. Every orchestrated result carries a
`nexus-selective-answer-v2` receipt and exactly one decision:

| Decision | What may produce it | Correctness confidence | Answer authority |
|---|---|---:|---:|
| `answered` | Fresh accepted output from `grounding_runtime.finalize_grounded_response`, bound to the request and exact output | unavailable; deterministic assurance is reported separately | Yes, only for the declared closed-world claim scope |
| `analysis_only` | Persona, ideation, template swarm, or template graph output | unavailable | No |
| `abstained` | No applicable verifier, untrained neural probe, or unavailable tool execution | unavailable | No |

The receipt includes the evidence class, reason, claim scope, explicit
limitations, inference protocol, closed verifier metadata, request/output
bindings, and authority bits. Its SHA-256 checksum detects mutation of the
receipt; it is not a signature or factual evidence and grants no tool,
permission, safety, memory, routing, activation, or promotion authority.

Answered receipts publish `confidence=null` and
`confidence_kind=deterministic_assurance_not_probability`. The fresh verifier
call reruns the same deterministic implementation, so
`algorithmically_independent=false` is explicit.

## Research basis

The implemented slice is deliberately smaller than a general reasoning or
test-time-scaling system:

- [T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models](https://arxiv.org/abs/2504.04718)
  motivates moving eligible calculation checks out of an untrusted generator.
  Nexus admits calculations through its deterministic grounder instead of a
  neural or template score.
- [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) places numeric
  checking in the renderer and binds numeric tokens to structured claims.
  Nexus v78.2 adopts that fail-closed presentation boundary for its small,
  allowlisted numeric domain.
- [Proof-Carrying Reasoning with Large Language Models](https://arxiv.org/abs/2511.08392)
  motivates explicit premises, rules, and conclusions. Nexus records supported
  derivation literals and methods, but does not claim a general formal-logic
  proof system.
- [Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2602.03975)
  combines structured feasibility gates with selective verification. Nexus
  implements the deterministic feasibility gate and fixed fresh revalidation;
  learned compute allocation remains future work.
- [Conformal Selective Prediction with General Risk Control](https://arxiv.org/abs/2603.24704)
  shows that finite-sample risk control needs a calibration procedure and
  assumptions such as exchangeability. Nexus therefore does not turn exact
  recomputation into a numeric confidence or risk guarantee.

These papers motivate the engineering direction; they do not validate this
implementation. The repository tests and live request-path checks are the
evidence for this codebase.

## Admission boundary

`source/nexus_engine.py` and direct `/v1/think`, `/v1/solve`,
`/v1/scientific`, or `/v1/chat` calls run a fresh grounding pass. An answer is
admitted only when the grounding reason is `explicit_arithmetic_exact` or
`verified_reasoning_solution`, the runtime and receipt schemas match the
current source, the receipt is selected, solved, and verification-passed, no
conflict is present, and the returned text is consistent with structured result
data. Scientific admission additionally requires a present, allowlisted science
plan.

The broader `NexusSolver` pattern library remains an audit and development
fixture. If it matches after the strict gate refuses the request, only bounded
match/formula/schema metadata is recorded; the full receipt and numeric
candidate are withheld. Regression families include negation, quotations,
mixed open-world requests, ambiguity, incomplete quantities, ignored physical
terms, and missing local constants.

`/v1/chat` now tries this strict boundary before persona scaffolding. Supported
closed-world math or science receives the same proof-carrying answer as Solver;
the exact turn deliberately does not update persona conversation state.
Unsupported and open-world chat remains `analysis_only`.

## Request and presentation binding

An answered v2 receipt binds SHA-256 digests of the exact request, exact public
output, current grounding receipt, and mandatory valid request nonce, plus an
allowlisted surface identifier. Unknown receipt, verifier, binding, or authority
fields fail closed.

Public answer authority also requires a passing algorithmically independent
checker witness. The currently supported witness families are arithmetic and
allowlisted science; a grounder-only geometry, rate, finance, or other result is
retained as diagnostic evidence but receives no proof capsule and must defer.

The stricter presentation contract is documented in
[`NEXUS_PROOF_CARRYING_CONVERSATION.md`](NEXUS_PROOF_CARRYING_CONVERSATION.md).
In summary, every numeric span must have an allowlisted role, and the Studio
does not reveal an answer until `/v1/verify` freshly reconstructs and exactly
matches the entire expected capsule.

## Experimental components retained

- **MiMo fast/deep:** a newly initialized architecture probe, not a trained
  language generator. It consumes at most 64 character ordinals in this path,
  publishes latent telemetry, and abstains.
- **Agent:** accepts tool declarations as metadata but has no tool executor.
  Declared tools are never counted as calls and cannot become evidence.
- **Swarm:** five deterministic role templates and replicator-style weight
  updates. The internal score means template agreement, not correctness.
- **Graph-of-Thoughts:** deterministic prefix templates with positional
  priorities. The selected path is neither an optimality proof nor an answer.
- **Ideation:** SCAMPER/TRIZ/analogy hypotheses. FNIR values are authored
  priorities, not measured benefits; proposals ask for matched evaluation and
  rollback evidence.
- **Persona chat:** conversation scaffolding with no factual answer authority;
  only the separate strict grounder path may answer eligible closed-world turns.

## Surface behavior

The API capability catalog exposes `nexus-exact-solver`,
`nexus-heuristic-suite`, and `nexus-experimental-neural-telemetry`. It does not
advertise context windows the active path does not consume. `/v1/feedback`
retains its compatibility route but rejects unverified policy updates.

The CLI renders “Verified Closed-World Answer,” “Analysis Only,” or “Answer
Withheld” and does not format unavailable confidence as a number. The same-origin
Studio sends a cryptographically generated 128-bit nonce, validates the v2
receipt and capsule shapes, submits the exact candidate to `/v1/verify`, and
only then reveals verified numeric output. A stale response, contract failure,
or network failure produces a fixed non-answer state without echoing the
candidate. The verified view itself is rebuilt from capsule-bound result fields
inside a fixed wrapper; unbound candidate prose, reasoning steps, aliases,
model labels, and latency do not inherit the verified badge.

## Verification

Focused checks:

```powershell
python -m pytest -q test_nexus_proof.py test_nexus_epistemics.py `
  test_nexus_engine.py test_nexus_api.py test_nexus_swarm.py `
  test_nexus_got.py test_nexus_ideation.py test_nexus_chat.py `
  test_nexus_solver.py test_nexus_studio_contract.py `
  test_nexus_hybrid_advancements.py
python -m py_compile source/nexus_proof.py source/nexus_epistemics.py `
  source/nexus_engine.py source/nexus_api.py source/nexus_cli.py `
  source/nexus_swarm.py source/nexus_got.py source/nexus_ideation.py
```

The focused suite covers closed-schema receipt invariants,
request/output/surface/nonce
binding, exact arithmetic and science, proof-span completeness, replay and
rehash attacks, Unicode numeric confusables, stale runtime/schema rejection,
false-positive abstention families, zero executed-tool accounting, no
self-training from internal scores, and removal of browser-local success
fallbacks.

## Non-claims and next experiments

- No open-world factual answer path was added.
- No empirical calibration, finite-sample risk guarantee, or numeric correctness
  probability is claimed.
- The capsule and receipt checksums are not signatures. A caller that only runs
  local integrity checking has not established provenance; the Studio therefore
  requires fresh `/v1/verify` comparison.
- Fresh verification uses the same implementation and is not algorithmically
  independent.
- Numeric claim coverage is not a general formal proof of natural-language
  reasoning.
- No neural checkpoint was loaded, trained, promoted, or activated by this
  change. Existing training data and background processes were left alone.
- The v82 adaptive telemetry probe and calibrated selective-risk lab stay
  shadow-only. Live influence still requires frozen held-out data,
  matched-compute baselines, risk and coverage intervals, OOD checks, protocol
  receipts, and explicit non-regression gates.
