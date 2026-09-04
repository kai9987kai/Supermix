# NexusMind proof-carrying conversation

Status: implemented as the source-tree v82.1 experimental presentation
contract on 2026-09-02. It changes no trained weights, active adapter, package,
or promotion pointer.

## What advanced

The v78.1 verifier decided whether a complete output could be called an answer.
V78.2 closed the gap between that decision and what a user actually sees: an
answered response is displayable in Nexus Studio only when every numeric span is
mechanically bound to the submitted request and survives a fresh renderer-side
verification round trip. V82.1 extends that boundary with mandatory valid nonce
bindings, fail-closed ledger capacity, and a passing independent witness for
every proof capsule.

```text
request + browser nonce
        |
        v
strict grounding recomputation
        |
        +-- unsupported or conflicted --> analysis_only / abstained
        |
        v
closed v2 answer receipt + numeric proof capsule
        |
        v
Studio withholds candidate and calls POST /v1/verify
        |
        +-- any mismatch or failure --> fixed rejection, candidate not rendered
        |
        v
fresh grounding + exact expected-capsule comparison
        |
        v
numeric output may be shown with deterministic-assurance labeling
```

This is a proof-carrying *presentation protocol* over an allowlisted
deterministic runtime. It is not a mathematical proof of arbitrary prose.

## Closed capsule contract

`source/nexus_proof.py` emits `nexus-proof-carrying-number-v2`. The capsule
binds:

- SHA-256 of the exact request text;
- SHA-256 of the exact public output text;
- SHA-256 of the selected display answer;
- SHA-256 of the current grounding receipt;
- SHA-256 of the mandatory valid request nonce;
- the allowlisted answer surface (`chat`, `solve`, `think`, `scientific`, or
  `engine`);
- grounding runtime and receipt schema versions;
- the problem class, method, unit, verified answer representations, and bounded
  derivation literals;
- exact byte offsets and digests for every numeric span; and
- a bounded, passing, algorithmically independent checker witness; unsupported
  claim families receive no capsule and must defer; and
- an overall capsule checksum.

All top-level and nested key sets are closed. Unknown fields fail instead of
being ignored. All tool, permission, safety, memory, route, model-activation,
and model-promotion authority bits are present and false.

The checksum is public and can be recomputed by an attacker. Local capsule
integrity is therefore necessary but insufficient: `/v1/verify` independently
reconstructs the expected capsule from a fresh strict grounding result and
requires full structural equality. A rejected response returns only a fixed
reason and an empty capsule digest; it does not echo the submitted candidate.
Surface-specific admission is reapplied during reconstruction; in particular,
a `scientific` capsule must pass the allowlisted science-plan gate and cannot
be minted from an arithmetic-only result.

## Numeric-span policy

Every ASCII numeric span in an admitted public output must resolve to exactly
one bounded role:

| Role | Allowed source |
|---|---|
| `derived_answer` | exact, fractional, approximate, or supported percentage representation from the accepted structured answer |
| `input_echo` | an exact numeric token present in the submitted request |
| `verified_unit_literal` | a number embedded in an accepted structured unit, such as an exponent |
| `verified_derivation_literal` | a number in an allowlisted structured reasoning step or method literal |

Coverage must be complete and contain at least one derived-answer span. An
extra number—even after the output and capsule checksums are recomputed—fails.
Unicode numeric characters outside the trusted ASCII representation fail
closed instead of being silently normalized.

## Renderer protocol

Nexus Studio uses `crypto.getRandomValues` to create a 128-bit nonce for every
Chat, Solver, and Think request. It validates the v2 answer receipt and proof
capsule shape but does not treat client-side validation as authority. The
candidate remains withheld while the browser submits its exact request, output,
display answer, surface, capsule, and nonce to `/v1/verify`.

The verifier permits a mark only when it returns all of:

- `status=verified`;
- `verified=true`;
- `renderer_may_mark_numeric_claims_verified=true`;
- `confidence=null`; and
- the exact expected `capsule_sha256`.

After that decision, the verified view is reconstructed from capsule-bound
result fields inside a fixed `Verified result:` wrapper. It does not render the
candidate's free-form prose, top-level aliases, reasoning steps, model labels,
or latency under the verified badge. Solver details are likewise limited to
capsule-bound result and receipt-binding fields.

Per-surface sequence counters discard late Chat, Solver, or Think responses so
an older request cannot overwrite a newer view. The API requires a 16-128
character ASCII request nonce for every authoritative answer and verification,
then consumes it after successful verification in a bounded, 15-minute
process-local ledger. Missing or malformed nonces fail before grounding, and a
duplicate nonce is rejected before another grounding pass. A full ledger rejects
new verification without evicting any unexpired nonce. The default ledger is
freshness defense, not authentication, and
does not mutate model, routing, or promotion state. A deployment may opt into
`NexusApiService(verification_nonce_db=...)` or
`--verification-nonce-db` for a bounded SQLite WAL ledger shared across local
workers and restarts. That store contains only nonce SHA-256 digests and
timestamps, never user prompts or answers; clock/availability failures must be
treated as a deployment fail-closed condition.

## Streaming contract

`POST /v1/think` accepts `stream=true` and returns Server-Sent Events under the
`nexus-sse-proof-carrying-v1` contract. The stream emits a start event, ordered
token chunks (`chunk_index` and `chunk_count`), one telemetry event, and a done
event. The telemetry event carries the same proof capsule as the non-streaming
response; clients must still call `/v1/verify` before displaying a verified
numeric result. The stream is therefore a transport optimization, not a
second answer-authority path. Non-exact responses carry an empty capsule and
remain analysis-only or abstained.

## Conversation behavior

Chat first invokes the same strict closed-world grounder used by Solver. Exact
arithmetic and allowlisted science can return a proof-carrying answer. That
branch does not update persona conversation state, preventing a verified number
from being reinterpreted by the unverified chat scaffold.

If strict grounding does not apply, recognizable persona behavior remains
available as `analysis_only`. Open-world prediction, unsupported factual claims,
and ambiguous or mixed-scope prompts receive no proof capsule and no answer
authority.

## Research mapping

- [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) motivates moving
  numeric claim verification to the renderer, using claim-bound spans, and
  defaulting to unverified. Those are the core implemented ideas.
- [PCRLLM](https://arxiv.org/abs/2511.08392) motivates explicit step-level
  premises, rules, and conclusions. V78.2 binds only the structured literals and
  method already emitted by the deterministic grounder; a general logical proof
  language is not claimed.
- [Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2602.03975)
  supports deterministic feasibility gating before spending verifier calls.
  V78.2 uses a fixed one-call renderer recheck rather than a learned allocation
  policy.
- [Conformal Selective Prediction with General Risk Control](https://arxiv.org/abs/2603.24704)
  separates selective trust from uncalibrated scores. Because this project has
  no frozen calibration experiment for these surfaces, answered responses use
  a categorical assurance label and `confidence=null`.

The papers motivate design choices; none is evidence that this implementation
inherits the papers' formal results.

## Independent arithmetic witness

For `problem_class=arithmetic`, the capsule carries a second implementation
witness from `source/nexus_independent_checker.py`. For
`problem_class=scientific_scenario`, it independently reparses the six
allowlisted kinematics and ideal-gas methods with its own unit table and exact
formula dispatch. The checker has no import path to the production grounder or
capsule builder and compares the exact displayed answer and unit. A mismatch
prevents capsule creation and renderer verification. This is deliberately a
scoped cross-check—not a signature, remote attestation, or claim that open-
world answers have independent proof.
The design follows recent work showing that test-time scaling without a
verifier is suboptimal and that factuality tests benefit from distribution-free
finite-sample guarantees: [Scaling Test-Time Compute Without Verification or
RL](https://proceedings.mlr.press/v267/setlur25a.html) and
[FactTest](https://proceedings.mlr.press/v267/nie25a.html).

## Adversarial regression boundary

The focused tests cover:

- appended contradictory or unrelated numbers, including public rehashing;
- changed query, output, display answer, or nonce;
- cross-query and cross-surface replay;
- repeated input/answer numerals with a canonical final answer span;
- unknown top-level and nested capsule fields;
- changed authority or limitation metadata followed by rehashing;
- Unicode numeric confusables;
- stale grounding runtime and receipt schema identifiers;
- fraction, percentage, formula, unit-exponent, conversion, and constant-
  acceleration output families; and
- the actual JSON `POST /v1/verify` route.

Run the focused evidence suite:

```powershell
python -m pytest -q test_nexus_proof.py test_nexus_epistemics.py `
  test_nexus_independent_checker.py `
  test_nexus_engine.py test_nexus_api.py test_nexus_swarm.py `
  test_nexus_got.py test_nexus_ideation.py test_nexus_chat.py `
  test_nexus_solver.py test_nexus_studio_contract.py `
  test_nexus_hybrid_advancements.py
```

## Explicit non-claims

- The capsule is not a signature, remote attestation, authenticated provenance,
  or cryptographic proof of correctness.
- The second call is a fresh recomputation by the same implementation, not an
  algorithmically independent verifier.
- `deterministic_assurance_not_probability` is not empirical calibration,
  selective-risk control, or open-world certainty.
- Numeric-span coverage does not verify every semantic implication in prose.
- Nonce replay defense is bounded freshness protection, not authentication or
  authorization. The default store is process-local; the optional SQLite store
  extends coverage across local workers and restarts but is not a distributed
  trust service.
- No new model checkpoint, decoder, training result, promotion, or deployment
  claim is introduced.
