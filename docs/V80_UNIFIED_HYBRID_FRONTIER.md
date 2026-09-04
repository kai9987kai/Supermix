# Supermix v80: bounded hybrid diagnostics

## Status

V80 is an experimental integration layer, not a promoted language model. It
combines architecture probes, deterministic analysis scaffolds, and the existing
closed-world grounding runtime behind the v78.2 proof-carrying selective-answer
contract.

Only fresh, strictly accepted output from
`grounding_runtime.finalize_grounded_response` can receive answer authority.
Every orchestrated reasoning result is otherwise either `analysis_only` or
`abstained`; diagnostic and health endpoints use their own non-answer schemas.

V82 extends this boundary with a separate
[`calibrated verify-or-defer lab`](NEXUS_CALIBRATED_VERIFY_OR_DEFER.md). Its
finite-sample receipts are conditional offline evidence only. The adaptive
mode reports shadow recommendations separately from the applied safe ACT cap;
it does not claim that authored heuristics or untrained optional mechanisms are
calibrated, efficient, or answer-producing.

## Runtime map

```text
Client / Nexus Studio
        |
        v
Nexus API admission boundary
   |                 |
   |                 +-- deterministic analysis scaffolds
   |                     (persona chat, ideation, swarm, graph search)
   |
   +-- fresh grounding recomputation
   |      +-- exact arithmetic
   |      +-- allowlisted science/reasoning plans
   |      +-- proof capsule + renderer revalidation
   |
   +-- untrained neural architecture diagnostics
          (fast/deep abstain; no text decoder)
```

## MiMo-lineage probe

The local module contains sparse-MoE, hybrid local/global attention, attention
sinks, MTP heads, and bounded latent-cycle machinery. In the Nexus path these
components are newly initialized and consume at most 64 character ordinals for
diagnostic execution. They do not load a trained Nexus text checkpoint and do
not emit an answer candidate.

Consequently, v80 makes no current claim of:

- infinite context or streaming;
- measured KV-cache reduction;
- measured 2–3x decoding acceleration;
- calibrated dynamic chain-of-thought allocation; or
- checkpoint-backed open-domain generation.

Those are architecture hypotheses that require a trained artifact and a fixed,
reproducible evaluation protocol.

## Software sampling and signal diagnostics

The entropy endpoint exposes four software mechanisms with explicit provenance:

1. `crypto`: Python `secrets`, delegating randomness to the host OS.
2. `seeded`: reproducible Python PRNG output for tests.
3. `os_csprng_transform`: `os.urandom` bytes followed by a numeric sine
   transform for visualization. It is not a QRNG, uses no quantum hardware, and
   makes no cryptographic-security claim for the transformed stream.
4. `chaotic`: a deterministic logistic-map sequence.

The cellular-automata grid is a deterministic visualization and is not a
cryptographic generator.

The RSI component computes a descriptive statistic over caller-supplied numeric
sequences. Raw `NexusEngine.process` telemetry identifies its synthetic
step-count sine probe, while `/v1/signals` identifies its constant `0.5` probe.
Both records state `is_live_reasoning_signal=false`; `/v1/think` intentionally
withholds that diagnostic. None measures reasoning quality, novelty, or
stability.

The v80 tabular Q-learning object is a disconnected experiment initialized from
authored priors. It is reported for inspection but does not control the live
Nexus process. Public feedback remains fail-closed, and unverified outputs do
not update either routing policy.

## Evidence and analysis contracts

- Exact arithmetic and allowlisted science/reasoning outputs require a selected,
  solved, verification-passed `supermix-verified-answer-receipt-v2` that is
  consistent with the grounding reason, method, result data, and rendered text.
- Answered outputs also require a closed `nexus-selective-answer-v2` receipt and
  complete `nexus-proof-carrying-number-v2` span capsule bound to request,
  output, grounding receipt, surface, and a mandatory valid nonce. The capsule
  must include a passing independent witness; unsupported families defer.
- `confidence=null` is deliberate. An accepted result reports
  `deterministic_assurance_not_probability`; fresh checking reruns the same
  implementation and is not empirical calibration or algorithmic independence.
- Persona chat first offers eligible closed-world turns to the strict grounder.
  Otherwise chat, ideation, swarm, and graph search expose authored internal
  priorities with no answer authority.
- Fast, deep, and agent modes abstain. Tool declarations are not executions.
- SHA-256 receipts detect accidental or post-creation mutation of metadata.
  They are not signatures, evidence, permission, or proof of factual truth.
- Nexus Studio withholds exact candidates until `POST /v1/verify` freshly
  reconstructs and exactly matches the expected capsule.

## API surface

| Endpoint | Role |
| --- | --- |
| `POST /v1/think` | Evidence-gated router; exact answers are recomputed again at the API boundary |
| `POST /v1/solve` | Strict exact arithmetic or allowlisted reasoning |
| `POST /v1/scientific` | Strict science-plan path |
| `POST /v1/verify` | Fresh renderer revalidation and exact proof-capsule comparison |
| `POST /v1/innovate` | Analysis-only authored concept scaffolds |
| `POST /v1/swarm` | Analysis-only deterministic role templates |
| `POST /v1/got` | Analysis-only deterministic graph scaffold |
| `POST /v1/chat` | Strict proof-carrying closed-world answer when eligible; otherwise analysis-only persona scaffold |
| `POST /v1/entropy` | Software sampling plus deterministic CA visualization |
| `GET /v1/signals` | Disconnected policy and synthetic RSI diagnostics |
| `GET /v1/telemetry` | Configuration and explicitly synthetic metric probes |
| `GET /v1/models` | Observed runtime capability catalog |
| `GET /health` | Service readiness |
| `GET /studio` | Same-origin experimental browser interface |

## Verification

Run the focused compatibility and evidence suite:

```powershell
python -m pytest -q test_nexus_proof.py test_nexus_epistemics.py `
  test_nexus_engine.py `
  test_nexus_api.py test_nexus_swarm.py test_nexus_got.py `
  test_nexus_ideation.py test_nexus_chat.py test_nexus_solver.py `
  test_nexus_studio_contract.py test_nexus_hybrid_advancements.py
```

Do not replace this with a hard-coded pass count; use the current test output as
the receipt for the exact tree being reviewed.
