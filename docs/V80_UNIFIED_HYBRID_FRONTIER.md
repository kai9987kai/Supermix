# Supermix v80: bounded hybrid diagnostics

## Status

V80 is an experimental integration layer, not a promoted language model. It
combines architecture probes, deterministic analysis scaffolds, and the existing
closed-world grounding runtime behind the v78.1 selective-answer contract.

Only fresh, strictly accepted output from
`grounding_runtime.finalize_grounded_response` can receive answer authority.
Everything else is either `analysis_only` or `abstained`.

## Runtime map

```text
Client / Nexus Studio
        |
        v
Nexus API admission boundary
   |                 |
   |                 +-- deterministic analysis scaffolds
   |                     (ideation, chat, swarm, graph search)
   |
   +-- fresh grounding recomputation
   |      +-- exact arithmetic
   |      +-- allowlisted science/reasoning plans
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
sequences. Nexus currently feeds it a synthetic step-count sine probe, while the
signals endpoint feeds a constant `0.5` probe. Both responses state their input
source and `is_live_reasoning_signal=false`; neither measures reasoning quality,
novelty, or stability.

The v80 tabular Q-learning object is a disconnected experiment initialized from
authored priors. It is reported for inspection but does not control the live
Nexus process. Public feedback remains fail-closed, and unverified outputs do
not update either routing policy.

## Evidence and analysis contracts

- Exact arithmetic and allowlisted science/reasoning outputs require a selected,
  solved, verification-passed `supermix-verified-answer-receipt-v2` that is
  consistent with the grounding reason, method, result data, and rendered text.
- `confidence=1.0` means deterministic agreement only within that accepted
  closed-world parse. It is not empirical calibration.
- Ideation, persona chat, swarm, and graph search expose authored internal
  priorities with `confidence=null` and no answer authority.
- Fast, deep, and agent modes abstain. Tool declarations are not executions.
- SHA-256 receipts detect accidental or post-creation mutation of metadata.
  They are not signatures, evidence, permission, or proof of factual truth.

## API surface

| Endpoint | Role |
| --- | --- |
| `POST /v1/think` | Evidence-gated router; exact answers are recomputed again at the API boundary |
| `POST /v1/solve` | Strict exact arithmetic or allowlisted reasoning |
| `POST /v1/scientific` | Strict science-plan path |
| `POST /v1/innovate` | Analysis-only authored concept scaffolds |
| `POST /v1/swarm` | Analysis-only deterministic role templates |
| `POST /v1/got` | Analysis-only deterministic graph scaffold |
| `POST /v1/chat` | Analysis-only persona scaffold |
| `POST /v1/entropy` | Software sampling plus deterministic CA visualization |
| `GET /v1/signals` | Disconnected policy and synthetic RSI diagnostics |
| `GET /v1/telemetry` | Configuration and explicitly synthetic metric probes |
| `GET /v1/models` | Observed runtime capability catalog |
| `GET /health` | Service readiness |
| `GET /studio` | Same-origin experimental browser interface |

## Verification

Run the focused compatibility and evidence suite:

```powershell
python -m pytest -q test_nexus_epistemics.py test_nexus_engine.py `
  test_nexus_api.py test_nexus_swarm.py test_nexus_got.py `
  test_nexus_ideation.py test_nexus_chat.py test_nexus_solver.py `
  test_nexus_studio_contract.py test_nexus_hybrid_advancements.py
```

Do not replace this with a hard-coded pass count; use the current test output as
the receipt for the exact tree being reviewed.
