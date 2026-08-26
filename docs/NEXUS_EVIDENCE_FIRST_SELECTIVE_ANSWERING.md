# NexusMind evidence-first selective answering

Status: implemented in the source-tree experimental Nexus surfaces on
2026-08-26. This change does not promote, activate, package, or alter a trained
model.

## Outcome

Nexus now distinguishes a verified closed-world answer from a useful but
non-authoritative artifact. Every orchestrated result carries a
`nexus-selective-answer-v1` receipt and exactly one decision:

| Decision | What may produce it | Correctness confidence | Answer authority |
|---|---|---:|---:|
| `answered` | Fresh accepted output from `grounding_runtime.finalize_grounded_response` | `1.0`, deterministic in-scope only | Yes, only for the declared closed-world claim scope |
| `analysis_only` | Persona, ideation, template swarm, or template graph output | unavailable | No |
| `abstained` | No applicable verifier, untrained neural probe, or unavailable tool execution | unavailable | No |

The receipt includes the evidence class, reason, claim scope, explicit
limitations, inference protocol, verifier metadata, and authority bits. Its
SHA-256 checksum detects mutation of the receipt; the checksum is not factual
evidence and grants no tool, permission, safety, memory, routing, activation,
or promotion authority.

## Research basis

The implemented slice is deliberately smaller than a full test-time scaling
system:

- [T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models](https://arxiv.org/abs/2504.04718)
  reports that small models struggle to verify memorization-heavy facts and
  calculations, and benefits from offloading those checks to external tools.
  Nexus therefore admits calculations through the existing deterministic
  verifier instead of trusting its neural or template score.
- [Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility](https://arxiv.org/abs/2608.04001)
  treats the complete inference system as the evaluated object and calls for
  protocol-matched compute and uncertainty reporting. Nexus receipts now state
  the regime, candidate count, generator/verifier calls, and executed tool
  count instead of collapsing them into one “confidence” number.
- [Uncertainty-Aware Abstention in Large Language Models with Provable Alignment Guarantees](https://arxiv.org/abs/2607.04430)
  explains why a heuristic uncertainty threshold alone does not establish a
  risk guarantee; held-out calibration and an error bound are required. Nexus
  therefore publishes no correctness confidence for heuristic outputs and
  makes unverified feedback fail closed.
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)
  motivates meaning-level uncertainty over multiple generations. This is a
  future evaluation candidate, not a current Nexus feature: the experimental
  MiMo path has no loaded text decoder or candidate-generation distribution to
  measure.

These papers motivate the engineering direction; they do not validate this
implementation. The local regression suite and future held-out evaluations are
the evidence for this codebase.

## Admission boundary

`source/nexus_engine.py` and direct `/v1/solve` or `/v1/scientific` calls run a
fresh grounding pass. An answer is admitted only when the grounding reason is
`explicit_arithmetic_exact` or `verified_reasoning_solution` and the returned
text is non-empty. Scientific admission additionally requires a present,
allowlisted science plan.

The broader `NexusSolver` pattern library remains available as an audit and
development fixture. If it matches after the strict gate refuses the request,
only bounded match/formula/schema metadata is recorded; the full receipt and
numeric candidate are withheld. Regression cases include:

- negated instructions;
- a valid calculation mixed with an open-world prediction;
- ambiguous alternative quantities;
- quoted or documentation examples that should only be explained;
- rolling-body energy that requires rotational terms;
- mechanical work with an angle the legacy pattern would ignore;
- lunar potential energy without an explicit local gravitational constant.

## Experimental components retained

- **MiMo fast/deep:** a newly initialized architecture probe, not a trained
  language generator. It consumes at most 64 character ordinals in this path,
  publishes latent telemetry, and abstains.
- **Agent:** accepts tool declarations as metadata but has no tool executor.
  Declared tools are never counted as calls and cannot become evidence.
- **Swarm:** five deterministic role templates and replicator-style weight
  updates. The internal score means template agreement, not correctness.
- **Graph-of-Thoughts:** deterministic prefix templates with positional
  priorities. Requested depth and beam bounds are enforced; the selected path
  is neither an optimality proof nor a verified answer.
- **Ideation:** SCAMPER/TRIZ/analogy concept hypotheses. FNIR values are authored
  priorities, projected benefits are framed as hypotheses, and every proposal
  asks for baseline, matched-compute, failure, rollback, and held-out metrics.
- **Persona chat:** conversation scaffolding with no factual answer authority.

## Surface behavior

The API model catalog now exposes three truthful capability identities:

- `nexus-exact-solver`;
- `nexus-heuristic-suite`;
- `nexus-experimental-neural-telemetry`.

It no longer advertises 262,144- or 1,048,576-token context windows for a path
that currently consumes 64 characters and has a configured sliding window of
128 tokens. `/v1/feedback` retains its route for compatibility but rejects
unverified policy updates. Diagnostic entropy and CHSH values are nested under
`synthetic_metric_probe` and explicitly identify that their fixed inputs are
not live model outputs.

The CLI renders “Verified Closed-World Answer,” “Analysis Only,” or “Answer
Withheld” and handles unavailable confidence without formatting it as a number.
The browser sends `query` to `/v1/solve` and `topic` to `/v1/innovate`, validates
analysis contracts, escapes returned text, and uses no local result generator.
Network failure produces `DEMO / BACKEND UNAVAILABLE` with no fabricated
answer, score, receipt, or telemetry. The API serves the page at `/studio` so
the UI and evidence endpoints share one origin.

## Verification

Focused checks:

```powershell
python -m pytest -q test_nexus_epistemics.py test_nexus_engine.py `
  test_nexus_api.py test_nexus_swarm.py test_nexus_got.py `
  test_nexus_ideation.py test_nexus_chat.py test_nexus_solver.py `
  test_nexus_studio_contract.py
python -m py_compile source/nexus_epistemics.py source/nexus_engine.py `
  source/nexus_api.py source/nexus_cli.py source/nexus_swarm.py `
  source/nexus_got.py source/nexus_ideation.py
```

The focused suite covers receipt invariants and tamper failure, admitted exact
arithmetic and science, false-positive abstention families, zero executed-tool
accounting, no self-training from internal scores, truthful API catalog data,
request-bound enforcement, non-quantified ideation claims, and removal of all
browser-local success fallbacks.

## Non-claims and next experiments

- No open-world factual answer path was added.
- No empirical calibration or finite-sample risk guarantee is claimed.
- No semantic-entropy estimator was implemented without a real generation
  distribution.
- No neural checkpoint was loaded, trained, promoted, or activated by this
  change. The concurrent v80 training process and its corpus were left alone.
- A future multi-candidate path should remain shadow-only until it has a frozen
  candidate generator, an eligible external verifier, matched-compute baselines,
  held-out calibration, risk/coverage curves with intervals, protocol receipts,
  and explicit non-regression gates.
