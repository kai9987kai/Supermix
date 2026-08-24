# Supermix v71: Verified Scientific Scenarios

## Outcome

V71 adds a bounded science capability to the active `source/` runtime and the
`runtime_python/` compatibility tree. It is a deterministic verifier, not a new
training claim and not a general scientific oracle.

The capability resolves a small set of fully specified, single-target
scenarios that v54 previously rejected even when the user supplied the formula
assumption and all quantities.

| scenario | targets | registry equation |
| --- | --- | --- |
| constant acceleration | final velocity, displacement | `v=u+a*t`; `s=u*t+(a*t^2)/2` |
| ideal gas | pressure, volume, temperature, amount | `P*V=n*R*T` |

## Trust boundary

The flow is intentionally one-way:

```text
user text
  -> strict parser and source-span bindings
  -> non-executable science-plan JSON
  -> versioned local formula registry
  -> exact SI normalisation and execution
  -> integrity, dimension, domain, and substitution checks
  -> model-conditional answer plus redacted public receipt
```

The local model is not an executor or verifier. No expression from a prompt is
passed to `eval`, `exec`, a shell, a network service, or generated Python. The
registry is immutable module data whose canonical JSON and SHA-256 digest are
included in every plan and internal verification receipt.

A successful plan must satisfy all of these conditions:

1. The request is one line, one supported scenario, and one target.
2. The model assumption is explicit: constant acceleration or ideal gas.
3. Every required input appears exactly once with a supported unit.
4. Every number and content word is consumed by the narrow grammar.
5. Input spans are disjoint and their SHA-256 bindings remain intact.
6. The plan and registry digests match their canonical representations.
7. SI dimensions, quantity domains, and equation substitution all pass.
8. Existing reasoning solvers find no conflicting verified result.
9. Grounding reparses the raw request and independently recomputes the trusted
   reasoning result before selecting it.

Failure at any boundary produces a fixed reason and no override.

## Exact arithmetic and units

Numeric literals enter as `Fraction` values through `Decimal`; binary floats are
not used for formula execution. Supported conversions include common velocity,
acceleration, time, pressure, volume, temperature, and amount units. Celsius is
handled as an affine conversion to kelvin. Result growth, literal length,
quantity count, plan steps, and query length are bounded.

The molar gas constant is represented as the exact SI decimal
`8.31446261815324`. A displayed terminating decimal may therefore be exact even
when it contains many digits; non-terminating results receive a bounded decimal
approximation while the rational value remains in the internal answer record.

## Receipts

The internal `supermix-science-plan-receipt-v1` contains only allowlisted
verification metadata:

- scenario, target, and formula identifiers;
- registry, query, plan, and input-span digests;
- bounded source offsets and input counts;
- pass/fail bits for registry integrity, plan integrity, bindings, dimensions,
  domain checks, and substitution;
- model-conditional and explicit-assumption flags; and
- authority bits, all false.

It contains no prompt text, source substring, answer, proof trace, retrieved
evidence, conversation history, or model-generated rationale. Before diagnostics
leave the grounding boundary, query and plan hashes are removed. The enclosing
public Verified Answer Receipt also excludes prompt-derived and input-span
digests while retaining allowlisted registry identity, bounded counts, check
bits, epistemic limits, and false authority flags.

## Prediction hardening shipped with v71

V71 separates the presence of forecast-shaped language from a justified
bounded estimate. A fabricated percentage with an irrelevant assumption no
longer satisfies the calibrated-prediction contract. A forecast must include a
topically relevant basis and an explicit uncertainty limit or abstention.

For the narrow empirical Bernoulli case, exact arithmetic can verify an observed
rate such as `7/10 = 70%`. Grounding may select that canonical estimate when a
candidate says `99%`, but only with these semantics:

- independent trials and a constant success probability are stated;
- the estimate is conditional on that model;
- it is not a guarantee; and
- calibration has not been established.

The selection does not grant open-world prediction authority.

## Fail-closed cases

The engine abstains on missing or duplicated quantities, implicit assumptions,
multiple targets, extra calculations, late corrections, quoted or injected
instructions, tampered plans, unsupported formulas, unsafe numeric growth, and
high-stakes or open-world contexts such as clinical dosing, pressure-vessel
operation, markets, or weather forecasts.

Unsupported science remains with the normal model/evidence path or receives a
clarification. It cannot borrow the science-plan receipt.

## Research basis and limits

The design follows tool-first and verifier-feedback results from SciAgent,
T1, and neuro-symbolic formal-verifier work. It also adopts the caution from
test-time-scaling research that repeated sampling against an imperfect verifier
can accumulate false positives. V71 therefore uses one deterministic parse and
one deterministic execution path; it does not sample until something passes.

Primary sources:

- SciAgent: <https://openreview.net/forum?id=N48b6pzMJc>
- T1: <https://arxiv.org/abs/2504.04718>
- Neuro-symbolic formal-verifier feedback: <https://arxiv.org/abs/2505.14479>
- Compute-optimal test-time scaling: <https://arxiv.org/abs/2408.03314>
- Limits of inference scaling via resampling: <https://openreview.net/forum?id=j8H84v6AZ1>
- SciBench: <https://proceedings.mlr.press/v235/wang24z.html>

These papers motivated the architecture; this repository does not claim to
reproduce their systems or results.

## Non-claims

V71 does not establish that the stated physical model describes reality. The
substitution check is not an independent empirical measurement. It does not
support arbitrary formulas, multi-step derivations, uncertainty propagation,
chemistry, clinical decisions, engineering safety margins, live data, or
open-world forecasts. It trains, promotes, or activates no checkpoint or
adapter, and this source change does not claim that local Windows artifacts were
rebuilt, signed, or release-verified.

## Verification

```powershell
python -m pytest -q test_science_plan.py test_reasoning_engine.py `
  test_grounding_runtime.py test_interaction_planner.py test_prompt_understanding.py
python source\sync_runtime_model_variants.py --check
python source\generate_studio_runtime_manifest.py --check
git diff --check
```
