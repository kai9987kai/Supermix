# Supermix v54 — Verified Probabilistic Scenarios

## What v54 is

Supermix v54 is an additive deterministic-runtime release. It adds one bounded
solver family for exact finite Bernoulli events while preserving the existing
v52 model line, v53 MiMoMix research stack, Deliberate Reasoning v3 behavior,
and Qwen Promotion Evidence v4 lifecycle.

V54 is not a new checkpoint or adapter. It does not claim broader model
intelligence, improved open-world forecasting, or validation of assumptions
supplied by a user.

| surface | v54 contract |
|---|---|
| product version | `54.0.0` |
| interaction planner | `supermix-plan-evaluate-v3` |
| reasoning schema | `supermix-reasoning-v2` (compatible) |
| reasoning engine | `supermix-reasoning-engine-v4` |
| scenario IR | `supermix-finite-bernoulli-scenario-v1` |
| source of truth | `source/reasoning_engine.py` |
| compatibility mirror | `runtime_python/reasoning_engine.py` |
| grounding authority gate | source and compatibility `grounding_runtime.py` |
| maximum trials | `200` |

## Accepted model

The parser must consume one complete request. It accepts a finite Bernoulli
model only when the user explicitly supplies one of these forms:

- IID Bernoulli trials with an exact success probability;
- independent trials with an explicitly `fixed`, `constant`, or `same` success
  probability; or
- IID fair-coin tosses or flips, which establish probability `1/2`.

The question must ask for `exactly`, `at least`, or `at most` a bounded count of
successes, heads, or tails. Trial counts range from 1 through 200, and the event
count must be between zero and the trial count. Probabilities may be written as
a fraction, bounded decimal, or percent and must lie in `[0, 1]`.

Examples within the grammar:

```text
Assuming 5 independent Bernoulli trials with fixed success probability of 1/2,
what is the probability of exactly 3 successes?
```

```text
Assuming 3 IID Bernoulli trials with success probability of 25%,
what is the probability of at least 1 success?
```

```text
Assuming 4 IID fair coin tosses,
what is the probability of at most 1 head?
```

The canonical scenario IR contains no raw prompt. It records the model kind,
trial count, event relation and count, outcome, exact probability numerator and
denominator, and a full-query-consumed flag.

## Exact computation

For `X ~ Binomial(n, p)`, the primary path evaluates the required point mass or
tail using exact integer combinations and rational arithmetic:

```text
P(X = k) = C(n, k) p^k (1-p)^(n-k)
```

`at least` and `at most` events sum the corresponding exact masses. No random
sampling or floating-point comparison decides correctness.

## Independent verification

The verifier does not replay the combination formula. It starts from mass one
at zero trials and repeatedly convolves with one Bernoulli distribution until
it reconstructs all `n + 1` outcome masses. The result passes only when:

1. the direct binomial result equals the convolution-derived event mass;
2. every reconstructed mass is non-negative;
3. the full distribution sums exactly to one; and
4. the event plus its complement sums exactly to one.

The runtime reports `bernoulli_convolution_and_mass_check` as the independent
verification method. Normal exhaustive solver consensus still applies before a
verified result can replace retrieved or generated text.

## Authority and abstention boundary

An accepted result is authoritative only for the mathematical model explicitly
stated in the request. Diagnostics mark it as model-conditional and record that
assumptions were explicit; they do not claim calibration.

The capability abstains on:

- dependent trials or changing, different, unknown, or unstated probabilities;
- sampling without replacement;
- multiple or incomplete event questions;
- out-of-range counts, probabilities, or trial budgets;
- quoted, negated, corrected, or unrelated trailing instructions;
- requests for certainty or guarantees; and
- medical, financial, legal, election, weather, or other high-stakes/open-world
  predictions.

The grounding layer provides a second authority boundary. When a reasoning
result claims `finite_binomial_event_probability`, grounding reparses the
original raw request using the loaded reasoning implementation and requires the
same canonical scenario schema and full-consumption marker. A stale or replaced
engine cannot gain rewrite authority merely by fabricating the method name.

## Runtime and packaging integration

The source and compatibility reasoning files are intended to remain
byte-identical. The Studio manifest binds both module hashes and their scenario,
engine, and result-schema constants. Its package guards state that Verified
Probabilistic Scenarios are available but have no open-world authority.

The Qwen desktop packaging already includes both reasoning files, so an embedded
solver needs no additional data module. The Studio application follows the
source import graph and includes its checked runtime manifest. If the solver is
ever split into another module, that file must also be added to the Qwen specs,
both Qwen build scripts, Studio runtime-module manifest, and CI compile list.

The Windows Studio installer and runtime manifest share product version
`54.0.0`. The installer helper hashes the resolved setup basename, including a
custom release name, rather than assuming the historical default filename.

## Release evidence required

Before v54 is published:

1. confirm source/runtime byte parity for reasoning and grounding;
2. regenerate and review `source/studio_runtime_manifest.json`;
3. run the model-snapshot and Studio-manifest checks;
4. run focused v54 regressions and the complete test suite;
5. parse changed PowerShell scripts and run `git diff --check`;
6. probe exact, tail, boundary, malformed, adversarial, and stale-engine cases
   through source, compatibility runtime, and live Studio/Qwen surfaces;
7. freeze and inspect the desktop application, then repeat live launches;
8. build, install, upgrade, and uninstall the per-user installer; and
9. independently recompute the release SHA-256 values.

Generated EXEs, installers, a passing manifest, or an installed application are
release evidence, not consequences of this design document. They must be
produced and verified separately from the source change.
