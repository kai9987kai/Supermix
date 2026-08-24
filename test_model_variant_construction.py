"""Every model variant `build_model` advertises must construct and run.

Two variants were broken and nobody noticed, because nothing in the suite ever
built one:

* `champion_expert_choice` lost the `def forward(self, x):` line at some point,
  stranding its body's `return x` at the end of `__init__`. Constructing it
  raised `NameError: name 'x' is not defined`, and the class had no forward at
  all.
* `megalarge` computed `extra_dim` and then passed the raw `extra_expansion_dim`
  argument, which defaults to `None`, so `build_model("megalarge")` reached
  `torch.empty(out_dim, None)` and raised `TypeError`.

The byte-parity gate did not help: it compares `runtime_python/model_variants.py`
against `source/model_variants.py`, so it faithfully reported the identical break
in both as "current". Parity is not correctness.

This is the cheapest possible guard -- construct each variant, push one batch
through, check the output shape -- and it is enough to catch both classes of
failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import model_variants  # noqa: E402

#: The names `build_model` dispatches on, excluding the ones below.
VARIANTS = (
    "active_inference_expert", "base", "cognitive_expert", "cognitive_leap_expert",
    "cognitive_leap_ultra_expert", "cognitive_leap_v52_expert", "consensus_expert",
    "deep_expert", "deliberative_expert", "expert_choice", "fractal_expert",
    "frontier_collective_expert", "frontier_expert", "frontier_verifier_expert",
    "hierarchical_expert", "holographic_state_space_expert", "large",
    "liquid_spiking_expert", "megalarge", "metacognitive_expert",
    "neurogenesis_expert", "omniscient_expert", "omniversal_expert",
    "paper_fusion_expert", "recursive_expert", "reflexive_expert", "smarter_expert",
    "test_time_diff_expert", "thought_expert", "transcendent_expert",
    "tree_of_thought_expert", "ultra_expert", "ultralarge", "xlarge", "xxlarge",
    "xxxlarge",
)

#: `titan_dreamer_expert` imports `model_frontier_v43`, which is not in this
#: repository. It is excluded explicitly rather than silently, so the gap is
#: visible instead of looking like coverage.
MISSING_DEPENDENCY = {"titan_dreamer_expert": "model_frontier_v43 is not present"}

#: The benchmark signature every variant is built against:
#: `(batch, 1, 128) -> (batch, 1, 10)`.
INPUT_SHAPE = (2, 1, 128)
OUTPUT_CLASSES = 10


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_constructs_and_runs(name):
    """Construct, forward, check shape.

    Gradients stay enabled: `liquid_spiking_expert` uses a surrogate-gradient
    spiking nonlinearity that needs autograd even in eval, and running it under
    `torch.no_grad()` raises. That is a property of the mechanism, not a defect,
    so the test accommodates it rather than excluding the variant.
    """

    torch.manual_seed(0)
    model = model_variants.build_model(name)
    model.eval()

    output = model(torch.randn(*INPUT_SHAPE))

    assert output.shape[0] == INPUT_SHAPE[0]
    assert output.shape[-1] == OUTPUT_CLASSES


@pytest.mark.parametrize("name", sorted(MISSING_DEPENDENCY))
def test_known_unbuildable_variants_are_still_unbuildable(name):
    """Pin the exclusions, so a fixed dependency shows up as a failing test.

    Without this the exclusion list would quietly outlive its reason.
    """

    with pytest.raises((ImportError, ModuleNotFoundError, NameError, AttributeError)):
        model_variants.build_model(name)


def test_expert_choice_has_a_forward():
    """The specific regression: a class body with no `forward`."""

    model = model_variants.build_model("expert_choice")

    assert type(model).forward is not torch.nn.Module.forward


def test_megalarge_accepts_default_dimensions():
    """The specific regression: a computed dim discarded for a None default."""

    model = model_variants.build_model("megalarge")

    assert sum(p.numel() for p in model.parameters()) > 0


def test_runtime_snapshot_matches_source():
    """Parity is not correctness, but a fix must reach the packaged copy too."""

    source = (SOURCE_DIR / "model_variants.py").read_bytes()
    runtime = (REPO_ROOT / "runtime_python" / "model_variants.py").read_bytes()

    assert source == runtime, (
        "runtime_python/model_variants.py is stale; run "
        "`python source/sync_runtime_model_variants.py`"
    )
