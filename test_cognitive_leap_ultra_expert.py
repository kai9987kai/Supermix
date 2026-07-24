import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'source'))

from model_variants import (
    ChampionNetCognitiveLeapUltraExpert,
    CognitiveLeapUltraExpertHead,
)
from benchmark_cognitive_leap_ultra_v51 import (
    DEFAULT_PREDICTION_STABILITY_MARGIN as BENCHMARK_STABILITY_MARGIN,
    DEFAULT_PREDICTION_STABILITY_RANK_DEPTH as BENCHMARK_STABILITY_RANK_DEPTH,
    evaluate as benchmark_evaluate,
)

def test_cognitive_leap_ultra():
    print("Starting smoke test for Cognitive Leap Ultra Expert (v51) architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model with 4 cores
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=64, core_dim=128, n_cycles=3, inner_steps=2, n_cores=4
    ).to(device)
    model.train()

    # Force alpha to 1.0 for gradient visibility
    for name, param in model.named_parameters():
        if name == 'layers.10.alpha':
            param.data.fill_(1.0)
            print(f"Set {name} to 1.0")

    dummy_input = torch.randn(2, 1, 128).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # 1. Forward Pass (Training)
    print("Running forward pass (train mode)...")
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 1, 10), f"Expected shape (2, 1, 10), got {logits.shape}"

    # Track diagnostics from the recursive head
    head = model.layers[10]
    ponder_cost = head.last_ponder_cost
    consistency_loss = head.last_consistency_loss
    gating_entropy = head.last_gating_entropy
    print(f"Ponder cost (expected cycles): {ponder_cost.item()}")
    print(f"Latent consistency loss: {consistency_loss.item()}")
    print(f"Gating entropy: {gating_entropy.item()}")
    assert ponder_cost.item() >= 1.0, "Ponder cost should be at least one cycle"
    assert ponder_cost.item() <= 3.0, "Ponder cost cannot exceed n_cycles"
    assert consistency_loss.item() > 0, "Consistency loss should be positive"
    assert gating_entropy.item() > 0, "Gating entropy should be positive"

    # 2. Backward Pass
    print("Running backward pass...")
    target = torch.randint(0, 10, (2, 1)).to(device)
    ce_loss = nn.functional.cross_entropy(logits.view(-1, 10), target.view(-1))

    # Deep improvement supervision over the cached per-cycle decodes
    dis_loss = model.deep_supervision_loss(target)
    print(f"Deep supervision loss: {dis_loss.item()}")
    assert dis_loss.item() > 0, "Deep supervision loss should be positive"

    # Include all paths in loss
    total_loss = ce_loss + 0.05 * consistency_loss + 0.1 * dis_loss - 0.01 * gating_entropy
    print(f"Total Loss value: {total_loss.item()}")
    total_loss.backward()

    print("Checking gradients...")
    # Check keys representing all core components of v51
    checks = {
        'thought_init': False,      # Scratchpad initializer
        'answer_init': False,       # Answer initializer
        'cores.0': False,           # Recurrent cores
        'cores.3': False,
        'core_router': False,       # Core gating router
        'cross_attn.qkv': False,    # Cross attention QKV
        'cross_attn.proj': False,   # Cross attention output projection
        'answer_update.0': False,   # Outer answer refinement
        'cycle_embed': False,       # Cycle identity conditioning
        'halt_head': False,         # ACT halting head
        'decode_head': False,       # Decoder
        'shared_up': False,         # Base shared expert
    }

    for name, param in model.named_parameters():
        for key in checks:
            if key in name and param.grad is not None and param.grad.abs().sum() > 0:
                checks[key] = True

    for key, found in checks.items():
        status = "OK" if found else "CRITICAL: NO GRADIENT!"
        print(f"  {key}: {status}")

    all_ok = all(checks.values())
    if not all_ok:
        failed = [k for k, v in checks.items() if not v]
        raise AssertionError(f"Missing gradients in: {failed}")

    print("All gradients verified successfully.")

    # 3. Test-time compute scaling
    print("Checking test-time compute scaling...")
    model.eval()
    with torch.no_grad():
        out_shallow = model(dummy_input, reasoning_cycles=1)
        out_deep = model(dummy_input, reasoning_cycles=8)
    assert out_shallow.shape == out_deep.shape == (2, 1, 10)
    diff = (out_shallow - out_deep).abs().max().item()
    print(f"Max |logit| difference between 1 and 8 cycles: {diff}")
    assert diff > 0, "Extra reasoning cycles should change the output"

    # 4. Confidence-based early exit (Entropy Exit)
    print("Checking entropy-based early exit...")
    head = model.layers[10]

    # Evaluate with very high entropy threshold (so it exits immediately)
    with torch.no_grad():
        model(
            dummy_input,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_entropy_threshold=10.0,
            prediction_stability_patience=0,
        )
    cycles_used_early = head.last_cycles_used.item()
    print(f"Cycles used with high entropy exit: {cycles_used_early} / 8 requested")
    assert cycles_used_early < 8, "High entropy exit threshold should stop cycling early"

    # Evaluate with zero entropy threshold (so it never exits early by entropy)
    with torch.no_grad():
        model(
            dummy_input,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_entropy_threshold=0.0,
            exit_tol=0.0,
            prediction_stability_patience=0,
        )
    cycles_used_late = head.last_cycles_used.item()
    print(f"Cycles used with zero thresholds: {cycles_used_late} / 8 requested")
    assert cycles_used_late == 8, "Zero threshold should never exit early"

    print("Entropy-based early exit verified.")
    print("\nSmoke test PASSED!")


def test_prediction_stability_verifier_controls_early_exit():
    torch.manual_seed(5102)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.zeros(1, 1, 128)
    head = model.layers[10]

    with torch.no_grad():
        stable_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_top_k=4,
        )

    assert stable_output.shape == (1, 1, 10)
    assert head.last_cycles_used.item() == 2
    assert head.last_exit_reason == "prediction_stable"
    assert head.last_decision_reference_cycles.item() == 3
    assert head.last_prediction_streak.item() >= 2
    assert head.last_prediction_confidence_delta.item() >= 0.0
    assert 0.0 <= head.last_prediction_margin.item() <= 1.0
    assert head.last_prediction_topk_js_divergence.item() >= 0.0
    assert (
        head.last_prediction_topk_js_divergence_max.item()
        >= head.last_prediction_topk_js_divergence.item()
    )
    assert torch.isfinite(head.last_prediction_topk_js_divergence)

    identical = torch.tensor([[0.6, 0.3, 0.1]])
    assert torch.equal(
        head._topk_js_divergence(identical, identical, top_k=2),
        torch.zeros(1),
    )
    generator = torch.Generator().manual_seed(510301)
    previous = torch.softmax(torch.randn(256, 10, generator=generator), dim=-1)
    current = torch.softmax(
        torch.log(previous) + 1e-5 * torch.randn(256, 10, generator=generator),
        dim=-1,
    )
    near_identical_js = head._topk_js_divergence(previous, current, top_k=5)
    assert torch.isfinite(near_identical_js).all()
    assert bool((near_identical_js >= 0.0).all().item())

    with torch.no_grad():
        full_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=0,
        )

    assert full_output.shape == stable_output.shape
    assert head.last_cycles_used.item() == 8
    assert head.last_exit_reason in {"max_cycles", "halt_mass"}


class _ScriptedCycleDecoder(nn.Module):
    def __init__(self, cycle_logits):
        super().__init__()
        self.register_buffer("cycle_logits", torch.tensor(cycle_logits, dtype=torch.float32))
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def forward(self, x):
        logits = self.cycle_logits[min(self.cursor, len(self.cycle_logits) - 1)]
        self.cursor += 1
        return logits.unsqueeze(0).expand(x.shape[0], -1)


def test_prediction_verifier_tracks_ordered_rank_tuple_and_blocks_legacy_bypass():
    head = CognitiveLeapUltraExpertHead(
        in_dim=4,
        out_dim=4,
        latent_dim=4,
        core_dim=8,
        n_cycles=3,
        inner_steps=1,
        max_cycles=8,
        dropout=0.0,
        n_cores=1,
    ).eval()
    decoder = _ScriptedCycleDecoder(
        [
            [4.0, 3.0, 2.0, 0.0],  # ordered top-3: (0, 1, 2)
            [4.0, 2.0, 3.0, 0.0],  # ordered top-3: (0, 2, 1)
            [4.0, 2.0, 3.0, 0.0],
        ]
    )
    head.decode_head = decoder
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.shared_scale.zero_()
        head.alpha.fill_(1.0)
        head.halt_head.weight.zero_()
        head.halt_head.bias.fill_(-100.0)

        head(
            torch.zeros(1, 4),
            reasoning_cycles=8,
            adaptive_compute=True,
            # Both legacy criteria would stop by cycle two if allowed to
            # bypass the active post-head decision verifier.
            exit_tol=10.0,
            exit_entropy_threshold=10.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_margin=0.0,
            prediction_stability_rank_depth=3,
            prediction_output_transform=nn.Identity(),
        )

    # The unchanged top-1 is insufficient: the cycle-two rank swap resets the
    # streak, and the request falls back to the trained reference budget.
    assert head.last_cycles_used.item() == 3
    assert head.last_exit_reason == "decision_reference_budget"
    assert head.last_prediction_streak.item() == 2
    assert head.last_decision_reference_cycles.item() == 3

    with torch.no_grad():
        decoder.cycle_logits[1:].copy_(
            decoder.cycle_logits[0].expand_as(decoder.cycle_logits[1:])
        )
    decoder.reset()
    with torch.no_grad():
        head(
            torch.zeros(1, 4),
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=10.0,
            exit_entropy_threshold=10.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_margin=0.0,
            prediction_stability_rank_depth=3,
            prediction_output_transform=nn.Identity(),
        )

    # A genuinely persistent, ordered rank-3 tuple still earns the intended
    # confident cycle-two exit.
    assert head.last_cycles_used.item() == 2
    assert head.last_exit_reason == "prediction_stable"

    decoder.reset()
    with torch.no_grad():
        head(
            torch.zeros(1, 4),
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=10.0,
            exit_entropy_threshold=10.0,
            prediction_stability_patience=0,
            prediction_stability_rank_depth=3,
            prediction_output_transform=nn.Identity(),
        )

    # Disabling the verifier preserves the historical legacy exit behavior.
    assert head.last_cycles_used.item() == 1
    assert head.last_exit_reason == "low_entropy"

    decoder.reset()
    with torch.no_grad():
        head(
            torch.zeros(1, 4),
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=10.0,
            exit_entropy_threshold=10.0,
            prediction_stability_patience=2,
            prediction_class_indices=[0],
            prediction_stability_rank_depth=3,
            prediction_output_transform=nn.Identity(),
        )

    # An invalid one-class decision scope fails closed for certification while
    # leaving the legacy fallback available.
    assert head.last_cycles_used.item() == 1
    assert head.last_exit_reason == "low_entropy"
    assert not bool(head.last_prediction_class_selection_valid.item())


def test_uncertified_prediction_falls_back_to_exact_reference_budget_output():
    torch.manual_seed(5110)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.randn(2, 1, 128)
    head = model.layers[10]

    with torch.no_grad():
        reference_output = model(
            sample,
            reasoning_cycles=head.n_cycles,
            adaptive_compute=False,
        )
        fallback_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            # These permissive legacy signals must not bypass the verifier.
            exit_tol=10.0,
            exit_entropy_threshold=10.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            # A probability gap cannot exceed one, so this run cannot certify
            # an early decision.
            prediction_stability_margin=1.0,
            prediction_stability_rank_depth=3,
        )

    assert torch.equal(fallback_output, reference_output)
    assert head.last_cycles_used.item() == head.n_cycles == 3
    assert head.last_exit_reason == "decision_reference_budget"
    assert head.last_decision_reference_cycles.item() == head.n_cycles


def test_prediction_stability_margin_guards_ambiguous_early_exit():
    torch.manual_seed(5103)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.zeros(1, 1, 128)
    head = model.layers[10]
    assert not any("last_prediction_margin" in key for key in model.state_dict())
    assert not any("last_prediction_class_count" in key for key in model.state_dict())
    assert not any(
        "last_prediction_class_selection_valid" in key for key in model.state_dict()
    )

    with torch.no_grad():
        default_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
        )
        default_cycles = head.last_cycles_used.item()
        default_reason = head.last_exit_reason

        explicit_zero_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_margin=0.0,
        )
        explicit_zero_cycles = head.last_cycles_used.item()
        explicit_zero_reason = head.last_exit_reason

        guarded_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_margin=1.0,
        )

    assert torch.equal(default_output, explicit_zero_output)
    assert default_cycles == explicit_zero_cycles == 2
    assert default_reason == explicit_zero_reason == "prediction_stable"
    assert guarded_output.shape == default_output.shape
    assert head.last_cycles_used.item() > default_cycles
    assert head.last_exit_reason != "prediction_stable"
    assert 0.0 <= head.last_prediction_margin.item() < 1.0


def test_prediction_verifier_uses_complete_post_head_output():
    torch.manual_seed(5105)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.randn(1, 1, 128)
    head = model.layers[10]

    with torch.no_grad():
        output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
        )
        final_probabilities = torch.softmax(output[0, 0], dim=-1)
        top_two = final_probabilities.topk(k=2).values
        expected_margin = top_two[0] - top_two[1]

    assert head.last_exit_reason == "prediction_stable"
    assert torch.allclose(head.last_prediction_margin, expected_margin, atol=1e-7)


def test_prediction_verifier_scopes_to_available_classes():
    torch.manual_seed(5106)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.zeros(1, 1, 128)
    head = model.layers[10]

    # Make an unavailable class dominate the returned logits. The verifier
    # must still measure prediction, drift, and margin only over classes 0/1.
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[0] = 2.0
        head.bias[9] = 100.0
        head.shared_scale.zero_()

        scoped_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_margin=0.0,
            prediction_class_indices=[0, 1],
        )
        scoped_cycles = head.last_cycles_used.item()
        scoped_reason = head.last_exit_reason
        scoped_margin = head.last_prediction_margin.clone()

        class_mask = torch.zeros(10, dtype=torch.bool)
        class_mask[:2] = True
        masked_output = model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=1.0,
            prediction_stability_margin=0.0,
            prediction_class_indices=class_mask,
        )
        masked_margin = head.last_prediction_margin.clone()

    assert scoped_output.argmax(dim=-1).item() == 9
    assert scoped_reason == "prediction_stable"
    assert scoped_cycles == 2
    scoped_probabilities = torch.softmax(scoped_output[0, 0, [0, 1]], dim=-1)
    expected_margin = scoped_probabilities.max() - scoped_probabilities.min()
    assert torch.allclose(scoped_margin, expected_margin, atol=1e-7)
    assert torch.equal(masked_output, scoped_output)
    assert torch.equal(masked_margin, scoped_margin)
    assert head.last_prediction_class_count.item() == 2
    assert bool(head.last_prediction_class_selection_valid.item())


def test_prediction_stability_rank_depth_guards_top_k_candidate_boundaries():
    torch.manual_seed(5109)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.zeros(1, 1, 128)
    head = model.layers[10]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-4.0)
        head.bias[0] = 4.0
        head.bias[1] = 2.0
        head.bias[2] = 1.0
        head.bias[3] = 0.9999
        head.shared_scale.zero_()
        head.decode_head.weight.zero_()
        head.halt_head.weight.zero_()
        head.halt_head.bias.fill_(-100.0)

        model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=0.0,
            prediction_stability_margin=1e-3,
            prediction_stability_rank_depth=1,
        )
        top1_cycles = head.last_cycles_used.item()
        top1_reason = head.last_exit_reason
        top1_margin = head.last_prediction_margin.item()

        model(
            sample,
            reasoning_cycles=8,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=2,
            prediction_stability_tol=0.0,
            prediction_stability_margin=1e-3,
            prediction_stability_rank_depth=3,
        )

    assert top1_reason == "prediction_stable"
    assert top1_cycles == 2
    assert top1_margin >= 1e-3
    assert head.last_exit_reason == "decision_reference_budget"
    assert head.last_cycles_used.item() == head.n_cycles == 3
    assert head.last_prediction_margin.item() >= 1e-3
    assert head.last_prediction_decision_margin.item() < 1e-3
    assert head.last_prediction_rank_depth.item() == 3


def test_prediction_verifier_all_class_scope_is_bit_identical_to_none():
    torch.manual_seed(5107)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.randn(2, 1, 128)
    head = model.layers[10]
    controls = dict(
        reasoning_cycles=6,
        adaptive_compute=True,
        exit_tol=0.0,
        exit_entropy_threshold=0.0,
        prediction_stability_patience=2,
        prediction_stability_tol=1.0,
        prediction_stability_margin=0.0,
    )

    with torch.no_grad():
        default_output = model(sample, **controls)
        default_cycles = head.last_cycles_used.clone()
        default_margin = head.last_prediction_margin.clone()
        default_js = head.last_prediction_topk_js_divergence.clone()

        complete_indices_output = model(
            sample,
            prediction_class_indices=list(reversed(range(10))),
            **controls,
        )
        complete_indices_cycles = head.last_cycles_used.clone()
        complete_indices_margin = head.last_prediction_margin.clone()
        complete_indices_js = head.last_prediction_topk_js_divergence.clone()

        complete_mask_output = model(
            sample,
            prediction_class_indices=torch.ones(10, dtype=torch.bool),
            **controls,
        )

    assert torch.equal(complete_indices_output, default_output)
    assert torch.equal(complete_mask_output, default_output)
    assert torch.equal(complete_indices_cycles, default_cycles)
    assert torch.equal(complete_indices_margin, default_margin)
    assert torch.equal(complete_indices_js, default_js)
    assert head.last_prediction_class_count.item() == 10
    assert bool(head.last_prediction_class_selection_valid.item())


def test_invalid_prediction_class_scopes_fail_closed_without_breaking_inference():
    torch.manual_seed(5108)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()
    sample = torch.zeros(1, 1, 128)
    head = model.layers[10]
    controls = dict(
        reasoning_cycles=4,
        adaptive_compute=True,
        exit_tol=0.0,
        exit_entropy_threshold=0.0,
        prediction_stability_patience=2,
        prediction_stability_tol=1.0,
        prediction_stability_margin=0.0,
    )
    invalid_scopes = (
        [],
        [0, 0],
        [-1, 1],
        [0, 10],
        [0.0, 1.0],
        torch.tensor([[0, 1]]),
        torch.tensor([True, False]),
    )

    with torch.no_grad():
        for invalid_scope in invalid_scopes:
            output = model(
                sample,
                prediction_class_indices=invalid_scope,
                **controls,
            )
            assert output.shape == (1, 1, 10)
            assert head.last_exit_reason != "prediction_stable"
            assert head.last_prediction_class_count.item() == 0
            assert not bool(head.last_prediction_class_selection_valid.item())

        one_class_output = model(
            sample,
            prediction_class_indices=[3],
            **controls,
        )
        empty_invalid_output = model(
            torch.empty(0, 1, 128),
            prediction_class_indices=[],
            **controls,
        )

    assert one_class_output.shape == (1, 1, 10)
    assert empty_invalid_output.shape == (0, 1, 10)
    assert head.last_exit_reason != "prediction_stable"
    assert head.last_prediction_class_count.item() == 0
    assert not bool(head.last_prediction_class_selection_valid.item())

    with torch.no_grad():
        model(sample, prediction_class_indices=[3], **controls)

    assert head.last_exit_reason != "prediction_stable"
    assert head.last_prediction_class_count.item() == 1
    assert not bool(head.last_prediction_class_selection_valid.item())
    assert head.last_prediction_margin.item() == 0.0


def test_prediction_margin_preserves_empty_batch_and_benchmark_contract():
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        output = model(torch.empty(0, 1, 128), reasoning_cycles=3)

    assert output.shape == (0, 1, 10)
    assert model.layers[10].last_prediction_margin.item() == 0.0
    assert model.layers[10].last_prediction_class_count.item() == 10
    assert bool(model.layers[10].last_prediction_class_selection_valid.item())
    assert torch.isfinite(model.layers[10].last_prediction_streak)
    assert torch.isfinite(model.layers[10].last_prediction_confidence_delta)
    assert torch.isfinite(model.layers[10].last_prediction_topk_js_divergence)
    assert BENCHMARK_STABILITY_MARGIN == 0.0005
    assert BENCHMARK_STABILITY_RANK_DEPTH == 3

    sample = torch.randn(2, 1, 128)
    labels = torch.tensor([0, 1])
    fixed = benchmark_evaluate(model, sample, labels, torch.device("cpu"))
    adaptive = benchmark_evaluate(
        model,
        sample,
        labels,
        torch.device("cpu"),
        reasoning_cycles=3,
        adaptive_compute=True,
        exit_tol=0.0,
        exit_entropy_threshold=0.0,
        prediction_stability_patience=2,
        prediction_stability_tol=1.0,
        prediction_stability_margin=0.0,
        prediction_stability_rank_depth=BENCHMARK_STABILITY_RANK_DEPTH,
    )
    invalid_scope = benchmark_evaluate(
        model,
        sample,
        labels,
        torch.device("cpu"),
        reasoning_cycles=3,
        adaptive_compute=True,
        prediction_stability_patience=2,
        prediction_stability_rank_depth=BENCHMARK_STABILITY_RANK_DEPTH,
        prediction_class_indices=[0],
    )

    assert fixed["prediction_verifier_active"] is False
    assert fixed["prediction_margin"] is None
    assert fixed["prediction_decision_margin"] is None
    assert fixed["prediction_rank_depth"] is None
    assert adaptive["prediction_verifier_active"] is True
    assert adaptive["prediction_margin"] is not None
    assert adaptive["prediction_decision_margin"] is not None
    assert adaptive["prediction_rank_depth"] == BENCHMARK_STABILITY_RANK_DEPTH
    assert invalid_scope["prediction_verifier_active"] is False
    assert invalid_scope["prediction_margin"] is None
    assert invalid_scope["prediction_decision_margin"] is None
    assert invalid_scope["prediction_rank_depth"] is None


def test_prediction_stability_margin_does_not_change_training():
    torch.manual_seed(5104)
    model = ChampionNetCognitiveLeapUltraExpert(
        latent_dim=32,
        core_dim=64,
        n_cycles=3,
        inner_steps=1,
        n_cores=2,
        dropout=0.0,
    ).train()
    sample = torch.randn(2, 1, 128)
    head = model.layers[10]

    baseline_output = model(
        sample,
        reasoning_cycles=4,
        adaptive_compute=True,
        exit_tol=0.0,
        exit_entropy_threshold=0.0,
        prediction_stability_patience=2,
        prediction_stability_tol=1.0,
        prediction_stability_margin=0.0,
    )
    guarded_output = model(
        sample,
        reasoning_cycles=4,
        adaptive_compute=True,
        exit_tol=0.0,
        exit_entropy_threshold=0.0,
        prediction_stability_patience=2,
        prediction_stability_tol=1.0,
        prediction_stability_margin=1.0,
        prediction_class_indices=[0, 1],
    )
    invalid_scope_output = model(
        sample,
        reasoning_cycles=4,
        adaptive_compute=True,
        exit_tol=0.0,
        exit_entropy_threshold=0.0,
        prediction_stability_patience=2,
        prediction_stability_tol=1.0,
        prediction_stability_margin=1.0,
        prediction_class_indices=[0, 0],
    )

    assert torch.equal(baseline_output, guarded_output)
    assert torch.equal(baseline_output, invalid_scope_output)
    assert head.last_cycles_used.item() == 4
    assert head.last_exit_reason == "max_cycles"
    assert head.last_prediction_margin.item() == 0.0
    assert not bool(head.last_prediction_class_selection_valid.item())
    state_keys = model.state_dict().keys()
    assert not any("last_prediction_class_count" in key for key in state_keys)
    assert not any("last_prediction_class_selection_valid" in key for key in state_keys)
    assert not any("last_prediction_decision_margin" in key for key in state_keys)
    assert not any("last_prediction_rank_depth" in key for key in state_keys)

if __name__ == "__main__":
    try:
        test_cognitive_leap_ultra()
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
