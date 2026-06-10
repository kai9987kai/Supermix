import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'source'))

from model_variants import ChampionNetCognitiveLeapExpert

def smoke_test_cognitive_leap():
    print("Starting smoke test for Cognitive Leap Expert (v50) architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ChampionNetCognitiveLeapExpert(
        latent_dim=64, core_dim=128, n_cycles=3, inner_steps=2
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
    print(f"Ponder cost (expected cycles): {ponder_cost.item()}")
    print(f"Latent consistency loss: {consistency_loss.item()}")
    assert ponder_cost.item() >= 1.0, "Ponder cost should be at least one cycle"
    assert ponder_cost.item() <= 3.0, "Ponder cost cannot exceed n_cycles"
    assert consistency_loss.item() > 0, "Consistency loss should be positive"

    # 2. Backward Pass
    print("Running backward pass...")
    target = torch.randint(0, 10, (2, 1)).to(device)
    ce_loss = nn.functional.cross_entropy(logits.view(-1, 10), target.view(-1))

    # Include the latent-convergence penalty so its path is exercised too
    total_loss = ce_loss + 0.05 * consistency_loss
    print(f"Total Loss value: {total_loss.item()}")
    total_loss.backward()

    print("Checking gradients...")
    # Explicitly check keys representing all new architectural components
    checks = {
        'thought_init': False,    # Scratchpad latent initializer
        'answer_init': False,     # Answer latent initializer
        'recur_core.0': False,    # Weight-tied recursive core
        'recur_core.3': False,
        'answer_update.0': False, # Outer answer refinement
        'answer_update.2': False,
        'latent_gain': False,     # Hypersphere normalization gain
        'halt_head': False,       # ACT-style adaptive halting
        'decode_head': False,     # Per-cycle decoder
        'shared_up': False,       # Base functionality
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

    # 3. Test-time compute scaling: more reasoning cycles must change the output
    print("Checking test-time compute scaling...")
    model.eval()
    with torch.no_grad():
        out_shallow = model(dummy_input, reasoning_cycles=1)
        out_deep = model(dummy_input, reasoning_cycles=8)
    assert out_shallow.shape == out_deep.shape == (2, 1, 10)
    diff = (out_shallow - out_deep).abs().max().item()
    print(f"Max |logit| difference between 1 and 8 cycles: {diff}")
    assert diff > 0, "Extra reasoning cycles should change the output"

    # Determinism in eval mode at fixed depth
    with torch.no_grad():
        out_a = model(dummy_input, reasoning_cycles=4)
        out_b = model(dummy_input, reasoning_cycles=4)
    assert torch.allclose(out_a, out_b), "Eval mode should be deterministic at fixed depth"
    print("Test-time compute scaling verified.")

    # 4. Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter count: {trainable:,} trainable / {total:,} total")

    print("\nSmoke test PASSED!")

if __name__ == "__main__":
    try:
        smoke_test_cognitive_leap()
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
