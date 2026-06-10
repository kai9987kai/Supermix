import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'source'))

from model_variants import build_model, detect_model_size_from_state_dict


def smoke_test_titan_dreamer():
    print("Starting smoke test for Titan-Dreamer Expert (v43)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_model("titan_dreamer_expert", dropout=0.0).to(device)
    model.train()

    # Force branch scales to 1.0 for gradient visibility
    for name, param in model.named_parameters():
        if name in ('layers.10.alpha', 'layers.10.beta', 'layers.10.shared_scale'):
            param.data.fill_(1.0)
            print(f"Set {name} to 1.0")
        elif name == 'layers.10.shared_down.weight':
            param.data.normal_(0, 0.05)
            print(f"Set {name} to random normal so gradients reach shared_up")

    dummy_input = torch.randn(2, 4, 128).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # 1. Forward Pass (invokes test-time memory updates + depth recursion)
    print("Running forward pass (train mode)...")
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 4, 10), f"Expected shape (2, 4, 10), got {logits.shape}"

    # 2. Backward Pass
    print("Running global backward pass...")
    target = torch.randint(0, 10, (2, 4)).to(device)
    loss = nn.functional.cross_entropy(logits.view(-1, 10), target.view(-1))
    print(f"Loss value: {loss.item()}")
    loss.backward()

    print("Checking gradients...")
    checks = {
        'titan_w1': False,
        'titan_w2': False,
        'mem_q.weight': False,
        'mem_gate.weight': False,
        'mem_out.weight': False,
        'persist_tokens': False,
        'depth_q.weight': False,
        'depth_cells.0.1.weight': False,
        'recursion_router.weight': False,
        'depth_out.weight': False,
        'shared_up.weight': False,
    }
    for name, param in model.named_parameters():
        for key in checks:
            if key in name and param.grad is not None and param.grad.abs().sum() > 0:
                checks[key] = True

    for key, found in checks.items():
        status = "OK" if found else "CRITICAL: NO GRADIENT!"
        print(f"  {key}: {status}")
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"Missing gradients in: {failed}")
    print("All gradients verified successfully.")

    # 3. Eval mode determinism + variant detection
    model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = model(dummy_input)
    assert torch.allclose(out1, out2, atol=1e-5), "Eval forward not deterministic"
    detected = detect_model_size_from_state_dict(model.state_dict())
    assert detected == "titan_dreamer_expert", f"Detection failed: {detected}"
    print(f"Variant detection OK: {detected}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter count: {trainable:,} trainable / {total:,} total")
    print("\nSmoke test PASSED!")


if __name__ == "__main__":
    try:
        smoke_test_titan_dreamer()
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
