import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'source'))

from model_variants import ChampionNetActiveInferenceExpert

def smoke_test_active_inference():
    print("Starting smoke test for Active Inference Expert (v27)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Lightweight initialization
    model = ChampionNetActiveInferenceExpert(inference_steps=3, lr=0.1, dropout=0.0).to(device)
    model.train()
    
    # Force alpha to 1.0 for gradient visibility
    for name, param in model.named_parameters():
        if name == 'layers.10.alpha':
            param.data.fill_(1.0)
            print(f"Set {name} to 1.0")
        elif 'experts_down' in name:
            param.data.normal_(0, 0.1)
            print(f"Set {name} to random normal to allow gradient flow backwards")

    dummy_input = torch.randn(2, 1, 128).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # 1. Forward Pass
    print("Running forward pass (train mode)...")
    # This invokes the test-time Free Energy optimization
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 1, 10), f"Expected shape (2, 1, 10), got {logits.shape}"
    
    # 2. Backward Pass
    print("Running global backward pass...")
    target = torch.randint(0, 10, (2, 1)).to(device)
    loss = nn.functional.cross_entropy(logits.view(-1, 10), target.view(-1))
    print(f"Loss value: {loss.item()}")
    loss.backward()
    
    print("Checking gradients...")
    checks = {
        'world_model.weight': False,
        'world_prior.weight': False,
        'precision_gate.weight': False,
        'experts_up.0.weight': False,
        'experts_down.0.weight': False,
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

    # 3. Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter count: {trainable:,} trainable / {total:,} total")

    print("\nSmoke test PASSED!")

if __name__ == "__main__":
    try:
        smoke_test_active_inference()
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
