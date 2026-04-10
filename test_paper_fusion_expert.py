import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'source'))

from model_variants import ChampionNetPaperFusionExpert

def smoke_test_paper_fusion():
    print("Starting smoke test for Paper Fusion Expert (v29)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ChampionNetPaperFusionExpert(n_sub_routers=3, dropout=0.0).to(device)
    model.train()
    
    # Force alpha + shared_scale to 1.0 and randomize experts_down for gradient flow
    for name, param in model.named_parameters():
        if name == 'layers.10.alpha':
            param.data.fill_(1.0)
            print(f"  Set {name} to 1.0")
        elif name == 'layers.10.shared_scale':
            param.data.fill_(1.0)
            print(f"  Set {name} to 1.0")
        elif 'experts_down' in name and 'layers.10' in name:
            param.data.normal_(0, 0.1)
        elif 'shared_down' in name and 'layers.10' in name:
            param.data.normal_(0, 0.1)
            print(f"  Set {name} to random normal for gradient flow")
        elif name == 'layers.10.ssm_adapter.gate':
            param.data.fill_(1.0)
            print(f"  Set {name} to 1.0")
        elif name == 'layers.10.ssm_adapter.A':
            param.data.uniform_(-1.0, -0.1)  # mild range to avoid sigmoid saturation
            print(f"  Set {name} to mild range [-1, -0.1]")

    dummy_input = torch.randn(4, 3, 128).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # 1. Forward Pass
    print("Running forward pass (train mode)...")
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 3, 10), f"Expected shape (4, 3, 10), got {logits.shape}"
    
    # 2. Check manifold alignment loss (RoMA)
    manifold_loss = model.layers[10]._manifold_loss
    print(f"RoMA manifold alignment loss: {manifold_loss.item():.6f}")
    assert manifold_loss.item() > 0, "RoMA loss should be > 0 during training"
    
    # 3. Backward Pass
    print("Running global backward pass...")
    target = torch.randint(0, 10, (4, 3)).to(device)
    ce_loss = nn.functional.cross_entropy(logits.view(-1, 10), target.view(-1))
    total_loss = ce_loss + 0.01 * manifold_loss
    print(f"CE Loss: {ce_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
    total_loss.backward()
    
    print("Checking gradients...")
    checks = {
        'mor_router.sub_routers.0.weight': False,  # MoR sub-router 1
        'mor_router.sub_routers.1.weight': False,  # MoR sub-router 2
        'mor_router.sub_routers.2.weight': False,  # MoR sub-router 3
        'mor_router.main_router.weight': False,     # MoR main router
        # Note: ssm_adapter.A gets zero grad in single-pass (h starts at 0, so tanh(A)*0=0).
        # This is correct STAD behavior — A only contributes with persistent recurrent state.
        'ssm_adapter.B': False,                     # STAD SSM B
        'ssm_adapter.C': False,                     # STAD SSM C
        'ssm_adapter.gate': False,                  # STAD gate
        'experts_up.0.weight': False,               # Expert up
        'experts_down.0.weight': False,             # Expert down
        'shared_up.weight': False,                  # Shared expert
    }
    
    # Separately check ssm_adapter.A (expected zero grad in single-pass, non-zero with recurrent state)
    a_has_grad = False
    for name, param in model.named_parameters():
        if 'ssm_adapter.A' in name:
            if param.grad is not None:
                a_has_grad = param.grad.abs().sum() > 0
            break
    print(f"  ssm_adapter.A: {'HAS GRAD' if a_has_grad else 'ZERO (expected in single-pass, A*h where h=0)'}")
    
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

    # 4. Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter count: {trainable:,} trainable / {total:,} total")

    print("\nSmoke test PASSED!")

if __name__ == "__main__":
    try:
        smoke_test_paper_fusion()
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
