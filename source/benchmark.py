import argparse
import json
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add source and runtime_python to path to allow imports if needed, 
# but they should be in the same dir or CWD for these tools.
source_dir = Path(__file__).parent
if str(source_dir) not in sys.path:
    sys.path.append(str(source_dir))

from model_variants import build_model, load_weights_for_model, detect_model_size_from_state_dict
from chat_pipeline import (
    text_to_model_input,
    load_conversation_examples,
    assign_labels,
    build_training_tensors
)
from torch.utils.data import DataLoader, TensorDataset

def run_compare(weights_a, meta_a, weights_b=None, meta_b=None, data_path=None, device="cpu"):
    print(f"Loading data from {data_path}...")
    examples = load_conversation_examples(data_path)
    print(f"Loaded {len(examples)} examples.")
    
    # Use assigning labels logic from pipeline
    labels, _ = assign_labels(examples, seed=42)
    x, y = build_training_tensors(examples, labels, feature_mode="context_v2")
    
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    models = [("Model A", weights_a, meta_a)]
    if weights_b and meta_b:
        models.append(("Model B", weights_b, meta_b))
        
    results = {}
    for name, w_path, m_path in models:
        print(f"Evaluating {name}: {w_path}")
        sd = torch.load(w_path, map_location=device)
        model_size = detect_model_size_from_state_dict(sd)
        print(f"  Detected size: {model_size}")
        
        # Build model and load weights
        model = build_model(model_size=model_size).to(device).eval()
        load_weights_for_model(model, sd, model_size=model_size)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.dtype != torch.float32:
                    xb = xb.float()
                logits = model(xb).squeeze(1)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        
        accuracy = correct / total if total > 0 else 0
        results[name] = accuracy
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_a", required=True)
    parser.add_argument("--meta_a", required=True)
    parser.add_argument("--weights_b", default=None)
    parser.add_argument("--meta_b", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    run_compare(args.weights_a, args.meta_a, args.weights_b, args.meta_b, args.data, args.device)
