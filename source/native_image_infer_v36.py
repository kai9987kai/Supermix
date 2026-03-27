#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_frontier_native_image_v36 import ChampionNetFrontierCollectiveNativeImage, save_prompt_image
from run import safe_load_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a native image from the v36 single-checkpoint model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_mode", default="")
    args = parser.parse_args()

    weights = Path(args.weights)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    feature_mode = str(args.feature_mode or meta.get("feature_mode") or "context_mix_v4")
    image_size = int(meta.get("image_size") or 64)

    device = torch.device(str(args.device))
    model = ChampionNetFrontierCollectiveNativeImage(image_size=image_size).to(device).eval()
    state_dict = safe_load_state_dict(str(weights))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        print(f"[v36-native-infer] unexpected_keys={len(incompatible.unexpected_keys)}")
    save_prompt_image(model, str(args.prompt), args.output, feature_mode=feature_mode, device=device)
    print(f"[v36-native-infer] output={args.output}")


if __name__ == "__main__":
    main()
