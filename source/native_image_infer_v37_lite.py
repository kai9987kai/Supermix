#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_native_image_lite_v37 import ChampionNetUltraExpertNativeImageLite, save_prompt_image
from run import safe_load_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an image with the v37 lite single-checkpoint model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    feature_mode = str(meta.get("feature_mode", "context_mix_v4"))
    image_size = int(meta.get("image_size", 64))

    device = torch.device(str(args.device))
    model = ChampionNetUltraExpertNativeImageLite(image_size=image_size).to(device).eval()
    state_dict = safe_load_state_dict(str(args.weights))
    model.load_state_dict(state_dict, strict=True)
    out_path = save_prompt_image(model, str(args.prompt), args.output, feature_mode=feature_mode, device=device)
    print(f"[v37-lite-infer] saved={out_path}")


if __name__ == "__main__":
    main()
