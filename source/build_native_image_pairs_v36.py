#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageEnhance

import build_science_image_test_dataset as science_images

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _make_prompt_templates(meta: Dict[str, str]) -> List[str]:
    concept = meta["concept"]
    caption = meta["caption"]
    tags = [part.strip() for part in meta.get("tags", "").split(",") if part.strip()]
    tag_phrase = ", ".join(tags[:3]) if tags else concept
    return [
        caption,
        f"a clean science diagram of {concept}",
        f"a minimal textbook illustration of {concept}",
        f"an educational icon showing {concept}",
        f"flat classroom poster art for {concept}",
        f"simple visual explanation of {concept}",
        f"{concept} diagram, {tag_phrase}",
        f"draw {concept} as a neat educational graphic",
    ]


def _variant_images(base_image: Image.Image) -> Iterable[Tuple[str, Image.Image]]:
    warm_overlay = Image.new("RGB", base_image.size, (255, 234, 210))
    cool_overlay = Image.new("RGB", base_image.size, (215, 232, 255))

    yield "base", base_image
    yield "warm", Image.blend(ImageEnhance.Color(base_image).enhance(1.10), warm_overlay, 0.08)
    yield "cool", Image.blend(ImageEnhance.Color(base_image).enhance(1.06), cool_overlay, 0.10)
    yield "crisp", ImageEnhance.Sharpness(ImageEnhance.Contrast(base_image).enhance(1.08)).enhance(1.35)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prompt->image dataset for the native v36 image head.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    output_path = Path(args.output)
    images_dir = Path(args.images_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for drawer_name, drawer_fn in science_images.DRAWERS:
        base_path = images_dir / f"{drawer_name}_base.png"
        meta = drawer_fn(base_path)
        base_image = Image.open(base_path).convert("RGB")
        prompts = _make_prompt_templates(meta)
        for variant_name, variant_image in _variant_images(base_image):
            variant_path = images_dir / f"{drawer_name}_{variant_name}.png"
            resized = variant_image.resize((int(args.image_size), int(args.image_size)), RESAMPLE_LANCZOS)
            resized.save(variant_path)
            prompt_order = prompts[:]
            rng.shuffle(prompt_order)
            for prompt in prompt_order:
                rows.append(
                    {
                        "prompt": prompt,
                        "image_path": str(variant_path),
                        "concept": meta["concept"],
                        "caption": meta["caption"],
                        "tags": meta["tags"],
                        "variant": variant_name,
                        "source": "v36_native_image_science",
                    }
                )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"[v36-native-build] rows={len(rows)}")
    print(f"[v36-native-build] images_dir={images_dir}")
    print(f"[v36-native-build] output={output_path}")


if __name__ == "__main__":
    main()
