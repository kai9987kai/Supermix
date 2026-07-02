import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


def _write_obj(path: Path, vertices: List[Tuple[float, float, float]], faces: List[Tuple[int, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# generated test model\n")
        for x, y, z in vertices:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")


def _cube(path: Path) -> Dict[str, str]:
    v = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    f = [
        (1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8),
    ]
    _write_obj(path, v, f)
    return {"concept": "cube", "caption": "3D cube mesh", "tags": "3d,geometry,cube,mesh,obj"}


def _pyramid(path: Path) -> Dict[str, str]:
    v = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 1.8)]
    f = [(1, 2, 3, 4), (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 1, 5)]
    _write_obj(path, v, f)
    return {"concept": "pyramid", "caption": "3D pyramid mesh with square base", "tags": "3d,geometry,pyramid,mesh,obj"}


def _triangular_prism(path: Path) -> Dict[str, str]:
    v = [(-1, -0.7, -1), (1, -0.7, -1), (0, 0.9, -1), (-1, -0.7, 1), (1, -0.7, 1), (0, 0.9, 1)]
    f = [(1, 2, 3), (4, 5, 6), (1, 2, 5, 4), (2, 3, 6, 5), (3, 1, 4, 6)]
    _write_obj(path, v, f)
    return {"concept": "triangular prism", "caption": "3D triangular prism mesh", "tags": "3d,geometry,prism,mesh,obj"}


def _tetrahedron(path: Path) -> Dict[str, str]:
    v = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
    f = [(1, 2, 3), (1, 4, 2), (1, 3, 4), (2, 4, 3)]
    _write_obj(path, v, f)
    return {"concept": "tetrahedron", "caption": "3D tetrahedron mesh", "tags": "3d,geometry,tetrahedron,mesh,obj"}


def _plane_grid(path: Path) -> Dict[str, str]:
    v = [(-1, -1, 0), (0, -1, 0), (1, -1, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0)]
    f = [(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)]
    _write_obj(path, v, f)
    return {"concept": "grid plane", "caption": "flat grid-like plane mesh", "tags": "3d,geometry,plane,mesh,obj,flat"}


MODEL_BUILDERS = [
    ("cube", _cube),
    ("pyramid", _pyramid),
    ("triangular_prism", _triangular_prism),
    ("tetrahedron", _tetrahedron),
    ("grid_plane", _plane_grid),
]


def _save_gif(frames: List[Image.Image], path: Path, duration: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=duration)


def _video_bouncing_ball(path: Path) -> Dict[str, str]:
    frames: List[Image.Image] = []
    w, h = 128, 96
    for t in range(18):
        im = Image.new("RGB", (w, h), (245, 248, 255))
        d = ImageDraw.Draw(im)
        x = 12 + t * 5
        y = 52 + int(18 * math.sin(t / 2.4))
        d.rectangle((0, 78, w, h), fill=(210, 230, 210))
        d.ellipse((x, y, x + 14, y + 14), fill=(220, 70, 70))
        frames.append(im)
    _save_gif(frames, path, duration=70)
    return {"concept": "motion", "caption": "bouncing ball animation showing changing position over time", "tags": "video,physics,motion,position,time,bouncing_ball"}


def _video_pendulum(path: Path) -> Dict[str, str]:
    frames: List[Image.Image] = []
    w, h = 128, 96
    anchor = (64, 12)
    for t in range(20):
        ang = 0.7 * math.sin(t / 3.0)
        L = 52
        x = int(anchor[0] + L * math.sin(ang))
        y = int(anchor[1] + L * math.cos(ang))
        im = Image.new("RGB", (w, h), (250, 250, 252))
        d = ImageDraw.Draw(im)
        d.line((anchor[0], 0, anchor[0], anchor[1]), fill=(80, 80, 80), width=3)
        d.line((anchor[0], anchor[1], x, y), fill=(70, 70, 120), width=3)
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(80, 140, 220))
        frames.append(im)
    _save_gif(frames, path, duration=70)
    return {"concept": "pendulum", "caption": "pendulum animation swinging side to side", "tags": "video,physics,pendulum,oscillation,motion"}


def _video_water_cycle(path: Path) -> Dict[str, str]:
    frames: List[Image.Image] = []
    w, h = 128, 96
    for t in range(16):
        im = Image.new("RGB", (w, h), (230, 244, 255))
        d = ImageDraw.Draw(im)
        d.ellipse((6, 70, 56, 92), fill=(70, 130, 220))
        d.ellipse((86, 10, 120, 28), fill=(245, 245, 245), outline=(180, 180, 180))
        d.ellipse((74, 14, 106, 32), fill=(245, 245, 245), outline=(180, 180, 180))
        d.ellipse((14, 8, 28, 22), fill=(245, 220, 70))
        for i in range(3):
            yy = 34 + ((t * 4 + i * 14) % 36)
            d.line((94 + i * 6, yy, 92 + i * 6, yy + 10), fill=(60, 120, 220), width=2)
        d.arc((20, 34, 72, 84), 250, 340, fill=(80, 170, 90), width=3)
        d.arc((40, 28, 92, 78), 60, 160, fill=(80, 170, 90), width=3)
        frames.append(im)
    _save_gif(frames, path, duration=90)
    return {"concept": "water cycle", "caption": "animated water cycle scene with water, cloud, rain, and arrows", "tags": "video,earth_science,water_cycle,evaporation,condensation,precipitation"}


def _video_orbit(path: Path) -> Dict[str, str]:
    frames: List[Image.Image] = []
    w, h = 128, 96
    cx, cy = 56, 48
    for t in range(24):
        a = 2 * math.pi * t / 24.0
        px = int(cx + 28 * math.cos(a))
        py = int(cy + 16 * math.sin(a))
        im = Image.new("RGB", (w, h), (15, 15, 28))
        d = ImageDraw.Draw(im)
        d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(250, 200, 70))
        d.arc((cx - 30, cy - 18, cx + 30, cy + 18), 0, 360, fill=(80, 90, 120), width=1)
        d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(90, 150, 235))
        frames.append(im)
    _save_gif(frames, path, duration=80)
    return {"concept": "orbit", "caption": "planet orbit animation around a central sun", "tags": "video,astronomy,orbit,planet,sun,solar_system"}


VIDEO_BUILDERS = [
    ("bouncing_ball.gif", _video_bouncing_ball),
    ("pendulum.gif", _video_pendulum),
    ("water_cycle.gif", _video_water_cycle),
    ("orbit.gif", _video_orbit),
]


def _rows_for_model(model_path: str, meta: Dict[str, str], rng: random.Random) -> List[Dict[str, str]]:
    concept = meta["concept"]
    caption = meta["caption"]
    tags = meta["tags"]
    rows = [
        ("What 3D shape/model is this?", f"It looks like a {concept} model.", "identify_3d"),
        ("Describe this 3D model in simple terms.", f"This is a {caption}.", "describe_3d"),
        ("What are a few clues that identify this 3D model?", f"Clues: {caption}; tags include {tags}.", "3d_clues"),
    ]
    if concept in {"cube", "pyramid", "tetrahedron", "triangular prism"}:
        rows.append(("Is this model more like a prism, pyramid, or polyhedron? Explain briefly.", f"It is a {concept}-type polyhedron based on its faces and overall shape.", "3d_classify"))
    out = []
    for user, assistant, task in rows:
        out.append(
            {
                "user": user if rng.random() > 0.25 else "3D model question: " + user,
                "assistant": assistant,
                "model3d_path": model_path,
                "model3d_caption": caption,
                "model3d_tags": tags,
                "topic": "multimodal_3d_tests",
                "concept": concept,
                "task": task,
            }
        )
    return out


def _rows_for_video(video_path: str, meta: Dict[str, str], rng: random.Random) -> List[Dict[str, str]]:
    concept = meta["concept"]
    caption = meta["caption"]
    tags = meta["tags"]
    rows = [
        ("What concept does this short video most likely demonstrate?", f"It most likely demonstrates {concept}.", "identify_video_concept"),
        ("Describe the motion or change shown in this test video.", f"The video shows {caption}.", "describe_video"),
        ("What visual clues support that interpretation?", f"Clues: {caption}; tags include {tags}.", "video_clues"),
    ]
    if concept in {"motion", "pendulum", "orbit"}:
        rows.append(("Is the motion in this video static or changing over time? Explain briefly.", "It is changing over time; the object's position changes across frames.", "motion_change"))
    out = []
    for user, assistant, task in rows:
        out.append(
            {
                "user": user if rng.random() > 0.25 else "Video question: " + user,
                "assistant": assistant,
                "video_path": video_path,
                "video_caption": caption,
                "video_tags": tags,
                "topic": "multimodal_video_tests",
                "concept": concept,
                "task": task,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate small 3D-model and test-video QA datasets.")
    ap.add_argument("--output", default="conversation_data.multimodal_3d_video_tests_v1.jsonl")
    ap.add_argument("--models_dir", default="test_3d_models")
    ap.add_argument("--videos_dir", default="test_videos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    rows: List[Dict[str, str]] = []

    model_paths: List[Tuple[str, Dict[str, str]]] = []
    for stem, fn in MODEL_BUILDERS:
        p = Path(args.models_dir) / f"{stem}.obj"
        meta = fn(p)
        model_paths.append((str(p), meta))

    video_paths: List[Tuple[str, Dict[str, str]]] = []
    for filename, fn in VIDEO_BUILDERS:
        p = Path(args.videos_dir) / filename
        meta = fn(p)
        video_paths.append((str(p), meta))

    repeats = max(1, int(args.repeats))
    for ridx in range(repeats):
        for model_path, meta in model_paths:
            for row in _rows_for_model(model_path, meta, rng):
                row["variant_pass"] = ridx
                rows.append(row)
        for video_path, meta in video_paths:
            for row in _rows_for_video(video_path, meta, rng):
                row["variant_pass"] = ridx
                rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"3D models: {len(model_paths)}")
    print(f"Videos: {len(video_paths)}")
    print(f"Repeats: {repeats}")
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
