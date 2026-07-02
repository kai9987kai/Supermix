from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageSequence, ImageStat


def _bucket(value: float, cuts: List[float], labels: List[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1] if labels else "mid"


def _analyze_video_impl(path: str) -> Tuple[float, ...]:
    p = Path(path)
    if not p.exists():
        return (0.0,) * 12

    suffix = p.suffix.lower()
    if suffix not in {".gif", ".webp"}:
        return (0.0,) * 12

    with Image.open(p) as im:
        frames: List[Image.Image] = []
        durations: List[int] = []
        for frame in ImageSequence.Iterator(im):
            fr = frame.convert("RGB").copy()
            frames.append(fr)
            durations.append(int(frame.info.get("duration", im.info.get("duration", 0)) or 0))
        if not frames:
            return (0.0,) * 12

        w, h = frames[0].size
        n = len(frames)
        avg_duration_ms = float(sum(durations)) / float(max(1, len(durations))) if durations else 0.0
        total_duration_ms = float(sum(durations)) if durations else avg_duration_ms * float(n)

        stat0 = ImageStat.Stat(frames[0])
        mean_r, mean_g, mean_b = [float(x) / 255.0 for x in stat0.mean[:3]]
        brightness = 0.2126 * mean_r + 0.7152 * mean_g + 0.0722 * mean_b
        contrast = sum(float(x) / 255.0 for x in stat0.stddev[:3]) / 3.0

        diff_vals: List[float] = []
        prev_gray = None
        sample_frames = frames[: min(n, 24)]
        for frame in sample_frames:
            gray = frame.convert("L").resize((min(64, max(1, w)), min(64, max(1, h))))
            px = list(gray.getdata())
            if prev_gray is not None:
                s = 0.0
                for a, b in zip(px, prev_gray):
                    s += abs(float(a) - float(b))
                diff_vals.append((s / max(1, len(px))) / 255.0)
            prev_gray = px
        motion = sum(diff_vals) / max(1, len(diff_vals)) if diff_vals else 0.0

        ratio = float(w) / float(max(1, h))
        try:
            import math

            log_aspect = max(-2.0, min(2.0, math.log(max(1e-6, ratio))))
        except Exception:
            log_aspect = 0.0

        fps_proxy = 0.0
        if avg_duration_ms > 0:
            fps_proxy = min(120.0, 1000.0 / avg_duration_ms) / 120.0
        frame_count_norm = min(1.0, float(n) / 64.0)
        total_dur_norm = min(1.0, total_duration_ms / 10000.0)
        avg_dur_norm = min(1.0, avg_duration_ms / 1000.0)
        area_proxy = min(1.0, (float(w) * float(h)) / float(1024 * 1024))
        max_c = max(mean_r, mean_g, mean_b)
        min_c = min(mean_r, mean_g, mean_b)
        saturation_proxy = 0.0 if max_c < 1e-6 else (max_c - min_c) / max_c

        return (
            float(w) / 1024.0,
            float(h) / 1024.0,
            log_aspect / 2.0,
            area_proxy,
            frame_count_norm,
            avg_dur_norm,
            total_dur_norm,
            fps_proxy,
            brightness,
            contrast,
            motion,
            saturation_proxy,
        )


@lru_cache(maxsize=256)
def extract_video_numeric_features(path: str) -> Tuple[float, ...]:
    """
    Compact numeric video features for multimodal fusion.
    """
    try:
        return _analyze_video_impl(path)
    except Exception:
        return (0.0,) * 12


@lru_cache(maxsize=256)
def describe_video_for_text(path: str) -> str:
    """
    Lightweight descriptor for small test videos/GIFs used in text-only training.
    """
    p = Path(path)
    if not p.exists():
        return f"missing_video:{path}"

    suffix = p.suffix.lower()
    if suffix not in {".gif", ".webp"}:
        # Keep generic support for future placeholders.
        return f"video_file {p.name}"

    nums = extract_video_numeric_features(path)
    with Image.open(p) as im:
        frames: List[Image.Image] = []
        durations: List[int] = []
        for frame in ImageSequence.Iterator(im):
            fr = frame.convert("RGB").copy()
            frames.append(fr)
            durations.append(int(frame.info.get("duration", im.info.get("duration", 0)) or 0))
        if not frames:
            return f"video_file {p.name} empty"

        w, h = frames[0].size
        n = len(frames)
        avg_duration_ms = int(sum(durations) / max(1, len(durations))) if durations else 0

        # Global color/brightness from first frame.
        stat0 = ImageStat.Stat(frames[0])
        mean_r, mean_g, mean_b = [float(x) for x in stat0.mean[:3]]
        brightness = float(nums[8])
        bright_label = _bucket(brightness, [0.25, 0.45, 0.70], ["dark", "dim", "mid", "bright"])

        # Motion proxy via average frame-to-frame grayscale difference.
        diff_vals: List[float] = []
        prev_gray = None
        for frame in frames[: min(n, 24)]:
            gray = frame.convert("L").resize((min(64, w), min(64, h)))
            px = list(gray.getdata())
            if prev_gray is not None:
                s = 0
                for a, b in zip(px, prev_gray):
                    s += abs(int(a) - int(b))
                diff_vals.append((s / max(1, len(px))) / 255.0)
            prev_gray = px
        motion = float(nums[10]) if len(nums) >= 11 else (sum(diff_vals) / max(1, len(diff_vals)) if diff_vals else 0.0)
        motion_label = _bucket(motion, [0.02, 0.06, 0.12], ["static", "low_motion", "medium_motion", "high_motion"])

        aspect = "square"
        ratio = float(w) / float(max(1, h))
        if ratio > 1.15:
            aspect = "landscape"
        elif ratio < 0.87:
            aspect = "portrait"

        # Dominant color family.
        if max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b) < 12:
            color = "gray"
        elif mean_r >= mean_g and mean_r >= mean_b:
            color = "red"
        elif mean_g >= mean_r and mean_g >= mean_b:
            color = "green"
        else:
            color = "blue"

        return (
            f"{aspect} {w}x{h}; frames={n}; avg_ms={avg_duration_ms}; "
            f"brightness={bright_label}; motion={motion_label}; color={color}"
        )
