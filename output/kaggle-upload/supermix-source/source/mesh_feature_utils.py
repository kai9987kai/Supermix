from functools import lru_cache
from pathlib import Path
from typing import List, Tuple


def _parse_obj(path: Path) -> Tuple[List[Tuple[float, float, float]], int]:
    verts: List[Tuple[float, float, float]] = []
    faces = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("v "):
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except Exception:
                        pass
            elif s.startswith("f "):
                faces += 1
    return verts, faces


def _analyze_3d_model_impl(path: str) -> Tuple[float, ...]:
    p = Path(path)
    if not p.exists():
        return (0.0,) * 12
    if p.suffix.lower() != ".obj":
        return (0.0,) * 12

    verts, face_count = _parse_obj(p)
    if not verts:
        return (0.0,) * 12

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z

    ex = abs(float(dx))
    ey = abs(float(dy))
    ez = abs(float(dz))
    longest = max(ex, ey, ez, 1e-6)
    shortest = max(min(ex, ey, ez), 1e-6)
    mid = sorted([ex, ey, ez])[1]
    volume_proxy = min(1.0, (ex * ey * ez) / 64.0)
    flatness = 1.0 - min(1.0, shortest / longest)
    compactness = min(1.0, shortest / longest)
    anisotropy = min(1.0, (longest - shortest) / longest)
    face_norm = min(1.0, float(face_count) / 256.0)
    vert_norm = min(1.0, float(len(verts)) / 256.0)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    return (
        vert_norm,
        face_norm,
        min(1.0, ex / 16.0),
        min(1.0, ey / 16.0),
        min(1.0, ez / 16.0),
        min(1.0, mid / longest),
        compactness,
        flatness,
        anisotropy,
        volume_proxy,
        max(-1.0, min(1.0, center_x / 8.0)),
        max(-1.0, min(1.0, (center_y + center_z) / 16.0)),
    )


@lru_cache(maxsize=256)
def extract_3d_model_numeric_features(path: str) -> Tuple[float, ...]:
    """
    Compact numeric mesh features for multimodal fusion.
    """
    try:
        return _analyze_3d_model_impl(path)
    except Exception:
        return (0.0,) * 12


@lru_cache(maxsize=256)
def describe_3d_model_for_text(path: str) -> str:
    """
    Tiny geometry descriptor for simple local OBJ test models.
    """
    p = Path(path)
    if not p.exists():
        return f"missing_3d_model:{path}"

    if p.suffix.lower() != ".obj":
        return f"3d_model_file {p.name}"

    nums = extract_3d_model_numeric_features(path)
    verts, face_count = _parse_obj(p)
    if not verts:
        return f"obj {p.name}; vertices=0; faces={face_count}"

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z

    axis_lengths = sorted([abs(dx), abs(dy), abs(dz)])
    if axis_lengths[2] < 1e-6:
        shape_hint = "flat"
    elif axis_lengths[0] > 0 and axis_lengths[2] / max(axis_lengths[0], 1e-6) < 1.6:
        shape_hint = "roughly_compact"
    elif axis_lengths[2] > 2.5 * max(axis_lengths[0], 1e-6):
        shape_hint = "elongated"
    else:
        shape_hint = "irregular"

    if nums[7] > 0.85:
        shape_hint = "flat"
    return (
        f"obj {p.name}; vertices={len(verts)}; faces={face_count}; "
        f"bbox=({dx:.2f},{dy:.2f},{dz:.2f}); shape={shape_hint}"
    )
