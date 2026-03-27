from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


BG = "#071019"
PANEL = "#0e1c2c"
PANEL_ALT = "#16324f"
TEXT = "#eef5ff"
MUTED = "#9eb3cf"
ACCENT = "#70b8ff"


def _font_candidates(size: int, bold: bool = False) -> List[str]:
    windir = Path.home().anchor + "Windows\\Fonts\\"
    if bold:
        return [
            windir + "bahnschrift.ttf",
            windir + "segoeuib.ttf",
            windir + "arialbd.ttf",
        ]
    return [
        windir + "bahnschrift.ttf",
        windir + "segoeui.ttf",
        windir + "arial.ttf",
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_candidates(size, bold=bold):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _resolve_destination(destination_hint: str, default_dir: Path, default_name: str) -> Path:
    cooked = str(destination_hint or "").strip()
    if not cooked:
        default_dir.mkdir(parents=True, exist_ok=True)
        return (default_dir / default_name).resolve()
    target = Path(cooked).expanduser()
    if target.suffix.lower() == ".png":
        target.parent.mkdir(parents=True, exist_ok=True)
        return target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    return (target / default_name).resolve()


def copy_generated_image(source_path: str, destination_hint: str, default_dir: Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Generated image not found: {source}")
    target = _resolve_destination(destination_hint, default_dir, source.name)
    shutil.copy2(source, target)
    return target


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text or "").replace("\r", "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def render_chat_transcript_image(
    transcript: Sequence[Dict[str, object]],
    *,
    destination_hint: str,
    default_dir: Path,
    title: str = "Supermix Studio Conversation",
    session_id: str = "",
) -> Path:
    if not transcript:
        raise ValueError("Transcript is empty.")

    title_font = load_font(34, bold=True)
    meta_font = load_font(18)
    who_font = load_font(18, bold=True)
    body_font = load_font(24)
    caption_font = load_font(18)
    scratch = Image.new("RGB", (1600, 200), BG)
    scratch_draw = ImageDraw.Draw(scratch)

    width = 1480
    pad_x = 42
    pad_y = 36
    bubble_max = 980
    items: List[Dict[str, object]] = []
    total_height = pad_y + 96

    for row in transcript:
        role = str(row.get("role") or "assistant")
        kind = str(row.get("kind") or "text")
        model_label = str(row.get("model_label") or ("You" if role == "user" else "Assistant"))
        body_text = str(row.get("response") or row.get("prompt_used") or "").strip()
        caption_text = str(row.get("prompt_used") or "").strip() if kind == "image" else ""
        image_path = str(row.get("output_path") or row.get("image_output_path") or "").strip()
        text_lines = _wrap_lines(scratch_draw, body_text, body_font, bubble_max - 40) if body_text else []
        caption_lines = _wrap_lines(scratch_draw, caption_text, caption_font, bubble_max - 40) if caption_text else []
        text_height = max(0, len(text_lines)) * 32
        caption_height = max(0, len(caption_lines)) * 24
        image_thumb = None
        image_height = 0
        if kind == "image" and image_path:
            try:
                with Image.open(image_path) as opened:
                    thumb = opened.convert("RGB")
                    thumb.thumbnail((bubble_max - 40, 520))
                    image_thumb = thumb.copy()
                    image_height = image_thumb.height + 16
            except Exception:
                image_thumb = None
        bubble_height = 56 + text_height + image_height + caption_height + 18
        items.append(
            {
                "role": role,
                "kind": kind,
                "model_label": model_label,
                "text_lines": text_lines,
                "caption_lines": caption_lines,
                "image_thumb": image_thumb,
                "bubble_height": bubble_height,
                "raw_text": body_text,
            }
        )
        total_height += bubble_height + 20

    total_height += pad_y
    image = Image.new("RGB", (width, total_height), BG)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((22, 22, width - 22, total_height - 22), radius=34, outline="#19314b", width=2, fill=BG)
    draw.text((pad_x, pad_y), title, font=title_font, fill=TEXT)
    meta_line = "Local packaged conversation export"
    if session_id:
        meta_line += f" | session {session_id[:12]}"
    draw.text((pad_x, pad_y + 48), meta_line, font=meta_font, fill=MUTED)

    cursor_y = pad_y + 96
    for item in items:
        is_user = item["role"] == "user"
        bubble_width = bubble_max
        x1 = width - pad_x - bubble_width if is_user else pad_x
        x2 = x1 + bubble_width
        y1 = cursor_y
        y2 = y1 + int(item["bubble_height"])
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=28,
            fill=PANEL_ALT if is_user else PANEL,
            outline="#1f4366" if is_user else "#19314b",
            width=2,
        )
        draw.text((x1 + 20, y1 + 16), str(item["model_label"]), font=who_font, fill=ACCENT if not is_user else TEXT)
        inner_y = y1 + 48
        for line in item["text_lines"]:
            draw.text((x1 + 20, inner_y), line, font=body_font, fill=TEXT)
            inner_y += 32
        if item["image_thumb"] is not None:
            image.paste(item["image_thumb"], (x1 + 20, inner_y))
            inner_y += item["image_thumb"].height + 16
        for line in item["caption_lines"]:
            draw.text((x1 + 20, inner_y), line, font=caption_font, fill=MUTED)
            inner_y += 24
        cursor_y = y2 + 20

    default_name = "supermix_conversation.png"
    target = _resolve_destination(destination_hint, default_dir, default_name)
    image.save(target, format="PNG")
    return target
