from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\bahnschrift.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_vertical_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1.0 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)
    return image


def add_soft_orb(base: Image.Image, center: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: int) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius // 2))
    base.alpha_composite(overlay)


def draw_chat_logo(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    card = make_vertical_gradient((size, size), (7, 17, 31), (12, 32, 57)).convert("RGBA")
    add_soft_orb(card, (size // 3, size // 3), size // 3, (61, 157, 255), 110)
    add_soft_orb(card, (size * 3 // 4, size * 7 // 10), size // 3, (255, 143, 74), 90)

    draw = ImageDraw.Draw(card)
    inset = max(24, size // 14)
    draw.rounded_rectangle(
        (inset, inset, size - inset, size - inset),
        radius=size // 5,
        outline=(143, 182, 236, 70),
        width=max(4, size // 42),
        fill=(11, 22, 39, 210),
    )

    aqua = (86, 187, 255, 255)
    amber = (255, 153, 86, 255)
    white = (242, 247, 255, 255)

    bubble1 = (
        size * 0.20,
        size * 0.24,
        size * 0.66,
        size * 0.53,
    )
    draw.rounded_rectangle(bubble1, radius=size * 0.12, fill=aqua)
    draw.polygon(
        [
            (size * 0.31, size * 0.51),
            (size * 0.25, size * 0.64),
            (size * 0.39, size * 0.55),
        ],
        fill=aqua,
    )

    bubble2 = (
        size * 0.35,
        size * 0.42,
        size * 0.79,
        size * 0.73,
    )
    draw.rounded_rectangle(bubble2, radius=size * 0.12, fill=amber)
    draw.polygon(
        [
            (size * 0.59, size * 0.71),
            (size * 0.69, size * 0.83),
            (size * 0.66, size * 0.68),
        ],
        fill=amber,
    )

    dot_r = size // 36
    dots1_y = size * 0.38
    for idx in range(3):
        cx = size * (0.31 + 0.09 * idx)
        draw.ellipse((cx - dot_r, dots1_y - dot_r, cx + dot_r, dots1_y + dot_r), fill=white)
    dots2_y = size * 0.56
    for idx in range(2):
        cx = size * (0.49 + 0.11 * idx)
        draw.rounded_rectangle(
            (cx - dot_r * 2, dots2_y - dot_r, cx + dot_r * 2, dots2_y + dot_r),
            radius=dot_r,
            fill=(35, 49, 75, 150),
        )

    shadow = card.filter(ImageFilter.GaussianBlur(size // 32))
    image.alpha_composite(shadow, dest=(0, 0))
    image.alpha_composite(card, dest=(0, 0))
    return image


def make_splash() -> Image.Image:
    width, height = 1200, 680
    image = make_vertical_gradient((width, height), (7, 16, 29), (9, 28, 45)).convert("RGBA")
    add_soft_orb(image, (240, 160), 220, (61, 157, 255), 120)
    add_soft_orb(image, (980, 520), 250, (255, 143, 74), 80)
    add_soft_orb(image, (760, 180), 180, (86, 255, 214), 45)

    draw = ImageDraw.Draw(image)
    for x in range(0, width, 48):
        draw.line((x, 0, x, height), fill=(255, 255, 255, 12), width=1)
    for y in range(0, height, 48):
        draw.line((0, y, width, y), fill=(255, 255, 255, 10), width=1)

    panel = (66, 72, width - 66, height - 72)
    draw.rounded_rectangle(panel, radius=36, fill=(11, 22, 39, 210), outline=(124, 164, 215, 48), width=2)

    brand = draw_chat_logo(320)
    image.alpha_composite(brand, dest=(760, 150))

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(28, bold=False)
    body_font = load_font(22, bold=False)
    label_font = load_font(18, bold=True)

    draw.text((120, 126), "Supermix Qwen", font=title_font, fill=(239, 246, 255))
    draw.text((120, 200), "Desktop chat launcher", font=subtitle_font, fill=(123, 186, 255))

    draw.rounded_rectangle((120, 248, 322, 286), radius=19, fill=(27, 83, 140))
    draw.text((146, 255), "LOCAL MODEL + WEBVIEW", font=label_font, fill=(236, 245, 255))

    lines = [
        "Starts the local chat server automatically.",
        "Opens the interface in its own desktop window.",
        "Uses the newest bundled adapter by default.",
    ]
    y = 334
    for line in lines:
        draw.rounded_rectangle((120, y + 8, 132, y + 20), radius=6, fill=(255, 153, 86))
        draw.text((156, y), line, font=body_font, fill=(214, 226, 245))
        y += 54

    footer = "Supermix_27"
    footer_w = draw.textbbox((0, 0), footer, font=label_font)[2]
    draw.text((width - 120 - footer_w, height - 120), footer, font=label_font, fill=(132, 157, 194))
    return image


def make_installer_banner(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = make_vertical_gradient(size, (8, 18, 32), (11, 30, 48)).convert("RGBA")
    add_soft_orb(image, (width // 2, height // 5), min(width, height) // 3, (61, 157, 255), 110)
    add_soft_orb(image, (width // 2, height * 4 // 5), min(width, height) // 3, (255, 143, 74), 80)
    logo = draw_chat_logo(min(width - 16, 148))
    logo = logo.resize((min(width - 18, 132), min(width - 18, 132)), Image.LANCZOS)
    image.alpha_composite(logo, dest=((width - logo.width) // 2, 18))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22, bold=True)
    body_font = load_font(14, bold=False)
    draw.text((18, 178), "Supermix", font=title_font, fill=(240, 246, 255))
    draw.text((18, 204), "Qwen Desktop", font=title_font, fill=(133, 194, 255))
    draw.text((18, 246), "Local model launcher", font=body_font, fill=(218, 228, 245))
    draw.text((18, 266), "with bundled adapter", font=body_font, fill=(218, 228, 245))
    return image


def make_small_installer_image(size: tuple[int, int]) -> Image.Image:
    logo = draw_chat_logo(128)
    return logo.resize(size, Image.LANCZOS)


def save_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    icon = draw_chat_logo(512)
    icon_png = ASSETS_DIR / "supermix_qwen_icon.png"
    icon_ico = ASSETS_DIR / "supermix_qwen_icon.ico"
    icon.save(icon_png)
    icon.save(icon_ico, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])

    splash = make_splash()
    splash.save(ASSETS_DIR / "supermix_qwen_splash.png")

    installer_wizard = make_installer_banner((164, 314)).convert("RGB")
    installer_wizard.save(ASSETS_DIR / "supermix_qwen_installer_wizard.bmp")

    installer_small = make_small_installer_image((55, 55)).convert("RGB")
    installer_small.save(ASSETS_DIR / "supermix_qwen_installer_small.bmp")

    print(icon_ico)
    print(ASSETS_DIR / "supermix_qwen_splash.png")
    print(ASSETS_DIR / "supermix_qwen_installer_wizard.bmp")


if __name__ == "__main__":
    save_assets()
