"""Generates chaselate.ico (and a standalone PNG) for the frozen exe / installer / shortcuts.

No external art assets, on purpose -- matches how the app already draws its own tray icon at
runtime (chaselate/ui/overlay.py:_make_icon) rather than sourcing something from outside the
project. Colors are pulled to match chaselate/ui/style.py's DARK palette exactly, so the exe
icon, the installer icon, and the in-app tray icon all read as the same product.

Design: three chevrons (>>>) fading from the app's neutral text color to its accent cyan,
inside a dark rounded-square "glass" tile with a hairline border and a soft top sheen. The
chevrons carry the name's double meaning at a glance -- forward motion for "chase" (real-time,
catching up to live speech), a colour handoff from neutral to accent for "translate" (one
thing becoming another) -- without needing literal text glyphs, which do not survive
downscaling to 16px.

Run: .venv\\Scripts\\python.exe packaging\\scripts\\make_icon.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# Supersampled canvas; every size in the .ico is downscaled from this with LANCZOS, which is
# what keeps the small sizes (16/24/32px) from looking muddy.
S = 1024

# -- palette, matching chaselate/ui/style.py's DARK dict exactly -----------------------------
PLATE = (14, 16, 22, 255)  # opaque version of DARK["plate"] rgba(16,18,24,216)
PLATE_TOP = (24, 28, 38, 255)  # subtle lighter tone for the gradient background
BORDER = (255, 255, 255, 45)  # DARK["border"] rgba(255,255,255,38), nudged for icon visibility
ACCENT = (0x6F, 0xD0, 0xFF, 255)  # DARK["accent"] #6fd0ff
ACCENT_DIM = (0x6F, 0xD0, 0xFF, 70)  # glow, echoes DARK["accent_dim"]
TEXT = (0xF2, 0xF4, 0xF8, 235)  # DARK["text"] #f2f4f8, the "original speech" chevron


def lerp_color(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))


def rounded_square_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def vertical_gradient(size, top, bottom):
    grad = Image.new("RGBA", (1, size), 0)
    for y in range(size):
        t = y / (size - 1)
        grad.putpixel((0, y), lerp_color(top, bottom, t))
    return grad.resize((size, size))


def chevron_points(cx, cy, half_w, half_h):
    """Three points describing a right-pointing '>' centred at (cx, cy)."""
    return [(cx - half_w, cy - half_h), (cx + half_w, cy), (cx - half_w, cy + half_h)]


def build_base():
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # Background tile: rounded square with a soft vertical gradient (top slightly lighter --
    # a cheap but effective "glass catching light" cue) and a hairline border.
    radius = int(S * 0.225)  # a Fluent/Win11-style squircle, not a hard rounded rect
    mask = rounded_square_mask(S, radius)
    bg = vertical_gradient(S, PLATE_TOP, PLATE)
    canvas.paste(bg, (0, 0), mask)

    border_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_layer)
    stroke = max(2, S // 170)
    bd.rounded_rectangle(
        [stroke, stroke, S - 1 - stroke, S - 1 - stroke],
        radius=radius - stroke,
        outline=BORDER,
        width=stroke,
    )
    canvas.alpha_composite(border_layer)

    # Soft top sheen: a low-opacity white ellipse, blurred, clipped to the tile -- the same
    # "glass" trick the app's own translucent panels use, just baked into a static image here.
    sheen = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sheen)
    sd.ellipse(
        [S * 0.08, -S * 0.35, S * 0.92, S * 0.35],
        fill=(255, 255, 255, 22),
    )
    sheen = sheen.filter(ImageFilter.GaussianBlur(S * 0.02))
    sheen.putalpha(Image.composite(sheen.split()[3], Image.new("L", (S, S), 0), mask))
    canvas.alpha_composite(sheen)

    # Chevron glow: a soft cyan blob behind the rightmost (accent-coloured) chevron, echoing
    # the accent_dim highlight the app itself puts behind the actively-translating caption.
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gcx, gcy = int(S * 0.665), int(S * 0.5)
    gr = int(S * 0.30)
    gd.ellipse([gcx - gr, gcy - gr, gcx + gr, gcy + gr], fill=ACCENT_DIM)
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.05))
    glow.putalpha(Image.composite(glow.split()[3], Image.new("L", (S, S), 0), mask))
    canvas.alpha_composite(glow)

    # Three chevrons, evenly spaced, colour fading from TEXT (original) to ACCENT (translated).
    chevrons = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chevrons)
    half_w = S * 0.115
    half_h = S * 0.195
    stroke_w = int(S * 0.075)
    spacing = S * 0.185
    start_x = S * 0.5 - spacing
    cy = S * 0.5
    for i in range(3):
        cx = start_x + spacing * i
        color = lerp_color(TEXT, ACCENT, i / 2.0)
        pts = chevron_points(cx, cy, half_w, half_h)
        cd.line(pts, fill=color, width=stroke_w, joint="curve")
        # Round caps: line() leaves flat ends, so cap each vertex with a filled circle in the
        # same colour to match the rest of the app's soft, no-hard-edges aesthetic.
        r = stroke_w / 2
        for px, py in pts:
            cd.ellipse([px - r, py - r, px + r, py + r], fill=color)
    canvas.alpha_composite(chevrons)

    return canvas


def main():
    base = build_base()

    png_path = os.path.join(OUT_DIR, "chaselate.png")
    base.resize((512, 512), Image.LANCZOS).save(png_path)
    print(f"wrote {png_path}")

    ico_path = os.path.join(OUT_DIR, "chaselate.ico")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    base.save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
    )
    print(f"wrote {ico_path} ({', '.join(str(s) for s in sizes)}px)")

    # A few standalone raster previews at the sizes that matter most for a sanity check --
    # small icons often look fine at 256px and fall apart at 16px, so render that directly
    # rather than trusting the .ico container's own downscaling only.
    for s in (16, 32, 256):
        preview_path = os.path.join(OUT_DIR, f"chaselate_{s}.png")
        base.resize((s, s), Image.LANCZOS).save(preview_path)
        print(f"wrote {preview_path}")


if __name__ == "__main__":
    main()
