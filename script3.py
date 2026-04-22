#!/usr/bin/env python3
"""
One Piece TCG Sim VOD Cutter - Version C (thumbnail generator)

Creates a YouTube-style thumbnail for each game clip showing:
- Both leader card arts side by side
- Which leader went first / second
- Leader names

Reads from segments_matched.json produced by script2.py.

Requirements:
    pip install pillow
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ── constants ────────────────────────────────────────────────────────────────

THUMB_W, THUMB_H = 1280, 720
BG_COLOR = (10, 10, 15)
FONT_PATH_IMPACT = r"C:\Windows\Fonts\impact.ttf"
FONT_PATH_BOLD   = r"C:\Windows\Fonts\arialbd.ttf"

BADGE_FIRST  = (212, 175, 55)   # gold
BADGE_SECOND = (160, 160, 170)  # silver
VS_COLOR     = (255, 255, 255)
NAME_COLOR   = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)


# ── log parsing ──────────────────────────────────────────────────────────────

_LEADER_LINE_RE = re.compile(
    r'\[([^\]]+)\] Leader is ([^[<\n]+?) \[.*?([A-Z]{1,5}\d*-\d+).*?\]',
)
_CHOSE_RE = re.compile(r'\[([^\]]+)\] Chose to go (First|Second)', re.IGNORECASE)


def _norm_player(name: str) -> str:
    """Strip zero-width spaces for reliable comparison."""
    return name.replace("​", "").strip()
_CARD_CODE_RE = re.compile(r'([A-Z0-9]+)-(\d+)')


def parse_log(log_path: str):
    """
    Returns:
        leaders: list of dicts [{player, name, card_code, went_first}]
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # {player_name: {name, card_code}}
    leader_map = {}
    for m in _LEADER_LINE_RE.finditer(text):
        player = _norm_player(m.group(1))
        leader_name, card_code = m.group(2).strip(), m.group(3)
        leader_map[player] = {"player": player, "name": leader_name, "card_code": card_code, "went_first": None}

    # Determine turn order
    for m in _CHOSE_RE.finditer(text):
        player = _norm_player(m.group(1))
        order = m.group(2).lower()
        if player in leader_map:
            leader_map[player]["went_first"] = (order == "first")
        for p, info in leader_map.items():
            if p != player and info["went_first"] is None:
                info["went_first"] = (order == "second")

    leaders = list(leader_map.values())
    # Put first-player first
    leaders.sort(key=lambda x: (not x["went_first"],))
    return leaders


def card_image_path(card_code: str, cards_dir: str) -> Optional[str]:
    m = _CARD_CODE_RE.match(card_code)
    if not m:
        return None
    set_name = m.group(1)
    path = os.path.join(cards_dir, set_name, f"{card_code}.jpg")
    return path if os.path.isfile(path) else None


# ── drawing helpers ──────────────────────────────────────────────────────────

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def dominant_color(img: Image.Image) -> Tuple[int, int, int]:
    """Extract the most vivid non-dark color from an image."""
    small = img.convert("RGB").resize((80, 80))
    quantized = small.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    colors = [(palette[i*3], palette[i*3+1], palette[i*3+2]) for i in range(8)]
    def vividness(c):
        r, g, b = c
        brightness = (r + g + b) / 3
        mx, mn = max(c), min(c)
        saturation = (mx - mn) / (mx + 1)
        return saturation * min(brightness, 180)  # vivid but not washed out
    colors.sort(key=vividness, reverse=True)
    return colors[0]


def make_bg(color_l: Tuple, color_r: Tuple) -> Image.Image:
    """Dark background with a subtle colored glow on each side."""
    bg = Image.new("RGBA", (THUMB_W, THUMB_H), (*BG_COLOR, 255))

    def add_glow(cx: int, color: Tuple, alpha: int = 90):
        layer = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = 520
        d.ellipse([cx - r, THUMB_H // 2 - r, cx + r, THUMB_H // 2 + r],
                  fill=(*color, alpha))
        blurred = layer.filter(ImageFilter.GaussianBlur(radius=110))
        bg.alpha_composite(blurred)

    add_glow(THUMB_W // 4, color_l)
    add_glow(3 * THUMB_W // 4, color_r)

    # Subtle dark vignette at edges
    vignette = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    dv = ImageDraw.Draw(vignette)
    for i in range(80):
        alpha = int((i / 80) ** 2 * 160)
        dv.rectangle([i, i, THUMB_W - i, THUMB_H - i],
                     outline=(0, 0, 0, 160 - alpha))
    bg.alpha_composite(vignette)
    return bg


def crop_art(card_img: Image.Image, art_fraction: float = 0.62) -> Image.Image:
    """Keep only the character art, removing the text box at the bottom."""
    h = int(card_img.height * art_fraction)
    return card_img.crop((0, 0, card_img.width, h))


def paste_card_with_glow(canvas: Image.Image, card_img: Image.Image,
                          cx: int, cy: int, target_h: int, angle: float,
                          glow_color: Tuple):
    scale = target_h / card_img.height
    card = card_img.resize((int(card_img.width * scale), target_h), Image.LANCZOS).convert("RGBA")
    if angle != 0:
        card = card.rotate(angle, expand=True, resample=Image.BICUBIC)

    x = cx - card.width // 2
    y = cy - card.height // 2

    # Colored glow behind card
    glow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow_mask = Image.new("RGBA", card.size, (*glow_color, 160))
    glow_layer.paste(glow_mask, (x, y), card)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=30))
    canvas.alpha_composite(glow_layer)

    # Drop shadow
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_mask = Image.new("RGBA", card.size, (0, 0, 0, 200))
    shadow_layer.paste(shadow_mask, (x + 16, y + 16), card)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=22))
    canvas.alpha_composite(shadow_layer)

    canvas.alpha_composite(card, (x, y))
    return x, y, card.width, card.height


def draw_badge(draw: ImageDraw.ImageDraw, cx: int, cy: int, label: str,
               color: Tuple, font: ImageFont.FreeTypeFont):
    bbox = font.getbbox(label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 22, 11
    rx0, ry0 = cx - tw // 2 - pad_x, cy - th // 2 - pad_y
    rx1, ry1 = cx + tw // 2 + pad_x, cy + th // 2 + pad_y
    # Shadow
    draw.rounded_rectangle([rx0+3, ry0+3, rx1+3, ry1+3], radius=14, fill=(0, 0, 0, 180))
    draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=14, fill=color)
    draw.text((cx, cy), label, font=font, fill=(10, 10, 10), anchor="mm")


# ── thumbnail generation ─────────────────────────────────────────────────────

def create_thumbnail(leaders: list, cards_dir: str, output_path: str):
    font_vs    = load_font(FONT_PATH_IMPACT, 130)
    font_badge = load_font(FONT_PATH_IMPACT, 46)
    font_name  = load_font(FONT_PATH_IMPACT, 52)

    # Load card images (art crop only)
    card_imgs = []
    colors = []
    for leader in leaders[:2]:
        path = card_image_path(leader["card_code"], cards_dir)
        if path:
            full = Image.open(path).convert("RGBA")
            art = crop_art(full)
            card_imgs.append(art)
            colors.append(dominant_color(art))
        else:
            card_imgs.append(None)
            colors.append((80, 80, 120))

    # Background with per-side color glow
    canvas = make_bg(colors[0], colors[1])

    # Center divider — thin bright line
    draw = ImageDraw.Draw(canvas)
    draw.line([(THUMB_W // 2, 0), (THUMB_W // 2, THUMB_H)], fill=(255, 255, 255, 30), width=2)

    positions = [
        (THUMB_W // 4 - 10,     THUMB_H // 2 + 30, -8.0),
        (3 * THUMB_W // 4 + 10, THUMB_H // 2 + 30,  8.0),
    ]
    card_h = int(THUMB_H * 0.92)

    for i, leader in enumerate(leaders[:2]):
        cx, cy, angle = positions[i]
        if card_imgs[i] is None:
            continue

        x, y, cw, ch = paste_card_with_glow(
            canvas, card_imgs[i], cx, cy, card_h, angle, colors[i]
        )

        draw = ImageDraw.Draw(canvas)

        # 1ST / 2ND badge and name — fixed positions near bottom of frame
        badge_label = "1ST" if leader.get("went_first") else "2ND"
        badge_color = BADGE_FIRST if leader.get("went_first") else BADGE_SECOND
        badge_cy = THUMB_H - 100
        draw_badge(draw, cx, badge_cy, badge_label, badge_color, font_badge)

        name_cy = THUMB_H - 52
        draw.text((cx, name_cy), leader["name"], font=font_name,
                  fill=(255, 255, 255), anchor="mm",
                  stroke_width=3, stroke_fill=(0, 0, 0))

    # VS — center, large, stroked
    draw.text((THUMB_W // 2, THUMB_H // 2 - 20), "VS", font=font_vs,
              fill=(255, 255, 255), anchor="mm",
              stroke_width=6, stroke_fill=(0, 0, 0))

    canvas.convert("RGB").save(output_path, "JPEG", quality=95)
    print(f"  Thumbnail: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thumbnails for OPTCG VOD clips")
    p.add_argument("--segments-json", required=True, help="segments_matched.json from script2.py")
    p.add_argument("--cards-dir", required=True, help="Root directory of card art (contains OP01/, OP02/, …)")
    p.add_argument("--output-dir", required=True, help="Directory to write thumbnail JPEGs into")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.segments_json, encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[INFO] Generating thumbnails for {len(segments)} segment(s)")
    os.makedirs(args.output_dir, exist_ok=True)

    for seg in segments:
        idx = seg["index"]
        log_path = seg.get("log_file")
        if not log_path or not os.path.isfile(log_path):
            print(f"[Game {idx}] No log file — skipping")
            continue

        print(f"\n[Game {idx}] Parsing {os.path.basename(log_path)}")
        leaders = parse_log(log_path)

        if not leaders:
            print(f"  [WARN] No leaders found in log")
            continue

        for l in leaders:
            order = "1st" if l.get("went_first") else ("2nd" if l.get("went_first") is False else "?")
            print(f"  {order}: {l['name']} ({l['card_code']})")

        slug = seg.get("clip_file") or f"game_{idx:02d}"
        base = os.path.splitext(os.path.basename(slug))[0]
        out_path = os.path.join(args.output_dir, f"{base}_thumb.jpg")

        create_thumbnail(leaders, args.cards_dir, out_path)

    print(f"\n[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
