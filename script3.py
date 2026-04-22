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


def draw_text_shadowed(draw: ImageDraw.ImageDraw, pos: Tuple[int, int], text: str,
                       font: ImageFont.FreeTypeFont, fill, shadow=SHADOW_COLOR,
                       offset: int = 3, anchor: str = "mm"):
    sx, sy = pos[0] + offset, pos[1] + offset
    draw.text((sx, sy), text, font=font, fill=shadow, anchor=anchor)
    draw.text(pos, text, font=font, fill=fill, anchor=anchor)


def draw_badge(draw: ImageDraw.ImageDraw, cx: int, cy: int, label: str,
               color: Tuple, font: ImageFont.FreeTypeFont):
    bbox = font.getbbox(label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 20, 10
    rx0 = cx - tw // 2 - pad_x
    ry0 = cy - th // 2 - pad_y
    rx1 = cx + tw // 2 + pad_x
    ry1 = cy + th // 2 + pad_y
    draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=12, fill=color)
    draw.text((cx, cy), label, font=font, fill=(10, 10, 10), anchor="mm")


def paste_card(canvas: Image.Image, card_img: Image.Image,
               center_x: int, center_y: int, target_h: int, angle: float):
    """Scale card to target_h, rotate, paste centered at (center_x, center_y)."""
    scale = target_h / card_img.height
    new_w = int(card_img.width * scale)
    card = card_img.resize((new_w, target_h), Image.LANCZOS).convert("RGBA")

    if angle != 0:
        card = card.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Drop shadow
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_card = Image.new("RGBA", card.size, (0, 0, 0, 180))
    shadow_x = center_x - card.width // 2 + 12
    shadow_y = center_y - card.height // 2 + 12
    shadow_layer.paste(shadow_card, (shadow_x, shadow_y), card)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=18))
    canvas.alpha_composite(shadow_layer)

    x = center_x - card.width // 2
    y = center_y - card.height // 2
    canvas.alpha_composite(card, (x, y))
    return x, y, card.width, card.height


# ── thumbnail generation ─────────────────────────────────────────────────────

def create_thumbnail(leaders: list, cards_dir: str, output_path: str):
    canvas = Image.new("RGBA", (THUMB_W, THUMB_H), (*BG_COLOR, 255))

    font_vs      = load_font(FONT_PATH_IMPACT, 96)
    font_badge   = load_font(FONT_PATH_IMPACT, 40)
    font_name    = load_font(FONT_PATH_BOLD,   36)

    draw = ImageDraw.Draw(canvas)

    # Layout: two cards, left and right thirds; VS in center
    positions = [
        (THUMB_W // 4,      THUMB_H // 2 - 20,  -6.0),   # left card (1st)
        (3 * THUMB_W // 4,  THUMB_H // 2 - 20,   6.0),   # right card (2nd)
    ]
    card_h = int(THUMB_H * 0.76)

    for i, leader in enumerate(leaders[:2]):
        cx, cy, angle = positions[i]
        card_path = card_image_path(leader["card_code"], cards_dir)

        if card_path:
            card_img = Image.open(card_path).convert("RGBA")
            x, y, cw, ch = paste_card(canvas, card_img, cx, cy, card_h, angle)
        else:
            x, y, cw, ch = cx - 160, cy - 230, 320, 460
            draw.rectangle([x, y, x + cw, y + ch], fill=(40, 40, 60))

        draw = ImageDraw.Draw(canvas)

        # Badge pinned inside the top of the card, always on-screen
        badge_label = "1ST" if leader.get("went_first") else "2ND"
        badge_color = BADGE_FIRST if leader.get("went_first") else BADGE_SECOND
        badge_cy = max(y + 36, 40)
        draw_badge(draw, cx, badge_cy, badge_label, badge_color, font_badge)

        # Leader name at bottom
        name_y = min(y + ch + 36, THUMB_H - 12)
        draw_text_shadowed(draw, (cx, name_y), leader["name"], font_name, NAME_COLOR, anchor="mt")

    # VS in center
    draw_text_shadowed(draw, (THUMB_W // 2, THUMB_H // 2), "VS", font_vs, VS_COLOR, offset=4)

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
