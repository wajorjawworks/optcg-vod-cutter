#!/usr/bin/env python3
"""
One Piece TCG Sim VOD Cutter — full pipeline

Runs three stages in sequence:
  1. Detect game boundaries via timer OCR, validate with chat OCR, cut clips with ffmpeg
  2. Match each clip to its combat log via room ID, rename clips with leader names
  3. Generate a YouTube-style thumbnail for each game

Usage:
    py pipeline.py --input VOD.mp4 --output-dir clips/ \
        --log-dir "C:/path/to/CombatLogs/AutoSaved" \
        --cards-dir "C:/path/to/Cards"

Skip flags:
    --no-cut          detect only, don't run ffmpeg
    --no-match        skip log matching / renaming
    --no-thumbnails   skip thumbnail generation

Requirements:
    pip install av numpy pillow pytesseract
    Tesseract OCR installed
    ffmpeg on PATH (unless --no-cut)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


# ── shared utilities ──────────────────────────────────────────────────────────

_TESSERACT_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]


def _resolve_tesseract(cmd: Optional[str]) -> Optional[str]:
    if cmd:
        return cmd
    if shutil.which("tesseract"):
        return None
    for path in _TESSERACT_FALLBACK_PATHS:
        if os.path.isfile(path):
            return path
    return None


def normalize_text(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("|", " ")
    text = re.sub(r"[^a-z0-9!#\[\]\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sec_to_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def crop_region(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = img.size
    x1 = int(round(w * left));  y1 = int(round(h * top))
    x2 = int(round(w * right)); y2 = int(round(h * bottom))
    x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1))
    x2 = max(x1 + 1, min(w, x2)); y2 = max(y1 + 1, min(h, y2))
    return img.crop((x1, y1, x2, y2))


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


# ── stage 1: detect and cut ───────────────────────────────────────────────────

@dataclass
class OCRHit:
    t: float
    raw_text: str
    norm_text: str


@dataclass
class Segment:
    index: int
    start: float
    end: float
    duration: float
    start_source: str
    end_source: str


def parse_timer_seconds(text: str) -> Optional[int]:
    text = text.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    m = re.search(r'(\d{1,2})[:\s](\d{2})', text)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if 0 <= mins <= 17 and 0 <= secs <= 59:
            return mins * 60 + secs
    return None


_TIMER_OCR_CFG = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:"


def _ocr_worker(task: Tuple[int, float, bytes, Optional[str]]) -> Tuple[int, float, str, str]:
    sample_index, t, png_bytes, tesseract_cmd = task
    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        img = Image.open(BytesIO(png_bytes)).convert("L")
        raw_text = (pytesseract.image_to_string(img, config=_TIMER_OCR_CFG) or "").strip()
        return sample_index, t, raw_text, normalize_text(raw_text)
    except Exception:
        return sample_index, t, "", ""


def scan_video_for_timer(
    video_path: str,
    sample_seconds: float,
    timer_box: Tuple[float, float, float, float],
    tesseract_cmd: Optional[str],
    max_workers: int,
) -> Tuple[List[OCRHit], float]:
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == "video")
    duration = float(container.duration / av.time_base) if container.duration is not None else 0.0

    tasks: List[Tuple[int, float, bytes, Optional[str]]] = []
    next_sample_t = 0.0
    sample_index = 0
    for frame in container.decode(video=stream.index):
        if frame.pts is None or frame.time_base is None:
            continue
        t = float(frame.pts * frame.time_base)
        if t + 1e-6 < next_sample_t:
            continue
        buf = BytesIO()
        crop_region(frame.to_image(), *timer_box).save(buf, format="PNG")
        tasks.append((sample_index, t, buf.getvalue(), tesseract_cmd))
        next_sample_t += sample_seconds
        sample_index += 1
    container.close()

    results: Dict[int, Tuple[int, float, str, str]] = {}
    with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {executor.submit(_ocr_worker, task): task[0] for task in tasks}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = (idx, 0.0, "", "")

    hits: List[OCRHit] = []
    for idx in sorted(results.keys()):
        _, t, raw_text, norm_text = results[idx]
        hits.append(OCRHit(t=t, raw_text=raw_text, norm_text=norm_text))
    return hits, duration


def find_timer_start_candidates(
    hits: List[OCRHit], low: int, high: int
) -> List[Tuple[float, str]]:
    return [(h.t, h.raw_text) for h in hits
            if (s := parse_timer_seconds(h.raw_text)) is not None and low <= s <= high]


def find_timer_end_candidates(
    hits: List[OCRHit], threshold: int
) -> List[Tuple[float, str]]:
    return [(h.t, h.raw_text) for h in hits
            if (s := parse_timer_seconds(h.raw_text)) is not None and s <= threshold]


def cluster_candidates(
    candidates: List[Tuple[float, str]], cluster_gap: float
) -> List[Tuple[float, str]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[0])
    clusters: List[List[Tuple[float, str]]] = [[candidates[0]]]
    for item in candidates[1:]:
        if item[0] - clusters[-1][-1][0] <= cluster_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return [(c[0][0], c[0][1]) for c in clusters]


def validate_start_with_chat(
    video_path: str,
    start_t: float,
    chat_box: Tuple[float, float, float, float],
    tesseract_cmd: Optional[str],
    window: float = 90.0,
    sample_seconds: float = 3.0,
) -> bool:
    """Return True if 'leader' appears in chat within window seconds after start_t."""
    try:
        container = av.open(video_path)
        stream = next(s for s in container.streams if s.type == "video")
        next_t = max(0.0, start_t - 10.0)
        check_to = start_t + window
        for frame in container.decode(video=stream.index):
            if frame.pts is None or frame.time_base is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t < next_t:
                continue
            if t > check_to:
                break
            chat_img = crop_region(frame.to_image(), *chat_box).convert("L")
            text = pytesseract.image_to_string(chat_img, config="--oem 3 --psm 6") or ""
            if "leader" in normalize_text(text):
                container.close()
                return True
            next_t = t + sample_seconds
        container.close()
        return False
    except Exception:
        return True


def build_segments(
    start_clusters: List[Tuple[float, str]],
    end_clusters: List[Tuple[float, str]],
    duration: float,
    start_pad: float,
    end_pad: float,
    min_duration: float,
    max_duration: float,
) -> List[Segment]:
    segments: List[Segment] = []
    overlap_tolerance = 3.0
    next_start_buffer = 2.0
    end_idx = 0

    for i, (start_t, start_text) in enumerate(start_clusters):
        next_start_t = start_clusters[i + 1][0] if i + 1 < len(start_clusters) else None
        chosen_end_t: Optional[float] = None
        chosen_end_text = ""

        while end_idx < len(end_clusters) and end_clusters[end_idx][0] <= start_t:
            end_idx += 1

        if end_idx < len(end_clusters):
            end_t, end_text = end_clusters[end_idx]
            if next_start_t is None or end_t < next_start_t:
                chosen_end_t, chosen_end_text = end_t, end_text
                end_idx += 1
            elif end_t <= next_start_t + overlap_tolerance:
                chosen_end_t, chosen_end_text = end_t, end_text
                end_idx += 1

        if chosen_end_t is None:
            if next_start_t is not None:
                chosen_end_t = max(start_t + min_duration, next_start_t - next_start_buffer)
                chosen_end_text = "next start fallback"
            else:
                chosen_end_t = duration
                chosen_end_text = "video end fallback"

        start = max(0.0, start_t - start_pad)
        end = min(duration, chosen_end_t + end_pad)
        dur = end - start

        if dur < min_duration or dur > max_duration:
            continue

        segments.append(Segment(
            index=len(segments) + 1,
            start=start, end=end, duration=dur,
            start_source=start_text, end_source=chosen_end_text,
        ))

    return segments


def write_segments_csv(output_dir: str, segments: List[Segment]) -> str:
    path = os.path.join(output_dir, "segments.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "start_seconds", "end_seconds", "duration_seconds",
                         "start_hms", "end_hms", "start_source", "end_source"])
        for s in segments:
            writer.writerow([
                s.index, f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}",
                sec_to_hms(s.start), sec_to_hms(s.end),
                s.start_source.replace("\n", " | ")[:200],
                s.end_source.replace("\n", " | ")[:200],
            ])
    return path


def cut_clips(
    video_path: str, output_dir: str, segments: List[Segment], fast_cut: bool
) -> None:
    for seg in segments:
        out_path = os.path.join(output_dir, f"game_{seg.index:02d}.mp4")
        if fast_cut:
            cmd = ["ffmpeg", "-y", "-ss", f"{seg.start:.3f}", "-to", f"{seg.end:.3f}",
                   "-i", video_path, "-c", "copy", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-ss", f"{seg.start:.3f}", "-to", f"{seg.end:.3f}",
                   "-i", video_path, "-c:v", "libx264", "-preset", "fast",
                   "-crf", "18", "-c:a", "aac", "-b:a", "192k", out_path]
        code, _, err = run_cmd(cmd)
        if code == 0:
            print(f"  Cut: {out_path}")
        else:
            print(f"  [WARN] ffmpeg failed for game {seg.index}: {err}", file=sys.stderr)


# ── stage 2: log matching ─────────────────────────────────────────────────────

_ROOM_ID_RE = re.compile(
    r'(?:connect(?:ion)?\s+(?:to\s+|with\s+room\s+id\s*[:\s]\s*))([A-Z0-9]{4,12})',
    re.IGNORECASE,
)
_LEADER_RE = re.compile(r'Leader is ([^[<\n]+?) \[', re.IGNORECASE)


def extract_room_id(text: str) -> Optional[str]:
    m = _ROOM_ID_RE.search(text)
    return m.group(1).upper() if m else None


def extract_leaders(log_path: str) -> List[str]:
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        return [m.group(1).strip() for m in _LEADER_RE.finditer(text)]
    except Exception:
        return []


def leaders_to_slug(leaders: List[str]) -> str:
    def sanitize(name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9 ]", "", name)
        return "".join(word.capitalize() for word in name.split())
    parts = [sanitize(l) for l in leaders if l]
    return "_vs_".join(parts) if parts else "Unknown"


def build_log_index(log_dir: str) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for log_path in sorted(Path(log_dir).glob("*.log")):
        try:
            first_line = log_path.read_text(encoding="utf-8", errors="ignore").split("\n")[0]
        except Exception:
            continue
        room_id = extract_room_id(first_line)
        if room_id:
            index.setdefault(room_id, []).append(log_path)
    return index


def scan_chat_for_room_id(
    video_path: str,
    start_t: float,
    chat_box: Tuple[float, float, float, float],
    tesseract_cmd: Optional[str],
    window: float = 60.0,
    sample_seconds: float = 2.0,
) -> Optional[str]:
    try:
        container = av.open(video_path)
        stream = next(s for s in container.streams if s.type == "video")
        check_from = max(0.0, start_t - 10.0)
        check_to = start_t + window
        next_t = check_from
        for frame in container.decode(video=stream.index):
            if frame.pts is None or frame.time_base is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t < check_from:
                continue
            if t > check_to:
                break
            if t < next_t:
                continue
            chat_img = crop_region(frame.to_image(), *chat_box).convert("L")
            raw = pytesseract.image_to_string(chat_img, config="--oem 3 --psm 6") or ""
            room_id = extract_room_id(raw)
            if room_id:
                container.close()
                return room_id
            next_t = t + sample_seconds
        container.close()
    except Exception as e:
        print(f"  [WARN] chat scan error: {e}")
    return None


def match_logs(
    video_path: str,
    segments: List[dict],
    log_dir: str,
    output_dir: str,
    chat_box: Tuple[float, float, float, float],
    tesseract_cmd: Optional[str],
) -> None:
    log_index = build_log_index(log_dir)
    print(f"[INFO] Indexed {len(log_index)} unique room IDs across "
          f"{sum(len(v) for v in log_index.values())} log files")

    for seg in segments:
        idx = seg["index"]
        start_t = seg["start"]
        print(f"\n[Game {idx}] start={start_t:.1f}s — scanning chat for room ID...")

        room_id = scan_chat_for_room_id(video_path, start_t, chat_box, tesseract_cmd)

        if not room_id:
            print(f"  [WARN] No room ID found — cannot match log")
            seg.update(room_id=None, log_file=None, log_files_all=[], leaders=[], clip_file=seg.get("clip_file"))
            continue

        print(f"  Room ID: {room_id}")
        seg["room_id"] = room_id

        matches = log_index.get(room_id, [])
        if not matches:
            print(f"  [WARN] No log file found for room ID {room_id}")
            seg.update(log_file=None, log_files_all=[])
            continue

        if len(matches) > 1:
            print(f"  [INFO] {len(matches)} log files share this room ID — using earliest")

        best = matches[0]
        leaders = extract_leaders(str(best))
        slug = leaders_to_slug(leaders)
        print(f"  Leaders: {' vs '.join(leaders) if leaders else 'unknown'}")

        new_clip = os.path.join(output_dir, f"game_{idx:02d}_{slug}.mp4")
        exact = os.path.join(output_dir, f"game_{idx:02d}.mp4")
        existing = sorted(Path(output_dir).glob(f"game_{idx:02d}*.mp4"))
        if os.path.isfile(exact):
            os.rename(exact, new_clip)
            print(f"  Renamed: game_{idx:02d}.mp4 -> {os.path.basename(new_clip)}")
        elif existing and str(existing[0]) != new_clip:
            os.rename(str(existing[0]), new_clip)
            print(f"  Renamed: {existing[0].name} -> {os.path.basename(new_clip)}")
        elif os.path.isfile(new_clip):
            print(f"  Clip already named correctly")
        else:
            print(f"  [WARN] No clip found for game {idx:02d}")
            new_clip = seg.get("clip_file", new_clip)

        dest_log = os.path.join(output_dir, f"game_{idx:02d}_{slug}.log")
        shutil.copy2(str(best), dest_log)
        print(f"  Log: {best.name} -> {os.path.basename(dest_log)}")

        seg.update(
            log_file=dest_log,
            log_files_all=[str(p) for p in matches],
            leaders=leaders,
            clip_file=new_clip,
        )


# ── stage 3: thumbnail generation ────────────────────────────────────────────

THUMB_W, THUMB_H = 1280, 720
BG_COLOR = (10, 10, 15)
FONT_PATH_IMPACT = r"C:\Windows\Fonts\impact.ttf"
BADGE_FIRST  = (212, 175, 55)
BADGE_SECOND = (160, 160, 170)

_LEADER_LINE_RE = re.compile(
    r'\[([^\]]+)\] Leader is ([^[<\n]+?) \[.*?([A-Z]{1,5}\d*-\d+).*?\]',
)
_CHOSE_RE = re.compile(r'\[([^\]]+)\] Chose to go (First|Second)', re.IGNORECASE)
_CARD_CODE_RE = re.compile(r'([A-Z0-9]+)-(\d+)')


def _norm_player(name: str) -> str:
    return name.replace("​", "").strip()


def parse_log_for_thumbnail(log_path: str) -> List[dict]:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    leader_map: dict = {}
    for m in _LEADER_LINE_RE.finditer(text):
        player = _norm_player(m.group(1))
        leader_map[player] = {
            "player": player,
            "name": m.group(2).strip(),
            "card_code": m.group(3),
            "went_first": None,
        }
    for m in _CHOSE_RE.finditer(text):
        player = _norm_player(m.group(1))
        order = m.group(2).lower()
        if player in leader_map:
            leader_map[player]["went_first"] = (order == "first")
        for p, info in leader_map.items():
            if p != player and info["went_first"] is None:
                info["went_first"] = (order == "second")
    leaders = list(leader_map.values())
    leaders.sort(key=lambda x: (not x["went_first"],))
    return leaders


def card_image_path(card_code: str, cards_dir: str) -> Optional[str]:
    m = _CARD_CODE_RE.match(card_code)
    if not m:
        return None
    path = os.path.join(cards_dir, m.group(1), f"{card_code}.jpg")
    return path if os.path.isfile(path) else None


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def dominant_color(img: Image.Image) -> Tuple[int, int, int]:
    small = img.convert("RGB").resize((80, 80))
    quantized = small.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    colors = [(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]) for i in range(8)]
    def vividness(c):
        r, g, b = c
        mx, mn = max(c), min(c)
        return ((mx - mn) / (mx + 1)) * min((r + g + b) / 3, 180)
    return max(colors, key=vividness)


def make_bg(color_l: Tuple, color_r: Tuple) -> Image.Image:
    bg = Image.new("RGBA", (THUMB_W, THUMB_H), (*BG_COLOR, 255))

    def add_glow(cx: int, color: Tuple, alpha: int = 90):
        layer = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        r = 520
        ImageDraw.Draw(layer).ellipse(
            [cx - r, THUMB_H // 2 - r, cx + r, THUMB_H // 2 + r], fill=(*color, alpha)
        )
        bg.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius=110)))

    add_glow(THUMB_W // 4, color_l)
    add_glow(3 * THUMB_W // 4, color_r)

    vignette = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    dv = ImageDraw.Draw(vignette)
    for i in range(80):
        alpha = int((i / 80) ** 2 * 160)
        dv.rectangle([i, i, THUMB_W - i, THUMB_H - i], outline=(0, 0, 0, 160 - alpha))
    bg.alpha_composite(vignette)
    return bg


def crop_art(card_img: Image.Image, art_fraction: float = 0.62) -> Image.Image:
    return card_img.crop((0, 0, card_img.width, int(card_img.height * art_fraction)))


def paste_card_with_glow(
    canvas: Image.Image, card_img: Image.Image,
    cx: int, cy: int, target_h: int, angle: float, glow_color: Tuple,
) -> None:
    scale = target_h / card_img.height
    card = card_img.resize((int(card_img.width * scale), target_h), Image.LANCZOS).convert("RGBA")
    if angle != 0:
        card = card.rotate(angle, expand=True, resample=Image.BICUBIC)

    x = cx - card.width // 2
    y = cy - card.height // 2

    glow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow_layer.paste(Image.new("RGBA", card.size, (*glow_color, 160)), (x, y), card)
    canvas.alpha_composite(glow_layer.filter(ImageFilter.GaussianBlur(radius=30)))

    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_layer.paste(Image.new("RGBA", card.size, (0, 0, 0, 200)), (x + 16, y + 16), card)
    canvas.alpha_composite(shadow_layer.filter(ImageFilter.GaussianBlur(radius=22)))

    canvas.alpha_composite(card, (x, y))


def draw_badge(draw: ImageDraw.ImageDraw, cx: int, cy: int, label: str,
               color: Tuple, font: ImageFont.FreeTypeFont) -> None:
    bbox = font.getbbox(label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 22, 11
    rx0, ry0 = cx - tw // 2 - pad_x, cy - th // 2 - pad_y
    rx1, ry1 = cx + tw // 2 + pad_x, cy + th // 2 + pad_y
    draw.rounded_rectangle([rx0 + 3, ry0 + 3, rx1 + 3, ry1 + 3], radius=14, fill=(0, 0, 0, 180))
    draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=14, fill=color)
    draw.text((cx, cy), label, font=font, fill=(10, 10, 10), anchor="mm")


def create_thumbnail(leaders: List[dict], cards_dir: str, output_path: str) -> None:
    font_vs    = load_font(FONT_PATH_IMPACT, 130)
    font_badge = load_font(FONT_PATH_IMPACT, 46)
    font_name  = load_font(FONT_PATH_IMPACT, 52)

    card_imgs = []
    colors = []
    for leader in leaders[:2]:
        path = card_image_path(leader["card_code"], cards_dir)
        if path:
            art = crop_art(Image.open(path).convert("RGBA"))
            card_imgs.append(art)
            colors.append(dominant_color(art))
        else:
            card_imgs.append(None)
            colors.append((80, 80, 120))

    canvas = make_bg(colors[0], colors[1])
    ImageDraw.Draw(canvas).line(
        [(THUMB_W // 2, 0), (THUMB_W // 2, THUMB_H)], fill=(255, 255, 255, 30), width=2
    )

    positions = [
        (THUMB_W // 4 - 10,      THUMB_H // 2 + 30, -8.0),
        (3 * THUMB_W // 4 + 10,  THUMB_H // 2 + 30,  8.0),
    ]
    card_h = int(THUMB_H * 0.92)

    # Draw 2nd leader first so 1st leader overlaps toward center
    for i in reversed(range(min(2, len(leaders)))):
        if card_imgs[i] is None:
            continue
        cx, cy, angle = positions[i]
        paste_card_with_glow(canvas, card_imgs[i], cx, cy, card_h, angle, colors[i])

    draw = ImageDraw.Draw(canvas)
    for i, leader in enumerate(leaders[:2]):
        cx = positions[i][0]
        badge_label = "1ST" if leader.get("went_first") else "2ND"
        badge_color = BADGE_FIRST if leader.get("went_first") else BADGE_SECOND
        draw_badge(draw, cx, THUMB_H - 100, badge_label, badge_color, font_badge)
        draw.text((cx, THUMB_H - 52), leader["name"], font=font_name,
                  fill=(255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))

    draw.text((THUMB_W // 2, THUMB_H // 2 - 20), "VS", font=font_vs,
              fill=(255, 255, 255), anchor="mm", stroke_width=6, stroke_fill=(0, 0, 0))

    canvas.convert("RGB").save(output_path, "JPEG", quality=95)
    print(f"  Thumbnail: {output_path}")


def generate_thumbnails(segments: List[dict], cards_dir: str, output_dir: str) -> None:
    for seg in segments:
        idx = seg["index"]
        log_path = seg.get("log_file")
        if not log_path or not os.path.isfile(log_path):
            print(f"[Game {idx}] No log file — skipping thumbnail")
            continue

        leaders = parse_log_for_thumbnail(log_path)
        if not leaders:
            print(f"[Game {idx}] No leaders found in log — skipping thumbnail")
            continue

        for l in leaders:
            order = "1st" if l.get("went_first") else ("2nd" if l.get("went_first") is False else "?")
            print(f"  {order}: {l['name']} ({l['card_code']})")

        slug = seg.get("clip_file") or f"game_{idx:02d}"
        base = os.path.splitext(os.path.basename(slug))[0]
        create_thumbnail(leaders, cards_dir, os.path.join(output_dir, f"{base}_thumb.jpg"))


# ── argument parsing and main ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OPTCG VOD Cutter — detect, cut, match logs, generate thumbnails"
    )

    # Input / output
    p.add_argument("--input",      required=True, help="Path to the VOD file")
    p.add_argument("--output-dir", required=True, help="Directory to write clips and reports")

    # Log matching
    p.add_argument("--log-dir", default=None,
                   help="Directory containing .log combat files (required for log matching)")

    # Thumbnail generation
    p.add_argument("--cards-dir", default=None,
                   help="Root dir of card art, e.g. Cards/ (required for thumbnails)")

    # Timer crop region
    p.add_argument("--timer-left",   type=float, default=0.60)
    p.add_argument("--timer-top",    type=float, default=0.0)
    p.add_argument("--timer-right",  type=float, default=0.72)
    p.add_argument("--timer-bottom", type=float, default=0.028)

    # Chat crop region
    p.add_argument("--chat-left",   type=float, default=0.054)
    p.add_argument("--chat-top",    type=float, default=0.262)
    p.add_argument("--chat-right",  type=float, default=0.270)
    p.add_argument("--chat-bottom", type=float, default=0.636)

    # Detection tuning
    p.add_argument("--sample-seconds",          type=float, default=0.25)
    p.add_argument("--timer-start-low",         type=int,   default=17 * 60 + 20)
    p.add_argument("--timer-start-high",        type=int,   default=17 * 60 + 35)
    p.add_argument("--timer-end-threshold",     type=int,   default=15)
    p.add_argument("--start-cluster-gap-seconds", type=float, default=30.0)
    p.add_argument("--end-cluster-gap-seconds",   type=float, default=10.0)
    p.add_argument("--start-pad-seconds",       type=float, default=3.0)
    p.add_argument("--end-pad-seconds",         type=float, default=2.0)
    p.add_argument("--min-duration-seconds",    type=float, default=120.0)
    p.add_argument("--max-duration-seconds",    type=float, default=7200.0)

    # System
    p.add_argument("--tesseract-cmd", default=None)
    p.add_argument("--max-workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 1))

    # Stage skips
    p.add_argument("--no-cut",        action="store_true", help="Detect only, don't run ffmpeg")
    p.add_argument("--fast-cut",      action="store_true", help="Stream copy (faster, less precise)")
    p.add_argument("--no-match",      action="store_true", help="Skip log matching")
    p.add_argument("--no-thumbnails", action="store_true", help="Skip thumbnail generation")
    p.add_argument("--dump-ocr",      action="store_true", help="Write raw OCR to ocr_dump.txt")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input VOD not found: {args.input}", file=sys.stderr)
        return 1
    if not args.no_cut and not ensure_ffmpeg():
        print("[ERROR] ffmpeg not found on PATH", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    tesseract_cmd = _resolve_tesseract(args.tesseract_cmd)
    if tesseract_cmd:
        print(f"[INFO] Using Tesseract at: {tesseract_cmd}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    timer_box = (args.timer_left, args.timer_top, args.timer_right, args.timer_bottom)
    chat_box  = (args.chat_left,  args.chat_top,  args.chat_right,  args.chat_bottom)

    # ── stage 1: detect ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[STAGE 1] Scanning timer: {args.input}")
    print(f"{'='*60}")

    hits, duration = scan_video_for_timer(
        args.input, args.sample_seconds, timer_box, tesseract_cmd, args.max_workers
    )
    print(f"[INFO] Duration: {sec_to_hms(duration)}, OCR samples: {len(hits)}")

    non_empty = [h for h in hits if h.raw_text.strip()]
    print(f"[INFO] Non-empty OCR: {len(non_empty)}")
    for h in non_empty[:5]:
        print(f"  t={sec_to_hms(h.t)} | {repr(h.raw_text[:60])}")

    if args.dump_ocr:
        dump = os.path.join(args.output_dir, "ocr_dump.txt")
        with open(dump, "w", encoding="utf-8") as f:
            for h in hits:
                f.write(f"{sec_to_hms(h.t)}\t{repr(h.raw_text)}\n")
        print(f"[INFO] OCR dump: {dump}")

    start_cands = find_timer_start_candidates(hits, args.timer_start_low, args.timer_start_high)
    end_cands   = find_timer_end_candidates(hits, args.timer_end_threshold)
    start_clusters = cluster_candidates(start_cands, args.start_cluster_gap_seconds)
    end_clusters   = cluster_candidates(end_cands,   args.end_cluster_gap_seconds)

    print(f"[INFO] Start clusters before chat validation: {len(start_clusters)}")

    validated = []
    for t, txt in start_clusters:
        print(f"  [CHECK] {sec_to_hms(t)} — scanning chat for 'leader'...")
        ok = validate_start_with_chat(args.input, t, chat_box, tesseract_cmd)
        print(f"    -> {'CONFIRMED' if ok else 'REJECTED (reconnection?)'}")
        if ok:
            validated.append((t, txt))
    start_clusters = validated

    print(f"[INFO] Confirmed starts: {len(start_clusters)}, end clusters: {len(end_clusters)}")

    segments = build_segments(
        start_clusters, end_clusters, duration,
        args.start_pad_seconds, args.end_pad_seconds,
        args.min_duration_seconds, args.max_duration_seconds,
    )

    print(f"[INFO] Built {len(segments)} segment(s):")
    for s in segments:
        print(f"  Game {s.index}: {sec_to_hms(s.start)} -> {sec_to_hms(s.end)} ({s.duration:.1f}s)")

    write_segments_csv(args.output_dir, segments)
    seg_dicts = [asdict(s) for s in segments]

    if not args.no_cut:
        cut_clips(args.input, args.output_dir, segments, args.fast_cut)

    # ── stage 2: log matching ─────────────────────────────────────────────────
    if not args.no_match:
        if not args.log_dir:
            print("\n[INFO] --log-dir not provided, skipping log matching")
        elif not os.path.isdir(args.log_dir):
            print(f"\n[WARN] Log directory not found: {args.log_dir} — skipping")
        else:
            print(f"\n{'='*60}")
            print(f"[STAGE 2] Matching combat logs from: {args.log_dir}")
            print(f"{'='*60}")
            match_logs(args.input, seg_dicts, args.log_dir, args.output_dir, chat_box, tesseract_cmd)
            matched = sum(1 for s in seg_dicts if s.get("log_file"))
            print(f"[INFO] Matched {matched}/{len(seg_dicts)} games to log files")

    matched_json = os.path.join(args.output_dir, "segments_matched.json")
    with open(matched_json, "w", encoding="utf-8") as f:
        json.dump(seg_dicts, f, indent=2)
    print(f"[INFO] Written: {matched_json}")

    # ── stage 3: thumbnails ───────────────────────────────────────────────────
    if not args.no_thumbnails:
        if not args.cards_dir:
            print("\n[INFO] --cards-dir not provided, skipping thumbnails")
        elif not os.path.isdir(args.cards_dir):
            print(f"\n[WARN] Cards directory not found: {args.cards_dir} — skipping")
        else:
            print(f"\n{'='*60}")
            print(f"[STAGE 3] Generating thumbnails")
            print(f"{'='*60}")
            generate_thumbnails(seg_dicts, args.cards_dir, args.output_dir)

    print(f"\n[INFO] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
