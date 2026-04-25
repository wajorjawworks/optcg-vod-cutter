#!/usr/bin/env python3
"""
One Piece TCG Sim VOD Cutter - Version B (log matcher)

Purpose:
- Take the segments detected by script1 (vod_cutter.py) and match each game
  clip to its combat log file using the room ID.

How room ID matching works:
- The first line of every log file contains the room ID in one of two forms:
    "Attempting to connect to ROOMID"
    "Waiting for a Connection with Room ID:ROOMID"
- The same message appears in the in-game chat at the start of each match.
- This script scans the chat panel of the VOD near each detected game start,
  extracts the room ID, and looks it up in the log directory.

Output:
- game_01.log, game_02.log … copied next to the clips in --output-dir
- segments_matched.json with a log_file field added to each segment

Requirements:
    pip install av pillow pytesseract

Also required:
- Tesseract OCR installed (auto-detected on Windows)
- segments.json produced by vod_cutter.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import pytesseract
from PIL import Image, ImageOps


# ── helpers shared with vod_cutter ──────────────────────────────────────────

_TESSERACT_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
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


def crop_region(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = img.size
    x1 = int(round(w * left));  y1 = int(round(h * top))
    x2 = int(round(w * right)); y2 = int(round(h * bottom))
    x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1))
    x2 = max(x1 + 1, min(w, x2)); y2 = max(y1 + 1, min(h, y2))
    return img.crop((x1, y1, x2, y2))


# ── room ID extraction ───────────────────────────────────────────────────────

_ROOM_ID_RE = re.compile(
    r'(?:connect(?:ion)?\s+(?:to\s+|with\s+room\s+id\s*[:\s]\s*))([A-Z0-9]{4,12})',
    re.IGNORECASE,
)


def extract_room_id(text: str) -> Optional[str]:
    """Pull the room ID out of a line like 'Attempting to connect to WLWTPJC'."""
    m = _ROOM_ID_RE.search(text)
    return m.group(1).upper() if m else None


# ── leader extraction ────────────────────────────────────────────────────────

_LEADER_RE = re.compile(r'Leader is ([^[<\n]+?) \[', re.IGNORECASE)


def extract_leaders(log_path: str) -> List[str]:
    """Return leader names in the order they appear in the log (opponent first, then you)."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        return [m.group(1).strip() for m in _LEADER_RE.finditer(text)]
    except Exception:
        return []


def leaders_to_slug(leaders: List[str]) -> str:
    """Turn ['Monkey D. Luffy', 'Imu'] into 'MonkeyDLuffy_vs_Imu'."""
    def sanitize(name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9 ]", "", name)  # keep letters, digits, spaces
        return "".join(word.capitalize() for word in name.split())

    parts = [sanitize(l) for l in leaders if l]
    return "_vs_".join(parts) if parts else "Unknown"


# ── log file index ───────────────────────────────────────────────────────────

def build_log_index(log_dir: str) -> Dict[str, List[Path]]:
    """Return {ROOM_ID: [sorted list of matching log paths]}."""
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


# ── chat scan for room ID ────────────────────────────────────────────────────

def scan_chat_for_room_id(
    video_path: str,
    start_t: float,
    chat_box: Tuple[float, float, float, float],
    tesseract_cmd: Optional[str],
    window: float = 60.0,
    sample_seconds: float = 2.0,
) -> Optional[str]:
    """
    Scan the chat panel of the video in [start_t - 10, start_t + window] and
    return the first room ID found, or None.
    """
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


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match VOD clips to combat log files via room ID")
    p.add_argument("--input", required=True, help="Path to the VOD file used by vod_cutter.py")
    p.add_argument("--segments-json", required=True, help="Path to segments.json from vod_cutter.py")
    p.add_argument("--log-dir", required=True, help="Directory containing .log combat files")
    p.add_argument("--output-dir", required=True, help="Directory where game_NN.mp4 clips live")
    p.add_argument("--chat-left",   type=float, default=0.054)
    p.add_argument("--chat-top",    type=float, default=0.262)
    p.add_argument("--chat-right",  type=float, default=0.270)
    p.add_argument("--chat-bottom", type=float, default=0.636)
    p.add_argument("--tesseract-cmd", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tesseract_cmd = _resolve_tesseract(args.tesseract_cmd)
    if tesseract_cmd:
        print(f"[INFO] Using Tesseract at: {tesseract_cmd}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    if not os.path.isfile(args.segments_json):
        print(f"[ERROR] segments.json not found: {args.segments_json}")
        return 1
    if not os.path.isdir(args.log_dir):
        print(f"[ERROR] Log directory not found: {args.log_dir}")
        return 1

    with open(args.segments_json, encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[INFO] Loaded {len(segments)} segments from {args.segments_json}")

    print(f"[INFO] Indexing log files in: {args.log_dir}")
    log_index = build_log_index(args.log_dir)
    print(f"[INFO] Indexed {len(log_index)} unique room IDs across {sum(len(v) for v in log_index.values())} log files")

    chat_box = (args.chat_left, args.chat_top, args.chat_right, args.chat_bottom)

    for seg in segments:
        idx = seg["index"]
        start_t = seg["start"]
        print(f"\n[Game {idx}] start={start_t:.1f}s — scanning chat for room ID...")

        room_id = scan_chat_for_room_id(args.input, start_t, chat_box, tesseract_cmd)

        if not room_id:
            print(f"  [WARN] No room ID found in chat — cannot match log")
            seg["room_id"] = None
            seg["log_file"] = None
            seg["log_files_all"] = []
            continue

        print(f"  Room ID: {room_id}")
        seg["room_id"] = room_id

        matches = log_index.get(room_id, [])
        if not matches:
            print(f"  [WARN] No log file found for room ID {room_id}")
            seg["log_file"] = None
            seg["log_files_all"] = []
            continue

        if len(matches) > 1:
            print(f"  [INFO] {len(matches)} log files share this room ID (reconnections) — using earliest")
            for m in matches:
                print(f"    {m.name}")

        best = matches[0]  # earliest by filename (ISO timestamp sorts correctly)

        # Extract leaders and build a descriptive slug for renaming
        leaders = extract_leaders(str(best))
        slug = leaders_to_slug(leaders)
        print(f"  Leaders: {' vs '.join(leaders) if leaders else 'unknown'}")

        # Find the clip — accept game_NN.mp4 or any already-renamed game_NN_*.mp4
        new_clip = os.path.join(args.output_dir, f"game_{idx:02d}_{slug}.mp4")
        exact = os.path.join(args.output_dir, f"game_{idx:02d}.mp4")
        existing = sorted(Path(args.output_dir).glob(f"game_{idx:02d}*.mp4"))
        if os.path.isfile(exact):
            os.rename(exact, new_clip)
            print(f"  Renamed clip: game_{idx:02d}.mp4 -> {os.path.basename(new_clip)}")
        elif existing and str(existing[0]) != new_clip:
            os.rename(str(existing[0]), new_clip)
            print(f"  Renamed clip: {existing[0].name} -> {os.path.basename(new_clip)}")
        elif os.path.isfile(new_clip):
            print(f"  Clip already named correctly: {os.path.basename(new_clip)}")
        else:
            print(f"  [WARN] No clip found for game {idx:02d}")

        # Copy the log with matching name
        dest_log = os.path.join(args.output_dir, f"game_{idx:02d}_{slug}.log")
        shutil.copy2(str(best), dest_log)
        print(f"  Matched log: {best.name} -> {os.path.basename(dest_log)}")

        seg["log_file"] = dest_log
        seg["log_files_all"] = [str(p) for p in matches]
        seg["leaders"] = leaders
        seg["clip_file"] = new_clip

    out_json = os.path.join(args.output_dir, "segments_matched.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"\n[INFO] Written: {out_json}")

    matched = sum(1 for s in segments if s.get("log_file"))
    print(f"[INFO] Matched {matched}/{len(segments)} games to log files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
