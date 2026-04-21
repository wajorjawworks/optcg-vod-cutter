#!/usr/bin/env python3
"""
One Piece TCG Sim VOD Cutter - Version A (clip splitter only)

Purpose:
- Split a VOD into separate game clips using chat OCR only.
- Do NOT try to identify leaders or match combat logs here.
- Focus only on getting game boundaries right.

Detection strategy:
- Fuzzy start keywords:
    * has connected
    * leader is
    * chose to go
    * waiting for a connection with room
    * will select turn order
- Fuzzy end keywords:
    * concedes
    * opponent has disconnected
- Nearby OCR hits are clustered so one game does not create many starts.
- Start/end pairing is chronological.

Requirements:
    pip install av numpy pillow pytesseract

Also required:
- Tesseract OCR installed and accessible
- ffmpeg on PATH
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
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps


START_KEYWORDS = [
    "has connected",
    "version",
    "leader is",
    "chose to go",
]

END_KEYWORDS = [
    "concedes",
    "opponent has disconnected",
]


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


def normalize_text(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("|", " ")
    text = re.sub(r"[^a-z0-9!#\[\]\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_timer_seconds(text: str) -> Optional[int]:
    """Parse MM:SS from noisy OCR of the in-game countdown timer."""
    text = text.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    m = re.search(r'(\d{1,2})[:\s](\d{2})', text)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if 0 <= mins <= 17 and 0 <= secs <= 59:
            return mins * 60 + secs
    return None


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


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def preprocess_for_ocr(chat_img: Image.Image, scale: int = 3) -> Image.Image:
    img = ImageOps.grayscale(chat_img)
    img = ImageOps.autocontrast(img)
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SHARPEN)
    arr = np.array(img)
    cutoff = max(120, int(np.percentile(arr, 68)))
    bw = np.where(arr > cutoff, 255, 0).astype(np.uint8)
    return Image.fromarray(bw, mode="L")


def crop_region(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = img.size
    if all(0.0 <= v <= 1.0 for v in (left, top, right, bottom)):
        x1 = int(round(w * left)); y1 = int(round(h * top)); x2 = int(round(w * right)); y2 = int(round(h * bottom))
    else:
        x1 = int(round(left)); y1 = int(round(top)); x2 = int(round(right)); y2 = int(round(bottom))
    x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1)); x2 = max(x1 + 1, min(w, x2)); y2 = max(y1 + 1, min(h, y2))
    return img.crop((x1, y1, x2, y2))


def _ocr_worker(task: Tuple[int, float, bytes, Optional[str]]) -> Tuple[int, float, str, str]:
    sample_index, t, png_bytes, tesseract_cmd = task
    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        processed = preprocess_for_ocr(img)
        text1 = pytesseract.image_to_string(processed, config="--oem 3 --psm 6") or ""
        text2 = pytesseract.image_to_string(processed, config="--oem 3 --psm 11") or ""
        raw_text = (text1 + "\n" + text2).strip()
        return sample_index, t, raw_text, normalize_text(raw_text)
    except Exception:
        return sample_index, t, "", ""


def scan_video_for_chat(video_path: str, sample_seconds: float, chat_box: Tuple[float, float, float, float], tesseract_cmd: Optional[str], max_workers: int) -> Tuple[List[OCRHit], float]:
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
        full_img = frame.to_image()
        chat_img = crop_region(full_img, *chat_box)
        buf = BytesIO()
        chat_img.save(buf, format="PNG")
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


def detect_start_cues(text: str) -> set[str]:
    """
    Detect broad start cue categories from noisy OCR.
    We only trust a start if multiple different cue categories appear nearby.
    """
    cues: set[str] = set()
    if "connect" in text:
        cues.add("connect")
    if "version" in text:
        cues.add("version")
    if "leader" in text:
        cues.add("leader")
    if "chose" in text:
        cues.add("chose")
    return cues


def end_score(text: str) -> int:
    score = 0
    for kw in END_KEYWORDS:
        if kw in text:
            score += 1
    if "concedes" in text:
        score += 2
    return score


def find_start_candidates(hits: List[OCRHit], window_seconds: float = 20.0, min_distinct_cues: int = 2) -> List[Tuple[float, str]]:
    """
    Build start candidates using multi-cue confirmation.
    A valid start window must contain at least N distinct cue categories.
    """
    out: List[Tuple[float, str]] = []
    n = len(hits)
    i = 0

    while i < n:
        base_t = hits[i].t
        earliest_t: Optional[float] = None
        earliest_text = ""
        cue_set: set[str] = set()

        j = i
        while j < n and hits[j].t <= base_t + window_seconds:
            text = hits[j].norm_text
            if text:
                cues = detect_start_cues(text)
                if cues:
                    cue_set.update(cues)
                    if earliest_t is None:
                        earliest_t = hits[j].t
                        earliest_text = hits[j].raw_text
            j += 1

        if earliest_t is not None and len(cue_set) >= min_distinct_cues:
            out.append((earliest_t, earliest_text))
            # Skip forward beyond this confirmed start window so we don't spawn many duplicates.
            while i < n and hits[i].t <= earliest_t + window_seconds:
                i += 1
            continue

        i += 1

    return out


def find_end_candidates(hits: List[OCRHit], threshold: int = 2) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for h in hits:
        score = end_score(h.norm_text)
        if score >= threshold:
            out.append((h.t, h.raw_text))
    return out


def find_timer_start_candidates(hits: List[OCRHit], low: int = 17 * 60 + 28, high: int = 17 * 60 + 30) -> List[Tuple[float, str]]:
    """Game start: timer reads 17:28–17:30."""
    out: List[Tuple[float, str]] = []
    for h in hits:
        s = parse_timer_seconds(h.raw_text)
        if s is not None and low <= s <= high:
            out.append((h.t, h.raw_text))
    return out


def find_timer_end_candidates(hits: List[OCRHit], threshold: int = 2) -> List[Tuple[float, str]]:
    """Game end: timer hits 00:00–00:02."""
    out: List[Tuple[float, str]] = []
    for h in hits:
        s = parse_timer_seconds(h.raw_text)
        if s is not None and s <= threshold:
            out.append((h.t, h.raw_text))
    return out


def cluster_candidates(candidates: List[Tuple[float, str]], cluster_gap_seconds: float) -> List[Tuple[float, str]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[0])
    clusters: List[List[Tuple[float, str]]] = [[candidates[0]]]
    for item in candidates[1:]:
        if item[0] - clusters[-1][-1][0] <= cluster_gap_seconds:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    # use earliest item in each cluster
    return [(cluster[0][0], cluster[0][1]) for cluster in clusters]


def build_segments_from_candidates(
    start_clusters: List[Tuple[float, str]],
    end_clusters: List[Tuple[float, str]],
    duration: float,
    start_pad: float,
    end_pad: float,
    min_duration: float,
    max_duration: float,
) -> List[Segment]:
    """
    Build segments from all start/end clusters across the whole VOD.

    Rules:
    - each start cluster is always considered
    - each end cluster can only be used once
    - prefer the first unused end after this start and before the next start
    - allow an end a few seconds after the next start to still belong to the previous game
    - if no end exists for a start:
        * use next start - small buffer
        * or video end for the final game
    """
    segments: List[Segment] = []
    overlap_tolerance = 3.0
    next_start_buffer = 2.0
    end_idx = 0

    for i, (start_t, start_text) in enumerate(start_clusters):
        next_start_t = start_clusters[i + 1][0] if i + 1 < len(start_clusters) else None
        chosen_end_t: Optional[float] = None
        chosen_end_text = ""

        # Advance past stale ends that occur before this start
        while end_idx < len(end_clusters) and end_clusters[end_idx][0] <= start_t:
            end_idx += 1

        # Try to consume the first valid UNUSED end after this start
        if end_idx < len(end_clusters):
            end_t, end_text = end_clusters[end_idx]

            # Best case: end before next start
            if next_start_t is None or end_t < next_start_t:
                chosen_end_t = end_t
                chosen_end_text = end_text
                end_idx += 1

            # Also okay if end is only slightly after next start
            elif end_t <= next_start_t + overlap_tolerance:
                chosen_end_t = end_t
                chosen_end_text = end_text
                end_idx += 1

        # No valid unused end found
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

        segments.append(
            Segment(
                index=len(segments) + 1,
                start=start,
                end=end,
                duration=dur,
                start_source=start_text,
                end_source=chosen_end_text,
            )
        )

    return segments


def write_segments_csv(output_dir: str, segments: List[Segment]) -> str:
    path = os.path.join(output_dir, "segments.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "start_seconds", "end_seconds", "duration_seconds", "start_hms", "end_hms", "start_source", "end_source"])
        for s in segments:
            writer.writerow([s.index, f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}", sec_to_hms(s.start), sec_to_hms(s.end), s.start_source.replace("\n", " | ")[:200], s.end_source.replace("\n", " | ")[:200]])
    return path


def write_segments_json(output_dir: str, segments: List[Segment]) -> str:
    path = os.path.join(output_dir, "segments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, indent=2)
    return path


def cut_segments(video_path: str, output_dir: str, segments: List[Segment], fast_cut: bool) -> List[str]:
    out_files: List[str] = []
    for seg in segments:
        out_path = os.path.join(output_dir, f"game_{seg.index:02d}.mp4")
        if fast_cut:
            cmd = ["ffmpeg", "-y", "-ss", f"{seg.start:.3f}", "-to", f"{seg.end:.3f}", "-i", video_path, "-c", "copy", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-ss", f"{seg.start:.3f}", "-to", f"{seg.end:.3f}", "-i", video_path, "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", out_path]
        code, _, err = run_cmd(cmd)
        if code == 0:
            out_files.append(out_path)
        else:
            print(f"[WARN] ffmpeg failed for game {seg.index}: {err}", file=sys.stderr)
    return out_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split One Piece TCG Sim VOD into separate games using fuzzy OCR only")
    p.add_argument("--input", required=True, help="Path to vod.mp4")
    p.add_argument("--output-dir", required=True, help="Directory for clips and reports")
    p.add_argument("--sample-seconds", type=float, default=0.25)
    p.add_argument("--timer-left", type=float, default=0.88, help="Timer crop: left edge as fraction of frame width")
    p.add_argument("--timer-top", type=float, default=0.005, help="Timer crop: top edge as fraction of frame height")
    p.add_argument("--timer-right", type=float, default=0.99, help="Timer crop: right edge as fraction of frame width")
    p.add_argument("--timer-bottom", type=float, default=0.055, help="Timer crop: bottom edge as fraction of frame height")
    p.add_argument("--timer-start-low", type=int, default=17 * 60 + 28, help="Game start: timer lower bound in seconds (default 17:28=1048)")
    p.add_argument("--timer-start-high", type=int, default=17 * 60 + 30, help="Game start: timer upper bound in seconds (default 17:30=1050)")
    p.add_argument("--timer-end-threshold", type=int, default=2, help="Game end: timer at or below this many seconds (default 2)")
    p.add_argument("--start-cluster-gap-seconds", type=float, default=30.0)
    p.add_argument("--end-cluster-gap-seconds", type=float, default=10.0)
    p.add_argument("--start-pad-seconds", type=float, default=3.0)
    p.add_argument("--end-pad-seconds", type=float, default=2.0)
    p.add_argument("--min-duration-seconds", type=float, default=120.0)
    p.add_argument("--max-duration-seconds", type=float, default=7200.0)
    p.add_argument("--tesseract-cmd", default=None)
    p.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--no-cut", action="store_true")
    p.add_argument("--fast-cut", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input VOD not found: {args.input}", file=sys.stderr)
        return 1
    if not args.no_cut and not ensure_ffmpeg():
        print("[ERROR] ffmpeg is not installed or not on PATH.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    timer_box = (args.timer_left, args.timer_top, args.timer_right, args.timer_bottom)
    print(f"[INFO] Scanning VOD timer: {args.input}")
    hits, duration = scan_video_for_chat(args.input, args.sample_seconds, timer_box, args.tesseract_cmd, args.max_workers)
    print(f"[INFO] Video duration: {sec_to_hms(duration)}")
    print(f"[INFO] OCR samples: {len(hits)}")

    start_candidates = find_timer_start_candidates(hits, low=args.timer_start_low, high=args.timer_start_high)
    end_candidates = find_timer_end_candidates(hits, threshold=args.timer_end_threshold)
    start_clusters = cluster_candidates(start_candidates, args.start_cluster_gap_seconds)
    end_clusters = cluster_candidates(end_candidates, args.end_cluster_gap_seconds)

    print(f"[INFO] Raw start candidates: {len(start_candidates)}")
    print(f"[INFO] Raw end candidates: {len(end_candidates)}")
    print(f"[INFO] Start clusters: {len(start_clusters)}")
    print(f"[INFO] End clusters: {len(end_clusters)}")
    for i, (t, txt) in enumerate(start_clusters[:10], start=1):
        print(f"  [START {i}] {sec_to_hms(t)} | {txt.replace(chr(10), ' | ')[:180]}")
    for i, (t, txt) in enumerate(end_clusters[:10], start=1):
        print(f"  [END {i}] {sec_to_hms(t)} | {txt.replace(chr(10), ' | ')[:180]}")

    segments = build_segments_from_candidates(
        start_clusters,
        end_clusters,
        duration,
        args.start_pad_seconds,
        args.end_pad_seconds,
        args.min_duration_seconds,
        args.max_duration_seconds,
    )

    print(f"[INFO] Built segments: {len(segments)}")
    for s in segments:
        print(f"  Game {s.index}: {sec_to_hms(s.start)} -> {sec_to_hms(s.end)} ({s.duration:.1f}s)")

    csv_path = write_segments_csv(args.output_dir, segments)
    json_path = write_segments_json(args.output_dir, segments)
    print(f"[INFO] Wrote: {csv_path}")
    print(f"[INFO] Wrote: {json_path}")

    if args.no_cut:
        print("[INFO] Dry run complete. No clips were cut.")
        return 0

    files = cut_segments(args.input, args.output_dir, segments, args.fast_cut)
    print(f"[INFO] Exported {len(files)} clip(s)")
    for f in files:
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
