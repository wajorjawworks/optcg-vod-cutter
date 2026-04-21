# optcg-vod-cutter

Automatically splits One Piece TCG Simulator VOD recordings into individual game clips using chat OCR — no manual timestamps needed.

## How it works

Scans the chat panel of the VOD frame-by-frame using Tesseract OCR. Game starts are detected when multiple cue phrases appear together (`has connected`, `leader is`, `chose to go`, etc.). Game ends are detected from `concedes` or `opponent has disconnected`. Detected boundaries are clustered and paired, then ffmpeg cuts the clips.

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on PATH (or pass `--tesseract-cmd`)
- ffmpeg on PATH

```
pip install -r requirements.txt
```

## Usage

```bash
python vod_cutter.py --input vod.mp4 --output-dir clips/
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to the VOD file |
| `--output-dir` | required | Where to write clips and reports |
| `--sample-seconds` | 0.25 | How often to sample a frame for OCR |
| `--chat-left/top/right/bottom` | 0.054 / 0.262 / 0.270 / 0.636 | Chat panel crop region (0–1 = fraction of frame) |
| `--start-pad-seconds` | 6.0 | Extra seconds before detected game start |
| `--end-pad-seconds` | 2.0 | Extra seconds after detected game end |
| `--min-duration-seconds` | 120.0 | Discard segments shorter than this |
| `--max-duration-seconds` | 7200.0 | Discard segments longer than this |
| `--no-cut` | off | Dry run: detect only, don't run ffmpeg |
| `--fast-cut` | off | Use stream copy instead of re-encode (faster, less precise) |
| `--tesseract-cmd` | auto | Full path to tesseract executable if not on PATH |
| `--max-workers` | cpu_count-1 | Parallel OCR workers |

### Outputs

- `segments.csv` — detected game boundaries with timestamps
- `segments.json` — same data in JSON format
- `game_01.mp4`, `game_02.mp4`, … — the cut clips

### Adjusting the chat crop

The default crop is tuned for the One Piece TCG Simulator layout at 1080p. If your layout differs, pass `--chat-left`, `--chat-top`, `--chat-right`, `--chat-bottom` as fractions of the frame width/height (0.0–1.0).

Use `--no-cut` first to iterate quickly on detection without cutting.
