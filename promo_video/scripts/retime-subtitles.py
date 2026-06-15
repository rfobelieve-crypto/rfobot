"""Stretch SUBTITLES timing to match the compact scene windows in
src/data/script.ts.

Algorithm (proportional per-scene stretch):
  1. Read ORIGINAL scene windows (hard-coded — they're known: the 12.5
     min design with the s(N) seconds I used originally).
  2. Read CURRENT scene windows from script.ts (after retime-to-vo.py).
  3. For each subtitle cue, identify which original scene it lived in.
  4. Map its start/end from old scene window → new scene window
     proportionally:
        ratio  = (new_end - new_start) / (old_end - old_start)
        new_t  = new_start + (old_t - old_start) * ratio

Output: src/data/subtitles.ts rewritten in place with new frame values.

Run AFTER retime-to-vo.py.  Subtitles end up ~aligned to VO within
< 1 sec drift (since each scene is ≤ 30s).  For frame-perfect timing
use Whisper transcription instead.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_TS = ROOT / "src" / "data" / "script.ts"
SUBS_TS = ROOT / "src" / "data" / "subtitles.ts"

FPS = 30


# ── ORIGINAL scene windows (the 12.5-min design before any retime) ───

ORIGINAL_WINDOWS: list[tuple[int, float, float]] = [
    # (scene_id, start_sec, end_sec)
    (1,   0.0,  20.0),
    (2,  20.0,  50.0),
    (3,  50.0,  90.0),
    (4,  90.0, 150.0),
    (5, 150.0, 210.0),
    (6, 210.0, 270.0),
    (7, 270.0, 330.0),
    (8, 330.0, 390.0),
    (9, 390.0, 450.0),
    (10, 450.0, 510.0),
    (11, 510.0, 570.0),
    (12, 570.0, 630.0),
    (13, 630.0, 690.0),
    (14, 690.0, 750.0),
]


def parse_current_windows() -> dict[int, tuple[float, float]]:
    """Parse SCENES from script.ts to get CURRENT (post-retime) windows."""
    src = SCRIPT_TS.read_text(encoding="utf-8")
    # Pattern: id: N,\n    title: '...',\n    start: s(X), end: s(Y)
    pat = re.compile(
        r"id:\s*(\d+),\s*\n\s*title:\s*'[^']*',\s*\n\s*"
        r"start:\s*s\(([\d.]+)\)\s*,\s*end:\s*s\(([\d.]+)\)",
    )
    out: dict[int, tuple[float, float]] = {}
    for m in pat.finditer(src):
        sid, st, en = int(m.group(1)), float(m.group(2)), float(m.group(3))
        out[sid] = (st, en)
    return out


def scene_of(frame: int, windows: list[tuple[int, float, float]]) -> int | None:
    """Which scene contains this frame (in seconds)?"""
    t = frame / FPS
    for sid, st, en in windows:
        if st <= t < en:
            return sid
    return None


def remap_frame(frame: int,
                 old_w: list[tuple[int, float, float]],
                 new_w: dict[int, tuple[float, float]]) -> int:
    """Map a frame from old timeline to new timeline."""
    t = frame / FPS
    sid = scene_of(frame, old_w)
    if sid is None:
        return frame    # fallback: leave unchanged
    old_st, old_en = [(s, e) for sid_, s, e in old_w if sid_ == sid][0]
    new_st, new_en = new_w[sid]
    old_dur = old_en - old_st
    new_dur = new_en - new_st
    if old_dur <= 0:
        return int(new_st * FPS)
    ratio = new_dur / old_dur
    new_t = new_st + (t - old_st) * ratio
    return int(round(new_t * FPS))


def parse_existing_cues() -> list[tuple[int, int, str]]:
    """Extract (start_frame, end_frame, text) from current subtitles.ts.

    We use the s() helper presence to detect cue lines.
    """
    src = SUBS_TS.read_text(encoding="utf-8")
    # Pattern: {start: s(X), end: s(Y), text: '...'}
    pat = re.compile(
        r"\{\s*start:\s*s\(([\d.]+)\)\s*,\s*end:\s*s\(([\d.]+)\)\s*,\s*"
        r"text:\s*'([^']*)'\s*\}",
    )
    out: list[tuple[int, int, str]] = []
    for m in pat.finditer(src):
        start_sec = float(m.group(1))
        end_sec = float(m.group(2))
        text = m.group(3)
        out.append((int(start_sec * FPS), int(end_sec * FPS), text))
    return out


def main() -> int:
    new_windows = parse_current_windows()
    if not new_windows:
        print("ERROR: couldn't parse new windows from script.ts")
        return 1

    # Sanity print
    print("Scene window mapping (old → new):")
    for sid, old_st, old_en in ORIGINAL_WINDOWS:
        if sid in new_windows:
            new_st, new_en = new_windows[sid]
            print(f"  scene {sid:2d}: {old_st:5.1f}-{old_en:5.1f}s "
                  f"({old_en - old_st:5.1f}s) → "
                  f"{new_st:5.1f}-{new_en:5.1f}s "
                  f"({new_en - new_st:5.1f}s)")

    cues = parse_existing_cues()
    print(f"\nParsed {len(cues)} cues from subtitles.ts")

    # Build new SUBTITLES content
    lines: list[str] = [
        "/**",
        " * Subtitle cues — RETIMED to compact scene windows by",
        " * scripts/retime-subtitles.py (proportional per-scene stretch).",
        " */",
        "import {FPS} from './script';",
        "",
        "export type Cue = {",
        "  start: number;",
        "  end: number;",
        "  text: string;",
        "};",
        "",
        "const s = (sec: number) => Math.round(sec * FPS);",
        "",
        "export const SUBTITLES: Cue[] = [",
    ]

    last_scene = None
    for old_start, old_end, text in cues:
        sid = scene_of(old_start, ORIGINAL_WINDOWS)
        if sid != last_scene:
            lines.append(f"  // Scene {sid}")
            last_scene = sid

        new_start = remap_frame(old_start, ORIGINAL_WINDOWS, new_windows)
        new_end = remap_frame(old_end, ORIGINAL_WINDOWS, new_windows)
        # Convert back to seconds for s() readability
        ns_sec = new_start / FPS
        ne_sec = new_end / FPS
        # Escape any single quotes in text
        safe_text = text.replace("'", "\\'")
        if safe_text:
            lines.append(
                f"  {{start: s({ns_sec:.2f}), end: s({ne_sec:.2f}), "
                f"text: '{safe_text}'}},"
            )
        else:
            lines.append(
                f"  {{start: s({ns_sec:.2f}), end: s({ne_sec:.2f}), text: ''}},"
            )

    lines.append("];")
    SUBS_TS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {SUBS_TS.name} with {len(cues)} retimed cues")
    print("\nRe-render:")
    print("  npx remotion render PromoVideo out/promo-compact.mp4 --concurrency=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
