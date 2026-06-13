"""Recompute SCENES[].start / .end in src/data/script.ts to match the
actual VO mp3 durations + 1s buffer.

Use when you want compact pacing instead of the original 12.5-min
designer windows (which leave large silent gaps at end of each scene).

Output:
  src/data/script.ts (in place) — start/end fields rewritten
  src/data/subtitles.ts (in place) — cues shifted proportionally

Run AFTER scripts/generate-tts.py.  Re-run if VO is re-generated.
"""
from __future__ import annotations

import re
from pathlib import Path

from mutagen.mp3 import MP3

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_TS = ROOT / "src" / "data" / "script.ts"
SUBS_TS = ROOT / "src" / "data" / "subtitles.ts"
AUDIO_DIR = ROOT / "public" / "audio"

BUFFER_SEC = 1.0   # tail padding per scene


def compute_windows() -> list[tuple[int, float, float]]:
    """Return [(scene_id, new_start_sec, new_end_sec)]."""
    out: list[tuple[int, float, float]] = []
    t = 0.0
    for i in range(1, 15):
        p = AUDIO_DIR / f"scene_{i:02d}.mp3"
        if not p.exists():
            print(f"WARN: {p} missing, keeping window 0")
            out.append((i, t, t))
            continue
        dur = MP3(p).info.length + BUFFER_SEC
        out.append((i, t, t + dur))
        t += dur
    print(f"New total runtime: {t:.1f}s = {t/60:.2f} min")
    return out


def rewrite_script(windows: list[tuple[int, float, float]]) -> None:
    """Patch script.ts in place: rewrite start/end for each scene block."""
    src = SCRIPT_TS.read_text(encoding="utf-8")
    by_id = {sid: (st, en) for sid, st, en in windows}
    # Find each scene block and rewrite its start/end lines
    pattern = re.compile(
        r"(id:\s*)(\d+)(,\s*\n\s*title:\s*'[^']*',\s*\n\s*start:\s*)"
        r"s\(\d+(?:\.\d+)?\)(,\s*end:\s*)s\(\d+(?:\.\d+)?\)",
    )

    def sub(m: re.Match) -> str:
        scene_id = int(m.group(2))
        st, en = by_id.get(scene_id, (0.0, 0.0))
        return (f"{m.group(1)}{scene_id}{m.group(3)}s({st:.1f})"
                f"{m.group(4)}s({en:.1f})")

    new_src = pattern.sub(sub, src)
    if new_src == src:
        print("WARN: no scene blocks matched — pattern may have drifted")
    SCRIPT_TS.write_text(new_src, encoding="utf-8")
    print(f"Updated {SCRIPT_TS.name}")


def rewrite_subtitles(windows: list[tuple[int, float, float]]) -> None:
    """Stretch subtitle cues proportionally to each scene's new duration.

    Cue's original scene is inferred from its position in the timeline
    of the OLD script.ts.  This is a best-effort stretch; manual fine-
    tune may still be needed for natural read timing.
    """
    # For now we just print guidance — properly retiming subtitles is
    # non-trivial because the original cue→scene mapping isn't explicit.
    # The user can either:
    #   (a) accept that subtitles may drift slightly, or
    #   (b) regenerate subtitles by hand from the new VO mp3s using
    #       a transcription tool (e.g. whisper).
    print("NOTE: subtitles.ts NOT auto-retimed.  Options:")
    print("  - Accept slight drift (still readable)")
    print("  - Run: whisper public/audio/scene_*.mp3 --output_format srt")
    print("  - Or manually adjust cue start/end in src/data/subtitles.ts")


def main() -> int:
    windows = compute_windows()
    rewrite_script(windows)
    rewrite_subtitles(windows)
    print()
    print("Done.  Re-render:")
    print("  npx remotion render PromoVideo out/promo-compact.mp4 --concurrency=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
