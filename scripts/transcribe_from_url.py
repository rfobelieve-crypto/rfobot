# -*- coding: utf-8 -*-
"""Transcribe remote lesson videos: fetch -> whisper -> delete the media.

Companion to transcribe_course.py (which works on local files). Here the media
is a means to an end: each file is pulled to a scratch directory, transcribed,
and deleted in a finally block, so only text survives. Nothing lands in the
project tree except the transcript.

Two environment details that cost time to discover (2026-07-28):
  * YouTube's default web client returns "The page needs to be reloaded" for
    these videos; player_client=android (or ios) extracts cleanly. Only format
    18 (360p mp4, AAC 44kHz) is offered there — no audio-only stream — which
    is fine, whisper downsamples to 16kHz mono anyway.
  * --cookies-from-browser chrome fails while Chrome is running (the cookie DB
    is locked). Not needed for unlisted videos, so it is not attempted.

Output goes to 外部技術借鑑/transcripts/, which .gitignore already excludes —
this is someone else's paid course and must not enter version control.

Usage:
    python scripts/transcribe_from_url.py --list lessons.json
    python scripts/transcribe_from_url.py --url https://youtu.be/XXXX --name "Part.1"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "外部技術借鑑" / "transcripts"

# The course is Traditional Chinese (Taiwan). Whisper's zh decoder drifts to
# Simplified without a hint, and trading jargon ("掛單", "吃單", "掃損") is far
# outside its usual prior, so the prompt seeds both the script and the domain.
INITIAL_PROMPT = (
    "以下是繁體中文的交易教學內容，主題包含訂單流、掛單簿、成交量、"
    "止損、假突破、流動性與市場結構分析。"
)


def hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fetch(url: str, dest_dir: Path) -> Path | None:
    """Pull the smallest stream that carries usable audio. Returns the path."""
    import yt_dlp

    opts = {
        "format": "18/bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
        # YouTube throttles these long videos to a standstill partway through
        # (observed: dead stop at 78 of 92 MB, zero bytes for minutes). A stalled
        # socket never errors, so retries alone do not help — throttled_rate is
        # what forces yt-dlp to re-extract a fresh URL when throughput collapses.
        "throttledratelimit": 51200,      # < 50 KB/s => re-extract
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="JSON file: [{name, url}, ...]")
    ap.add_argument("--url")
    ap.add_argument("--name")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        items = json.loads(Path(args.list).read_text(encoding="utf-8"))
    elif args.url:
        items = [{"name": args.name or "lesson", "url": args.url}]
    else:
        print("need --list or --url")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items
            if args.force or not (OUT_DIR / f"{it['name']}.txt").exists()]
    print(f"{len(items)} lesson(s), {len(todo)} to do -> {OUT_DIR}\n", flush=True)
    if not todo:
        print("all done")
        return 0

    from faster_whisper import WhisperModel
    import numpy as np
    print(f"loading whisper '{args.model}' on GPU ...", flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    list(model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
    print("  ready\n", flush=True)

    scratch = Path(tempfile.mkdtemp(prefix="lesson_"))
    total_audio = total_wall = 0.0
    try:
        for i, it in enumerate(todo, 1):
            name, url = it["name"], it["url"]
            print(f"[{i}/{len(todo)}] {name}", flush=True)
            media = None
            t0 = time.time()
            try:
                media = fetch(url, scratch)
                mb = media.stat().st_size / 1048576
                print(f"  fetched {mb:.0f} MB in {time.time() - t0:.0f}s",
                      flush=True)

                t1 = time.time()
                segments, info = model.transcribe(
                    str(media), language=args.lang, beam_size=5,
                    initial_prompt=INITIAL_PROMPT, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                segs = [dict(start=s.start, end=s.end, text=s.text.strip())
                        for s in segments]
                wall = time.time() - t1
                dur = segs[-1]["end"] if segs else 0.0
                total_audio += dur
                total_wall += wall

                (OUT_DIR / f"{name}.txt").write_text(
                    "\n".join(f"[{hms(s['start'])}] {s['text']}" for s in segs),
                    encoding="utf-8")
                (OUT_DIR / f"{name}.json").write_text(
                    json.dumps(dict(name=name, source=url, model=args.model,
                                    language=info.language, duration=dur,
                                    segments=segs),
                               ensure_ascii=False, indent=1),
                    encoding="utf-8")
                print(f"  {hms(dur)} audio -> {len(segs)} segments in "
                      f"{wall:.0f}s ({dur / wall if wall else 0:.0f}x)",
                      flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                print("  ! failed, continuing", flush=True)
            finally:
                # The media is scratch, not an artefact. Delete it whether the
                # transcription succeeded, failed, or raised halfway.
                if media is not None and media.exists():
                    media.unlink()
                    print("  media deleted", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if total_wall:
        print(f"\ndone: {hms(total_audio)} of audio in {hms(total_wall)} "
              f"({total_audio / total_wall:.0f}x realtime)")
    print(f"transcripts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
