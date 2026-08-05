# -*- coding: utf-8 -*-
"""Transcribe local video/audio files with faster-whisper (GPU).

Built for working through an external trading course: point it at the folder
of downloaded lessons and it writes one timestamped .txt per file so the
material can be read, searched and quoted precisely rather than re-watched.

Environment notes (set up 2026-07-27, worth keeping because none of it is
obvious):
  * ffmpeg.exe is NOT required. faster-whisper decodes through PyAV, which
    statically bundles its own FFmpeg. A standalone ffmpeg install was started
    and abandoned once that was verified.
  * ctranslate2 4.8.1 ships cudnn64_9.dll but not cuBLAS, so `pip install
    --no-deps nvidia-cublas-cu12` supplies cublas64_12.dll / cublasLt64_12.dll.
    --no-deps matters: a plain install can drag in a numpy that breaks
    pandas 1.4.2 (see mistake log, numpy 2 ABI break).
  * ctranslate2's __init__ only adds its OWN directory to the Windows DLL
    search path, so those two DLLs are HARD-LINKED into site-packages/
    ctranslate2/ (0 extra bytes; its existing glob("*.dll") loop picks them
    up). If ctranslate2 is ever reinstalled the links vanish and CUDA will
    fail again with "Library cublas64_12.dll is not found" — re-run:
        mklink /H <ct2_dir>\\cublas64_12.dll <nvidia\\cublas\\bin\\...>

Measured on the RTX 3070 Laptop: small ~96x realtime, medium ~66x realtime.

Usage:
    python scripts/transcribe_course.py <folder-or-file> [--model medium]
                                        [--lang en] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "外部技術借鑑" / "transcripts"

MEDIA_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts",
             ".mp3", ".m4a", ".wav", ".aac", ".opus", ".flac", ".ogg"}


def hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*")
                  if p.is_file() and p.suffix.lower() in MEDIA_EXT)


def load_model(name: str):
    """Prefer GPU, but prove it can actually encode before committing to it.

    Constructing WhisperModel(device="cuda") succeeds even when cuBLAS is
    missing — the failure only surfaces on the first encode. Probing with one
    second of silence turns that into a clean CPU fallback here, instead of a
    crash three files into a long batch.
    """
    import numpy as np
    from faster_whisper import WhisperModel

    try:
        m = WhisperModel(name, device="cuda", compute_type="float16")
        list(m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
        return m, "cuda/float16"
    except Exception as exc:
        print(f"  ! GPU unusable ({type(exc).__name__}: {str(exc)[:80]}) "
              f"— falling back to CPU (roughly 1x realtime)")
        return WhisperModel(name, device="cpu", compute_type="int8"), "cpu/int8"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="video/audio file, or a folder of them")
    ap.add_argument("--model", default="medium",
                    help="tiny|base|small|medium|large-v3 (default: medium)")
    ap.add_argument("--lang", default=None,
                    help="force a language code, e.g. en / zh (default: auto)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe files that already have output")
    args = ap.parse_args()

    target = Path(args.target).expanduser()
    if not target.exists():
        print(f"not found: {target}")
        return 1

    files = collect(target)
    if not files:
        print(f"no media files under {target}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [f for f in files
            if args.force or not (out_dir / f"{f.stem}.txt").exists()]
    print(f"{len(files)} media file(s), {len(todo)} to transcribe "
          f"-> {out_dir}\n")
    if not todo:
        print("all done already (use --force to redo)")
        return 0

    print(f"loading model '{args.model}' ...")
    model, dev = load_model(args.model)
    print(f"  device: {dev}\n")

    total_audio = total_wall = 0.0
    for i, f in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {f.name}")
        t0 = time.time()
        try:
            segments, info = model.transcribe(
                str(f), beam_size=5, language=args.lang,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            segs = [dict(start=s.start, end=s.end, text=s.text.strip())
                    for s in segments]
        except Exception:
            import traceback
            traceback.print_exc()
            print("  ! failed, skipping\n")
            continue

        wall = time.time() - t0
        dur = segs[-1]["end"] if segs else 0.0
        total_audio += dur
        total_wall += wall

        (out_dir / f"{f.stem}.txt").write_text(
            "\n".join(f"[{hms(s['start'])}] {s['text']}" for s in segs),
            encoding="utf-8")
        (out_dir / f"{f.stem}.json").write_text(
            json.dumps(dict(source=f.name, language=info.language,
                            language_probability=round(info.language_probability, 3),
                            duration=dur, model=args.model, segments=segs),
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"  {hms(dur)} audio, lang={info.language} "
              f"({info.language_probability:.0%}), {len(segs)} segments, "
              f"{wall:.0f}s wall = {dur / wall if wall else 0:.0f}x\n")

    if total_wall:
        print(f"done: {hms(total_audio)} of audio in {hms(total_wall)} "
              f"({total_audio / total_wall:.0f}x realtime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
