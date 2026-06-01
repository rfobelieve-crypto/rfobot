# V7 Promo Video (Remotion)

12.5-minute promotional video for the V7 BTC quant trading system,
generated programmatically with Remotion 4.  All voiceover, subtitles,
and visuals are declared as TypeScript / React — `npx remotion render`
emits MP4.

```
promo_video/
├── package.json            # Remotion 4 + React 18
├── remotion.config.ts      # codec / quality settings (h264 / CRF 18)
├── tsconfig.json
├── src/
│   ├── index.ts            # registerRoot
│   ├── Root.tsx            # <Composition> declaration
│   ├── Composition.tsx     # sequences 14 scenes + global subtitle layer
│   ├── data/
│   │   ├── script.ts       # 14 scenes: timing + VO Chinese + visual hint
│   │   └── subtitles.ts    # ~80 cues matched to VO
│   ├── components/
│   │   ├── Subtitles.tsx   # bottom-anchored Chinese subtitle layer
│   │   └── Primitives.tsx  # SceneTitle / CountUp / GradientBG / AccentLine
│   └── scenes/
│       ├── Scene01ColdOpen.tsx        # full impl (template)
│       ├── Scene02Hook.tsx            # full impl
│       ├── Scene03Problem.tsx         # full impl
│       ├── Scene04Architecture.tsx    # full impl
│       └── SceneStub.tsx              # placeholder for 5-14
├── scripts/
│   └── export-srt.ts       # data/subtitles.ts → out/subtitles.srt
├── public/audio/           # drop scene_NN.mp3 voiceovers here
└── out/                    # render output (gitignored)
```

## One-time setup

```bash
cd promo_video
npm install
```

## Iteration loop (no audio yet, just visuals)

```bash
npx remotion studio       # opens http://localhost:3000
# Scrub through frames, tweak scenes, hot-reload edits.
```

The 4 scenes already implemented (1, 2, 3, 4) render cleanly.
Scenes 5-14 show the `SceneStub` placeholder — read each scene's
`visual_hint` in `src/data/script.ts` and clone the Scene01 template.

## Voiceover production (Chinese TTS)

The voiceover text lives in `src/data/script.ts` per scene.  Generate
MP3s with any neural TTS service that handles Mandarin well:

### Option A: ElevenLabs (paid, highest quality)
```bash
# Sign up → API key → pick a Mandarin voice
# Loop scenes, POST text to /v1/text-to-speech/<voice_id>
# Save outputs as public/audio/scene_01.mp3 … scene_14.mp3
```

### Option B: Azure Speech (cheap, neural)
```bash
# https://speech.microsoft.com → zh-CN-XiaoxiaoNeural recommended
# Or zh-TW-HsiaoChenNeural for Taiwanese tone
```

### Option C: Local OS TTS (free, lower quality, fine for draft)
```bash
# macOS: say -v Mei-Jia -o public/audio/scene_01.aiff "..."
# Then convert: ffmpeg -i scene_01.aiff scene_01.mp3
```

Quick helper script idea (not committed; write it once you pick a TTS):
```ts
// scripts/generate-tts.ts
import {SCENES} from '../src/data/script';
for (const s of SCENES) {
  // call your TTS API with s.vo_zh
  // write result to public/audio/scene_NN.mp3
}
```

## Final render

```bash
# 1080p (Twitter / YouTube SD)
npm run build              # → out/promo.mp4

# 4K (YouTube hero / landing page)
npm run build-4k           # → out/promo-4k.mp4

# Single frame preview
npm run still              # → out/preview.png
```

## SRT export (for YouTube auto-captions or external editors)

```bash
npx tsx scripts/export-srt.ts  # → out/subtitles.srt
```

The video itself bakes subtitles in via `<Subtitles>` component.  Export
SRT only if you want to upload them separately (e.g. YouTube CC track).

## Editing checklist before final render

| Item | Where |
|---|---|
| Adjust scene length | `src/data/script.ts` → SCENES[].start / .end |
| Tweak VO wording | `src/data/script.ts` → SCENES[].vo_zh |
| Re-time a subtitle cue | `src/data/subtitles.ts` |
| Change scene visuals | `src/scenes/SceneNN_*.tsx` |
| Adjust subtitle font / size | `src/components/Subtitles.tsx` |
| Codec / quality | `remotion.config.ts` |

## Known limitations

1. Scenes 5-14 are stubs.  Total runtime is correct; only the visuals
   for those 10 scenes need building out (~30 min each at this style).
2. Mock screenshots of `/okx-perf` / `/paper-perf` need to be added
   to `public/screenshots/` and rendered via `<Img>` once you grab
   them from the live system.
3. Background music isn't wired — add `public/audio/bgm.mp3` and
   include it as a separate `<Audio>` at the Composition root with
   `volume={0.15}` to layer under VO.

## Honest scope check

This scaffolds the *engine*.  Producing a truly polished 12-minute
promo video is still substantial work:
- Each remaining scene: ~30 min to design + implement well
- Voiceover production: ~1 hour (TTS + listen + retake)
- Final pass: BGM + transitions + colour grading

Realistic timeline for one person: **1-2 weekends** to ship a
broadcast-quality version.  This scaffold takes you from zero to
"4 scenes already render + framework for the rest".
