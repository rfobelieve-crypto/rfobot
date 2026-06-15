/**
 * Generates `out/subtitles.srt` from src/data/subtitles.ts.
 *
 * Use:    npx tsx scripts/export-srt.ts
 * Output: industry-standard SRT consumable by YouTube / Premiere / DaVinci
 * if you ever export the video and want to switch the subtitle layer
 * from baked-in to platform-side.
 */
import {writeFileSync, mkdirSync} from 'fs';
import {SUBTITLES} from '../src/data/subtitles';
import {FPS} from '../src/data/script';

const frameToSrt = (frame: number) => {
  const totalMs = (frame / FPS) * 1000;
  const ms = Math.floor(totalMs % 1000);
  const s = Math.floor(totalMs / 1000) % 60;
  const m = Math.floor(totalMs / 60000) % 60;
  const h = Math.floor(totalMs / 3600000);
  const pad = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
};

mkdirSync('out', {recursive: true});

const lines: string[] = [];
let n = 1;
for (const c of SUBTITLES) {
  if (!c.text) continue;   // skip empty cues
  lines.push(String(n++));
  lines.push(`${frameToSrt(c.start)} --> ${frameToSrt(c.end)}`);
  lines.push(c.text);
  lines.push('');
}
writeFileSync('out/subtitles.srt', lines.join('\n'), 'utf-8');
console.log(`Wrote out/subtitles.srt (${n - 1} cues)`);
