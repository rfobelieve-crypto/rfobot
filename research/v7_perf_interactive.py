# -*- coding: utf-8 -*-
"""V7 績效累積 — 互動版（Lightweight Charts），四個獨立面板。

取代原本的四宮格 PNG。操作者要求兩件事，這裡一次解決：
  1 四個圖表分開，不要擠在一起
  2 跟站上其他圖表一樣可縮放/平移/十字準星，不是靜態圖

四個面板各自是獨立的 chart 物件、各自佔滿整個寬度，時間軸互相同步
（拖動任何一個，其他跟著走）。資料與 v7_perf_accum.py 完全一致，只是
換渲染層——同一組定義不會因為換圖而漂移。

面板：
  ① 訊號 edge   累積方向報酬（Strong / Moderate），未扣成本
  ② 濾網對照   T0-T3 的「每筆平均」（濾網會減少筆數，總和不可比）
  ③ 滾動勝率   30 / 90 筆，含 50% 硬幣線
  ④ 實盤真錢   equity_ret_pct 累積（唯一有真錢的面板）

Run: python research/v7_perf_interactive.py [--out PATH]
Out: research/results/v7_perf_interactive.html
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

from v7_perf_accum import (WALL, SUP, TRIGGER, annotate, load_live,  # noqa: E402
                           load_signals, terrain_rows)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_perf_interactive.html"


def series(points):
    """[(unix_ts, value)] -> Lightweight-Charts line data, de-duplicated
    on time (the library rejects duplicate timestamps, and two signals can
    share a bar)."""
    out, seen = [], set()
    for ts, v in points:
        t = int(ts)
        while t in seen:
            t += 1          # nudge collisions, keeps ordering stable
        seen.add(t)
        out.append({"time": t, "value": round(float(v), 4)})
    return out


def rolling(vals, w):
    out = []
    acc = 0.0
    for i, v in enumerate(vals):
        acc += v
        if i >= w:
            acc -= vals[i - w]
        if i >= w - 1:
            out.append(100 * acc / w)
        else:
            out.append(None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    sigs = load_signals()
    live = load_live()
    for s in sigs:
        s["ts"] = int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        sgn = 1 if s["direction"] == "UP" else -1
        s["r"] = (100 * float(s["actual_return_4h"]) * sgn
                  if s["actual_return_4h"] is not None else 0.0)
    strong = [s for s in sigs if s["strength"] == "Strong"]
    mod = [s for s in sigs if s["strength"] == "Moderate"]

    def cum(rows):
        acc = 0.0
        pts = []
        for r in rows:
            acc += r["r"]
            pts.append((r["ts"], acc))
        return series(pts)

    def running_mean(rows):
        acc = 0.0
        pts = []
        for i, r in enumerate(rows, 1):
            acc += r["r"]
            pts.append((r["ts"], acc / i))
        return series(pts)

    # ② tiers
    ctx = terrain_rows()
    tiers: dict[str, list] = {}
    tier_meta: dict[str, dict] = {}
    if ctx:
        keep = {"T0 全部 Strong": [], "T1 +追突破 veto": [],
                "T2 +前方有牆扣": [], "T3 +要求背後支撐": []}
        for s in strong:
            a = annotate(ctx, s["ts"], s["direction"])
            if a is None:
                continue
            ahead, behind, cb = a
            keep["T0 全部 Strong"].append(s)
            if cb == "follow":
                continue
            keep["T1 +追突破 veto"].append(s)
            if ahead is not None and ahead <= WALL:
                continue
            keep["T2 +前方有牆扣"].append(s)
            if behind is None or behind > SUP:
                continue
            keep["T3 +要求背後支撐"].append(s)
        for lab, rows in keep.items():
            if len(rows) >= 5:
                tiers[lab] = running_mean(rows)
                tot = sum(r["r"] for r in rows)
                tier_meta[lab] = {"n": len(rows),
                                  "mean": round(tot / len(rows), 3),
                                  "total": round(tot, 1)}

    # ③ rolling win rate
    roll = {}
    for rows, lab in ((strong, "Strong"), (mod, "Moderate")):
        for w in (30, 90):
            vals = rolling([int(r["correct"]) for r in rows], w)
            pts = [(r["ts"], v) for r, v in zip(rows, vals) if v is not None]
            if pts:
                roll[f"{lab} {w} 筆"] = series(pts)

    # ④ live money
    for t in live:
        t["ts"] = int((t["exit_time"] or t["entry_time"])
                      .replace(tzinfo=timezone.utc).timestamp())
    live.sort(key=lambda x: x["ts"])
    acc = 0.0
    live_pts = []
    for t in live:
        acc += float(t["equity_ret_pct"])
        live_pts.append((t["ts"], acc))
    live_wr = (100 * sum(1 for t in live if float(t["equity_ret_pct"]) > 0)
               / len(live)) if live else 0

    payload = {
        "edge": {"Strong": cum(strong), "Moderate": cum(mod)},
        "edge_meta": {"Strong": len(strong), "Moderate": len(mod)},
        "tiers": tiers, "tier_meta": tier_meta,
        "roll": roll,
        "live": series(live_pts),
        "live_meta": {"n": len(live), "wr": round(live_wr),
                      "cum": round(acc, 2)},
        "trigger": TRIGGER,
    }
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"  Strong {len(strong)} · Moderate {len(mod)} · live {len(live)}"
          f" · tiers {len(tiers)}")
    print(f"  wrote {args.out}")
    return 0


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V7 績效累積</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  body { margin:0; background:#0e1116; color:#d7dce3;
         font-family:'Microsoft JhengHei','Noto Sans CJK TC',sans-serif; }
  .panel { border-bottom:1px solid #1c222b; }
  .head { display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 14px;
          padding:9px 14px 4px; }
  .t { font-size:13px; color:#fff; }
  .k { font-size:11px; color:#8b93a1; }
  .dot { display:inline-block; width:9px; height:2px; vertical-align:middle;
         margin-right:5px; }
  .note { padding:10px 14px 16px; font-size:11px; line-height:1.8;
          color:#6b7280; }
</style></head><body>
<div id="panels"></div>
<div class="note">
  ①③ 是<b>訊號品質</b>（未扣成本、未套停損、未計倉位），不是交易績效。
  ② 比的是<b>每筆平均</b>——濾網會減少筆數，總和不可比；扳機線左側是
  濾網的<b>推導資料</b>，不算證據。只有 ④ 是真錢（equity_ret_pct，已含 2x 名目）。
  四個面板時間軸同步，可縮放/平移。
</div>
<script>
const D = __DATA__;
const OPTS = {
  layout: { background: { color:'#0e1116' }, textColor:'#8b93a1' },
  grid: { vertLines:{ color:'#171d25' }, horzLines:{ color:'#171d25' } },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  timeScale: { timeVisible:false, secondsVisible:false, borderColor:'#1c222b' },
  rightPriceScale: { borderColor:'#1c222b' },
  handleScale: { axisPressedMouseMove: true },
};
const COLORS = ['#00d1b2','#7b6cff','#f0b90b','#ff9f43','#00ffa3','#ff5c5c'];
const charts = [];
let TMIN = Infinity, TMAX = -Infinity;   // union of every series' own data

function panel(title, note, seriesMap, opts) {
  opts = opts || {};
  const wrap = document.createElement('div');
  wrap.className = 'panel';
  const head = document.createElement('div');
  head.className = 'head';
  head.innerHTML = '<span class="t">' + title + '</span>';
  const box = document.createElement('div');
  wrap.appendChild(head); wrap.appendChild(box);
  document.getElementById('panels').appendChild(wrap);

  const h = Math.max(200, Math.round(window.innerHeight * 0.46));
  box.style.height = h + 'px';
  const ch = LightweightCharts.createChart(box, Object.assign({}, OPTS,
      { width: box.clientWidth || window.innerWidth, height: h }));
  let i = 0;
  for (const [name, data] of Object.entries(seriesMap)) {
    if (!data || !data.length) continue;
    const color = COLORS[i % COLORS.length];
    const s = ch.addLineSeries({ color, lineWidth: 2, priceLineVisible:false,
                                 lastValueVisible:true });
    s.setData(data);
    TMIN = Math.min(TMIN, data[0].time);
    TMAX = Math.max(TMAX, data[data.length - 1].time);
    const lbl = (opts.legend && opts.legend[name]) || name;
    head.innerHTML += '<span class="k"><i class="dot" style="background:'
                      + color + '"></i>' + lbl + '</span>';
    i++;
  }
  if (opts.zero) {
    const z = ch.addLineSeries({ color:'#2a3039', lineWidth:1,
      priceLineVisible:false, lastValueVisible:false });
    const all = Object.values(seriesMap).flat();
    if (all.length) {
      const t0 = Math.min(...all.map(p => p.time));
      const t1 = Math.max(...all.map(p => p.time));
      z.setData([{ time:t0, value:opts.zero }, { time:t1, value:opts.zero }]);
    }
  }
  if (note) head.innerHTML += '<span class="k">' + note + '</span>';
  charts.push(ch); ch.timeScale().fitContent();
  return ch;
}

const em = D.edge_meta;
panel('① 訊號 edge — 累積方向報酬 %（未扣成本）', '',
      { ['Strong n=' + em.Strong]: D.edge.Strong,
        ['Moderate n=' + em.Moderate]: D.edge.Moderate }, { zero: 0 });

const tl = {};
for (const [k, v] of Object.entries(D.tiers)) {
  const m = D.tier_meta[k];
  tl[k + ' n=' + m.n + ' 均' + (m.mean >= 0 ? '+' : '') + m.mean + '%'] = v;
}
if (Object.keys(tl).length) {
  panel('② 濾網對照 — 每筆平均方向報酬 %（非總和）',
        '扳機起算 ' + D.trigger + '（左側為推導資料）', tl, { zero: 0 });
}

panel('③ 滾動勝率 %（紅線 50% 是硬幣線）', '', D.roll, { zero: 50 });

const lm = D.live_meta;
panel('④ 實盤累積帳戶報酬 %（真錢）',
      'n=' + lm.n + ' · 勝率 ' + lm.wr + '% · 累積 '
      + (lm.cum >= 0 ? '+' : '') + lm.cum + '%',
      { ['實盤 equity_ret_pct']: D.live }, { zero: 0 });

// NOTE (2026-08-02): cross-panel time sync REMOVED after measuring it.
// Lightweight Charts dispatches range events asynchronously, so a
// `syncing` latch never covers them: every programmatic setVisibleRange
// re-triggered every other panel's subscriber, they pushed their ranges
// back at each other, and the loop settled on the narrowest common
// window — 2026-06-24..07-28, which silently hid eight months of the
// Strong series and ignored every fitContent/setVisibleRange call I made
// afterwards. Each panel now simply fits its own data. Independent
// zooming is a smaller loss than a page that lies about its range; a
// guarded, gesture-only sync can come back once it is verified rather
// than assumed.
setTimeout(() => charts.forEach(c => c.timeScale().fitContent()), 60);

window.addEventListener('resize', () => {
  const h = Math.max(200, Math.round(window.innerHeight * 0.46));
  charts.forEach(c => c.applyOptions({ width: window.innerWidth, height: h }));
  document.querySelectorAll('.panel > div:last-child')
          .forEach(d => d.style.height = h + 'px');
});
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
