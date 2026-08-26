# -*- coding: utf-8 -*-
"""Calibrate the frozen pool definition against a reference indicator.

The operator supplied "Liquidity Pools (SABAI SABAI FX)" by FX365_Thailand
and asked to recalibrate. This file draws BOTH definitions on the same
window so the difference is visible rather than argued about.

THE REFERENCE INDICATOR, read from its Pine source:
  len_l = 4                       pivot left/right bars
  swing_h = ta.pivothigh(h,4,4)   confirmed 4 bars after the extreme
  LSH = valuewhen(swing_h, ...)   ONLY THE MOST RECENT swing high is kept
  pool = [LSH, LSH*thresh]        a ZONE, not a line
  raid = ta.crossover(high, LSH)  same trigger as the frozen rule

  thresh on 1h resolves to 1.001 (0.1%). The source intends a special case
  for 60/240 but writes
      (timeframe.period == "60" and timeframe.period == "240")
  which is never true — a period cannot be both — so 1h silently takes the
  final fallback. Noted because the band width matters to the comparison,
  not to criticise the author.

THE FROZEN DEFINITION (sweep_core / level_types):
  PIVOT = 10, every UNSWEPT pivot stays live, pools are lines, and three
  more pool families exist (session / PDH-PDL / PWH-PWL).

WHAT THIS FILE IS NOT: a change. Every Gate F number, the §0.58
decomposition, the §0.59 pre-registration and the 8,262-event backtest all
rest on the frozen definition. Swapping it would void that evidence, so
the frozen rule is left untouched and this only measures the gap. If the
reference turns out to be better, that is a NEW pre-registration with its
own forward sample, not an edit.

Run: python research/liquidity_calibrate.py [SYMBOL] [HOURS]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib import font_manager        # noqa: E402

for _f in ("Microsoft JhengHei", "Microsoft YaHei", "SimHei", "MS Gothic"):
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from research.liquidity_map_check import (                 # noqa: E402
    swing_levels, first_hit,
)

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUTDIR = ROOT / "research" / "results"
REF_PIVOT = 4
REF_THRESH = 1.001          # what 1h actually resolves to in the Pine source


def ref_pivots(bars, P=REF_PIVOT):
    """ta.pivothigh/pivotlow(P, P) — every pivot, with its confirm bar.

    Tie handling follows the frozen rule (>= all, > at least one) so the
    only intentional difference from flow_system here is P, not the
    comparison operator. TradingView's exact tie behaviour on flat bars is
    undocumented; on hourly crypto exact ties are rare enough not to move
    the comparison.
    """
    n = len(bars)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    out = []
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if (all(h[i] >= h[k] for k in seg)
                and any(h[i] > h[k] for k in seg if k != i)):
            out.append((i + P, h[i], 1))
        if (all(lo[i] <= lo[k] for k in seg)
                and any(lo[i] < lo[k] for k in seg if k != i)):
            out.append((i + P, lo[i], -1))
    out.sort()
    return out


def ref_latest_only(pivots, n):
    """The indicator keeps ONLY the most recent SH and SL at any time.

    Returns, per bar, (active_buy_level, active_sell_level) — which is a
    fundamentally smaller pool inventory than "every unswept pivot".
    """
    buy = [None] * n
    sell = [None] * n
    cb = cs = None
    by_bar = {}
    for est, price, side in pivots:
        by_bar.setdefault(est, []).append((price, side))
    for i in range(n):
        for price, side in by_bar.get(i, []):
            if side == 1:
                cb = price
            else:
                cs = price
        buy[i], sell[i] = cb, cs
    return buy, sell


def main() -> int:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 72
    bars = SC.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    n = len(bars)
    lo_i = max(0, n - hours)
    px = bars[-1][SC.C]
    dts = [datetime.fromtimestamp(b[0], timezone.utc) for b in bars[lo_i:]]

    froz = swing_levels(bars)                 # PIVOT=10, all
    refp = ref_pivots(bars)                   # PIVOT=4, all
    rbuy, rsell = ref_latest_only(refp, n)

    def live_in_window(items):
        out = []
        for est, price, side in items:
            if est >= n:
                continue
            hit = first_hit(bars, est, price, side)
            if hit is not None and hit < lo_i:
                continue
            out.append((est, price, side, hit))
        return out

    fz = live_in_window(froz)
    rf = live_in_window(refp)
    band = 3.0
    fzb = [x for x in fz if abs(x[1] - px) / px * 100 <= band]
    rfb = [x for x in rf if abs(x[1] - px) / px * 100 <= band]

    print(f"§0.68b 流動性定義校準 — {sym}，最近 {hours} 小時，現價 {px:,.2f}\n")
    print(f"{'':<30} {'±3% 內未掃':>12} {'全區間未掃':>12} {'視窗內總數':>12}")
    print(f"{'凍結版 swing（PIVOT=10、全部保留）':<30} "
          f"{sum(1 for x in fzb if x[3] is None):12d} "
          f"{sum(1 for x in fz if x[3] is None):12d} {len(fz):12d}")
    print(f"{'參考指標（PIVOT=4、全部保留）':<30} "
          f"{sum(1 for x in rfb if x[3] is None):12d} "
          f"{sum(1 for x in rf if x[3] is None):12d} {len(rf):12d}")
    ab, asl = rbuy[-1], rsell[-1]
    print(f"{'參考指標（PIVOT=4、只留最近一個）':<30} "
          f"{sum(1 for v in (ab, asl) if v is not None):12d} "
          f"{'（依定義最多 2 個）':>18}")

    print(f"\n  參考指標此刻的兩個池（含 ±0.1% 區間）：")
    if ab:
        print(f"    買方 {ab:,.2f} ~ {ab*REF_THRESH:,.2f}  "
              f"({100*(ab-px)/px:+.2f}%)")
    if asl:
        print(f"    賣方 {asl/REF_THRESH:,.2f} ~ {asl:,.2f}  "
              f"({100*(asl-px)/px:+.2f}%)")

    # how often do the two definitions agree on a level?
    fzset = {round(x[1], 2) for x in fz}
    rfset = {round(x[1], 2) for x in rf}
    inter = fzset & rfset
    print(f"\n  視窗內價位重合：凍結 {len(fzset)} 個、參考 {len(rfset)} 個、"
          f"**完全相同的 {len(inter)} 個**")
    print(f"  → 凍結版的每個 swing 位置，參考指標{'都找得到' if len(inter) == len(fzset) else '只找到部分'}"
          f"（PIVOT 越小池越多，大 pivot 是小 pivot 的子集）")

    # ── chart ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(21, 9), facecolor="#11141a",
                             sharey=True)
    for ax, (title, items, latest) in zip(axes, [
            (f"凍結定義  PIVOT=10 · 全部未掃保留", fzb, None),
            (f"參考指標  PIVOT=4 · 只留最近一個 · ±0.1% 區間", rfb,
             (ab, asl))]):
        ax.set_facecolor("#11141a")
        w = 0.028
        for d, b in zip(dts, bars[lo_i:]):
            up = b[SC.C] >= b[SC.O]
            c = "#3fb950" if up else "#f85149"
            ax.plot([d, d], [b[SC.L], b[SC.H]], color=c, lw=0.9, zorder=3)
            ax.add_patch(plt.Rectangle(
                (mdates.date2num(d) - w / 2, min(b[SC.O], b[SC.C])),
                w, max(abs(b[SC.C] - b[SC.O]), 1e-9),
                facecolor=c, edgecolor=c, zorder=3))
        for est, price, side, hit in items:
            x0 = dts[max(0, est - lo_i)] if est >= lo_i else dts[0]
            x1 = dts[min(len(dts) - 1, hit - lo_i)] if hit is not None else dts[-1]
            if x1 <= x0:
                x1 = dts[-1]
            col = "#ff5252" if side == 1 else "#0ef30e"
            ax.hlines(price, x0, x1, colors=col,
                      lw=1.4 if hit is None else 0.7,
                      linestyles="-" if hit is None else (0, (3, 3)),
                      alpha=0.9 if hit is None else 0.3, zorder=2)
        if latest:
            for v, side in ((latest[0], 1), (latest[1], -1)):
                if not v:
                    continue
                yl, yh = ((v, v * REF_THRESH) if side == 1
                          else (v / REF_THRESH, v))
                col = "#ff5252" if side == 1 else "#0ef30e"
                ax.axhspan(yl, yh, color=col, alpha=0.28, zorder=1)
        ax.axhline(px, color="#c9d1d9", lw=0.8, ls=":", alpha=0.7, zorder=5)
        ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
        ax.tick_params(colors="#8b949e", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.grid(alpha=0.12, color="#8b949e", lw=0.5)
    fig.suptitle(f"{sym} 流動性池定義校準 · 最近 {hours} 小時 "
                 f"（紅=買方 綠=賣方，實線未掃／虛線已掃）",
                 color="#e6edf3", fontsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = OUTDIR / f"liquidity_calibrate_{sym}.png"
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    print(f"\nwritten {out}")

    OUTDIR.joinpath("liquidity_calibrate.json").write_text(json.dumps({
        "symbol": sym, "px": px,
        "frozen_live_band": sum(1 for x in fzb if x[3] is None),
        "ref_all_live_band": sum(1 for x in rfb if x[3] is None),
        "ref_latest_only": sum(1 for v in (ab, asl) if v is not None),
        "price_overlap": len(inter), "frozen_prices": len(fzset),
        "ref_prices": len(rfset),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
