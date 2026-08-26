# -*- coding: utf-8 -*-
"""Draw every liquidity pool of the last 72h — a definition sanity check.

The operator's reasoning, and it is correct: if the pool definitions are
wrong, everything built on top of them is wrong too, and no amount of
statistics downstream will reveal it. So the definitions get looked at
BEFORE any model is built.

Every level here is read from the FROZEN code, not re-derived:
  swing     sweep_core.detect_sweeps — a PIVOT(10) extreme, confirmed only
            PIVOT bars after it forms, sweepable strictly after that
  session   level_types.build_levels — Asia 00-08 / London 07-16 /
            NY 12-21 UTC; a COMPLETED session's high and low become two
            pools at its close
  pdh_pdl   previous UTC day's high/low, live from the new day's first bar
  pwh_pwl   previous ISO week's high/low, live from the new week's first bar

A pool is drawn from the bar it becomes live until price trades through it
(solid = still resting, dashed = already taken). That is the picture the
strategy actually acts on, so it is the picture worth checking.

Deliberately NOT re-implemented here: if this file computed its own
levels it would be checking my reconstruction rather than the system's
rules, which is the failure it exists to prevent.

Run: python research/liquidity_map_check.py [SYMBOL] [HOURS]
Out: research/results/liquidity_map_<SYM>.png + a printed inventory
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib import font_manager        # noqa: E402

# CJK font or every Chinese label renders as a box — a chart drawn to be
# eyeballed is useless if its legend is unreadable.
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
import level_types as LT                                   # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUTDIR = ROOT / "research" / "results"

COLORS = {"swing": "#e8c547", "session": "#5aa9e6",
          "pdh_pdl": "#7ed957", "pwh_pwl": "#e06c9f"}
LABEL = {"swing": "swing 波段高低", "session": "session 時段高低",
         "pdh_pdl": "PDH/PDL 昨日高低", "pwh_pwl": "PWH/PWL 上週高低"}


def swing_levels(bars):
    """(established_bar, price, side) for every confirmed PIVOT extreme.

    Mirrors detect_sweeps' pivot test exactly; that function only returns
    levels that were eventually swept, so the test is repeated here to
    also surface pools still resting — the ones that matter for a live map.
    """
    n = len(bars)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    out = []
    P = SC.PIVOT
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if (all(h[i] >= h[k] for k in seg)
                and any(h[i] > h[k] for k in seg if k != i)):
            out.append((i + P, h[i], 1))       # live only after confirmation
        if (all(lo[i] <= lo[k] for k in seg)
                and any(lo[i] < lo[k] for k in seg if k != i)):
            out.append((i + P, lo[i], -1))
    return out


def first_hit(bars, start_i, price, side):
    """Bar index where price first trades through the pool, else None."""
    for j in range(start_i + 1, len(bars)):
        if side == 1 and bars[j][SC.H] > price:
            return j
        if side == -1 and bars[j][SC.L] < price:
            return j
    return None


def main() -> int:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 72
    fp = CACHE / f"{sym}USDT_1h.csv"
    if not fp.exists():
        raise SystemExit(f"no cache for {sym}: {fp}")
    bars = SC.load_csv(str(fp))

    lv = {"swing": swing_levels(bars)}
    lv.update({k: list(v) for k, v in LT.build_levels(bars).items()})

    n = len(bars)
    lo_i = max(0, n - hours)
    t0, t1 = bars[lo_i][0], bars[-1][0]
    px_now = bars[-1][SC.C]
    atr = SC.atr(bars)[-1] if hasattr(SC, "atr") else None

    # A pool established at ANY point in history and never traded through is
    # still resting, so the raw count runs into the hundreds and says more
    # about how long the cache is than about what price can reach. The band
    # below is a DISPLAY filter only — no strategy rule is changed by it —
    # and the full count is printed alongside so the filtering is visible.
    band = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0     # percent
    shown, wide = defaultdict(list), defaultdict(list)
    for kind, items in lv.items():
        for est_i, price, side in items:
            if est_i >= n:
                continue
            hit = first_hit(bars, est_i, price, side)
            if hit is not None and hit < lo_i:
                continue                       # taken before the window
            wide[kind].append((est_i, price, side, hit))
            if abs(price - px_now) / px_now * 100 <= band:
                shown[kind].append((est_i, price, side, hit))

    print(f"§0.68 流動性位置目視檢查 — {sym}，最近 {hours} 小時")
    print(f"  視窗 {datetime.fromtimestamp(t0, timezone.utc):%Y-%m-%d %H:%M}"
          f" → {datetime.fromtimestamp(t1, timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"  現價 {px_now:,.2f}" + (f"｜ATR {atr:,.2f}" if atr else "") + "\n")

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#11141a")
    ax.set_facecolor("#11141a")
    dts = [datetime.fromtimestamp(b[0], timezone.utc) for b in bars[lo_i:]]
    w = 0.028
    for d, b in zip(dts, bars[lo_i:]):
        up = b[SC.C] >= b[SC.O]
        c = "#3fb950" if up else "#f85149"
        ax.plot([d, d], [b[SC.L], b[SC.H]], color=c, lw=0.9, zorder=3)
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(d) - w / 2, min(b[SC.O], b[SC.C])),
            w, max(abs(b[SC.C] - b[SC.O]), 1e-9),
            facecolor=c, edgecolor=c, zorder=3))

    total_live = 0
    print(f"  {'池種':<22} {'±' + str(band) + '% 內未掃':>12} {'已掃':>6} "
          f"{'全價格區間未掃':>14}")
    for kind in ("pwh_pwl", "pdh_pdl", "session", "swing"):
        items = shown.get(kind, [])
        live = [x for x in items if x[3] is None]
        taken = [x for x in items if x[3] is not None]
        allive = [x for x in wide.get(kind, []) if x[3] is None]
        total_live += len(live)
        print(f"  {LABEL[kind]:<22} {len(live):12d} {len(taken):6d} "
              f"{len(allive):14d}")
        for est_i, price, side, hit in items:
            x0 = dts[max(0, est_i - lo_i)] if est_i >= lo_i else dts[0]
            x1 = dts[min(len(dts) - 1, hit - lo_i)] if hit is not None else dts[-1]
            if x1 <= x0:
                x1 = dts[-1]
            ax.hlines(price, x0, x1, colors=COLORS[kind],
                      lw=1.5 if hit is None else 0.8,
                      linestyles="-" if hit is None else (0, (3, 3)),
                      alpha=0.95 if hit is None else 0.35, zorder=2)
            if hit is None:
                ax.plot(x1, price, marker="<", ms=5,
                        color=COLORS[kind], zorder=4)

    ax.axhline(px_now, color="#c9d1d9", lw=0.8, ls=":", alpha=0.7, zorder=5)
    ax.text(dts[0], px_now, f" 現價 {px_now:,.0f}", color="#c9d1d9",
            fontsize=9, va="bottom")

    handles = [plt.Line2D([], [], color=COLORS[k], lw=2, label=LABEL[k])
               for k in ("swing", "session", "pdh_pdl", "pwh_pwl")]
    handles += [plt.Line2D([], [], color="#c9d1d9", lw=2, label="實線 = 未掃（still resting）"),
                plt.Line2D([], [], color="#c9d1d9", lw=1, ls=(0, (3, 3)),
                           label="虛線 = 已被掃過")]
    leg = ax.legend(handles=handles, loc="upper left", framealpha=0.25,
                    facecolor="#11141a", edgecolor="#30363d", fontsize=9)
    for t in leg.get_texts():
        t.set_color("#c9d1d9")

    ax.set_title(f"{sym}  流動性位置（凍結定義）  最近 {hours} 小時"
                 f"   ±{band}% 內未掃池 {total_live} 個",
                 color="#e6edf3", fontsize=13, pad=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.tick_params(colors="#8b949e", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.grid(alpha=0.12, color="#8b949e", lw=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = OUTDIR / f"liquidity_map_{sym}.png"
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"\n  未掃池總數 {total_live}")
    print(f"\nwritten {out}")

    # nearest resting pools — the ones a sweep could actually reach
    print("\n  離現價最近的未掃池：")
    near = sorted(((abs(p - px_now), kind, p, side)
                   for kind, items in wide.items()
                   for _e, p, side, hit in items if hit is None))[:12]
    for d, kind, p, side in near:
        rel = 100 * (p - px_now) / px_now
        print(f"    {LABEL[kind]:<22} {p:>12,.2f}  ({rel:+.2f}%)  "
              f"{'上方 buy-side' if side == 1 else '下方 sell-side'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
