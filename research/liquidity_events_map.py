# -*- coding: utf-8 -*-
"""7-day liquidity map WITH the events the strategy actually traded.

The operator wants to verify the pool positions themselves: "是不是位置都
跑掉了導致模型學錯或是學不到東西". Yesterday's map showed levels only. A
level can sit in a plausible-looking place and still be wrong, so this
version overlays what the frozen rule DID:

  · every pool live in the window (solid = resting, dashed = taken)
  · the SWEEP bar of each traded event  (the moment h[j] > h[i])
  · the FILL bar                        (retest touch, the actual entry)
  · the outcome                         (green = won, red = lost)

If the triangles do not land on visible sweeps, or the fills are nowhere
near their level, the event definition is broken and nothing downstream
can be trusted.

Plus one check a picture cannot make, run before drawing anything:

  ANCHOR CHECK — every pool price must equal some bar's actual high (buy
  side) or low (sell side) in the coin's own history. A level that matches
  no bar means the price came from somewhere it should not have: bad bar
  alignment, a stale cache, or an off-by-one in the pivot index. This is
  the cheap version of the price-alignment proof mistake.md 2026-07-28
  demands before trusting any harness that joins bars to events.

Run: python research/liquidity_events_map.py [SYMBOL] [HOURS]
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
from research.liquidity_map_check import swing_levels, first_hit  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUTDIR = ROOT / "research" / "results"
COLORS = {"swing": "#e8c547", "session": "#5aa9e6",
          "pdh_pdl": "#7ed957", "pwh_pwl": "#e06c9f"}
LABEL = {"swing": "swing 波段", "session": "session 時段",
         "pdh_pdl": "PDH/PDL 昨日", "pwh_pwl": "PWH/PWL 上週"}


def main() -> int:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 168
    # The band must follow the WINDOW's own price action, not a percentage
    # of the last close. 4% hid 59 of 72 pivots (the operator circled three
    # "missing" points, two of which were mine). 100% drew 311 pools across
    # 38k-130k and squashed the candles into a strip. Neither is a
    # verification chart. What matters is every pool price could plausibly
    # touch inside this window, so the range comes from the bars.
    margin = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    bars = SC.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    n = len(bars)
    lo_i = max(0, n - hours)
    px = bars[-1][SC.C]
    idx = {b[0]: i for i, b in enumerate(bars)}

    lv = {"swing": swing_levels(bars)}
    lv.update({k: list(v) for k, v in LT.build_levels(bars).items()})
    w_hi = max(b[SC.H] for b in bars[lo_i:])
    w_lo = min(b[SC.L] for b in bars[lo_i:])
    y_hi = w_hi * (1 + margin / 100.0)
    y_lo = w_lo * (1 - margin / 100.0)

    # ── ANCHOR CHECK, before anything is drawn ──────────────────────────
    highs = {round(b[SC.H], 8) for b in bars}
    lows = {round(b[SC.L], 8) for b in bars}
    print(f"§0.72 流動性位置驗證 — {sym}，最近 {hours} 小時\n")
    print("── 錨定檢查：每個池價必須等於歷史上某根 K 的實際高/低 ──")
    bad_total = 0
    for kind, items in lv.items():
        bad = [(e, p, s) for e, p, s in items
               if round(p, 8) not in (highs if s == 1 else lows)]
        bad_total += len(bad)
        print(f"   {LABEL[kind]:<16} {len(items):5d} 個，對不上 K 棒的 "
              f"**{len(bad)}** 個"
              + ("" if not bad else f"  例：{bad[0][1]:,.4f}"))
    if bad_total:
        print(f"\n   ⚠ 共 {bad_total} 個池價找不到對應的 K 棒 —— "
              "定義或資料錯位，先修這個再看圖")
    else:
        print("   → 全部對得上，池價沒有跑掉\n")

    # ── the events the frozen rule actually traded in this window ───────
    sw_by_lvl = defaultdict(list)
    for e in SC.detect_sweeps(bars):
        sw_by_lvl[round(float(e["level"]), 8)].append(e["j"])
    events = []
    for fill_ts, exit_ts, R, lvl, A, stopped, pierce, side in \
            SC.backtest_symbol(bars):
        fi = idx.get(fill_ts)
        if fi is None or fi < lo_i:
            continue
        cands = [j for j in sw_by_lvl.get(round(float(lvl), 8), [])
                 if j < fi and fi - j <= SC.W]
        if not cands:
            continue
        events.append({"j": max(cands), "fill": fi,
                       "exit": idx.get(exit_ts, fi), "R": R, "lvl": lvl,
                       "pierce": pierce, "side": side, "stopped": stopped})
    vb = [e for e in events if e["pierce"] <= 0.25]
    print(f"── 視窗內凍結規則實際交易的事件 ──")
    print(f"   全部 {len(events)} 筆，其中變體 B（穿越≤0.25 ATR）"
          f"**{len(vb)}** 筆")
    for e in sorted(events, key=lambda z: z["fill"]):
        d = datetime.fromtimestamp(bars[e["fill"]][0], timezone.utc)
        tag = "B" if e["pierce"] <= 0.25 else " "
        print(f"   [{tag}] {d:%m-%d %Hh}  {e['side']:<5} "
              f"價位 {e['lvl']:>12,.2f}  穿越 {e['pierce']:.3f} ATR  "
              f"開掃→成交 {e['fill']-e['j']} 根  "
              f"R {e['R']:+.3f}{'  停損' if e['stopped'] else ''}")

    # ── chart ────────────────────────────────────────────────────────────
    dts = [datetime.fromtimestamp(b[0], timezone.utc) for b in bars[lo_i:]]
    fig, ax = plt.subplots(figsize=(20, 10.6), facecolor="#11141a")
    ax.set_facecolor("#11141a")
    w = 0.012
    for d, b in zip(dts, bars[lo_i:]):
        up = b[SC.C] >= b[SC.O]
        c = "#3fb950" if up else "#f85149"
        ax.plot([d, d], [b[SC.L], b[SC.H]], color=c, lw=0.7, zorder=3)
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(d) - w / 2, min(b[SC.O], b[SC.C])),
            w, max(abs(b[SC.C] - b[SC.O]), 1e-9),
            facecolor=c, edgecolor=c, zorder=3))

    # Levels that were actually TRADED must always be drawn, whatever the
    # display band says — otherwise a sweep marker floats with no line
    # under it and the picture stops being checkable. That happened on the
    # first render: the 74,514 event sat 4.5% away, outside the band, so
    # its triangle appeared with nothing attached.
    traded_px = {round(e["lvl"], 8) for e in events}
    live = 0
    tags: list[tuple[float, str]] = []
    for kind, items in lv.items():
        for est, price, side in items:
            if est >= n:
                continue
            is_traded = round(price, 8) in traded_px
            if not is_traded and not (y_lo <= price <= y_hi):
                continue
            hit = first_hit(bars, est, price, side)
            if hit is not None and hit < lo_i:
                continue
            x0 = dts[max(0, est - lo_i)] if est >= lo_i else dts[0]
            x1 = dts[min(len(dts) - 1, hit - lo_i)] if hit is not None else dts[-1]
            if x1 <= x0:
                x1 = dts[-1]
            if hit is None:
                live += 1
            ax.hlines(price, x0, x1, colors=COLORS[kind],
                      lw=(2.2 if is_traded else (1.0 if hit is None else 0.5)),
                      linestyles="-" if hit is None else (0, (3, 3)),
                      alpha=(1.0 if is_traded
                             else (0.55 if hit is None else 0.16)),
                      zorder=4 if is_traded else 2)
            # price tag on the right edge for pools still resting — those
            # are the only ones price can still act on, and reading one off
            # the y-axis by eye is what made the first version hard to check
            if hit is None and not is_traded:
                tags.append((price, COLORS[kind]))

    # A pivot is only CONFIRMED PIVOT bars after its extreme, so its line
    # starts 10 bars to the right of the peak a reader is looking at. That
    # is correct (no look-ahead) but it reads as "the peak was not
    # recorded". Mark the extreme and connect it with a faint stub so the
    # eye can follow peak -> confirmation -> line.
    for kind, items in lv.items():
        if kind != "swing":
            continue
        for est, price, side in items:
            if est >= n or not (y_lo <= price <= y_hi):
                continue
            ext = est - SC.PIVOT
            if ext < lo_i:
                continue
            hit = first_hit(bars, est, price, side)
            if hit is not None and hit < lo_i:
                continue
            ax.plot(dts[ext - lo_i], price, marker="x", ms=5,
                    color=COLORS["swing"], alpha=0.75, zorder=5)
            ax.hlines(price, dts[ext - lo_i], dts[max(0, est - lo_i)],
                      colors=COLORS["swing"], lw=0.7, alpha=0.35,
                      linestyles=(0, (1, 2)), zorder=1)

    # establishment bar per traded level, so the segment can start there
    est_of = {}
    for kind, items in lv.items():
        for est, price, side in items:
            k = round(price, 8)
            if k in traded_px and (k not in est_of or est > est_of[k][0]):
                est_of[k] = (est, kind)

    # Right-edge price tags, thinned so they cannot collide. The first
    # version stacked five labels on top of each other above 82k — a label
    # that cannot be read is worse than no label, because it still costs
    # the reader a glance.
    span = (y_hi - y_lo) or 1.0
    min_gap = span * 0.016
    last_y = None
    for price, col in sorted(tags, reverse=True):
        if last_y is not None and abs(price - last_y) < min_gap:
            continue
        last_y = price
        ax.annotate(f"{price:,.0f}", (dts[-1], price),
                    xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=col, alpha=0.8)

    for e in events:
        jx = dts[e["j"] - lo_i] if e["j"] >= lo_i else dts[0]
        fx = dts[e["fill"] - lo_i]
        # the level itself, from where it became live to the sweep bar —
        # the line must END exactly under the triangle
        ek = est_of.get(round(e["lvl"], 8))
        if ek:
            e0 = dts[max(0, ek[0] - lo_i)] if ek[0] >= lo_i else dts[0]
            if e0 < jx:
                ax.hlines(e["lvl"], e0, jx, colors=COLORS[ek[1]],
                          lw=2.0, alpha=0.95, zorder=4)
        won = e["R"] > 0
        col = "#3fb950" if won else "#f85149"
        big = e["pierce"] <= 0.25
        # sweep marker on the level, fill marker at the entry
        ax.plot(jx, e["lvl"], marker="v" if e["side"] == "SHORT" else "^",
                ms=11 if big else 7, color="#ffd166",
                mec="#11141a", mew=0.8, zorder=6)
        ax.plot(fx, e["lvl"], marker="o", ms=8 if big else 5, color=col,
                mec="#11141a", mew=0.8, zorder=6)
        ax.plot([jx, fx], [e["lvl"], e["lvl"]], color=col,
                lw=2.6 if big else 1.2, alpha=0.6, zorder=5)
        # the two numbers that decide whether an event is even eligible and
        # what it did — reading them off the console while looking at the
        # picture was the slow part of verifying this chart
        ax.annotate(f"{'B' if big else '·'} {e['pierce']:.2f}ATR  "
                    f"R{e['R']:+.2f}",
                    (fx, e["lvl"]), xytext=(9, -11 if e["side"] == "SHORT" else 7),
                    textcoords="offset points", fontsize=7.5,
                    color=col, weight="bold" if big else "normal",
                    zorder=8,
                    bbox=dict(boxstyle="square,pad=0.22", fc="#11141a",
                              ec="none", alpha=0.82))

    ax.axhline(px, color="#c9d1d9", lw=0.8, ls=":", alpha=0.7, zorder=7)
    handles = [plt.Line2D([], [], color=COLORS[k], lw=2, label=LABEL[k])
               for k in ("swing", "session", "pdh_pdl", "pwh_pwl")]
    handles += [
        plt.Line2D([], [], marker="^", color="#ffd166", ls="", ms=10,
                   label="掃單發生（開掃棒）"),
        plt.Line2D([], [], marker="o", color="#3fb950", ls="", ms=8,
                   label="成交點 · 賺"),
        plt.Line2D([], [], marker="o", color="#f85149", ls="", ms=8,
                   label="成交點 · 賠"),
        plt.Line2D([], [], color="#c9d1d9", lw=2, label="實線 = 未掃"),
        plt.Line2D([], [], color="#c9d1d9", lw=1, ls=(0, (3, 3)),
                   label="虛線 = 已掃"),
        plt.Line2D([], [], marker="x", color="#e8c547", ls="", ms=6,
                   label="× = 擺盪極值（線在其右 10 根才確認）"),
    ]
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.09), framealpha=0.0,
                    facecolor="#11141a", edgecolor="none", fontsize=8.5,
                    ncol=4, handlelength=1.8, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color("#c9d1d9")
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(f"{sym}  流動性位置 + 凍結規則實際交易的事件  "
                 f"最近 {hours} 小時（視窗價格範圍內未掃 {live} 個，"
                 f"事件 {len(events)} 筆／變體B {len(vb)} 筆）",
                 color="#e6edf3", fontsize=13, pad=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.tick_params(colors="#8b949e", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.grid(alpha=0.12, color="#8b949e", lw=0.5)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.05, 0.965, 1))   # room for tags + legend
    out = OUTDIR / f"liquidity_events_{sym}.png"
    fig.savefig(out, dpi=125, facecolor=fig.get_facecolor())
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
