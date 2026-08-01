# -*- coding: utf-8 -*-
"""V7 績效累積圖 — 訊號 edge、濾網對照、滾動勝率、實盤真錢。

Why a cumulative view: every V7 performance surface today is a snapshot
(a win-rate number, a /perf text block). Snapshots hide the shape — when
the edge worked, when it stalled, and whether a filter would have helped
or just cut signals. The raid line has had cumulative netR curves for
weeks; V7 had none.

Panels
  1 累積方向報酬: sum of actual_return_4h signed by the signal's own
    direction, Strong vs Moderate. This is the SIGNAL edge, not trading
    P&L — no stop, no cost, no sizing. Retrains are marked, because a
    curve that spans model versions is a curve of several models.
  2 濾網對照 (T0-T3): the same cumulative curve after applying the
    terrain filters. Everything LEFT of the trigger line is the data the
    filters were derived on — in-sample by construction and drawn in a
    shaded band so it can never be read as evidence. Only the segment
    RIGHT of the line is forward.
  3 滾動勝率 30/90 筆 + 逐月柱狀，with the 50% coin line.
  4 實盤累積 net%: v7_okx_positions, the only panel with real money in
    it. Baseline resets are annotated because the account was manually
    blown up twice and re-funded; a naive equity curve across those
    events would be fiction.

Run:  python research/v7_perf_accum.py
Out:  research/results/v7_perf_accum.png
"""
from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_perf_accum.png"
BG, FG, GRID = "#0e1116", "#d7dce3", "#1c222b"
C_S, C_M, C_T1, C_T2, C_T3, C_LIVE = ("#00d1b2", "#7b6cff", "#f0b90b",
                                      "#ff9f43", "#00ffa3", "#00d1b2")
TRIGGER = "2026-08-02"          # both adoption clocks start here
BUFFER_FIX = "2026-04-19"       # warmup-buffer fix: decode changed
WALL, SUP = 1.4, 1.8

for k, v in {"figure.facecolor": BG, "axes.facecolor": BG,
             "savefig.facecolor": BG, "text.color": FG,
             "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
             "axes.edgecolor": GRID, "grid.color": GRID}.items():
    matplotlib.rcParams[k] = v
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_signals():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, strength, correct, "
                "actual_return_4h, model_version FROM tracked_signals "
                "WHERE correct IS NOT NULL AND strength IN "
                "('Strong','Moderate') ORDER BY signal_time")
            return cur.fetchall()
    finally:
        conn.close()


def load_live():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entry_time, exit_time, net_pct, equity_ret_pct, "
                "exit_reason FROM v7_okx_positions WHERE status='CLOSED' "
                "AND equity_ret_pct IS NOT NULL ORDER BY entry_time")
            return cur.fetchall()
    finally:
        conn.close()


def terrain_rows():
    """Per-signal terrain annotation. Optional: needs the pool machinery
    and a BTC 1h cache, which may be absent in a fresh image — the panel
    degrades to a note instead of failing the whole render."""
    try:
        import sweep_core as SC
        from shadow_review import ensure_bars
        from v7_price_location import pool_lifecycle
        from sweep_raid_postflow import raids_with_fill
        from collections import defaultdict
        ensure_bars("BTC")
        import level_types as LT
        bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
        ts2i = {b[0]: i for i, b in enumerate(bars)}
        atr = SC.atr14(bars)
        cl = [b[SC.C] for b in bars]
        pools = pool_lifecycle(bars)
        by_hh = defaultdict(list)
        for r in raids_with_fill("BTC"):
            by_hh[r["ts"] // 3600].append(r["side"])
        out = {}
        for ts, j in ts2i.items():
            pass
        return {"ts2i": ts2i, "atr": atr, "cl": cl, "pools": pools,
                "by_hh": by_hh}
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] terrain unavailable: {e}")
        return None


def annotate(ctx, ts, direction):
    ts2i, atr, cl, pools, by_hh = (ctx["ts2i"], ctx["atr"], ctx["cl"],
                                   ctx["pools"], ctx["by_hh"])
    j = ts2i.get(ts)
    if j is None or atr[j] in (None, 0):
        return None
    up = direction == "UP"
    c = cl[j]
    above = [p[2] for p in pools if p[0] <= j
             and (p[1] is None or p[1] > j) and p[2] > c]
    below = [p[2] for p in pools if p[0] <= j
             and (p[1] is None or p[1] > j) and p[2] < c]
    ahead = ((min(above) - c) / atr[j] if up and above else
             (c - max(below)) / atr[j] if (not up) and below else None)
    behind = ((c - max(below)) / atr[j] if up and below else
              (min(above) - c) / atr[j] if (not up) and above else None)
    ctxb = "none"
    for k in range(0, 5):
        sides = by_hh.get(ts // 3600 - k)
        if sides:
            ctxb = "fade" if ((sides[0] == 1 and not up)
                              or (sides[0] == -1 and up)) else "follow"
            break
    return ahead, behind, ctxb


def main() -> int:
    sigs = load_signals()
    live = load_live()
    for s in sigs:
        s["t"] = pd.Timestamp(s["signal_time"], tz="UTC")
        sgn = 1 if s["direction"] == "UP" else -1
        s["r"] = (100 * float(s["actual_return_4h"]) * sgn
                  if s["actual_return_4h"] is not None else 0.0)
    strong = [s for s in sigs if s["strength"] == "Strong"]
    mod = [s for s in sigs if s["strength"] == "Moderate"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("V7 績效累積 — 訊號 edge / 濾網對照 / 滾動勝率 / 實盤真錢",
                 color=FG, fontsize=14, y=0.98)
    trig = pd.Timestamp(TRIGGER, tz="UTC")

    # ① cumulative directional return
    ax = axes[0][0]
    for rows, lab, col in ((strong, "Strong", C_S), (mod, "Moderate", C_M)):
        if not rows:
            continue
        ax.plot([r["t"] for r in rows], np.cumsum([r["r"] for r in rows]),
                color=col, lw=1.6, label=f"{lab} (n={len(rows)})")
    vers = {}
    for s in sigs:
        v = s.get("model_version")
        if v and v not in vers:
            vers[v] = s["t"]
    for i, (v, t) in enumerate(sorted(vers.items(), key=lambda x: x[1])[1:]):
        ax.axvline(t, color="#8b93a1", ls=":", lw=.9)
        ax.text(t, ax.get_ylim()[1], f" 重訓", rotation=90, fontsize=7,
                color="#8b93a1", va="top")
    ax.axvline(pd.Timestamp(BUFFER_FIX, tz="UTC"), color="#ff5c5c",
               ls="--", lw=1)
    ax.text(pd.Timestamp(BUFFER_FIX, tz="UTC"), 0, " buffer 修復",
            color="#ff5c5c", fontsize=8, rotation=90, va="bottom")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_ylabel("累積方向報酬 %（未扣成本）")
    ax.set_title("① 訊號 edge：每筆 4h 報酬按自己的方向累加", fontsize=10)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.grid(alpha=.25)

    # ② filter tiers on the same curve
    ax = axes[0][1]
    ctx = terrain_rows()
    if ctx:
        keep = {"T0 全部 Strong": [], "T1 +追突破 veto": [],
                "T2 +前方有牆扣": [], "T3 +要求背後支撐": []}
        for s in strong:
            ts = int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
            a = annotate(ctx, ts, s["direction"])
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
        # Running MEAN, not cumulative sum: T3 keeps 194 of 766 signals,
        # so its cumulative sum is lower by construction and would read
        # as "the filter is worse". The mean per signal is what the
        # filter actually claims to improve.
        for (lab, rows), col in zip(keep.items(),
                                    (C_S, C_T1, C_T2, C_T3)):
            if len(rows) < 5:
                continue
            rr = np.array([r["r"] for r in rows])
            run = np.cumsum(rr) / np.arange(1, len(rr) + 1)
            ax.plot([r["t"] for r in rows], run, color=col, lw=1.5,
                    label=f"{lab} n={len(rows)} 均{rr.mean():+.3f}% "
                          f"總{rr.sum():+.0f}%")
        lo_x = min(s["t"] for s in strong)
        ax.axvspan(lo_x, trig, color="#ffffff", alpha=.045)
        ax.text(lo_x, 0, "  ← 濾網推導區（同一批資料，非證據）",
                color="#8b93a1", fontsize=8, va="bottom")
        ax.set_ylim(-0.35, 0.65)
        ax.axvline(trig, color="#00ffa3", ls="--", lw=1.2)
        ax.text(trig, ax.get_ylim()[0], " 扳機起算 →", color="#00ffa3",
                fontsize=8, va="bottom")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    else:
        ax.text(.5, .5, "地形資料不可用（缺 BTC K 線快取）",
                ha="center", va="center", color="#8b93a1", fontsize=11)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_ylabel("每筆平均方向報酬 %（滾動）")
    ax.set_title("② 濾網對照（每筆平均，非總和）：陰影區是推導資料", fontsize=10)
    ax.grid(alpha=.25)

    # ③ rolling win rate
    ax = axes[1][0]
    for rows, lab, col in ((strong, "Strong", C_S), (mod, "Moderate", C_M)):
        if len(rows) < 30:
            continue
        ser = pd.Series([int(r["correct"]) for r in rows],
                        index=[r["t"] for r in rows])
        ax.plot(ser.index, 100 * ser.rolling(30).mean(), color=col, lw=1.3,
                label=f"{lab} 30 筆")
        ax.plot(ser.index, 100 * ser.rolling(90).mean(), color=col, lw=2.2,
                alpha=.55, label=f"{lab} 90 筆")
    ax.axhline(50, color="#ff5c5c", ls="--", lw=1)
    ax.axhline(65, color="#8b93a1", ls=":", lw=1)
    ax.text(ax.get_xlim()[0], 65.5, " Strong 目標 65%", color="#8b93a1",
            fontsize=7)
    ax.set_ylabel("方向準確率 %")
    ax.set_title("③ 滾動勝率：紅線 50% 是硬幣線", fontsize=10)
    ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG, ncol=2)
    ax.grid(alpha=.25)

    # ④ live money
    ax = axes[1][1]
    if live:
        for t in live:
            t["t"] = pd.Timestamp(t["exit_time"] or t["entry_time"], tz="UTC")
        live.sort(key=lambda x: x["t"])
        # equity_ret_pct is the PERCENT column (executor writes
        # equity_ret * 100). net_pct is a FRACTION (-0.0177 = -1.77%) —
        # summing it as a percent understates the curve ~100x, which is
        # what the first render of this chart did.
        cum = np.cumsum([float(x["equity_ret_pct"]) for x in live])
        ax.plot([x["t"] for x in live], cum, color=C_LIVE, lw=1.8,
                marker="o", ms=3.5)
        ax.axhline(0, color=GRID, lw=1)
        for lab, day in (("基準 $274", "2026-07-28"),):
            d = pd.Timestamp(day, tz="UTC")
            if d >= min(x["t"] for x in live):
                ax.axvline(d, color="#ff5c5c", ls="--", lw=1)
                ax.text(d, cum.min(), f" {lab}", color="#ff5c5c", fontsize=8)
        wins = sum(1 for x in live if float(x["equity_ret_pct"]) > 0)
        ax.set_title(f"④ 實盤累積 net%（n={len(live)} 筆 · 勝率 "
                     f"{100*wins/len(live):.0f}% · 累積 {cum[-1]:+.2f}%）",
                     fontsize=10)
    else:
        ax.text(.5, .5, "尚無已平倉的實盤交易", ha="center", va="center",
                color="#8b93a1", fontsize=11)
        ax.set_title("④ 實盤累積 net%", fontsize=10)
    ax.set_ylabel("累積帳戶報酬 %（equity_ret_pct 相加）")
    ax.grid(alpha=.25)

    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.text(0.5, 0.006,
             "①③ 是訊號品質（未扣成本、未套停損、未計 sizing），不是交易績效；"
             "② 比的是每筆平均（濾網會減少筆數，總和不可比）；只有 ④ 是真錢"
             "（equity_ret_pct，已含 2x 名目）。②的陰影區為推導資料，非證據。",
             ha="center", color="#8b93a1", fontsize=8)
    fig.savefig(OUT, dpi=140)
    print(f"  Strong {len(strong)} · Moderate {len(mod)} · live {len(live)}")
    if strong:
        print(f"  Strong 累積方向報酬 {np.sum([s['r'] for s in strong]):+.1f}%"
              f" · 勝率 {100*np.mean([s['correct'] for s in strong]):.1f}%")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
