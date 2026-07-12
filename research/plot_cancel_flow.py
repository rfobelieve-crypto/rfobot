"""Cancellation-flow monitor chart — research/eyeball tool (NOT production).

Visualises depth_deltas_1m (the per-side add/cancel stream that starts
2026-07-09) against mid price, so撤單 asymmetry can be eyeballed next to
price moves. This is the data that IS stronger than net OB Depth: it
separates撤單 from加單/fill, which the net-imbalance oscillator cannot.

DISCIPLINE: monitoring/intuition tool, not a signal. Edge is unproven
until the 2026-08-10 cancel_lead_ic verdict. Read-only; no production
import; does not touch the Telegram static/interactive charts.

Usage:
    python research/plot_cancel_flow.py                # full depth era
    python research/plot_cancel_flow.py --hours 24     # last 24h
    python research/plot_cancel_flow.py --smooth 30    # 30-min skew smoothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from shared.db import get_db_conn

BG = "#0e1116"; GREEN = "#26a269"; RED = "#e01b24"; TXT = "#e3e3e3"; SUB = "#9aa0a6"; GRID = "#2a2f38"


def load(hours: int | None) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        dd = pd.read_sql(
            "SELECT minute_start_ms ms, bid_add_qty, bid_cancel_qty, "
            "ask_add_qty, ask_cancel_qty FROM depth_deltas_1m "
            "WHERE canonical_symbol='BTC-USD' ORDER BY minute_start_ms", conn)
        ob = pd.read_sql(
            "SELECT ts_ms ms, mid_price FROM orderbook_snapshots_1m "
            "WHERE canonical_symbol='BTC-USD' ORDER BY ts_ms", conn)
    finally:
        conn.close()
    if dd.empty:
        return dd
    dd["m"] = (dd["ms"] // 60000).astype("int64")
    ob["m"] = (ob["ms"] // 60000).astype("int64")
    dd = dd.groupby("m").last()
    mid = ob.groupby("m")["mid_price"].last().astype(float)
    df = dd.join(mid.rename("mid"), how="left")
    for c in ("bid_add_qty", "bid_cancel_qty", "ask_add_qty", "ask_cancel_qty"):
        df[c] = df[c].astype(float)
    df["mid"] = df["mid"].ffill()
    df.index = pd.to_datetime(df.index * 60, unit="s")
    if hours:
        df = df.loc[df.index.max() - pd.Timedelta(hours=hours):]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=0, help="0 = full depth era")
    ap.add_argument("--smooth", type=int, default=20, help="skew rolling minutes")
    args = ap.parse_args()

    df = load(args.hours or None)
    if df.empty or len(df) < 30:
        print(f"depth_deltas too young ({len(df)} min) — collector live since 7/9")
        return 0

    tc = df["bid_cancel_qty"] + df["ask_cancel_qty"]
    # 撤單不對稱: +1 = 賣側被抽(向上真空)  -1 = 買側被抽(向下真空)
    skew = ((df["ask_cancel_qty"] - df["bid_cancel_qty"]) / tc.replace(0, np.nan))
    # 去均值:扣掉結構性小偏差(ask 天生撤多 ~+0.06),只留「相對平常的偏離」
    baseline = skew.mean()
    skew = skew - baseline
    skew_s = skew.rolling(args.smooth, min_periods=max(3, args.smooth // 3)).mean()
    # 淨抽離 = 撤 − 加(真的被抽走、非換手)
    bid_net = df["bid_cancel_qty"] - df["bid_add_qty"]
    ask_net = df["ask_cancel_qty"] - df["ask_add_qty"]
    intensity = tc.rolling(args.smooth, min_periods=3).mean()

    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(13, 10.5), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [3, 2.4, 1.8], "hspace": 0.12})
    fig.patch.set_facecolor(BG)
    span_h = (df.index.max() - df.index.min()).total_seconds() / 3600
    fig.suptitle(f"撤單流監控  BTC-USD   ·   {df.index.min():%m-%d %H:%M} → "
                 f"{df.index.max():%m-%d %H:%M}  ({span_h:.0f}h, n={len(df)})   "
                 f"·   研究工具, 非信號 (edge 待 8/10)",
                 color=TXT, fontsize=13, y=0.955)

    for ax in (a1, a2, a3):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=SUB, labelsize=9)

    # A: price + strong-skew event marks
    a1.plot(df.index, df["mid"], color="#d0d0d0", lw=1.2)
    strong = skew_s.abs() > 0.5
    up = strong & (skew_s > 0); dn = strong & (skew_s < 0)
    a1.scatter(df.index[up], df["mid"][up], s=14, color=GREEN, zorder=5,
               label="賣側被抽 (向上真空)")
    a1.scatter(df.index[dn], df["mid"][dn], s=14, color=RED, zorder=5,
               label="買側被抽 (向下真空)")
    a1.set_ylabel("Mid price", color=TXT, fontsize=10)
    a1.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=TXT, fontsize=9, loc="upper left")

    # B: cancel skew (smoothed diverging fill)
    x = df.index; z = skew_s.fillna(0).values
    a2.axhline(0, color=SUB, lw=0.8)
    a2.fill_between(x, 0, z, where=z >= 0, color=GREEN, alpha=0.55, interpolate=True)
    a2.fill_between(x, 0, z, where=z < 0, color=RED, alpha=0.55, interpolate=True)
    a2.scatter(x, skew.values, s=3, color=SUB, alpha=0.35)
    a2.set_ylim(-1.05, 1.05)
    a2.set_ylabel(f"撤單不對稱\n({args.smooth}m 平滑·去均值 {baseline:+.2f})", color=TXT, fontsize=10)
    a2.text(0.005, 0.9, "＋賣側被抽 (偏多真空)", transform=a2.transAxes, color=GREEN, fontsize=8.5)
    a2.text(0.005, 0.06, "－買側被抽 (偏空真空)", transform=a2.transAxes, color=RED, fontsize=8.5)

    # C: cancellation intensity (total pulled qty, smoothed)
    a3.fill_between(x, 0, intensity.values, color="#8ab4f8", alpha=0.5)
    a3.plot(x, intensity.values, color="#8ab4f8", lw=1.0)
    a3.set_ylabel(f"撤單強度\n(兩側總量)", color=TXT, fontsize=10)
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    out = PROJECT_ROOT / "research" / "results" / "cancel_flow_monitor.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
