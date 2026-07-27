"""Cancellation-flow monitor chart — research/eyeball tool (NOT production).

Visualises depth_deltas_1m (the per-side add/cancel stream that starts
2026-07-09) against mid price, so撤單 asymmetry can be eyeballed next to
price moves. This is the data that IS stronger than net OB Depth: it
separates撤單 from加單/fill, which the net-imbalance oscillator cannot.

DISCIPLINE: monitoring/intuition tool, not a signal. Edge is unproven
until the 2026-08-10 cancel_lead_ic verdict. Read-only; no production
import; does not touch the Telegram static/interactive charts.

Usage:
    python research/plot_cancel_flow.py                # last 24h (default)
    python research/plot_cancel_flow.py --hours 0      # full depth era (debug)
    python research/plot_cancel_flow.py --smooth 60    # heavier skew smoothing
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

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei",
                                   "Noto Sans CJK TC", "Noto Sans CJK SC",
                                   "WenQuanYi Zen Hei"]  # MS = local Win; Noto = Railway image
plt.rcParams["axes.unicode_minus"] = False

from shared.db import get_db_conn

BG = "#0e1116"; GREEN = "#26a269"; RED = "#e01b24"; TXT = "#e3e3e3"; SUB = "#9aa0a6"; GRID = "#2a2f38"


def _q(conn, sql: str, params=None) -> pd.DataFrame:
    """DB → DataFrame without pd.read_sql: its handling of DictCursor rows
    differs across pandas versions (the container's newer pandas turned the
    column ALIAS into row values). dict rows via pd.DataFrame() are stable."""
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        rows = cur.fetchall() or []
    return pd.DataFrame(rows)


def load(hours: int | None) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        dd = _q(conn,
            "SELECT minute_start_ms ms, bid_add_qty, bid_cancel_qty, "
            "ask_add_qty, ask_cancel_qty FROM depth_deltas_1m "
            "WHERE canonical_symbol='BTC-USD' AND exchange='binance' "
            "ORDER BY minute_start_ms")
        ob = _q(conn,
            "SELECT ts_ms ms, mid_price FROM orderbook_snapshots_1m "
            "WHERE canonical_symbol='BTC-USD' ORDER BY ts_ms")
    finally:
        conn.close()
    if dd.empty:
        return dd
    # Newer pandas (Railway image) reads DB numerics as arrow-backed str;
    # local Windows pandas reads them as ints. Coerce so // works on both.
    dd["ms"] = pd.to_numeric(dd["ms"])
    ob["ms"] = pd.to_numeric(ob["ms"])
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


def _overlay_signals(ax, df: pd.DataFrame) -> None:
    """Overlay v7 Strong signals on the price panel — read-only, best-effort."""
    try:
        conn = get_db_conn()
        try:
            sig = _q(conn,
                "SELECT signal_time, direction FROM tracked_signals "
                "WHERE strength='Strong' AND direction IN ('UP','DOWN') "
                "AND signal_time >= %s AND signal_time <= %s",
                params=(str(df.index.min()), str(df.index.max())))
        finally:
            conn.close()
        if sig.empty:
            return
        sig["signal_time"] = pd.to_datetime(sig["signal_time"])
        px = df["mid"].reindex(
            df.index[df.index.searchsorted(sig["signal_time"]).clip(0, len(df) - 1)]).values
        up = sig["direction"].values == "UP"
        ax.scatter(sig["signal_time"][up], px[up], marker="^", s=70,
                   color=GREEN, edgecolor="white", lw=0.6, zorder=6)
        ax.scatter(sig["signal_time"][~up], px[~up], marker="v", s=70,
                   color=RED, edgecolor="white", lw=0.6, zorder=6)
    except Exception as e:
        print(f"(signal overlay skipped: {e})")


# Machine-detected playbook events (cancel_playbook_watcher) — the events the
# WATCHER pushed on its own, as opposed to _overlay_signals' v7 Strong entries.
# Marker = playbook, fill = direction, white ring = actually alerted to
# Telegram (vs silently logged). No-direction playbooks (gate_only /
# two_sided) are deliberately NOT drawn: ~50% of rows, no directional claim,
# they'd just bury the readable ones. Read-only + best-effort, same as above:
# a failure here must never take the chart down.
_PB_MARKER = {"true_break": "^", "absorption": "D", "vacuum": "*",
              "vacuum_lead": "P"}


def _overlay_playbooks(ax, df: pd.DataFrame) -> None:
    try:
        conn = get_db_conn()
        try:
            ev = _q(conn,
                "SELECT minute_start_ms ms, playbook, direction, alerted, "
                "def_version FROM cancel_playbook_events "
                "WHERE direction IN ('UP','DOWN') "
                "AND minute_start_ms BETWEEN %s AND %s",
                params=(int(df.index.min().timestamp() * 1000),
                        int(df.index.max().timestamp() * 1000)))
        finally:
            conn.close()
        if ev.empty:
            return
        ev["ts"] = pd.to_datetime(pd.to_numeric(ev["ms"]) // 1000, unit="s")
        ev["alerted"] = pd.to_numeric(ev["alerted"]).fillna(0).astype(int)
        px = df["mid"].reindex(
            df.index[df.index.searchsorted(ev["ts"]).clip(0, len(df) - 1)]).values
        for pb, mk in _PB_MARKER.items():
            for alerted in (0, 1):
                m = ((ev["playbook"] == pb) & (ev["alerted"] == alerted)).values
                if not m.any():
                    continue
                up = (ev["direction"].values == "UP") & m
                dn = (ev["direction"].values == "DOWN") & m
                # alerted → white ring + opaque; silent → no edge, faded
                kw = (dict(edgecolor="white", lw=1.1, alpha=0.95, s=64)
                      if alerted else dict(lw=0, alpha=0.45, s=40))
                if up.any():
                    ax.scatter(ev["ts"][up], px[up], marker=mk,
                               color=GREEN, zorder=5, **kw)
                if dn.any():
                    ax.scatter(ev["ts"][dn], px[dn], marker=mk,
                               color=RED, zorder=5, **kw)
    except Exception as e:
        print(f"(playbook overlay skipped: {e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    # 24h window / 15m smooth: info half-life ≤60m means only the right edge
    # is actionable; the rest of the 24h is baseline + review context. Longer
    # windows compress 1m spikes below pixel resolution (0 = full era, debug).
    ap.add_argument("--hours", type=int, default=24, help="0 = full depth era")
    ap.add_argument("--smooth", type=int, default=15, help="skew rolling minutes")
    ap.add_argument("--candle", type=int, default=0,
                    help="K線分鐘數 (0 = 自動: ≤30h→15m, ≤80h→30m, 其餘→60m)")
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

    x = df.index
    z = skew_s.fillna(0).values
    EP = 0.30                                    # 顯著不對稱門檻(顯示用)
    # 用連續色帶標「該時段哪一側被抽」,跨三格對齊(取代雜亂散點)
    def bands(vals, cond, color):
        m = cond
        i = 0; n = len(m)
        while i < n:
            if m[i]:
                j = i
                while j + 1 < n and m[j + 1]:
                    j += 1
                if j - i >= max(3, args.smooth // 4):   # 只畫夠長的時段
                    for ax in (a1, a2, a3):
                        ax.axvspan(x[i], x[j], color=color, alpha=0.10, zorder=0)
                i = j + 1
            else:
                i += 1
    bands(z, z >= EP, GREEN)
    bands(z, z <= -EP, RED)

    # A: 價格 K 線(1m mid 重採樣 OHLC — 無逐筆成交價,mid 是最誠實的近似)
    c_min = args.candle if args.candle > 0 else (
        15 if span_h <= 30 else 30 if span_h <= 80 else 60)
    ohlc = df["mid"].resample(f"{c_min}min").agg(
        ["first", "max", "min", "last"]).dropna()
    cx = ohlc.index + pd.Timedelta(minutes=c_min / 2)   # bar 中心
    body_w = (c_min * 0.7) / 1440.0                     # datetime 軸: 天為單位
    eps = float(df["mid"].max()) * 2e-5                 # 十字星最小可見實體
    rising = (ohlc["last"] >= ohlc["first"]).values
    for m, col in ((rising, GREEN), (~rising, RED)):
        seg = ohlc[m]
        if seg.empty:
            continue
        cc = cx[m]
        a1.vlines(cc, seg["min"], seg["max"], color=col, lw=0.9, zorder=3)
        bot = np.minimum(seg["first"].values, seg["last"].values)
        h = np.maximum(np.abs(seg["last"].values - seg["first"].values), eps)
        a1.bar(cc, h, bottom=bot, width=body_w, color=col,
               edgecolor=col, lw=0.4, zorder=4)
    a1.set_ylabel(f"價格 ({c_min}m K · mid)", color=TXT, fontsize=10)
    _overlay_playbooks(a1, df)   # 機器自推劇本(先畫,壓在 v7 訊號底下)
    _overlay_signals(a1, df)
    a1.text(0.006, 0.93, "▲UP ▼DOWN = v7 Strong 信號   綠帶=賣側被抽 紅帶=買側被抽",
            transform=a1.transAxes, color=SUB, fontsize=9)
    # 第二行圖例:機器劇本(與 v7 訊號分開講,免得覆盤時混為一談)
    a1.text(0.006, 0.875,
            "機器劇本(非我方進場): ^真破  ◆吸收  +撤單先行   "
            "白框=有推播 / 淡色=僅記錄   綠=看漲 紅=看跌",
            transform=a1.transAxes, color=SUB, fontsize=8.5)

    # B: cancel skew(只留平滑填色,無散點)
    a2.axhline(0, color=SUB, lw=0.8)
    for lv in (0.3, -0.3):
        a2.axhline(lv, color=GRID, lw=0.7, ls="--")
    a2.fill_between(x, 0, z, where=z >= 0, color=GREEN, alpha=0.65, interpolate=True, lw=0)
    a2.fill_between(x, 0, z, where=z < 0, color=RED, alpha=0.65, interpolate=True, lw=0)
    a2.set_ylim(-0.8, 0.8)
    a2.set_ylabel(f"撤單不對稱\n({args.smooth}m 平滑)", color=TXT, fontsize=10)
    a2.text(0.006, 0.88, "＋ 賣側撤多", transform=a2.transAxes, color=GREEN, fontsize=9)
    a2.text(0.006, 0.06, "－ 買側撤多", transform=a2.transAxes, color=RED, fontsize=9)

    # C: cancellation intensity(乾淨面積)
    a3.fill_between(x, 0, intensity.values, color="#8ab4f8", alpha=0.45, lw=0)
    a3.plot(x, intensity.values, color="#8ab4f8", lw=1.1)
    a3.set_ylabel("撤單強度\n(兩側總量)", color=TXT, fontsize=10)
    a3.margins(y=0.05)
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lb in a3.get_xticklabels():
        lb.set_rotation(0)

    out = PROJECT_ROOT / "research" / "results" / "cancel_flow_monitor.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
