"""
Market Intelligence Indicator Chart — 4h prediction visualization.

Layout (matching 目標範例.png):
  Top bar:    Confidence heatmap (horizontal color strip)
  Main chart: Candlestick + direction triangles (size/opacity by confidence)
  Sub panel:  Bull/Bear Power histogram
  Text:       Per-bar prediction summary

Usage:
    python research/indicator_chart.py
    python research/indicator_chart.py --last 200
    python research/indicator_chart.py --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "eda_charts"


def load_indicator() -> pd.DataFrame:
    path = ROOT / "ml_data" / "BTC_USD_indicator_4h.parquet"
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df = df.set_index("dt")
    return df


def load_ohlc() -> pd.DataFrame:
    path = ROOT / "ml_data" / "BTC_USD_1h_enhanced.parquet"
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df = df.set_index("dt")
        elif "ts_open" in df.columns:
            df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
            df = df.set_index("dt")
    return df[["open", "high", "low", "close"]]


def plot_indicator_chart(ind: pd.DataFrame, ohlc: pd.DataFrame,
                          last_n: int = 200, save: bool = False):
    # Slice to last_n bars
    sig = ind.tail(last_n).copy()
    oh  = ohlc.reindex(sig.index).dropna()
    sig = sig.reindex(oh.index)

    n = len(oh)
    x = np.arange(n)
    dates = oh.index

    fig = plt.figure(figsize=(20, 11), facecolor="white")
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 8, 3], hspace=0.05)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1: Confidence heatmap strip
    # ═══════════════════════════════════════════════════════════════════════
    ax_conf = fig.add_subplot(gs[0])

    conf = sig["confidence_score"].fillna(0).values
    conf_norm = np.clip(conf / 100.0, 0, 1)

    colors = []
    for i in range(n):
        c = conf_norm[i]
        colors.append(plt.cm.Purples(0.15 + 0.85 * c))

    ax_conf.bar(x, np.ones(n), width=1.0, color=colors, edgecolor="none")
    ax_conf.set_xlim(-0.5, n - 0.5)
    ax_conf.set_ylim(0, 1)
    ax_conf.set_yticks([])
    ax_conf.set_xticks([])
    ax_conf.set_title("BTC Market Intelligence Indicator  (4h prediction horizon)",
                       fontsize=14, fontweight="bold", loc="left")
    ax_conf.text(n + 1, 0.5, "Confidence", fontsize=8, va="center",
                  transform=ax_conf.transData)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2: Candlestick + Direction Triangles
    # ═══════════════════════════════════════════════════════════════════════
    ax_price = fig.add_subplot(gs[1])

    opens  = oh["open"].values
    highs  = oh["high"].values
    lows   = oh["low"].values
    closes = oh["close"].values

    up   = closes >= opens
    down = ~up

    body_width = 0.6
    ax_price.bar(x[up],  closes[up] - opens[up], bottom=opens[up],
                  width=body_width, color="#26a69a", edgecolor="#26a69a", linewidth=0.5)
    ax_price.bar(x[down], opens[down] - closes[down], bottom=closes[down],
                  width=body_width, color="#ef5350", edgecolor="#ef5350", linewidth=0.5)

    ax_price.vlines(x[up],  lows[up],  highs[up],  color="#26a69a", linewidth=0.5)
    ax_price.vlines(x[down], lows[down], highs[down], color="#ef5350", linewidth=0.5)

    # Direction triangles
    price_range = highs.max() - lows.min()
    offset = price_range * 0.02

    for i in range(n):
        d = sig.iloc[i]["pred_direction"]
        c = sig.iloc[i]["confidence_score"]
        s = sig.iloc[i]["strength_score"]

        if d == "NEUTRAL" or pd.isna(c) or c < 40:
            continue

        # Strong (>=70): large dark triangle; Moderate 50~70: small light triangle
        if s == "Strong":
            msize = 12
        else:
            msize = 7

        alpha = max(0.3, min(c / 100, 1.0))

        if d == "UP":
            y_pos = lows[i] - offset
            marker = "^"
            color = "#004d40" if s == "Strong" else "#26a69a"
        else:
            y_pos = highs[i] + offset
            marker = "v"
            color = "#b71c1c" if s == "Strong" else "#ef5350"

        ax_price.scatter(i, y_pos, marker=marker, s=msize**2,
                          color=color, alpha=alpha, edgecolors="none", zorder=5)

    ax_price.set_ylabel("Price (USD)", fontsize=10)
    ax_price.grid(True, alpha=0.15)
    ax_price.set_xlim(-0.5, n - 0.5)

    # Date labels on price panel
    tick_pos_price = np.linspace(0, n - 1, min(12, n)).astype(int)
    ax_price.set_xticks(tick_pos_price)
    ax_price.set_xticklabels(
        [dates[i].strftime("%m/%d %H:%M") for i in tick_pos_price],
        fontsize=7, rotation=30, ha="right"
    )

    # Legend
    legend_elements = [
        plt.scatter([], [], marker="^", color="#004d40", s=144, label="Strong UP"),
        plt.scatter([], [], marker="^", color="#26a69a", s=49, label="UP"),
        plt.scatter([], [], marker="v", color="#b71c1c", s=144, label="Strong DOWN"),
        plt.scatter([], [], marker="v", color="#ef5350", s=49, label="DOWN"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=7,
                     framealpha=0.8, ncol=4)

    # Latest prediction annotation (Strong only)
    strong_sigs = sig[(sig["pred_direction"] != "NEUTRAL") & (sig["strength_score"] == "Strong")]
    last_active = strong_sigs.iloc[-1] if len(strong_sigs) > 0 else None
    if last_active is not None:
        d = last_active["pred_direction"]
        ret = last_active["pred_return_4h"] * 100
        conf = last_active["confidence_score"]
        strength = last_active["strength_score"]
        regime = last_active["regime"]
        txt = (f"4h forecast: {d}  |  pred: {ret:+.2f}%  |  "
               f"confidence: {conf:.0f}  |  {strength}  |  {regime}")
        ax_price.text(0.98, 0.02, txt, transform=ax_price.transAxes,
                       fontsize=8, ha="right", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 3: Bull/Bear Power
    # ═══════════════════════════════════════════════════════════════════════
    ax_bbp = fig.add_subplot(gs[2])

    bbp = sig["bull_bear_power"].fillna(0).values
    bbp_colors = ["#26a69a" if v > 0 else "#ef5350" for v in bbp]

    ax_bbp.bar(x, bbp, width=0.8, color=bbp_colors, alpha=0.7, edgecolor="none")
    ax_bbp.axhline(0, color="black", linewidth=0.5)
    ax_bbp.set_ylabel("Bull/Bear\nPower", fontsize=9)
    ax_bbp.set_ylim(-1, 1)
    ax_bbp.set_xlim(-0.5, n - 0.5)
    ax_bbp.grid(True, alpha=0.15)

    # X-axis labels on BBP panel
    tick_positions = np.linspace(0, n - 1, min(12, n)).astype(int)
    ax_bbp.set_xticks(tick_positions)
    ax_bbp.set_xticklabels(
        [dates[i].strftime("%m/%d %H:%M") for i in tick_positions],
        fontsize=7, rotation=30, ha="right"
    )
    ax_conf.set_xticks([])

    plt.tight_layout()

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "indicator_4h.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=200)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ind  = load_indicator()
    ohlc = load_ohlc()
    plot_indicator_chart(ind, ohlc, last_n=args.last, save=args.save)


if __name__ == "__main__":
    main()
