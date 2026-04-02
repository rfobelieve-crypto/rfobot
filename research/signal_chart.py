"""
Signal Visualization — matches target chart layout.

Layout:
  Top bar:    Confidence heatmap (horizontal color strip)
  Main chart: Candlestick + signal arrows (triangles)
  Sub panel:  Bull/Bear Power bar chart

Usage:
    python research/signal_chart.py
    python research/signal_chart.py --last 200
    python research/signal_chart.py --save
    python research/signal_chart.py --horizon 4h
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "eda_charts"


def load_signals(horizon: str = "1h") -> pd.DataFrame:
    path = ROOT / "ml_data" / f"BTC_USD_signals_{horizon}.parquet"
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df = df.set_index("dt")
    return df


def load_ohlc() -> pd.DataFrame:
    path = ROOT / "ml_data" / "BTC_USD_15m_enhanced.parquet"
    df = pd.read_parquet(path)
    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt")
    return df[["open", "high", "low", "close"]]


def plot_signal_chart(signals: pd.DataFrame, ohlc: pd.DataFrame,
                       last_n: int = 200, horizon: str = "1h",
                       save: bool = False):
    """
    3-panel chart matching target layout:
      1. Confidence heatmap strip
      2. Candlestick + signal triangles
      3. Bull/Bear Power bars
    """
    # Slice to last_n bars
    sig = signals.tail(last_n).copy()
    oh  = ohlc.reindex(sig.index).dropna()
    sig = sig.reindex(oh.index)

    n = len(oh)
    x = np.arange(n)
    dates = oh.index

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 8, 3],
                           hspace=0.05)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1: Confidence heatmap strip
    # ═══════════════════════════════════════════════════════════════════════
    ax_conf = fig.add_subplot(gs[0])

    conf = sig["confidence"].fillna(0).values
    # Colour: low confidence = grey, high = orange/red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "conf", ["#e0e0e0", "#ffcc80", "#ff9800", "#e65100", "#b71c1c"]
    )
    conf_norm = conf / 100.0  # normalize to 0-1

    # Direction overlay: green for UP, red for DOWN
    colors = []
    for i in range(n):
        d = sig.iloc[i]["direction"]
        c = conf_norm[i]
        if d == "UP":
            colors.append(plt.cm.Greens(0.3 + 0.7 * c))
        elif d == "DOWN":
            colors.append(plt.cm.Reds(0.3 + 0.7 * c))
        else:
            colors.append((0.88, 0.88, 0.88, 1.0))

    ax_conf.bar(x, np.ones(n), width=1.0, color=colors, edgecolor="none")
    ax_conf.set_xlim(-0.5, n - 0.5)
    ax_conf.set_ylim(0, 1)
    ax_conf.set_yticks([])
    ax_conf.set_xticks([])
    ax_conf.set_title(f"BTC 15-min AI Trading Signals  (horizon: {horizon})",
                       fontsize=14, fontweight="bold", loc="left")

    # Add "Confidence" label
    ax_conf.text(n + 1, 0.5, "Confidence", fontsize=8, va="center",
                  transform=ax_conf.transData)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2: Candlestick + Signal Arrows
    # ═══════════════════════════════════════════════════════════════════════
    ax_price = fig.add_subplot(gs[1], sharex=ax_conf)

    # Draw candlesticks
    opens  = oh["open"].values
    highs  = oh["high"].values
    lows   = oh["low"].values
    closes = oh["close"].values

    up   = closes >= opens
    down = ~up

    # Bodies
    body_width = 0.6
    ax_price.bar(x[up],  closes[up] - opens[up], bottom=opens[up],
                  width=body_width, color="#26a69a", edgecolor="#26a69a", linewidth=0.5)
    ax_price.bar(x[down], opens[down] - closes[down], bottom=closes[down],
                  width=body_width, color="#ef5350", edgecolor="#ef5350", linewidth=0.5)

    # Wicks
    ax_price.vlines(x[up],  lows[up],  highs[up],  color="#26a69a", linewidth=0.5)
    ax_price.vlines(x[down], lows[down], highs[down], color="#ef5350", linewidth=0.5)

    # Signal triangles
    price_range = highs.max() - lows.min()
    offset = price_range * 0.02

    for i in range(n):
        d = sig.iloc[i]["direction"]
        c = sig.iloc[i]["confidence"]
        s = sig.iloc[i]["strength"]

        if d not in ("UP", "DOWN"):
            continue
        if pd.isna(c):
            continue

        # Size based on confidence
        msize = 4 + c / 10  # 4~14

        # Colour based on strength
        if d == "UP":
            y_pos = lows[i] - offset
            marker = "^"
            color = "#26a69a" if s != "Strong" else "#004d40"
            alpha = 0.5 + c / 200
        else:
            y_pos = highs[i] + offset
            marker = "v"
            color = "#ef5350" if s != "Strong" else "#b71c1c"
            alpha = 0.5 + c / 200

        ax_price.scatter(i, y_pos, marker=marker, s=msize**2,
                          color=color, alpha=min(alpha, 1.0),
                          edgecolors="none", zorder=5)

    ax_price.set_ylabel("Price", fontsize=10)
    ax_price.grid(True, alpha=0.15)
    ax_price.set_xlim(-0.5, n - 0.5)

    # Legend
    legend_elements = [
        plt.scatter([], [], marker="v", color="#ef5350", s=40, label="Down"),
        plt.scatter([], [], marker="^", color="#26a69a", s=40, label="Up"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=8,
                     framealpha=0.7, ncol=2)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 3: Bull/Bear Power
    # ═══════════════════════════════════════════════════════════════════════
    ax_bbp = fig.add_subplot(gs[2], sharex=ax_conf)

    bbp = sig["bull_bear_power"].fillna(0).values
    bbp_colors = ["#26a69a" if v > 0 else "#ef5350" for v in bbp]

    ax_bbp.bar(x, bbp, width=0.8, color=bbp_colors, alpha=0.7, edgecolor="none")
    ax_bbp.axhline(0, color="black", linewidth=0.5)
    ax_bbp.set_ylabel("Bull/Bear\nPower", fontsize=9)
    ax_bbp.set_ylim(-1, 1)
    ax_bbp.grid(True, alpha=0.15)

    # X-axis: show dates
    tick_positions = np.linspace(0, n - 1, min(10, n)).astype(int)
    ax_bbp.set_xticks(tick_positions)
    ax_bbp.set_xticklabels(
        [dates[i].strftime("%m/%d\n%H:%M") for i in tick_positions],
        fontsize=7, rotation=0
    )

    ax_conf.set_xticks([])

    plt.tight_layout()

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"signal_chart_{horizon}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="1h", choices=["1h", "2h", "4h"])
    ap.add_argument("--last", type=int, default=200,
                    help="Number of bars to show")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    signals = load_signals(args.horizon)
    ohlc    = load_ohlc()

    plot_signal_chart(signals, ohlc, last_n=args.last,
                       horizon=args.horizon, save=args.save)


if __name__ == "__main__":
    main()
