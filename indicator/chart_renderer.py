"""
Chart renderer — generates indicator PNG from prediction DataFrame.
v3: Dual-model layout with 5 panels:
    1. Confidence heatmap
    2. Candlestick + direction triangles
    3. Magnitude bar chart (predicted |return|)
    4. Bull/Bear Power histogram
    5. Direction P(UP) probability
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

CHART_PATH = Path("/tmp/indicator_chart.png")


def render_chart(ind: pd.DataFrame, last_n: int = 100) -> bytes:
    """
    Render indicator chart and return PNG bytes.

    ind must have: open, high, low, close, pred_direction, confidence_score,
                   strength_score, pred_return_4h, bull_bear_power, regime
    Optional: dir_prob_up, mag_pred
    """
    sig = ind.tail(last_n).copy()
    sig = sig.dropna(subset=["open", "high", "low", "close"])

    n = len(sig)
    if n == 0:
        return b""

    # Convert to UTC+8 for display
    TZ_UTC8 = timezone(timedelta(hours=8))
    sig.index = sig.index.tz_convert(TZ_UTC8)

    x = np.arange(n)
    dates = sig.index

    has_dir_prob = "dir_prob_up" in sig.columns and sig["dir_prob_up"].notna().any()
    has_mag = "mag_pred" in sig.columns and sig["mag_pred"].notna().any()

    # Layout: 5 panels (full dual-model), 4 (no mag), or 3 (legacy)
    panels = [0.8, 8]  # confidence + candlestick
    if has_mag:
        panels.append(2.0)   # magnitude
    panels.append(2.0)       # BBP
    if has_dir_prob:
        panels.append(2.5)   # direction P(UP)

    n_panels = len(panels)
    fig_h = 9 + 2 * (n_panels - 3)
    fig = plt.figure(figsize=(20, fig_h), facecolor="white")
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=panels, hspace=0.05)

    panel_idx = 0

    # ── Panel 1: Confidence heatmap ──────────────────────────────────────
    ax_conf = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    conf = sig["confidence_score"].fillna(0).values
    conf_norm = np.clip(conf / 100.0, 0, 1)

    colors = [plt.cm.Purples(0.15 + 0.85 * c) for c in conf_norm]
    ax_conf.bar(x, np.ones(n), width=1.0, color=colors, edgecolor="none")
    ax_conf.set_xlim(-0.5, n - 0.5)
    ax_conf.set_ylim(0, 1)
    ax_conf.set_yticks([])
    ax_conf.set_xticks([])

    now_str = datetime.now(TZ_UTC8).strftime("%Y-%m-%d %H:%M UTC+8")
    ax_conf.set_title(
        f"BTC Market Intelligence Indicator  (4h prediction)  |  Updated: {now_str}",
        fontsize=14, fontweight="bold", loc="left"
    )
    ax_conf.text(n + 1, 0.5, "Confidence", fontsize=8, va="center",
                 transform=ax_conf.transData)

    # ── Panel 2: Candlestick + direction triangles ───────────────────────
    ax_price = fig.add_subplot(gs[panel_idx]); panel_idx += 1

    opens = sig["open"].values.astype(float)
    highs = sig["high"].values.astype(float)
    lows = sig["low"].values.astype(float)
    closes = sig["close"].values.astype(float)

    up = closes >= opens
    down = ~up
    body_width = 0.6

    ax_price.bar(x[up], closes[up] - opens[up], bottom=opens[up],
                 width=body_width, color="#26a69a", edgecolor="#26a69a", linewidth=0.5)
    ax_price.bar(x[down], opens[down] - closes[down], bottom=closes[down],
                 width=body_width, color="#ef5350", edgecolor="#ef5350", linewidth=0.5)
    ax_price.vlines(x[up], lows[up], highs[up], color="#26a69a", linewidth=0.5)
    ax_price.vlines(x[down], lows[down], highs[down], color="#ef5350", linewidth=0.5)

    # Direction triangles — direction model only (shape = direction, alpha = confidence)
    price_range = highs.max() - lows.min()
    offset = price_range * 0.02

    for i in range(n):
        d = sig.iloc[i]["pred_direction"]
        c = sig.iloc[i]["confidence_score"]
        s = sig.iloc[i]["strength_score"]

        if d == "NEUTRAL" or pd.isna(c):
            continue

        msize = 11
        alpha = max(0.4, min(c / 100, 1.0))

        if d == "UP":
            y_pos = lows[i] - offset
            marker = "^"
            color = "#004d40" if s == "Strong" else "#26a69a"
        else:
            y_pos = highs[i] + offset
            marker = "v"
            color = "#b71c1c" if s == "Strong" else "#ef5350"

        # Strong: bigger + white edge
        sz = 13 if s == "Strong" else msize
        edge = "white" if s == "Strong" else "none"
        lw = 1.2 if s == "Strong" else 0
        ax_price.scatter(i, y_pos, marker=marker, s=sz**2,
                         color=color, alpha=alpha, edgecolors=edge,
                         linewidths=lw, zorder=5)

    ax_price.set_ylabel("Price (USD)", fontsize=10)
    ax_price.grid(True, alpha=0.15)
    ax_price.set_xlim(-0.5, n - 0.5)
    ax_price.set_xticks([])

    # Legend
    legend_elements = [
        plt.scatter([], [], marker="^", color="#004d40", s=169, edgecolors="white",
                    linewidths=1.2, label="Strong UP"),
        plt.scatter([], [], marker="^", color="#26a69a", s=121, label="UP"),
        plt.scatter([], [], marker="v", color="#ef5350", s=121, label="DOWN"),
        plt.scatter([], [], marker="v", color="#b71c1c", s=169, edgecolors="white",
                    linewidths=1.2, label="Strong DOWN"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=7,
                    framealpha=0.8, ncol=4)

    # Date labels — shared tick positions for all panels
    tick_pos = np.linspace(0, n - 1, min(12, n)).astype(int)

    # ── Panel 3: Magnitude (predicted |return_4h|) ───────────────────────
    if has_mag:
        ax_mag = fig.add_subplot(gs[panel_idx]); panel_idx += 1
        mag_vals = sig["mag_pred"].fillna(0).values.astype(float) * 100  # to %
        mag_dirs = sig["pred_direction"].fillna("NEUTRAL").values

        mag_colors = []
        for i in range(n):
            d = mag_dirs[i]
            m = mag_vals[i]
            if d == "UP":
                mag_colors.append("#004d40" if m > 0.5 else "#26a69a")
            elif d == "DOWN":
                mag_colors.append("#b71c1c" if m > 0.5 else "#ef5350")
            else:
                mag_colors.append("#9e9e9e")

        ax_mag.bar(x, mag_vals, width=0.8, color=mag_colors, alpha=0.8, edgecolor="none")
        ax_mag.axhline(0.5, color="#666666", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_mag.set_ylabel("Magnitude\n(%)", fontsize=9)
        ax_mag.set_xlim(-0.5, n - 0.5)
        ax_mag.set_ylim(0, max(mag_vals.max() * 1.2, 1.0))
        ax_mag.grid(True, alpha=0.15)
        ax_mag.set_xticks([])

    # ── Panel: Bull/Bear Power ───────────────────────────────────────────
    ax_bbp = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    bbp = sig["bull_bear_power"].fillna(0).values
    bbp_colors = ["#26a69a" if v > 0 else "#ef5350" for v in bbp]

    ax_bbp.bar(x, bbp, width=0.8, color=bbp_colors, alpha=0.7, edgecolor="none")
    ax_bbp.axhline(0, color="black", linewidth=0.5)
    ax_bbp.set_ylabel("Bull/Bear\nPower", fontsize=9)
    ax_bbp.set_ylim(-1, 1)
    ax_bbp.set_xlim(-0.5, n - 0.5)
    ax_bbp.grid(True, alpha=0.15)

    if not has_dir_prob:
        # This is the bottom panel — add date labels
        ax_bbp.set_xticks(tick_pos)
        ax_bbp.set_xticklabels(
            [dates[i].strftime("%m/%d %H:%M") for i in tick_pos],
            fontsize=7, rotation=30, ha="right"
        )
    else:
        ax_bbp.set_xticks([])

    # ── Panel: Direction Probability P(UP) ───────────────────────────────
    if has_dir_prob:
        ax_dir = fig.add_subplot(gs[panel_idx]); panel_idx += 1
        dir_prob = sig["dir_prob_up"].fillna(0.5).values

        dir_colors = []
        for p in dir_prob:
            if p > 0.65:
                dir_colors.append("#004d40")
            elif p > 0.5:
                dir_colors.append("#26a69a")
            elif p < 0.35:
                dir_colors.append("#b71c1c")
            else:
                dir_colors.append("#ef5350")

        ax_dir.bar(x, dir_prob - 0.5, bottom=0.5, width=0.8,
                   color=dir_colors, alpha=0.8, edgecolor="none")
        ax_dir.axhline(0.5, color="black", linewidth=0.8)
        ax_dir.axhline(0.65, color="#004d40", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_dir.axhline(0.35, color="#b71c1c", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_dir.set_ylabel("Direction\nP(UP)", fontsize=9)
        ax_dir.set_ylim(0.2, 0.8)
        ax_dir.set_xlim(-0.5, n - 0.5)
        ax_dir.set_yticks([0.35, 0.5, 0.65])
        ax_dir.set_yticklabels(["0.35", "0.50", "0.65"], fontsize=7)
        ax_dir.grid(True, alpha=0.15)

        # Bottom panel — add date labels
        ax_dir.set_xticks(tick_pos)
        ax_dir.set_xticklabels(
            [dates[i].strftime("%m/%d %H:%M") for i in tick_pos],
            fontsize=7, rotation=30, ha="right"
        )

    # Watermark
    fig.text(0.98, 0.01, "source@rfo", fontsize=9, color="gray",
             alpha=0.5, ha="right", va="bottom")

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    # Also save to disk
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHART_PATH, "wb") as f:
        f.write(png_bytes)

    logger.info("Chart rendered: %d bars, %.1f KB", n, len(png_bytes) / 1024)
    return png_bytes
