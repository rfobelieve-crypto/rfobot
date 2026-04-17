"""
Chart renderer — generates indicator PNG from prediction DataFrame.
v4: 4-panel layout (removed Direction P(UP) panel):
    1. Confidence heatmap
    2. Candlestick + direction triangles (Moderate + Strong)
    3. Magnitude bar chart (predicted |return|)
    4. Bull/Bear Power histogram
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
    Optional: mag_pred
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

    has_mag = "mag_pred" in sig.columns and sig["mag_pred"].notna().any()

    # Layout: 4 panels (with magnitude) or 3 (legacy)
    panels = [0.8, 8]  # confidence + candlestick
    if has_mag:
        panels.append(2.0)   # magnitude
    panels.append(2.0)       # BBP

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

    # Direction triangles — Strong signals only
    price_range = highs.max() - lows.min()
    offset = price_range * 0.02

    for i in range(n):
        d = sig.iloc[i]["pred_direction"]
        c = sig.iloc[i]["confidence_score"]
        s = sig.iloc[i]["strength_score"]

        if d == "NEUTRAL" or pd.isna(c) or s != "Strong":
            continue

        alpha = max(0.5, min(c / 100, 1.0))

        if d == "UP":
            y_pos = lows[i] - offset
            marker = "^"
            color = "#004d40"
        else:
            y_pos = highs[i] + offset
            marker = "v"
            color = "#b71c1c"

        ax_price.scatter(i, y_pos, marker=marker, s=13**2,
                         color=color, alpha=alpha, edgecolors="white",
                         linewidths=1.2, zorder=5)

    ax_price.set_ylabel("Price (USD)", fontsize=10)
    ax_price.grid(True, alpha=0.15)
    ax_price.set_xlim(-0.5, n - 0.5)
    ax_price.set_xticks([])

    # Legend
    legend_elements = [
        plt.scatter([], [], marker="^", color="#004d40", s=169, edgecolors="white",
                    linewidths=1.2, label="Strong UP"),
        plt.scatter([], [], marker="v", color="#b71c1c", s=169, edgecolors="white",
                    linewidths=1.2, label="Strong DOWN"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=7,
                    framealpha=0.8, ncol=2)

    # Date labels — shared tick positions for all panels
    tick_pos = np.linspace(0, n - 1, min(12, n)).astype(int)

    # ── Panel 3: Magnitude (predicted |return_4h|, signed by regression lean) ──
    if has_mag:
        ax_mag = fig.add_subplot(gs[panel_idx]); panel_idx += 1
        mag_raw = sig["mag_pred"].fillna(0).values.astype(float) * 100  # to %
        strength = sig["strength_score"].fillna("Weak").values
        # Prefer regression lean (dir_pred_ret) so every bar gets a direction,
        # not just the top-15% tiers. Fall back to pred_direction for the
        # legacy binary path which has no dir_pred_ret column.
        if "dir_pred_ret" in sig.columns:
            lean = sig["dir_pred_ret"].fillna(0).values.astype(float)
            sign_arr = np.sign(lean)
        else:
            dirs = sig["pred_direction"].fillna("NEUTRAL").values
            sign_arr = np.where(dirs == "UP", 1.0,
                                np.where(dirs == "DOWN", -1.0, 0.0))

        # Tier → colour intensity. Strong=dark, others=grey.
        UP_STRONG = "#004d40"
        DN_STRONG = "#b71c1c"
        NEUTRAL_GREY = "#bdbdbd"

        # All bars plot above 0; direction is encoded purely by colour.
        mag_signed = np.abs(mag_raw).astype(float)
        mag_colors = []
        for i in range(n):
            s = strength[i]
            sgn = sign_arr[i]
            if s != "Strong":
                mag_colors.append(NEUTRAL_GREY)
            elif sgn > 0:
                mag_colors.append(UP_STRONG)
            elif sgn < 0:
                mag_colors.append(DN_STRONG)
            else:
                mag_colors.append(NEUTRAL_GREY)

        ax_mag.bar(x, mag_signed, width=0.8, color=mag_colors, alpha=0.8, edgecolor="none")

        # Reference lines: 80th / 90th percentile of |mag| within the window.
        # p80 is where Direction Strong WR starts to lift (base 72.9% → 77.6%);
        # p90 marks the high-conviction tier.
        abs_nonzero = mag_signed[mag_signed > 1e-9]
        if abs_nonzero.size >= 10:
            p80 = float(np.quantile(abs_nonzero, 0.80))
            p90 = float(np.quantile(abs_nonzero, 0.90))
            for y, color, label in [
                (p80, "#f9a825", "p80"),
                (p90, "#ef6c00", "p90"),
            ]:
                ax_mag.axhline(y, color=color, linewidth=0.8,
                               linestyle="--", alpha=0.7)
                ax_mag.text(n - 0.5, y, f" {label}", fontsize=6,
                            color=color, va="center", ha="left")

        ax_mag.set_ylabel("Magnitude\n(%)", fontsize=9)
        ax_mag.set_xlim(-0.5, n - 0.5)
        # Scale: pure data-driven — max(mag) × 1.1.
        peak = float(np.nanmax(mag_signed)) if mag_signed.size else 0.0
        mag_max = peak * 1.1 if peak > 1e-6 else 0.05
        ax_mag.set_ylim(0, mag_max)

        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax_mag.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if mag_max < 0.1:
            fmt = lambda v, _: f"{v:.3f}"
        elif mag_max < 1.0:
            fmt = lambda v, _: f"{v:.2f}"
        else:
            fmt = lambda v, _: f"{v:.1f}"
        ax_mag.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax_mag.tick_params(axis="y", labelsize=7)
        ax_mag.grid(True, alpha=0.2, axis="both", which="major")
        ax_mag.set_xticks([])

    # ── Panel: Bull/Bear Power (bottom) ──────────────────────────────────
    ax_bbp = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    bbp = sig["bull_bear_power"].fillna(0).values
    bbp_colors = ["#26a69a" if v > 0 else "#ef5350" for v in bbp]

    ax_bbp.bar(x, bbp, width=0.8, color=bbp_colors, alpha=0.7, edgecolor="none")
    ax_bbp.axhline(0, color="black", linewidth=0.5)
    ax_bbp.set_ylabel("Bull/Bear\nPower", fontsize=9)
    ax_bbp.set_ylim(-1, 1)
    ax_bbp.set_xlim(-0.5, n - 0.5)
    ax_bbp.grid(True, alpha=0.15)

    # Bottom panel — date labels
    ax_bbp.set_xticks(tick_pos)
    ax_bbp.set_xticklabels(
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
