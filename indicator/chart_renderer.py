"""
Chart renderer — generates indicator PNG from prediction DataFrame.
v5: panel layout:
    1. Confidence heatmap
    2. Regime strip (when regime column present)
    3. Candlestick + direction triangles (Moderate + Strong)
    4. Magnitude bar chart (when mag_pred present)
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

# Regime color mapping for the regime strip (Panel 2)
REGIME_COLORS = {
    "TRENDING_BULL": "#26a69a",
    "TRENDING_BEAR": "#ef5350",
    "CHOPPY":        "#78828f",
    "WARMUP":        "#eeeeee",
}


def render_chart(ind: pd.DataFrame, last_n: int = 100) -> bytes:
    """
    Render indicator chart and return PNG bytes.

    ind must have: open, high, low, close, pred_direction, confidence_score,
                   strength_score, pred_return_4h, regime
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
    has_regime = "regime" in sig.columns and sig["regime"].notna().any()

    # Layout panels (top → bottom):
    #   confidence heatmap (0.8) → regime strip (0.3, optional)
    #   → candlestick (8) → magnitude (2.0, optional)
    panels = [0.8]                       # confidence heatmap
    if has_regime:
        panels.append(0.3)               # regime strip
    panels.append(8)                     # candlestick
    if has_mag:
        panels.append(2.0)               # magnitude

    n_panels = len(panels)
    fig_h = 9 + 2 * (n_panels - 2 - (1 if has_regime else 0))
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

    # ── Panel 2: Regime strip ────────────────────────────────────────────
    # Thin horizontal band showing regime over time. Aligned with candlestick
    # x-axis (same bar indices) so readers can immediately see which bars
    # fall in TRENDING_BULL / TRENDING_BEAR / CHOPPY windows — useful for
    # diagnosing contra-trend signal clusters.
    if has_regime:
        ax_regime = fig.add_subplot(gs[panel_idx]); panel_idx += 1
        regimes = sig["regime"].fillna("CHOPPY").astype(str).values
        reg_colors = [REGIME_COLORS.get(r, REGIME_COLORS["CHOPPY"]) for r in regimes]
        ax_regime.bar(x, np.ones(n), width=1.0, color=reg_colors, edgecolor="none")
        ax_regime.set_xlim(-0.5, n - 0.5)
        ax_regime.set_ylim(0, 1)
        ax_regime.set_yticks([])
        ax_regime.set_xticks([])
        ax_regime.text(n + 1, 0.5, "Regime", fontsize=8, va="center",
                       transform=ax_regime.transData)
        # Current regime as a small label at the right edge
        last_reg = regimes[-1]
        ax_regime.text(
            n - 0.5, 0.5, last_reg.replace("TRENDING_", ""),
            fontsize=7, va="center", ha="right", color="white",
            fontweight="bold",
            transform=ax_regime.transData,
        )

    # ── Panel 3: Candlestick + direction triangles ───────────────────────
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

    # Rolling 200-bar percentile of |mag_pred| — used to scale triangle
    # size by the user's mental filter: Strong + mag>p90 → extra large
    # (🔥 strongest conviction), Strong + mag<p80 → downsized to show
    # "direction strong but magnitude weak", and mirrors for Moderate.
    if "mag_pred" in sig.columns:
        _abs_mag = np.abs(sig["mag_pred"].fillna(0).values.astype(float))
        mag_pct_arr = np.full(n, 50.0)
        for i in range(n):
            start = max(0, i - 199)
            w = _abs_mag[start:i + 1]
            w_nz = w[w > 1e-9]
            if len(w_nz) >= 10 and _abs_mag[i] > 1e-9:
                mag_pct_arr[i] = (w_nz <= _abs_mag[i]).sum() / len(w_nz) * 100
    else:
        mag_pct_arr = np.full(n, 50.0)

    # Direction triangles — Strong + Moderate signals
    price_range = highs.max() - lows.min()
    offset = price_range * 0.02

    for i in range(n):
        d = sig.iloc[i]["pred_direction"]
        c = sig.iloc[i]["confidence_score"]
        s = sig.iloc[i]["strength_score"]

        if d == "NEUTRAL" or pd.isna(c) or s not in ("Strong", "Moderate"):
            continue

        is_strong = (s == "Strong")
        mag_p = mag_pct_arr[i]
        # Size by (strength × mag_pct) — the user's dual-gate mental rule
        if is_strong:
            if mag_p >= 90:   tri_size = 17**2    # extra large: 🔥 Go
            elif mag_p >= 80: tri_size = 13**2    # regular Strong
            else:             tri_size = 9**2     # downsized: dir strong, mag weak
        else:  # Moderate
            if mag_p >= 90:   tri_size = 13**2    # upsized: dir mod, mag strong 🎯
            elif mag_p >= 80: tri_size = 9**2     # regular Moderate
            else:             tri_size = 6**2     # downsized: both weak
        alpha = max(0.5, min(c / 100, 1.0)) if is_strong else max(0.35, min(c / 100, 0.7))

        if d == "UP":
            y_pos = lows[i] - offset
            marker = "^"
            color = "#004d40" if is_strong else "#66bb6a"
        else:
            y_pos = highs[i] + offset
            marker = "v"
            color = "#b71c1c" if is_strong else "#ef5350"

        ax_price.scatter(i, y_pos, marker=marker, s=tri_size,
                         color=color, alpha=alpha, edgecolors="white",
                         linewidths=1.2 if is_strong else 0.8, zorder=5)

    ax_price.set_ylabel("Price (USD)", fontsize=10)
    ax_price.grid(True, alpha=0.15)
    ax_price.set_xlim(-0.5, n - 0.5)
    # x-tick labels will be set on the bottom-most panel below (either
    # ax_mag if has_mag, or ax_price itself). Suppress here conditionally.
    if has_mag:
        ax_price.set_xticks([])

    # Legend
    legend_elements = [
        plt.scatter([], [], marker="^", color="#004d40", s=169, edgecolors="white",
                    linewidths=1.2, label="Strong UP"),
        plt.scatter([], [], marker="v", color="#b71c1c", s=169, edgecolors="white",
                    linewidths=1.2, label="Strong DOWN"),
        plt.scatter([], [], marker="^", color="#66bb6a", s=81, edgecolors="white",
                    linewidths=0.8, label="Moderate UP"),
        plt.scatter([], [], marker="v", color="#ef5350", s=81, edgecolors="white",
                    linewidths=0.8, label="Moderate DOWN"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=7,
                    framealpha=0.8, ncol=4)

    # Date labels — shared tick positions for all panels
    tick_pos = np.linspace(0, n - 1, min(12, n)).astype(int)

    # ── Panel 3: Magnitude (predicted |return_4h|, signed by regression lean) ──
    if has_mag:
        ax_mag = fig.add_subplot(gs[panel_idx]); panel_idx += 1
        mag_raw = sig["mag_pred"].fillna(0).values.astype(float) * 100  # to %
        strength = sig["strength_score"].fillna("Weak").values
        # Magnitude bar colour encodes the regression lean sign (UP/DOWN/0).
        # We use pred_return_4h as the lean source — it carries the same
        # value as the engine's raw dir_pred_ret (inference.py:365) but IS
        # persisted to indicator_history MySQL, so historical bars survive
        # a process restart with their direction colour intact.
        # Per-bar fallback to pred_direction handles legacy rows that
        # somehow lack pred_return_4h (extreme edge case after migrations).
        lean_col = sig.get("pred_return_4h")
        if lean_col is None:
            lean_col = sig.get("dir_pred_ret")
        if lean_col is not None:
            lean_vals = lean_col.values.astype(float)
            dirs = sig["pred_direction"].fillna("NEUTRAL").values
            fallback_sign = np.where(dirs == "UP", 1.0,
                                     np.where(dirs == "DOWN", -1.0, 0.0))
            sign_arr = np.where(np.isnan(lean_vals),
                                fallback_sign,
                                np.sign(lean_vals))
        else:
            dirs = sig["pred_direction"].fillna("NEUTRAL").values
            sign_arr = np.where(dirs == "UP", 1.0,
                                np.where(dirs == "DOWN", -1.0, 0.0))

        # Tier → colour intensity. Strong=dark, Moderate=light, others=grey.
        UP_STRONG = "#004d40"
        UP_MODERATE = "#66bb6a"
        DN_STRONG = "#b71c1c"
        DN_MODERATE = "#ef5350"
        NEUTRAL_GREY = "#bdbdbd"

        # All bars plot above 0; direction is encoded purely by colour.
        mag_signed = np.abs(mag_raw).astype(float)
        mag_colors = []
        for i in range(n):
            s = strength[i]
            sgn = sign_arr[i]
            if s == "Strong":
                if sgn > 0:
                    mag_colors.append(UP_STRONG)
                elif sgn < 0:
                    mag_colors.append(DN_STRONG)
                else:
                    mag_colors.append(NEUTRAL_GREY)
            elif s == "Moderate":
                if sgn > 0:
                    mag_colors.append(UP_MODERATE)
                elif sgn < 0:
                    mag_colors.append(DN_MODERATE)
                else:
                    mag_colors.append(NEUTRAL_GREY)
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

    # Date labels on the bottom-most existing panel.
    bottom_ax = ax_mag if has_mag else ax_price
    bottom_ax.set_xticks(tick_pos)
    bottom_ax.set_xticklabels(
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
