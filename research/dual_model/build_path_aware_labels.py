"""
Build path-clipped regression target (Phase 1.1).

Replaces y_path_ret_4h (TWAP endpoint) with a path-aware first-touch return:

    For each bar t with entry = close[t]:
        Look at bars t+1..t+H (H = horizon = 4).  Find first bar with:
          (A) high[t+k] >= entry × (1 + TP_dist)   → +TP hit
          (B) low[t+k]  <= entry × (1 - SL_dist)   → -SL hit
        First-touch logic:
          - A only       → target = +TP_dist     (label_type = 'tp')
          - B only       → target = -SL_dist     (label_type = 'sl')
          - A and B same → target = -SL_dist     (label_type = 'ambig'; conservative
                                                  — closer barrier wins)
          - Neither      → target = endpoint return (label_type = 'timeout')

Configuration (per user, 2026-05-10):
    TP = 0.5%, SL = 0.3%, horizon = 4 bars.

Output:
    research/dual_model/.cache/labels_path_clip_TP50_SL30.parquet
    Columns: y_path_clip_4h, first_touch_bar, label_type, plus copy of close/high/low.

Sanity checks printed:
    1. label_type distribution (timeout < 30% means barriers active)
    2. first_touch_bar distribution (most early = aligned with order-flow horizon)
    3. target distribution (mean / std / min / max)
    4. Spearman correlation vs existing y_path_ret_4h (low ≠ same target)
    5. Per-month timeout ratio (drift check)
"""
from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
OUT_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"
HORIZON = 4

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_path_clipped_target(
    klines: pd.DataFrame,
    tp_dist: float,
    sl_dist: float,
    horizon: int,
) -> pd.DataFrame:
    n = len(klines)
    close = klines["close"].values.astype(float)
    high = klines["high"].values.astype(float)
    low = klines["low"].values.astype(float)

    target = np.full(n, np.nan)
    first_touch_bar = np.full(n, np.nan)
    label_type = np.full(n, "", dtype=object)

    # SL distance < TP distance → if same bar both touched, SL wins (closer)
    # If your config has TP < SL, swap the ambig logic — but with TP=0.5/SL=0.3
    # the closer barrier is SL.
    closer_is_sl = sl_dist < tp_dist

    for t in range(n - horizon):
        entry = close[t]
        tp_price = entry * (1.0 + tp_dist)
        sl_price = entry * (1.0 - sl_dist)

        triggered = False
        for k in range(1, horizon + 1):
            hi = high[t + k]
            lo = low[t + k]
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price

            if tp_hit and sl_hit:
                if closer_is_sl:
                    target[t] = -sl_dist
                    label_type[t] = "ambig_sl"
                else:
                    target[t] = +tp_dist
                    label_type[t] = "ambig_tp"
                first_touch_bar[t] = k
                triggered = True
                break
            if sl_hit:
                target[t] = -sl_dist
                label_type[t] = "sl"
                first_touch_bar[t] = k
                triggered = True
                break
            if tp_hit:
                target[t] = +tp_dist
                label_type[t] = "tp"
                first_touch_bar[t] = k
                triggered = True
                break

        if not triggered:
            target[t] = (close[t + horizon] / entry) - 1.0
            label_type[t] = "timeout"
            first_touch_bar[t] = horizon

    return pd.DataFrame({
        "y_path_clip_4h": target,
        "first_touch_bar": first_touch_bar,
        "label_type": label_type,
        "close": close,
    }, index=klines.index)


def build_baseline_target(klines: pd.DataFrame, horizon: int) -> pd.Series:
    """Reproduce y_path_ret_4h (TWAP) for spearman comparison."""
    close = klines["close"].values.astype(float)
    n = len(close)
    y = np.full(n, np.nan)
    for t in range(n - horizon):
        future_sum = sum(close[t + k] for k in range(1, horizon + 1))
        y[t] = (future_sum / horizon) / close[t] - 1.0
    return pd.Series(y, index=klines.index, name="y_path_ret_4h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=float, default=0.005,
                        help="TP distance as fraction (default 0.005 = 0.5%%)")
    parser.add_argument("--sl", type=float, default=0.003,
                        help="SL distance as fraction (default 0.003 = 0.3%%)")
    args = parser.parse_args()
    tp_dist = args.tp
    sl_dist = args.sl

    logger.info("Loading klines from %s", KLINES_PATH)
    klines = pd.read_parquet(KLINES_PATH)
    klines = klines[["open", "high", "low", "close"]].copy()
    klines = klines.dropna()
    klines = klines[~klines.index.duplicated(keep="last")].sort_index()
    logger.info("Klines: n=%d, range %s ~ %s",
                len(klines), klines.index.min(), klines.index.max())

    logger.info("Building path-clipped target: TP=%.2f%%, SL=%.2f%%, H=%d",
                tp_dist * 100, sl_dist * 100, HORIZON)
    new_label = build_path_clipped_target(klines, tp_dist, sl_dist, HORIZON)
    baseline = build_baseline_target(klines, HORIZON)
    combined = new_label.join(baseline)
    combined = combined.dropna(subset=["y_path_clip_4h"])

    print(f"\n{'='*80}")
    print(f"Sanity 1: label_type distribution (n={len(combined)})")
    print(f"{'='*80}")
    counts = combined["label_type"].value_counts()
    for lt, cnt in counts.items():
        pct = cnt / len(combined) * 100
        print(f"  {lt:<10} {cnt:>5} ({pct:>5.1f}%)")
    timeout_pct = (combined["label_type"] == "timeout").mean() * 100
    print(f"\n  timeout ratio = {timeout_pct:.1f}%  "
          f"({'OK' if 10 < timeout_pct < 50 else 'check barriers'})")

    print(f"\n{'='*80}")
    print("Sanity 2: first_touch_bar distribution (1=fastest, 4=horizon-end)")
    print(f"{'='*80}")
    bar_dist = combined["first_touch_bar"].value_counts().sort_index()
    for bar, cnt in bar_dist.items():
        pct = cnt / len(combined) * 100
        print(f"  bar+{int(bar)}h : {cnt:>5} ({pct:>5.1f}%)")
    print(f"\n  Mean bars to first touch = {combined['first_touch_bar'].mean():.2f}")
    fast_pct = (combined["first_touch_bar"] <= 1).sum() / len(combined) * 100
    print(f"  Triggered within 1h: {fast_pct:.1f}% "
          f"(higher = better aligned with short-horizon order-flow features)")

    print(f"\n{'='*80}")
    print("Sanity 3: target distribution")
    print(f"{'='*80}")
    print(f"  y_path_clip_4h:")
    print(f"    mean   = {combined['y_path_clip_4h'].mean():+.5f}")
    print(f"    std    = {combined['y_path_clip_4h'].std():.5f}")
    print(f"    min    = {combined['y_path_clip_4h'].min():+.5f}")
    print(f"    max    = {combined['y_path_clip_4h'].max():+.5f}")
    print(f"    >0     = {(combined['y_path_clip_4h']>0).mean()*100:.1f}%")
    print(f"  y_path_ret_4h (baseline TWAP):")
    print(f"    mean   = {combined['y_path_ret_4h'].mean():+.5f}")
    print(f"    std    = {combined['y_path_ret_4h'].std():.5f}")

    print(f"\n{'='*80}")
    print("Sanity 4: Spearman vs existing y_path_ret_4h")
    print(f"{'='*80}")
    rho, _ = spearmanr(combined["y_path_clip_4h"], combined["y_path_ret_4h"])
    print(f"  rho = {rho:.4f}  ({'high — labels similar' if rho > 0.85 else 'OK — labels differ enough to be worth retraining'})")

    print(f"\n{'='*80}")
    print("Sanity 5: per-month timeout ratio (drift check)")
    print(f"{'='*80}")
    combined_idx = combined.copy()
    combined_idx["month"] = combined_idx.index.tz_localize(None).to_period("M")
    monthly = combined_idx.groupby("month").apply(
        lambda g: pd.Series({
            "n": len(g),
            "tp_pct": (g["label_type"] == "tp").mean() * 100,
            "sl_pct": ((g["label_type"] == "sl") | (g["label_type"] == "ambig_sl")).mean() * 100,
            "timeout_pct": (g["label_type"] == "timeout").mean() * 100,
            "mean_target_bps": g["y_path_clip_4h"].mean() * 10000,
        })
    )
    print(monthly.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"labels_path_clip_TP{int(tp_dist*10000)}_SL{int(sl_dist*10000)}.parquet"
    combined.to_parquet(out_path)
    logger.info("Saved labels → %s (%d rows)", out_path, len(combined))


if __name__ == "__main__":
    main()
