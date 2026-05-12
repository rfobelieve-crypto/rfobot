"""
v9 Dual Binary Win-Rate Labels.

For each bar t (entry = close[t]):

    y_long_win[t] ∈ {0, 1}:
        1 if 4h-forward path first touches +TP_dist
        0 if 4h-forward path first touches -SL_dist
        timeout (neither touched): 1 if close[t+4] > close[t] else 0

    y_short_win[t] ∈ {0, 1}:  (symmetric, short trade frame)
        1 if 4h-forward path first touches -TP_dist (price drops, short wins)
        0 if 4h-forward path first touches +SL_dist (price rises, short loses)
        timeout: 1 if close[t+4] < close[t] else 0

Configuration (per user 2026-05-11):
    TP_dist = 0.5% (50 bps)
    SL_dist = 0.3% (30 bps)
    Horizon = 4 bars (4h)

This is the target alignment for AUTO TRADING:  what we want to know is
"if I enter here, will I hit TP before SL?"  — exactly what the classifier
will predict.

Ambig case (same bar high≥+TP AND low≤-SL): conservative — assume SL hits
first (same as paper_trading_tpsl.py).

Output:
    research/dual_model/.cache/labels_winrate_TP50_SL30.parquet
        ts, close, y_long_win, y_short_win, long_label_type, short_label_type,
        long_first_touch_bar, short_first_touch_bar
"""
from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
OUT_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"

TP_DIST = 0.005  # 0.5%
SL_DIST = 0.003  # 0.3%

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_winrate_labels(klines: pd.DataFrame, horizon: int) -> pd.DataFrame:
    n = len(klines)
    close = klines["close"].values.astype(float)
    high = klines["high"].values.astype(float)
    low = klines["low"].values.astype(float)

    y_long = np.full(n, np.nan)
    y_short = np.full(n, np.nan)
    long_type = np.full(n, "", dtype=object)
    short_type = np.full(n, "", dtype=object)
    long_bar = np.full(n, np.nan)
    short_bar = np.full(n, np.nan)

    for t in range(n - horizon):
        entry = close[t]
        # LONG frame
        long_tp_price = entry * (1.0 + TP_DIST)
        long_sl_price = entry * (1.0 - SL_DIST)
        long_done = False
        for k in range(1, horizon + 1):
            hi = high[t + k]
            lo = low[t + k]
            tp_hit_long = hi >= long_tp_price
            sl_hit_long = lo <= long_sl_price
            if tp_hit_long and sl_hit_long:
                # Conservative: SL first → loss
                y_long[t] = 0
                long_type[t] = "ambig_sl"
                long_bar[t] = k
                long_done = True
                break
            if sl_hit_long:
                y_long[t] = 0
                long_type[t] = "sl"
                long_bar[t] = k
                long_done = True
                break
            if tp_hit_long:
                y_long[t] = 1
                long_type[t] = "tp"
                long_bar[t] = k
                long_done = True
                break
        if not long_done:
            # Timeout — use endpoint sign
            endpoint = close[t + horizon]
            y_long[t] = 1 if endpoint > entry else 0
            long_type[t] = "timeout_win" if endpoint > entry else "timeout_loss"
            long_bar[t] = horizon

        # SHORT frame (mirror of LONG)
        short_tp_price = entry * (1.0 - TP_DIST)  # price drops 0.5% — short wins
        short_sl_price = entry * (1.0 + SL_DIST)  # price rises 0.3% — short loses
        short_done = False
        for k in range(1, horizon + 1):
            hi = high[t + k]
            lo = low[t + k]
            tp_hit_short = lo <= short_tp_price  # price drops to TP
            sl_hit_short = hi >= short_sl_price  # price rises to SL
            if tp_hit_short and sl_hit_short:
                y_short[t] = 0
                short_type[t] = "ambig_sl"
                short_bar[t] = k
                short_done = True
                break
            if sl_hit_short:
                y_short[t] = 0
                short_type[t] = "sl"
                short_bar[t] = k
                short_done = True
                break
            if tp_hit_short:
                y_short[t] = 1
                short_type[t] = "tp"
                short_bar[t] = k
                short_done = True
                break
        if not short_done:
            endpoint = close[t + horizon]
            y_short[t] = 1 if endpoint < entry else 0
            short_type[t] = "timeout_win" if endpoint < entry else "timeout_loss"
            short_bar[t] = horizon

    return pd.DataFrame({
        "close": close,
        "y_long_win": y_long,
        "y_short_win": y_short,
        "long_label_type": long_type,
        "short_label_type": short_type,
        "long_first_touch_bar": long_bar,
        "short_first_touch_bar": short_bar,
    }, index=klines.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=4,
                        help="Forward bars for TP/SL barrier check (default 4)")
    args = parser.parse_args()
    horizon = args.horizon

    logger.info("Loading klines from %s", KLINES_PATH)
    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]]
    klines = klines.dropna()
    klines = klines[~klines.index.duplicated(keep="last")].sort_index()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    logger.info("Klines: n=%d, range %s ~ %s", len(klines), klines.index.min(), klines.index.max())

    logger.info("Building dual binary win-rate labels (TP=%.1f%% SL=%.1f%% H=%d)",
                TP_DIST*100, SL_DIST*100, horizon)
    labels = build_winrate_labels(klines, horizon)
    labels = labels.dropna(subset=["y_long_win", "y_short_win"])
    logger.info("Labels: n=%d", len(labels))

    # Sanity 1: long/short win rate (base rate)
    print(f"\n{'='*80}")
    print("Sanity 1: base win rate (random entry on every bar)")
    print(f"{'='*80}")
    print(f"  Long  win rate: {labels['y_long_win'].mean()*100:.2f}%  (target: cover 13bps cost)")
    print(f"  Short win rate: {labels['y_short_win'].mean()*100:.2f}%")
    print(f"  BE WR with TP=0.5%/SL=0.3%, cost 13bps: 51.0% (after cost)")
    print(f"  BE WR with TP=0.5%/SL=0.3%, no cost:    37.5%")
    print()
    print(f"  Long  P(both win+short win) = {((labels['y_long_win']==1) & (labels['y_short_win']==1)).mean()*100:.2f}%")
    print(f"  Long  P(both lose)          = {((labels['y_long_win']==0) & (labels['y_short_win']==0)).mean()*100:.2f}%")

    # Sanity 2: label_type distribution
    print(f"\n{'='*80}")
    print("Sanity 2: label_type distribution")
    print(f"{'='*80}")
    print(f"  LONG side:")
    print(labels["long_label_type"].value_counts().to_string())
    print(f"\n  SHORT side:")
    print(labels["short_label_type"].value_counts().to_string())

    # Sanity 3: first-touch bar timing
    print(f"\n{'='*80}")
    print("Sanity 3: first touch bar (when does TP or SL hit?)")
    print(f"{'='*80}")
    for col, lbl in [("long_first_touch_bar", "LONG"),
                     ("short_first_touch_bar", "SHORT")]:
        print(f"  {lbl} mean bars to event: {labels[col].mean():.2f}")
        # Bucket bars into rough ranges scaling with horizon
        for bar in sorted(set([1, 2, 4, horizon//2, horizon])):
            if bar < 1:
                continue
            pct = (labels[col] == bar).mean() * 100
            print(f"    bar+{bar}h: {pct:.1f}%")

    # Sanity 4: cross-class correlation
    rho = labels[["y_long_win", "y_short_win"]].corr().iloc[0, 1]
    print(f"\n{'='*80}")
    print("Sanity 4: cross-class correlation")
    print(f"{'='*80}")
    print(f"  corr(y_long_win, y_short_win) = {rho:+.4f}")
    print(f"  -> if strongly negative (~-0.6 to -0.9): healthy (long wins vs short loses)")
    print(f"  -> if weak (-0.3 to +0.3): both can win simultaneously (timeout overlap)")

    # Sanity 5: per-month distribution
    print(f"\n{'='*80}")
    print("Sanity 5: per-month base win rate (drift check)")
    print(f"{'='*80}")
    idx = labels.copy()
    idx["month"] = idx.index.to_period("M")
    monthly = idx.groupby("month").agg(
        n=("y_long_win", "count"),
        long_wr=("y_long_win", lambda x: x.mean()*100),
        short_wr=("y_short_win", lambda x: x.mean()*100),
    )
    print(monthly.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"labels_winrate_TP{int(TP_DIST*10000)}_SL{int(SL_DIST*10000)}_H{horizon}.parquet"
    labels.to_parquet(out_path)
    logger.info("Saved -> %s (%d rows)", out_path, len(labels))


if __name__ == "__main__":
    main()
