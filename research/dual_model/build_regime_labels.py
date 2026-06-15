"""
v12 3-class regime transition labels.

Concept (user idea 2, 2026-05-12):
    Reframe target — don't predict "TP before SL" (v9, hit AUC ceiling
    0.54 with current features). Predict "will the market trend in this
    window, and in which direction?"

Label (per bar t, looking forward HORIZON bars):
    0 = no_trend   : neither +TH nor -TH touched in window
    1 = up_trend   : high[t+k] first crosses +TH before low touches -TH
    2 = down_trend : low[t+k]  first crosses -TH before high touches +TH

When both barriers touched same bar: conservative — go with lower side
(short-favored, since path usually flickers; mirrors paper_trading_tpsl).

Hyperparameters (configurable via CLI):
    threshold : barrier distance, default 0.006 (0.6%)
    horizon   : bars forward, default 8

Output:
    research/dual_model/.cache/labels_regime_T{bp}_H{h}.parquet
        ts (index), y_regime (0/1/2), label_type, first_touch_bar, close

Why this might work where v9 didn't:
    - 3-class with multi:softprob lets model output P(up) + P(down) +
      P(no_trend) probabilities — more informative than binary
    - Filter trade based on max(P_up, P_down) > T AND P_no_trend < T'
      — model self-selects high-confidence triggers
    - Asymmetry of regimes (up_rare / down_rare / no_trend_common) might
      be learnable from existing features in a way binary win wasn't
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_regime_labels(klines: pd.DataFrame, threshold: float,
                          horizon: int) -> pd.DataFrame:
    n = len(klines)
    close = klines["close"].values.astype(float)
    high = klines["high"].values.astype(float)
    low = klines["low"].values.astype(float)

    y = np.full(n, np.nan)
    label_type = np.full(n, "", dtype=object)
    bar = np.full(n, np.nan)

    for t in range(n - horizon):
        entry = close[t]
        up_price = entry * (1.0 + threshold)
        dn_price = entry * (1.0 - threshold)
        done = False
        for k in range(1, horizon + 1):
            hi = high[t + k]
            lo = low[t + k]
            up_hit = hi >= up_price
            dn_hit = lo <= dn_price
            if up_hit and dn_hit:
                # Both same bar - conservative pick lower side (short)
                y[t] = 2
                label_type[t] = "ambig_dn"
                bar[t] = k
                done = True
                break
            if up_hit:
                y[t] = 1
                label_type[t] = "up"
                bar[t] = k
                done = True
                break
            if dn_hit:
                y[t] = 2
                label_type[t] = "dn"
                bar[t] = k
                done = True
                break
        if not done:
            y[t] = 0
            label_type[t] = "no_trend"
            bar[t] = horizon

    return pd.DataFrame({
        "close": close,
        "y_regime": y,
        "label_type": label_type,
        "first_touch_bar": bar,
    }, index=klines.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.006,
                        help="Barrier distance, fraction (0.006 = 0.6%%)")
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]]
    klines = klines.dropna()
    klines = klines[~klines.index.duplicated(keep="last")].sort_index()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    logger.info("Klines: n=%d, range %s ~ %s",
                len(klines), klines.index.min(), klines.index.max())

    logger.info("Building regime labels: threshold=%.2f%% H=%d",
                args.threshold * 100, args.horizon)
    labels = build_regime_labels(klines, args.threshold, args.horizon)
    labels = labels.dropna(subset=["y_regime"])
    labels["y_regime"] = labels["y_regime"].astype(int)
    logger.info("Labels: n=%d", len(labels))

    print(f"\n{'='*70}")
    print(f"Class distribution (threshold={args.threshold*100:.2f}%, H={args.horizon})")
    print(f"{'='*70}")
    counts = labels["y_regime"].value_counts().sort_index()
    label_names = {0: "no_trend", 1: "up_trend", 2: "down_trend"}
    for cls, cnt in counts.items():
        pct = cnt / len(labels) * 100
        print(f"  {cls} ({label_names[cls]:<10}): {cnt:>5} ({pct:>5.1f}%)")

    print(f"\nLabel type breakdown:")
    for lt, cnt in labels["label_type"].value_counts().items():
        print(f"  {lt}: {cnt} ({cnt/len(labels)*100:.1f}%)")

    print(f"\nFirst-touch bar distribution:")
    for bar_idx in range(1, args.horizon + 1):
        n = (labels["first_touch_bar"] == bar_idx).sum()
        pct = n / len(labels) * 100
        print(f"  bar+{bar_idx}h: {n} ({pct:.1f}%)")

    print(f"\nPer-month distribution (drift check):")
    idx = labels.copy()
    idx["month"] = idx.index.to_period("M")
    monthly = idx.groupby("month").agg(
        n=("y_regime", "count"),
        no_trend_pct=("y_regime", lambda x: (x == 0).mean() * 100),
        up_pct=("y_regime", lambda x: (x == 1).mean() * 100),
        dn_pct=("y_regime", lambda x: (x == 2).mean() * 100),
    )
    print(monthly.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (OUT_DIR /
                f"labels_regime_T{int(args.threshold*10000)}_H{args.horizon}.parquet")
    labels.to_parquet(out_path)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
