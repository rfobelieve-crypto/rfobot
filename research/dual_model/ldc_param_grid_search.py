"""
LDC parameter grid search for BTC 1h (user 2026-05-12).

Coordinate-descent style — fix all params at jdehorty defaults, sweep one
at a time.  Identify best per parameter.  Final combined run uses the
winning combo.

Parameters tested:
    1. neighbors_count k:  5, 8, 12, 16
    2. kernel_h:           4, 6, 8, 12
    3. kernel_r:           1.0, 4.0, 8.0, 16.0
    4. ema_period:         100, 150, 200, 300, OFF
    5. feature_spec:       jdehorty default vs faster/slower/2feat/4feat
    6. use_volatility_filter, use_regime_filter on/off
    7. regime_threshold:   -0.3, -0.1, 0.0, +0.1

Backtest: dynamic_cross exit + min hold 4h (production setting).
Cost: maker 5 bps.

Metrics reported: n trades, WR, net bps, cum %, MDD %, PF, hold hours.
"""
from __future__ import annotations
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.ldc_python import compute_ldc_signals
from research.dual_model.ldc_swing_backtest import simulate_swing

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
MAKER_COST = 0.0005

# Default = jdehorty (user's current settings)
DEFAULT_PARAMS = dict(
    neighbors_count=8,
    max_bars_back=2000,
    feature_spec=(("rsi", 14, 1), ("wt", 10, 11), ("cci", 20, 1)),
    use_volatility_filter=True,
    use_regime_filter=True,
    regime_threshold=-0.1,
    use_ema_filter=True,
    ema_period=200,
    kernel_h=8,
    kernel_r=8.0,
    kernel_x=25,
    kernel_lag=2,
)


def load_klines():
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    return k


def evaluate_config(klines: pd.DataFrame, params: dict, label: str) -> dict:
    """Compute LDC with params, run swing backtest, return summary."""
    t0 = time.time()
    ldc = compute_ldc_signals(klines, verbose=False, **params)
    t_ldc = time.time() - t0
    n_long_cand = int(ldc["start_long_trade"].sum())
    n_short_cand = int(ldc["start_short_trade"].sum())

    if n_long_cand + n_short_cand == 0:
        return {
            "label": label, "n": 0, "wr": np.nan,
            "net_bps": np.nan, "cum_1x": np.nan, "cum_3x": np.nan,
            "mdd_1x": np.nan, "mdd_3x": np.nan,
            "pf": np.nan, "hold_h": np.nan,
            "long_cand": n_long_cand, "short_cand": n_short_cand,
            "ldc_secs": round(t_ldc, 1),
        }

    trades = simulate_swing(
        ldc, klines,
        exit_mode="dynamic_cross",
        min_hold_bars=4,
        cost=MAKER_COST,
    )
    if trades.empty:
        return {"label": label, "n": 0, "long_cand": n_long_cand,
                "short_cand": n_short_cand, "ldc_secs": round(t_ldc, 1)}
    n = len(trades)
    wr = trades["win"].mean()
    net = trades["net_pct"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    return {
        "label": label, "n": n, "wr": wr,
        "net_bps": net.mean() * 10000,
        "cum_1x": cum[-1] * 100, "cum_3x": cum[-1] * 100 * 3,
        "mdd_1x": mdd, "mdd_3x": mdd * 3,
        "pf": pf, "hold_h": trades["held_hours"].mean(),
        "long_cand": n_long_cand, "short_cand": n_short_cand,
        "ldc_secs": round(t_ldc, 1),
    }


def print_row(r: dict):
    if r.get("n", 0) == 0:
        print(f"  {r['label']:<50} n=0  cand L={r.get('long_cand',0)} S={r.get('short_cand',0)}  ldc={r.get('ldc_secs',0)}s")
        return
    cum_1x_str = f"{r['cum_1x']:+5.1f}"
    cum_3x_str = f"{r['cum_3x']:+5.1f}"
    print(
        f"  {r['label']:<50} n={r['n']:>3} WR={r['wr']*100:>5.1f}% "
        f"net={r['net_bps']:>+6.1f}bp "
        f"cum={cum_1x_str}%/{cum_3x_str}%(3x) "
        f"MDD={r['mdd_1x']:>+5.1f}%/{r['mdd_3x']:>+5.1f}%(3x) "
        f"PF={r['pf']:>4.2f} hold={r['hold_h']:>3.0f}h"
    )


def with_param(**overrides):
    p = dict(DEFAULT_PARAMS)
    p.update(overrides)
    return p


def main():
    klines = load_klines()
    print(f"\nKlines: {len(klines)} bars ({klines.index.min()} ~ {klines.index.max()})")
    print(f"Exit: dynamic_cross + min hold 4h | Cost: 5bp maker\n")

    print("=" * 140)
    print("BASELINE (jdehorty default — current production)")
    print("=" * 140)
    base = evaluate_config(klines, DEFAULT_PARAMS, "BASELINE jdehorty default")
    print_row(base)

    # 1) neighbors_count
    print()
    print("=" * 140)
    print("1) Neighbors count (k) sweep — current = 8")
    print("=" * 140)
    for k in [3, 5, 8, 12, 16, 20]:
        r = evaluate_config(klines, with_param(neighbors_count=k), f"k = {k}")
        print_row(r)

    # 2) kernel_h
    print()
    print("=" * 140)
    print("2) Kernel lookback (h) sweep — current = 8")
    print("=" * 140)
    for h in [4, 6, 8, 10, 12, 16]:
        r = evaluate_config(klines, with_param(kernel_h=h), f"kernel_h = {h}")
        print_row(r)

    # 3) kernel_r
    print()
    print("=" * 140)
    print("3) Kernel relative_weight (r) sweep — current = 8.0")
    print("=" * 140)
    for r_val in [1.0, 4.0, 8.0, 16.0, 32.0]:
        r = evaluate_config(klines, with_param(kernel_r=r_val), f"kernel_r = {r_val:.1f}")
        print_row(r)

    # 4) ema_period
    print()
    print("=" * 140)
    print("4) EMA filter period sweep — current = 200, OFF available")
    print("=" * 140)
    for ep in [50, 100, 150, 200, 300, None]:
        if ep is None:
            r = evaluate_config(klines, with_param(use_ema_filter=False), "EMA filter OFF")
        else:
            r = evaluate_config(klines, with_param(ema_period=ep), f"EMA period = {ep}")
        print_row(r)

    # 5) feature_spec variants
    print()
    print("=" * 140)
    print("5) Feature combo sweep — current = (rsi 14/1, wt 10/11, cci 20/1)")
    print("=" * 140)
    feature_variants = [
        ("3 jdehorty default", (("rsi", 14, 1), ("wt", 10, 11), ("cci", 20, 1))),
        ("3 faster periods",   (("rsi", 7, 1), ("wt", 5, 8), ("cci", 14, 1))),
        ("3 slower periods",   (("rsi", 21, 1), ("wt", 14, 14), ("cci", 28, 1))),
        ("2 features (rsi+wt)", (("rsi", 14, 1), ("wt", 10, 11))),
        ("2 features (rsi+cci)", (("rsi", 14, 1), ("cci", 20, 1))),
        ("4 features (+ rsi9)", (("rsi", 14, 1), ("wt", 10, 11), ("cci", 20, 1), ("rsi", 9, 1))),
        ("5 features",         (("rsi", 14, 1), ("wt", 10, 11), ("cci", 20, 1), ("rsi", 9, 1), ("cci", 14, 1))),
    ]
    for name, fs in feature_variants:
        r = evaluate_config(klines, with_param(feature_spec=fs), name)
        print_row(r)

    # 6) regime/volatility filter
    print()
    print("=" * 140)
    print("6) Volatility & Regime filter on/off — current = both ON")
    print("=" * 140)
    for vol, reg, label in [
        (True, True, "vol=ON  reg=ON (current)"),
        (True, False, "vol=ON  reg=OFF"),
        (False, True, "vol=OFF reg=ON"),
        (False, False, "vol=OFF reg=OFF"),
    ]:
        r = evaluate_config(klines, with_param(
            use_volatility_filter=vol, use_regime_filter=reg), label)
        print_row(r)

    # 7) regime threshold
    print()
    print("=" * 140)
    print("7) Regime filter threshold sweep — current = -0.1")
    print("=" * 140)
    for th in [-0.5, -0.3, -0.1, 0.0, +0.1, +0.3]:
        r = evaluate_config(klines, with_param(regime_threshold=th), f"regime thr = {th:+.2f}")
        print_row(r)


if __name__ == "__main__":
    main()
