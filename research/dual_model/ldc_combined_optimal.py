"""
Run combined "best of each parameter" LDC configs to verify they stack.

Coordinate-descent winners individually:
  kernel_h = 10
  kernel_r = 1.0 (max cum)  or  32.0 (best risk-adjusted)
  ema_period = 100  or  OFF
  k, regime, vol, features: keep baseline

Test combinations and compare to baseline + best single tweak.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.ldc_param_grid_search import (
    load_klines, evaluate_config, print_row, DEFAULT_PARAMS, with_param,
)


CONFIGS = [
    ("BASELINE (current production)", DEFAULT_PARAMS),
    # Single-param winners
    ("only kernel_h=10", with_param(kernel_h=10)),
    ("only kernel_r=1.0", with_param(kernel_r=1.0)),
    ("only kernel_r=32.0", with_param(kernel_r=32.0)),
    ("only EMA=100", with_param(ema_period=100)),
    ("only EMA=OFF", with_param(use_ema_filter=False)),
    # Two-param combinations
    ("h=10 + r=32", with_param(kernel_h=10, kernel_r=32.0)),
    ("h=10 + EMA=100", with_param(kernel_h=10, ema_period=100)),
    ("h=10 + EMA=OFF", with_param(kernel_h=10, use_ema_filter=False)),
    ("r=32 + EMA=100", with_param(kernel_r=32.0, ema_period=100)),
    ("r=32 + EMA=OFF", with_param(kernel_r=32.0, use_ema_filter=False)),
    # All three
    ("h=10 + r=32 + EMA=100 (low-risk combo)",
        with_param(kernel_h=10, kernel_r=32.0, ema_period=100)),
    ("h=10 + r=32 + EMA=OFF (high-cum combo)",
        with_param(kernel_h=10, kernel_r=32.0, use_ema_filter=False)),
    ("h=10 + r=1.0 + EMA=OFF (max cum aggressive)",
        with_param(kernel_h=10, kernel_r=1.0, use_ema_filter=False)),
]


def main():
    klines = load_klines()
    print(f"\nKlines: {len(klines)} bars, exit dynamic_cross + min hold 4h, cost 5bp\n")
    print("=" * 145)
    print(f"  {'config':<50} {'n':>4} {'WR':>6} {'net':>10} {'cum%(1x/3x)':>22} {'MDD%(1x/3x)':>22} {'PF':>5} {'hold':>5}")
    print("=" * 145)
    results = []
    for label, params in CONFIGS:
        r = evaluate_config(klines, params, label)
        print_row(r)
        results.append(r)

    # Rank by Sharpe-like: cum_1x / |MDD_1x|
    print()
    print("=" * 100)
    print("RANKED by risk-adjusted (cum/|MDD|):")
    print("=" * 100)
    ranked = [r for r in results if r.get("n", 0) > 0]
    ranked.sort(key=lambda r: r["cum_1x"] / max(abs(r["mdd_1x"]), 1), reverse=True)
    for i, r in enumerate(ranked[:5], 1):
        ratio = r["cum_1x"] / max(abs(r["mdd_1x"]), 1)
        print(f"  #{i}: {r['label']:<48} cum/|MDD| = {ratio:.2f}  "
              f"cum_3x={r['cum_3x']:+.1f}% MDD_3x={r['mdd_3x']:+.1f}%")


if __name__ == "__main__":
    main()
