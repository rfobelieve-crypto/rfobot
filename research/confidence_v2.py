"""
E5: Confidence v2 — Magnitude + Regime composite confidence.

Components:
  1. mag_score: percentile of |pred| in expanding OOS distribution (0-100)
  2. regime_score: trailing IC in current regime (1.0 / 0.6 / 0.0)
  3. composite: mag_score × regime_score

Validation:
  - Quintile monotonicity: higher confidence → higher dir_acc & |actual_return|
  - Compare v1 (current percentile) vs v2 (composite) calibration

Usage:
    python -m research.confidence_v2
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from research.prediction_indicator_v2 import (
    load_data, select_features_burnin, walk_forward_predict,
    assign_direction_raw, calibrate_confidence_v2, assign_strength,
    BURN_IN_BARS, N_FOLDS, TARGET, DEADZONE,
)
from research.regime_analysis import assign_regime

ROOT = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════════════
# Component 1: Magnitude Score (expanding OOS percentile)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mag_score(pred: np.ndarray) -> np.ndarray:
    """
    Percentile of |pred| in expanding OOS distribution.
    Range: 0-100. Higher = model more opinionated.
    """
    abs_pred = np.abs(pred)
    mag_score = np.full(len(pred), np.nan)
    history = []

    for i in range(len(pred)):
        if np.isnan(pred[i]):
            continue
        history.append(abs_pred[i])
        if len(history) < 30:
            continue
        hist_arr = np.array(history[:-1])
        mag_score[i] = (hist_arr < abs_pred[i]).sum() / len(hist_arr) * 100

    return mag_score


# ═══════════════════════════════════════════════════════════════════════════
# Component 2: Regime Score (trailing IC → gate)
# ═══════════════════════════════════════════════════════════════════════════

def compute_regime_score(pred: np.ndarray, y: np.ndarray,
                         regime: np.ndarray) -> np.ndarray:
    """
    Trailing regime IC → regime_score.
    Uses ONLY past data (expanding window) to compute IC per regime.

    Score mapping:
      IC > 0.03  → 1.0 (high confidence in this regime)
      0 < IC ≤ 0.03 → 0.6 (moderate)
      IC ≤ 0     → 0.0 (abstain)
      insufficient data → 0.6 (neutral)
    """
    regime_score = np.full(len(pred), np.nan)

    # Track per-regime pred/actual history (trailing only)
    regime_history = {}  # regime_name → [(pred, actual), ...]

    for i in range(len(pred)):
        if np.isnan(pred[i]) or np.isnan(y[i]):
            continue

        r = regime[i]
        if r == "WARMUP":
            regime_score[i] = 0.6
            continue

        # Compute trailing IC for current regime BEFORE adding current bar
        if r in regime_history and len(regime_history[r]) >= 50:
            hist = regime_history[r]
            preds = np.array([h[0] for h in hist])
            actuals = np.array([h[1] for h in hist])
            ic, _ = spearmanr(preds, actuals)

            if ic > 0.03:
                regime_score[i] = 1.0
            elif ic > 0:
                regime_score[i] = 0.6
            else:
                regime_score[i] = 0.0
        else:
            # Not enough history for this regime yet
            regime_score[i] = 0.6

        # Add current bar to history
        if r not in regime_history:
            regime_history[r] = []
        regime_history[r].append((pred[i], y[i]))

    return regime_score


# ═══════════════════════════════════════════════════════════════════════════
# Composite Confidence v2
# ═══════════════════════════════════════════════════════════════════════════

def compute_confidence_v2(mag_score: np.ndarray,
                          regime_score: np.ndarray) -> np.ndarray:
    """
    Composite: mag_score × regime_score.
    Range: 0-100 (regime_score scales mag_score down).
    """
    confidence = mag_score * regime_score
    # Clip to 0-100
    confidence = np.clip(confidence, 0, 100)
    return confidence


# ═══════════════════════════════════════════════════════════════════════════
# Validation: Quintile Monotonicity
# ═══════════════════════════════════════════════════════════════════════════

def validate_confidence(label: str, confidence: np.ndarray,
                        pred: np.ndarray, y: np.ndarray,
                        direction: np.ndarray, n_bins: int = 5):
    """
    Validate confidence calibration:
    1. Bin into quintiles
    2. Check: higher confidence → higher dir_acc
    3. Check: higher confidence → higher |actual_return|
    """
    valid = ~np.isnan(confidence) & ~np.isnan(pred) & ~np.isnan(y)
    active = valid & (direction != "NEUTRAL")

    if active.sum() < 50:
        print(f"\n  [{label}] Insufficient active bars ({active.sum()})")
        return {}

    conf_a = confidence[active]
    pred_a = pred[active]
    y_a = y[active]
    dir_a = direction[active]

    dir_pred = np.where(dir_a == "UP", 1, -1)
    dir_actual = np.sign(y_a)
    dir_correct = (dir_pred == dir_actual)

    # Quintile bins
    try:
        bins = pd.qcut(conf_a, n_bins, labels=False, duplicates="drop")
    except ValueError:
        print(f"\n  [{label}] Cannot create quintiles (too few unique values)")
        return {}

    print(f"\n  [{label}] Quintile Calibration (n={active.sum()} active bars)")
    print(f"  {'Quintile':>10s}  {'Conf Range':>14s}  {'n':>5s}  {'Dir Acc':>8s}  {'|Ret|':>8s}  {'IC':>8s}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")

    quintile_dir_acc = []
    quintile_abs_ret = []
    quintile_ic = []

    for q in sorted(set(bins)):
        mask = bins == q
        n = mask.sum()
        if n < 10:
            continue

        da = dir_correct[mask].mean()
        abs_ret = np.abs(y_a[mask]).mean()
        ic_q, _ = spearmanr(pred_a[mask], y_a[mask]) if n >= 15 else (np.nan, np.nan)
        lo = conf_a[mask].min()
        hi = conf_a[mask].max()

        quintile_dir_acc.append(da)
        quintile_abs_ret.append(abs_ret)
        quintile_ic.append(ic_q)

        print(f"  {int(q):>10d}  {lo:6.1f}–{hi:6.1f}  {n:5d}  {da:8.1%}  {abs_ret*100:7.3f}%  {ic_q:+8.4f}" if not np.isnan(ic_q)
              else f"  {int(q):>10d}  {lo:6.1f}–{hi:6.1f}  {n:5d}  {da:8.1%}  {abs_ret*100:7.3f}%  {'N/A':>8s}")

    # Check monotonicity
    if len(quintile_dir_acc) >= 3:
        diffs_acc = np.diff(quintile_dir_acc)
        mono_acc = (diffs_acc > 0).sum() / len(diffs_acc)
        diffs_ret = np.diff(quintile_abs_ret)
        mono_ret = (diffs_ret > 0).sum() / len(diffs_ret)

        print(f"\n  Dir Acc monotonicity:  {mono_acc:.0%} ({(diffs_acc > 0).sum()}/{len(diffs_acc)} increasing)")
        print(f"  |Return| monotonicity: {mono_ret:.0%} ({(diffs_ret > 0).sum()}/{len(diffs_ret)} increasing)")

        # Overall IC
        ic_all, pval = spearmanr(pred_a, y_a)
        print(f"  Overall IC: {ic_all:+.4f} (p={pval:.3f})")

        return {
            "mono_acc": mono_acc,
            "mono_ret": mono_ret,
            "quintile_dir_acc": quintile_dir_acc,
            "ic": ic_all,
        }

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Strength tiers comparison
# ═══════════════════════════════════════════════════════════════════════════

def compare_strength_tiers(label: str, confidence: np.ndarray,
                           pred: np.ndarray, y: np.ndarray,
                           direction: np.ndarray):
    """Compare dir_acc and IC across Weak/Moderate/Strong tiers."""
    valid = ~np.isnan(confidence) & ~np.isnan(pred) & ~np.isnan(y)
    active = valid & (direction != "NEUTRAL")

    if active.sum() < 30:
        return

    conf_a = confidence[active]
    pred_a = pred[active]
    y_a = y[active]
    dir_a = direction[active]

    dir_pred = np.where(dir_a == "UP", 1, -1)
    dir_actual = np.sign(y_a)
    dir_correct = (dir_pred == dir_actual)

    print(f"\n  [{label}] Strength Tier Performance")
    print(f"  {'Tier':>10s}  {'n':>5s}  {'Dir Acc':>8s}  {'IC':>8s}  {'Mean |Ret|':>11s}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*11}")

    for lo, hi, name in [(0, 40, "Weak"), (40, 70, "Moderate"), (70, 101, "Strong")]:
        mask = (conf_a >= lo) & (conf_a < hi)
        n = mask.sum()
        if n < 10:
            print(f"  {name:>10s}  {n:5d}  {'(few)':>8s}")
            continue
        da = dir_correct[mask].mean()
        ic_t, _ = spearmanr(pred_a[mask], y_a[mask]) if n >= 15 else (np.nan, np.nan)
        abs_ret = np.abs(y_a[mask]).mean()
        ic_str = f"{ic_t:+8.4f}" if not np.isnan(ic_t) else f"{'N/A':>8s}"
        print(f"  {name:>10s}  {n:5d}  {da:8.1%}  {ic_str}  {abs_ret*100:10.3f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  E5: Confidence v2 — Magnitude × Regime Composite")
    print("=" * 70)

    # Reuse v2 pipeline
    df, feat_cols = load_data()
    feat_cols = select_features_burnin(df, feat_cols, top_k=40)

    print(f"\n{'='*70}")
    print(f"  Walk-Forward OOS Prediction")
    print(f"{'='*70}")
    oos_pred = walk_forward_predict(df, feat_cols)

    direction = assign_direction_raw(oos_pred)
    y = df[TARGET].values

    # Regime detection
    regime = assign_regime(df)

    # ── Component 1: Magnitude Score ──
    print(f"\n{'='*70}")
    print(f"  Component 1: Magnitude Score")
    print(f"{'='*70}")
    mag_score = compute_mag_score(oos_pred)
    valid = ~np.isnan(mag_score)
    print(f"  Valid mag_score bars: {valid.sum()}")
    print(f"  Mean: {np.nanmean(mag_score):.1f}  Median: {np.nanmedian(mag_score):.1f}")

    # ── Component 2: Regime Score ──
    print(f"\n{'='*70}")
    print(f"  Component 2: Regime Score (trailing IC)")
    print(f"{'='*70}")
    regime_score = compute_regime_score(oos_pred, y, regime)
    valid_rs = ~np.isnan(regime_score)
    for s in [0.0, 0.6, 1.0]:
        n = (regime_score == s).sum()
        print(f"  regime_score={s:.1f}: {n:5d} bars ({n/valid_rs.sum():.1%})")

    # ── Composite v2 ──
    print(f"\n{'='*70}")
    print(f"  Composite Confidence v2 = mag_score × regime_score")
    print(f"{'='*70}")
    conf_v2 = compute_confidence_v2(mag_score, regime_score)
    valid_v2 = ~np.isnan(conf_v2)
    print(f"  Valid bars: {valid_v2.sum()}")
    print(f"  Mean: {np.nanmean(conf_v2):.1f}  Median: {np.nanmedian(conf_v2):.1f}")
    print(f"  Zeros (abstained): {(conf_v2 == 0).sum()} ({(conf_v2[valid_v2] == 0).sum()/valid_v2.sum():.1%})")

    # ── v1 Confidence (current system: expanding percentile only) ──
    conf_v1 = calibrate_confidence_v2(oos_pred, BURN_IN_BARS)

    # ═══════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VALIDATION: v1 vs v2 Confidence Calibration")
    print(f"{'='*70}")

    # v1 quintile calibration
    v1_result = validate_confidence("v1 (mag only)", conf_v1, oos_pred, y, direction)

    # v2 quintile calibration
    v2_result = validate_confidence("v2 (mag×regime)", conf_v2, oos_pred, y, direction)

    # Magnitude-only (raw mag_score, no regime)
    mag_result = validate_confidence("mag_score (raw)", mag_score, oos_pred, y, direction)

    # ── Strength tier comparison ──
    print(f"\n{'='*70}")
    print(f"  STRENGTH TIER COMPARISON")
    print(f"{'='*70}")

    compare_strength_tiers("v1 (mag only)", conf_v1, oos_pred, y, direction)
    compare_strength_tiers("v2 (mag×regime)", conf_v2, oos_pred, y, direction)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'System':>20s}  {'Dir Acc Mono':>14s}  {'|Ret| Mono':>12s}  {'Overall IC':>12s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*12}  {'-'*12}")

    for name, res in [("v1 (mag only)", v1_result),
                       ("v2 (mag×regime)", v2_result),
                       ("mag_score (raw)", mag_result)]:
        if res:
            print(f"  {name:>20s}  {res['mono_acc']:14.0%}  {res['mono_ret']:12.0%}  {res['ic']:+12.4f}")
        else:
            print(f"  {name:>20s}  {'N/A':>14s}  {'N/A':>12s}  {'N/A':>12s}")

    # ── Blueprint success criteria check ──
    print(f"\n  Success criteria:")
    if v2_result:
        mono_ok = v2_result.get("mono_acc", 0) >= 0.5
        print(f"    Confidence monotonicity (dir_acc): {'PASS' if mono_ok else 'FAIL'} ({v2_result.get('mono_acc', 0):.0%})")
        # Check Strong > Moderate > Weak
    print(f"    (See strength tier table above for Strong > Moderate > Weak check)")


def main():
    run()


if __name__ == "__main__":
    main()
