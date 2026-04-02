"""
E3+E4: Regime-Aware Analysis & Selective Abstention.

E3: 依 trailing-only regime 分割 OOS 表現，找出拖累 IC 的 regime。
E4: 在負 IC regime 中 abstain，驗證精度提升。

Usage:
    python -m research.regime_analysis
    python -m research.regime_analysis --save
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── 匯入 v2 共用模組 ──────────────────────────────────────────────────
from research.prediction_indicator_v2 import (
    load_data, select_features_burnin, walk_forward_predict,
    assign_direction_raw, calibrate_confidence_v2, assign_strength,
    compute_bull_bear_power,
    BURN_IN_BARS, N_FOLDS, TARGET, DEADZONE,
    STRONG_THRESHOLD, MODERATE_THRESHOLD,
)

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "ml_data"


# ═══════════════════════════════════════════════════════════════════════════
# E3: Trailing-Only Regime Detection (no lookahead)
# ═══════════════════════════════════════════════════════════════════════════

def assign_regime(df: pd.DataFrame) -> np.ndarray:
    """
    Trailing-only regime classification.
    Uses only past data (expanding percentile rank).

    Returns: string array of regime labels per bar.
    """
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    # 24h trailing metrics
    ret_24h = close.pct_change(24)
    vol_24h = log_ret.rolling(24).std()

    # Expanding percentile rank — only uses past data
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)

    # Classification
    regime = np.full(len(df), "CHOPPY", dtype=object)
    regime[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
    regime[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"

    # First 168 bars (1 week) don't have enough data for percentile
    regime[:168] = "WARMUP"

    print(f"\n  Regime distribution:")
    for r in ["WARMUP", "TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        n = (regime == r).sum()
        print(f"    {r:18s}: {n:5d} ({n/len(df):.1%})")

    return regime


# ═══════════════════════════════════════════════════════════════════════════
# E3: Regime-Sliced OOS Analysis
# ═══════════════════════════════════════════════════════════════════════════

def regime_sliced_analysis(df, pred, direction, regime):
    """Compute per-regime OOS metrics."""
    y = df[TARGET].values
    valid = ~np.isnan(pred) & ~np.isnan(y) & (regime != "WARMUP")

    print(f"\n{'='*70}")
    print(f"  E3: Regime-Sliced OOS Analysis")
    print(f"{'='*70}")

    results = {}
    for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        mask = valid & (regime == r)
        n = mask.sum()
        if n < 30:
            print(f"\n  {r}: n={n} (too few, skip)")
            results[r] = {"n": n, "ic": np.nan, "dir_acc": np.nan}
            continue

        ic_r, pval_r = spearmanr(y[mask], pred[mask])

        # Direction accuracy (active signals only)
        active = mask & (direction != "NEUTRAL")
        if active.sum() > 10:
            dir_actual = np.sign(y[active])
            dir_pred = np.where(direction[active] == "UP", 1, -1)
            dir_acc = (dir_pred == dir_actual).mean()
        else:
            dir_acc = np.nan

        # Mean actual return
        mean_ret = y[mask].mean()
        std_ret = y[mask].std()

        # Mean |pred|
        mean_abs_pred = np.abs(pred[mask]).mean()

        sig = "***" if pval_r < 0.01 else "* " if pval_r < 0.05 else "  "
        print(f"\n  {r}  (n={n})")
        print(f"    IC         = {ic_r:+.4f} {sig}  (p={pval_r:.3f})")
        print(f"    Dir acc    = {dir_acc:.1%}" if not np.isnan(dir_acc) else "    Dir acc    = N/A")
        print(f"    Mean ret   = {mean_ret*100:+.4f}%")
        print(f"    Ret std    = {std_ret*100:.4f}%")
        print(f"    Mean |pred|= {mean_abs_pred*100:.4f}%")

        results[r] = {"n": n, "ic": ic_r, "pval": pval_r,
                       "dir_acc": dir_acc, "mean_ret": mean_ret}

    # ── Per-fold × per-regime breakdown ──
    print(f"\n  Fold × Regime IC matrix:")
    n_total = len(df)
    oos_data = n_total - BURN_IN_BARS
    fold_size = oos_data // N_FOLDS

    header = f"  {'Fold':>6s}"
    for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        header += f"  {r:>16s}"
    print(header)

    for k in range(N_FOLDS):
        fs = BURN_IN_BARS + fold_size * k
        fe = min(BURN_IN_BARS + fold_size * (k + 1), n_total)
        fold_mask = np.zeros(n_total, dtype=bool)
        fold_mask[fs:fe] = True
        fold_mask = fold_mask & valid

        row = f"  {k+1:>6d}"
        for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
            m = fold_mask & (regime == r)
            if m.sum() < 10:
                row += f"  {'n<10':>16s}"
            else:
                ic_k, _ = spearmanr(y[m], pred[m])
                row += f"  {ic_k:+16.4f}"
        print(row)

    # ── Monthly × regime breakdown ──
    print(f"\n  Monthly × Regime IC matrix:")
    df_oos = df[valid].copy()
    df_oos["pred"] = pred[valid]
    df_oos["regime"] = regime[valid]
    df_oos["month"] = df_oos.index.strftime("%Y-%m")

    months = sorted(df_oos["month"].unique())
    header = f"  {'Month':>8s}"
    for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        header += f"  {r:>16s}"
    print(header)

    for mo in months:
        row = f"  {mo:>8s}"
        for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
            sub = df_oos[(df_oos["month"] == mo) & (df_oos["regime"] == r)]
            if len(sub) < 10:
                row += f"  {'n<10':>16s}"
            else:
                ic_m, _ = spearmanr(sub[TARGET], sub["pred"])
                row += f"  {ic_m:+16.4f}"
        print(row)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# E4: Selective Abstention
# ═══════════════════════════════════════════════════════════════════════════

def apply_abstention(df, pred, direction, confidence, regime,
                     regime_results: dict):
    """
    Apply regime-aware abstention.

    Strategy:
    1. Abstain in regimes with IC < 0
    2. Abstain when |pred| too small (below 25th percentile of OOS |pred|)
    3. Measure improvement in emitted signals
    """
    y = df[TARGET].values
    valid = ~np.isnan(pred) & ~np.isnan(y) & (regime != "WARMUP")

    print(f"\n{'='*70}")
    print(f"  E4: Selective Abstention")
    print(f"{'='*70}")

    # ── Layer 1: Regime gate ──
    bad_regimes = [r for r, v in regime_results.items()
                   if not np.isnan(v.get("ic", np.nan)) and v["ic"] < 0]
    good_regimes = [r for r, v in regime_results.items()
                    if not np.isnan(v.get("ic", np.nan)) and v["ic"] >= 0]

    print(f"\n  [Layer 1] Regime Gate")
    print(f"    Bad regimes (IC<0, abstain): {bad_regimes}")
    print(f"    Good regimes (IC>=0, keep):  {good_regimes}")

    regime_pass = valid.copy()
    for r in bad_regimes:
        regime_pass = regime_pass & (regime != r)

    n_after_regime = regime_pass.sum()
    print(f"    Before: {valid.sum()} bars")
    print(f"    After:  {n_after_regime} bars ({n_after_regime/valid.sum():.1%} retained)")

    if n_after_regime >= 50:
        ic_after, pval = spearmanr(y[regime_pass], pred[regime_pass])
        active = regime_pass & (direction != "NEUTRAL")
        if active.sum() > 0:
            dir_acc = (np.where(direction[active] == "UP", 1, -1) == np.sign(y[active])).mean()
        else:
            dir_acc = 0
        print(f"    IC after regime gate: {ic_after:+.4f}  (p={pval:.3f})")
        print(f"    Dir acc:             {dir_acc:.1%}  (active={active.sum()})")

    # ── Layer 2: Prediction magnitude gate ──
    print(f"\n  [Layer 2] Prediction Magnitude Gate")
    oos_abs_pred = np.abs(pred[valid & ~np.isnan(pred)])
    thresholds = [25, 50, 75]

    for pct in thresholds:
        thr = np.percentile(oos_abs_pred, pct)
        mag_pass = regime_pass & (np.abs(pred) >= thr)
        n_mag = mag_pass.sum()

        if n_mag < 30:
            print(f"    |pred| >= {pct}th pct ({thr*100:.4f}%): n={n_mag} (too few)")
            continue

        ic_mag, pval_mag = spearmanr(y[mag_pass], pred[mag_pass])
        active = mag_pass & (direction != "NEUTRAL")
        if active.sum() > 0:
            da_mag = (np.where(direction[active] == "UP", 1, -1) == np.sign(y[active])).mean()
        else:
            da_mag = 0
        sig = "***" if pval_mag < 0.01 else "* " if pval_mag < 0.05 else "  "
        print(f"    |pred| >= {pct}th pct ({thr*100:.4f}%): "
              f"n={n_mag} ({n_mag/valid.sum():.1%})  "
              f"IC={ic_mag:+.4f} {sig}  dir={da_mag:.1%}")

    # ── Composite abstention: Regime + 50th pct magnitude ──
    print(f"\n  [Composite] Regime Gate + 50th Percentile Magnitude")
    thr_50 = np.percentile(oos_abs_pred, 50)
    composite = regime_pass & (np.abs(pred) >= thr_50)
    n_comp = composite.sum()

    if n_comp >= 30:
        ic_comp, pval_comp = spearmanr(y[composite], pred[composite])
        active_comp = composite & (direction != "NEUTRAL")
        if active_comp.sum() > 0:
            da_comp = (np.where(direction[active_comp] == "UP", 1, -1) == np.sign(y[active_comp])).mean()
        else:
            da_comp = 0
        print(f"    Bars:    {n_comp} ({n_comp/valid.sum():.1%} of total OOS)")
        print(f"    IC:      {ic_comp:+.4f}  (p={pval_comp:.3f})")
        print(f"    Dir acc: {da_comp:.1%}  (active={active_comp.sum()})")

    # ── Composite: Regime + 75th pct ──
    print(f"\n  [Composite] Regime Gate + 75th Percentile Magnitude")
    thr_75 = np.percentile(oos_abs_pred, 75)
    composite75 = regime_pass & (np.abs(pred) >= thr_75)
    n_comp75 = composite75.sum()

    if n_comp75 >= 30:
        ic_comp75, pval_comp75 = spearmanr(y[composite75], pred[composite75])
        active_75 = composite75 & (direction != "NEUTRAL")
        if active_75.sum() > 0:
            da_75 = (np.where(direction[active_75] == "UP", 1, -1) == np.sign(y[active_75])).mean()
        else:
            da_75 = 0
        print(f"    Bars:    {n_comp75} ({n_comp75/valid.sum():.1%} of total OOS)")
        print(f"    IC:      {ic_comp75:+.4f}  (p={pval_comp75:.3f})")
        print(f"    Dir acc: {da_75:.1%}  (active={active_75.sum()})")

    # ── Confidence tier analysis AFTER abstention ──
    print(f"\n  [After Abstention] Confidence Tier Analysis (Regime + 50th pct)")
    if n_comp >= 30:
        active_comp = composite & (direction != "NEUTRAL")
        if active_comp.sum() > 0:
            conf_a = confidence[active_comp]
            dir_actual = np.sign(y[active_comp])
            dir_pred = np.where(direction[active_comp] == "UP", 1, -1)
            dir_correct = (dir_pred == dir_actual)

            for lo, hi, label in [(0, 40, "Weak"), (40, 70, "Moderate"), (70, 101, "Strong")]:
                mask = (conf_a >= lo) & (conf_a < hi) & ~np.isnan(conf_a)
                if mask.sum() >= 5:
                    acc = dir_correct[mask].mean()
                    ic_b, _ = spearmanr(y[active_comp][mask], pred[active_comp][mask]) if mask.sum() >= 10 else (np.nan, np.nan)
                    print(f"    {label:10s}  n={mask.sum():5d}  "
                          f"dir_acc={acc:.1%}  IC={ic_b:+.4f}" if not np.isnan(ic_b) else
                          f"    {label:10s}  n={mask.sum():5d}  dir_acc={acc:.1%}")
                else:
                    print(f"    {label:10s}  n={mask.sum():5d}  (too few)")

    # ── Monthly performance AFTER abstention ──
    print(f"\n  [After Abstention] Monthly IC (Regime + 50th pct)")
    if n_comp >= 30:
        idx_comp = composite
        df_comp = df[idx_comp].copy()
        df_comp["pred"] = pred[idx_comp]
        df_comp["month"] = df_comp.index.strftime("%Y-%m")
        for mo, grp in df_comp.groupby("month"):
            if len(grp) < 15:
                ic_m_str = f"n={len(grp):3d} (few)"
            else:
                ic_m, p_m = spearmanr(grp[TARGET], grp["pred"])
                sig = "***" if p_m < 0.01 else "* " if p_m < 0.05 else "  "
                ic_m_str = f"IC={ic_m:+.4f} {sig}  n={len(grp)}"
            print(f"    {mo}  {ic_m_str}")

    # ── Summary comparison ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Before vs After Abstention")
    print(f"{'='*70}")

    # Before
    ic_before, _ = spearmanr(y[valid], pred[valid])
    active_before = valid & (direction != "NEUTRAL")
    if active_before.sum() > 0:
        da_before = (np.where(direction[active_before] == "UP", 1, -1) == np.sign(y[active_before])).mean()
    else:
        da_before = 0

    print(f"\n  {'Metric':20s}  {'Before':>12s}  {'Regime Only':>12s}  {'Regime+50pct':>12s}  {'Regime+75pct':>12s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    # IC
    row_ic = f"  {'OOS IC':20s}  {ic_before:+12.4f}"
    if n_after_regime >= 50:
        ic_r, _ = spearmanr(y[regime_pass], pred[regime_pass])
        row_ic += f"  {ic_r:+12.4f}"
    else:
        row_ic += f"  {'N/A':>12s}"
    if n_comp >= 30:
        row_ic += f"  {ic_comp:+12.4f}"
    else:
        row_ic += f"  {'N/A':>12s}"
    if n_comp75 >= 30:
        row_ic += f"  {ic_comp75:+12.4f}"
    else:
        row_ic += f"  {'N/A':>12s}"
    print(row_ic)

    # Dir acc
    row_da = f"  {'Dir Accuracy':20s}  {da_before:12.1%}"
    if n_after_regime >= 50:
        ar = regime_pass & (direction != "NEUTRAL")
        da_r = (np.where(direction[ar] == "UP", 1, -1) == np.sign(y[ar])).mean() if ar.sum() > 0 else 0
        row_da += f"  {da_r:12.1%}"
    else:
        row_da += f"  {'N/A':>12s}"
    if n_comp >= 30:
        row_da += f"  {da_comp:12.1%}"
    else:
        row_da += f"  {'N/A':>12s}"
    if n_comp75 >= 30:
        row_da += f"  {da_75:12.1%}"
    else:
        row_da += f"  {'N/A':>12s}"
    print(row_da)

    # N bars
    row_n = f"  {'Bars':20s}  {valid.sum():12d}"
    row_n += f"  {n_after_regime:12d}" if n_after_regime >= 50 else f"  {'N/A':>12s}"
    row_n += f"  {n_comp:12d}" if n_comp >= 30 else f"  {'N/A':>12s}"
    row_n += f"  {n_comp75:12d}" if n_comp75 >= 30 else f"  {'N/A':>12s}"
    print(row_n)

    # Retention
    row_pct = f"  {'Retention':20s}  {'100.0%':>12s}"
    row_pct += f"  {n_after_regime/valid.sum():12.1%}" if n_after_regime >= 50 else f"  {'N/A':>12s}"
    row_pct += f"  {n_comp/valid.sum():12.1%}" if n_comp >= 30 else f"  {'N/A':>12s}"
    row_pct += f"  {n_comp75/valid.sum():12.1%}" if n_comp75 >= 30 else f"  {'N/A':>12s}"
    print(row_pct)

    # ── Fold-level analysis after best abstention ──
    best_composite = composite  # default to regime+50pct
    best_label = "Regime+50pct"
    print(f"\n  Per-fold IC ({best_label}):")
    n_total = len(df)
    oos_data = n_total - BURN_IN_BARS
    fold_size = oos_data // N_FOLDS
    fold_ics_before = []
    fold_ics_after = []

    for k in range(N_FOLDS):
        fs = BURN_IN_BARS + fold_size * k
        fe = min(BURN_IN_BARS + fold_size * (k + 1), n_total)
        fold_mask = np.zeros(n_total, dtype=bool)
        fold_mask[fs:fe] = True

        # Before
        fb = fold_mask & valid
        if fb.sum() >= 20:
            ic_fb, _ = spearmanr(y[fb], pred[fb])
            fold_ics_before.append(ic_fb)
        else:
            ic_fb = np.nan
            fold_ics_before.append(np.nan)

        # After
        fa = fold_mask & best_composite
        if fa.sum() >= 20:
            ic_fa, _ = spearmanr(y[fa], pred[fa])
            fold_ics_after.append(ic_fa)
        else:
            ic_fa = np.nan
            fold_ics_after.append(np.nan)

        dates = df.index[fold_mask & valid]
        d0 = dates[0].strftime('%m/%d') if len(dates) > 0 else "?"
        d1 = dates[-1].strftime('%m/%d') if len(dates) > 0 else "?"

        delta = ""
        if not np.isnan(ic_fb) and not np.isnan(ic_fa):
            d = ic_fa - ic_fb
            delta = f"  Δ={d:+.4f}"

        ic_fa_str = f"{ic_fa:+.4f}" if not np.isnan(ic_fa) else "    N/A"
        print(f"    Fold {k+1}: Before IC={ic_fb:+.4f}  After IC={ic_fa_str}{delta}  "
              f"({d0}~{d1}  n_before={fb.sum()}  n_after={fa.sum()})")

    fib = np.array([x for x in fold_ics_before if not np.isnan(x)])
    fia = np.array([x for x in fold_ics_after if not np.isnan(x)])
    if len(fib) > 1 and len(fia) > 1:
        icir_b = fib.mean() / fib.std() if fib.std() > 0 else 0
        icir_a = fia.mean() / fia.std() if fia.std() > 0 else 0
        neg_b = (fib < 0).sum()
        neg_a = (fia < 0).sum()
        print(f"\n    ICIR:  Before={icir_b:+.2f}  After={icir_a:+.2f}")
        print(f"    Neg folds: Before={neg_b}/{len(fib)}  After={neg_a}/{len(fia)}")

    return {
        "regime_pass": regime_pass,
        "composite_50": composite,
        "composite_75": composite75,
        "bad_regimes": bad_regimes,
        "good_regimes": good_regimes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(save: bool = False):
    print("=" * 70)
    print("  BTC Market Intelligence Indicator — E3+E4")
    print("  E3: Regime-Sliced OOS Analysis")
    print("  E4: Selective Abstention (Regime + Magnitude)")
    print("=" * 70)

    # Reuse v2 pipeline
    df, feat_cols = load_data()
    feat_cols = select_features_burnin(df, feat_cols, top_k=40)

    print(f"\n{'='*70}")
    print(f"  Walk-Forward OOS Prediction (reusing v2 pipeline)")
    print(f"{'='*70}")
    oos_pred = walk_forward_predict(df, feat_cols)

    direction = assign_direction_raw(oos_pred)
    confidence = calibrate_confidence_v2(oos_pred, BURN_IN_BARS)

    # E3: Regime detection
    regime = assign_regime(df)

    # E3: Regime-sliced analysis
    regime_results = regime_sliced_analysis(df, oos_pred, direction, regime)

    # E4: Selective abstention
    abstention_results = apply_abstention(
        df, oos_pred, direction, confidence, regime, regime_results
    )

    if save:
        out = pd.DataFrame(index=df.index)
        out["pred_return_4h"] = oos_pred
        out["pred_direction"] = direction
        out["confidence_score"] = confidence
        out["strength_score"] = assign_strength(confidence)
        out["regime"] = regime
        out["regime_pass"] = abstention_results["regime_pass"]
        out["actual_return_4h"] = df[TARGET].values
        out["close"] = df["close"].values
        out_path = OUT_DIR / "BTC_USD_indicator_4h_e3e4.parquet"
        out.to_parquet(out_path, index=True)
        print(f"\n  Saved: {out_path}")

    return abstention_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=args.save)


if __name__ == "__main__":
    main()
