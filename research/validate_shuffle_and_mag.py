"""
Two critical validations for direction-regression:

V5 — Label-shuffle leakage probe:
    Permute y_path_ret_4h, retrain walk-forward, re-evaluate.
    A clean model should collapse to:
        IC ~ 0, AUC ~ 0.5, Strong WR ~ 50%
    Any residual signal would imply feature leakage.

MAG — Adding magnitude gate on top of regression direction:
    Load existing production magnitude predictions (via the current mag
    model, which is UNTOUCHED) and combine:
        Strong = (top-5% rolling |pred_ret|)  AND  (mag_pct >= X)
    Sweep X in {0.0, 0.5, 0.6, 0.7, 0.8} to see if Mag filter adds WR.
    Mistake-log note: check_no_mag_gate showed Mag had no real value for
    the Init head in true OOS — but this is a DIFFERENT direction head
    (regression), so we re-test rather than assume.

Usage:
    python -m research.validate_shuffle_and_mag
"""
from __future__ import annotations

import sys
import logging
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, walk_forward_splits, RESULTS_DIR,
)
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import (
    FULL_DIRECTION, filter_available,
)
from research.dual_model.train_direction_reg_4h import BASE_PARAMS, OBJECTIVE_MAP

import xgboost as xgb

logging.getLogger().setLevel(logging.WARNING)

ROLL = 500
STRONG_FRAC = 0.05


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def rolling_top5(score, y):
    q_up = pd.Series(score).rolling(ROLL, min_periods=100).quantile(
        1 - STRONG_FRAC / 2).values
    q_dn = pd.Series(score).rolling(ROLL, min_periods=100).quantile(
        STRONG_FRAC / 2).values
    valid = ~np.isnan(q_up)
    long_fire = valid & (score >= q_up)
    short_fire = valid & (score <= q_dn)
    fire = long_fire | short_fire
    win = (long_fire & (y > 0)) | (short_fire & (y < 0))
    return fire, win


# ────────────────────────────────────────────────────────────────────
# V5: Label shuffle
# ────────────────────────────────────────────────────────────────────
def v5_label_shuffle():
    print("\n" + "=" * 76)
    print("  V5: LABEL-SHUFFLE LEAKAGE PROBE")
    print("=" * 76)
    print("  Expected on clean model: IC~0  AUC~0.5  WR~50%")

    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    features = filter_available(FULL_DIRECTION, list(df.columns))

    mask = df["y_path_ret_4h"].notna()
    clean_y = df.loc[mask, "y_path_ret_4h"].values.astype(float)

    rng = np.random.default_rng(20260415)
    shuffled = rng.permutation(clean_y)
    df_shuf = df.copy()
    df_shuf.loc[mask, "y_path_ret_4h"] = shuffled

    splits = walk_forward_splits(len(df_shuf), initial_train=288,
                                  test_size=48, step=48)
    params = BASE_PARAMS.copy()
    params.update(OBJECTIVE_MAP["mse"])
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = 150   # fewer trees = faster, fine for sanity

    all_pred, all_y = [], []
    for fold_i, (tr, te) in enumerate(splits):
        tr_df = df_shuf.iloc[tr]
        te_df = df_shuf.iloc[te]
        tr_m = tr_df["y_path_ret_4h"].notna()
        te_m = te_df["y_path_ret_4h"].notna()
        X_tr = tr_df.loc[tr_m, features].fillna(0)
        y_tr = tr_df.loc[tr_m, "y_path_ret_4h"].values.astype(float)
        X_te = te_df.loc[te_m, features].fillna(0)
        y_te = te_df.loc[te_m, "y_path_ret_4h"].values.astype(float)
        if len(y_tr) < 50 or len(y_te) < 5:
            continue
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, verbose=False)
        all_pred.append(m.predict(X_te))
        all_y.append(y_te)

    pred = np.concatenate(all_pred)
    y = np.concatenate(all_y)

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    ic = float(spearmanr(pred, y).correlation)
    auc_mask = y != 0
    auc = float(roc_auc_score((y[auc_mask] > 0).astype(int), pred[auc_mask]))

    fire, win = rolling_top5(pred, y)
    n_fire = int(fire.sum())
    n_win = int(win.sum())
    wr = n_win / n_fire if n_fire else float("nan")
    lo, hi = wilson_ci(n_win, n_fire)

    print(f"\n  Shuffled model OOS:  n={len(pred)}")
    print(f"    Spearman IC : {ic:+.4f}")
    print(f"    Sign AUC    : {auc:.4f}")
    print(f"    Top-5% WR   : {wr * 100:.1f}%  [{lo * 100:.1f}%, "
          f"{hi * 100:.1f}%]  (n={n_fire})")

    verdict_ic = "OK" if abs(ic) < 0.05 else "LEAK RISK"
    verdict_auc = "OK" if abs(auc - 0.5) < 0.03 else "LEAK RISK"
    verdict_wr = "OK" if 0.42 < wr < 0.58 else "LEAK RISK"
    print(f"\n  Verdict: IC={verdict_ic}  AUC={verdict_auc}  WR={verdict_wr}")
    if any(v == "LEAK RISK" for v in (verdict_ic, verdict_auc, verdict_wr)):
        print("  WARNING: Residual signal on shuffled labels — possible leakage.")
    else:
        print("  CLEAN: shuffled model collapses as expected. No leakage detected.")


# ────────────────────────────────────────────────────────────────────
# MAG: overlay test — Direction-Reg × Mag gate
# ────────────────────────────────────────────────────────────────────
def mag_overlay_test():
    print("\n" + "=" * 76)
    print("  MAG OVERLAY: Direction-REG + Mag gate sweep")
    print("=" * 76)
    print("  Baseline: Direction-REG only (no Mag filter)")
    print("  Variants: require mag_pct >= X for Strong")

    oos = pd.read_parquet(
        RESULTS_DIR / "direction_reg_oos_mse.parquet"
    ).sort_index()
    pred = oos["pred_ret"].values
    y = oos["y_path_ret_4h"].values

    # Need magnitude percentile aligned to the same OOS bars. Use production
    # mag model via IndicatorEngine.backfill_mag_pred.
    df = load_and_cache_data()
    from indicator.inference import IndicatorEngine
    eng = IndicatorEngine()
    if not hasattr(eng, "dual_mag_model"):
        print("  ERROR: no mag model loaded")
        return
    mag_series = eng.backfill_mag_pred(df)
    # Convert to rolling percentile of |mag| (analogous to the engine's
    # mag_score but computed here against the whole series for testing)
    mag_full = mag_series.reindex(oos.index).fillna(method="ffill").values
    # Rolling percentile within 500-bar window (past-only)
    mag_pct = np.full(len(mag_full), np.nan)
    for i in range(len(mag_full)):
        lo_i = max(0, i - ROLL + 1)
        window = mag_full[lo_i:i + 1]
        window = window[np.isfinite(window)]
        if len(window) < 50:
            continue
        mag_pct[i] = (window < mag_full[i]).sum() / len(window)

    # Base: rolling top-5% direction
    fire_dir, win_dir = rolling_top5(pred, y)

    def apply_mag_gate(threshold):
        mg_ok = mag_pct >= threshold if threshold > 0 else np.ones_like(mag_pct, bool)
        mg_ok = np.where(np.isnan(mag_pct), False, mg_ok)
        fire = fire_dir & mg_ok
        win = win_dir & mg_ok
        n = int(fire.sum())
        w = int(win.sum())
        wr = w / n if n else float("nan")
        lo, hi = wilson_ci(w, n)
        return n, w, wr, lo, hi

    print(f"\n  {'Mag gate':<14}{'n_fire':>10}{'wins':>8}{'WR':>10}{'CI':>22}"
          f"{'delta':>10}")
    print("  " + "-" * 72)
    base_n, base_w, base_wr, base_lo, base_hi = apply_mag_gate(0)
    print(f"  {'(none)':<14}{base_n:>10d}{base_w:>8d}{base_wr*100:>9.1f}%"
          f"  [{base_lo*100:5.1f}%, {base_hi*100:5.1f}%]{'(base)':>10}")
    for thr in [0.30, 0.50, 0.60, 0.70, 0.80]:
        n, w, wr, lo, hi = apply_mag_gate(thr)
        if n < 5:
            print(f"  mag>={thr:<7.2f}{n:>10d}{'  too few':>40}")
            continue
        delta = (wr - base_wr) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  mag>={thr:<7.2f}{n:>10d}{w:>8d}{wr*100:>9.1f}%"
              f"  [{lo*100:5.1f}%, {hi*100:5.1f}%]   {sign}{delta:+.1f}pp")

    print("\n  Interpretation:")
    print("    - If delta stays near 0 -> Mag gate adds nothing on top of")
    print("      Direction-Reg (same finding as Init head)")
    print("    - If delta >> 0 and n not too crushed -> Mag worth adding")
    print("    - If delta > 0 but fires collapse -> tradeoff, probably not worth")


def main():
    v5_label_shuffle()
    mag_overlay_test()


if __name__ == "__main__":
    main()
