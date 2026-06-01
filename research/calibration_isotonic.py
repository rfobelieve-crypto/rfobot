"""Walk-forward isotonic calibration of V7 direction regressor.

Reads existing OOS predictions, fits per-fold isotonic regression on
prior folds' (pred, true) pairs, applies to current fold's predictions.

Goal: improve Strong-tier precision at given top-K cutoff WITHOUT
retraining V7.  Calibration is rank-preserving (so sign_AUC unchanged)
but magnitude-corrective — over-confident predictions get shrunk
toward realized average, under-confident gets stretched.

Optional: per-side (long / short) separate calibration since mistake log
documents asymmetric sign-acc (DOWN ≈ 59% vs UP ≈ 53%).

Output:
  research/results/dual_model/direction_reg_oos_mse_calibrated.parquet
  research/results/calibration_isotonic_compare.csv

Usage:
    python research/calibration_isotonic.py
    python research/calibration_isotonic.py --per-side
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OOS_PATH = PROJECT_ROOT / "research" / "results" / "dual_model" \
                       / "direction_reg_oos_mse.parquet"
OUT_PARQUET = PROJECT_ROOT / "research" / "results" / "dual_model" \
                          / "direction_reg_oos_mse_calibrated.parquet"
COMPARE_CSV = PROJECT_ROOT / "research" / "results" \
                          / "calibration_isotonic_compare.csv"


def walk_forward_calibrate(df: pd.DataFrame,
                            per_side: bool = False,
                            min_train: int = 200) -> pd.DataFrame:
    """For each fold k, fit isotonic on folds < k, apply to fold k.

    per_side=True fits separate isotonic for positive vs negative preds.
    Set min_train=200 to ensure each fit has enough samples.
    """
    out = df.copy()
    out["pred_calibrated"] = np.nan
    folds = sorted(df["fold"].unique())
    for k in folds:
        train = df[df["fold"] < k]
        test = df[df["fold"] == k]
        if len(train) < min_train:
            # Cold start: not enough history to calibrate; pass through
            out.loc[test.index, "pred_calibrated"] = test["pred_ret"].values
            continue
        if per_side:
            cal = np.full(len(test), np.nan)
            for sign, mask in (("pos", train["pred_ret"] >= 0),
                                ("neg", train["pred_ret"] < 0)):
                side_train = train[mask]
                if len(side_train) < min_train // 2:
                    continue
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(side_train["pred_ret"].values,
                        side_train["y_path_ret_4h"].values)
                if sign == "pos":
                    test_mask = test["pred_ret"].values >= 0
                else:
                    test_mask = test["pred_ret"].values < 0
                if test_mask.any():
                    cal[test_mask] = iso.predict(
                        test["pred_ret"].values[test_mask])
            out.loc[test.index, "pred_calibrated"] = cal
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(train["pred_ret"].values,
                    train["y_path_ret_4h"].values)
            out.loc[test.index, "pred_calibrated"] = iso.predict(
                test["pred_ret"].values)
    return out


def metrics_block(df: pd.DataFrame, pred_col: str,
                   label: str) -> dict:
    """IC + sign_AUC + Strong WR at thresholds + Top-K precision."""
    d = df.dropna(subset=[pred_col, "y_path_ret_4h"])
    if d.empty:
        return {"label": label}
    ic, _ = spearmanr(d[pred_col], d["y_path_ret_4h"])
    sign_y = (d["y_path_ret_4h"] > 0).astype(int)
    try:
        auc = roc_auc_score(sign_y, d[pred_col])
    except Exception:
        auc = np.nan
    out = {"label": label, "n": len(d),
           "ic": float(ic),
           "sign_auc": float(auc) if not np.isnan(auc) else np.nan}
    # Strong WR at fixed thresholds
    for thr in (0.001, 0.002, 0.003, 0.005, 0.008):
        m = d[pred_col].abs() >= thr
        n_strong = int(m.sum())
        if n_strong == 0:
            out[f"thr_{thr}_n"] = 0
            out[f"thr_{thr}_wr"] = np.nan
            continue
        correct = (np.sign(d[pred_col][m])
                    == np.sign(d["y_path_ret_4h"][m])).sum()
        out[f"thr_{thr}_n"] = n_strong
        out[f"thr_{thr}_wr"] = float(correct / n_strong)
    # Top-K precision
    for pct in (0.05, 0.10, 0.20):
        n_top = max(1, int(len(d) * pct))
        top = d.assign(abs_pred=d[pred_col].abs()).nlargest(n_top, "abs_pred")
        correct = (np.sign(top[pred_col])
                    == np.sign(top["y_path_ret_4h"])).sum()
        out[f"top_{int(pct*100)}pct_n"] = len(top)
        out[f"top_{int(pct*100)}pct_wr"] = float(correct / len(top))
    return out


def main(per_side: bool = False) -> int:
    print(f"Loading OOS predictions from {OOS_PATH.name}…")
    df = pd.read_parquet(OOS_PATH)
    print(f"  {len(df)} rows × {len(df['fold'].unique())} folds")
    print(f"  raw pred range: [{df['pred_ret'].min():+.5f}, "
          f"{df['pred_ret'].max():+.5f}]")
    print(f"  raw pred std:   {df['pred_ret'].std():.5f}")
    print(f"  realized std:   {df['y_path_ret_4h'].std():.5f}")

    print(f"\nFitting walk-forward isotonic"
          f"{' (per-side)' if per_side else ''}…")
    cal = walk_forward_calibrate(df, per_side=per_side)
    cal_valid = cal.dropna(subset=["pred_calibrated"])
    print(f"  calibrated {len(cal_valid)}/{len(cal)} rows")
    print(f"  cal pred range: [{cal_valid['pred_calibrated'].min():+.5f}, "
          f"{cal_valid['pred_calibrated'].max():+.5f}]")
    print(f"  cal pred std:   {cal_valid['pred_calibrated'].std():.5f}")

    print("\n=== METRICS — RAW vs CALIBRATED ===")
    m_raw = metrics_block(df, "pred_ret", "RAW")
    m_cal = metrics_block(cal, "pred_calibrated",
                            "CAL_per_side" if per_side else "CAL")

    keys_order = ["n", "ic", "sign_auc",
                  "thr_0.001_n", "thr_0.001_wr",
                  "thr_0.002_n", "thr_0.002_wr",
                  "thr_0.003_n", "thr_0.003_wr",
                  "thr_0.005_n", "thr_0.005_wr",
                  "thr_0.008_n", "thr_0.008_wr",
                  "top_5pct_n", "top_5pct_wr",
                  "top_10pct_n", "top_10pct_wr",
                  "top_20pct_n", "top_20pct_wr"]

    rows = []
    print(f"{'Metric':22s}  {'RAW':>14s}  {'CAL':>14s}  {'Δ':>10s}")
    print("-" * 70)
    for k in keys_order:
        a = m_raw.get(k)
        b = m_cal.get(k)
        if a is None or b is None:
            continue
        try:
            if isinstance(a, int):
                d_str = f"{int(b - a):+d}"
                a_str, b_str = f"{a}", f"{int(b)}"
            else:
                if np.isnan(a) or np.isnan(b):
                    d_str = "n/a"
                else:
                    d_str = f"{b - a:+.4f}"
                a_str = f"{a:.4f}" if not np.isnan(a) else "n/a"
                b_str = f"{b:.4f}" if not np.isnan(b) else "n/a"
        except Exception:
            a_str, b_str, d_str = str(a), str(b), "n/a"
        print(f"{k:22s}  {a_str:>14s}  {b_str:>14s}  {d_str:>10s}")
        rows.append({"metric": k, "raw": a, "cal": b})

    pd.DataFrame(rows).to_csv(COMPARE_CSV, index=False)
    print(f"\nWrote {COMPARE_CSV}")

    cal[["pred_ret", "pred_calibrated", "y_path_ret_4h", "fold"]] \
        .to_parquet(OUT_PARQUET)
    print(f"Wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-side", action="store_true",
                        help="Fit separate isotonic for pos/neg predictions")
    args = parser.parse_args()
    raise SystemExit(main(per_side=args.per_side))
