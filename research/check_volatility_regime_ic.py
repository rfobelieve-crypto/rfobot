"""
A2: Slice walk-forward OOS IC by realized volatility regime.

Loads the OOS predictions saved by concept_drift_monthly_ic.py and slices
two ways:
  1. By |y| quintile — direct test of "high realized vol → IC reverses".
  2. By |pred| quintile — feasible inference-time gate test.

If high-|y| quintile shows significantly negative IC, the model is regime-
conditionally broken (not concept-drift). If high-|pred| quintile also
shows it, we have an inference-time gate available.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_concept_drift_oos.parquet"


def boot_ic_ci(p: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(p)
    if n < 30:
        return (np.nan, np.nan)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ic, _ = spearmanr(p[idx], y[idx])
        out[i] = ic if np.isfinite(ic) else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def slice_table(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = pd.qcut(df[by], q=5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    rows = []
    for bucket, g in df.groupby("bucket", observed=True):
        ic, _ = spearmanr(g["pred"], g["y"])
        lo, hi = boot_ic_ci(g["pred"].values, g["y"].values)
        rows.append({
            "slice_by": label,
            "bucket": bucket,
            "n": len(g),
            f"min_{by}": g[by].min(),
            f"max_{by}": g[by].max(),
            f"mean_{by}": g[by].mean(),
            "ic": ic,
            "ic_lo": lo,
            "ic_hi": hi,
            "sign_acc": float((np.sign(g["pred"]) == np.sign(g["y"])).mean()),
            "mean_y": float(g["y"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(OOS)
    df["abs_y"] = df["y"].abs()
    df["abs_pred"] = df["pred"].abs()

    print(f"Loaded OOS: n={len(df)}, range {df['ts'].min()} ~ {df['ts'].max()}")
    print()

    for col, lab in [("abs_y", "|y|  (direct realized-vol regime)"),
                     ("abs_pred", "|pred|  (inference-time vol gate)")]:
        t = slice_table(df, col, lab)
        print("=" * 100)
        print(f"Slice by {lab}")
        print("=" * 100)
        print(f"{'bucket':<8} {'n':>5} {f'mean_{col}':>12} "
              f"{'IC':>8} {'CI_lo':>8} {'CI_hi':>8} {'sign_acc':>9} {'mean_y':>10}")
        print("-" * 100)
        for _, r in t.iterrows():
            print(f"{r['bucket']:<8} {r['n']:>5d} {r[f'mean_{col}']:>12.5f} "
                  f"{r['ic']:>+8.4f} {r['ic_lo']:>+8.3f} {r['ic_hi']:>+8.3f} "
                  f"{r['sign_acc']:>9.3f} {r['mean_y']:>+10.5f}")
        print()


if __name__ == "__main__":
    main()
