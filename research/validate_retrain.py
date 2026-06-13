"""
Post-retrain validation: bootstrap CI for IC/AUC + monthly stability +
production-relevant top-K precision (rolling percentile decoder).

Acceptance criteria (per CLAUDE.md feedback_validation_protocol):
  - Direction WF IC bootstrap CI lower bound >= 0.13
  - Direction WF AUC bootstrap CI lower bound >= 0.55
  - Top-5% precision >= 60% AND CI lower bound >= 55%
  - Monthly IC: no month should have IC < 50% of overall
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def bootstrap_ci(values, statfn, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(statfn(values[idx]))
    boots = np.sort(boots)
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 + ci) / 2 * n_boot)]
    return float(lo), float(hi)


def bootstrap_paired(arr_a, arr_b, statfn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap stat using paired indices (for IC/AUC where both arrays needed)."""
    rng = np.random.default_rng(seed)
    boots = []
    n = len(arr_a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(statfn(arr_a[idx], arr_b[idx]))
    boots = np.sort(boots)
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 + ci) / 2 * n_boot)]
    return float(lo), float(hi)


def spearman_ic(pred, y):
    return float(pd.Series(pred).corr(pd.Series(y), method="spearman"))


def auc_sign(pred, y):
    """AUC for sign(y) using pred as score."""
    from sklearn.metrics import roc_auc_score
    yb = (y > 0).astype(int)
    if yb.sum() == 0 or yb.sum() == len(yb):
        return float("nan")
    return float(roc_auc_score(yb, pred))


def top_k_precision(pred, y, k_frac=0.05):
    """Symmetric top-K: top k/2 predictions become 'long', bottom k/2 become 'short'.
    Win = (long & y>0) or (short & y<0)."""
    n = len(pred)
    k = int(n * k_frac / 2.0)
    if k < 1:
        return float("nan"), 0
    order = np.argsort(pred)
    short_idx = order[:k]
    long_idx = order[-k:]
    long_wins = int(np.sum(y[long_idx] > 0))
    short_wins = int(np.sum(y[short_idx] < 0))
    total = 2 * k
    return (long_wins + short_wins) / total, total


def evaluate(parquet_path: Path, label: str):
    df = pd.read_parquet(parquet_path).sort_index()
    pred = df["pred_ret"].values.astype(float)
    y = df["y_path_ret_4h"].values.astype(float)

    print(f"\n{'='*70}")
    print(f"  {label}   n={len(df)}  range={df.index.min()} ~ {df.index.max()}")
    print(f"{'='*70}")

    ic = spearman_ic(pred, y)
    ic_lo, ic_hi = bootstrap_paired(pred, y, spearman_ic, n_boot=1000)
    print(f"  Spearman IC   : {ic:+.4f}    bootstrap 95% CI = [{ic_lo:+.4f}, {ic_hi:+.4f}]")

    auc = auc_sign(pred, y)
    auc_lo, auc_hi = bootstrap_paired(pred, y, auc_sign, n_boot=1000)
    print(f"  AUC (sign)    : {auc:.4f}    bootstrap 95% CI = [{auc_lo:.4f}, {auc_hi:.4f}]")

    print(f"\n  Top-K precision (production-relevant rolling-percentile decoder):")
    for kf in [0.02, 0.05, 0.10, 0.15, 0.20]:
        prec, n_signals = top_k_precision(pred, y, kf)
        prec_lo, prec_hi = bootstrap_paired(
            pred, y, lambda p, y, kf=kf: top_k_precision(p, y, kf)[0], n_boot=500
        )
        tier = "Strong" if kf == 0.05 else ("Moderate" if kf == 0.15 else "")
        print(f"    top-{int(kf*100):2d}% ({n_signals:3d} signals)  precision={prec:.3f}  "
              f"CI=[{prec_lo:.3f}, {prec_hi:.3f}]  {tier}")

    # Monthly stability
    print(f"\n  Monthly IC stability:")
    df_m = df.copy()
    df_m["month"] = pd.to_datetime(df_m.index).strftime("%Y-%m")
    for month, g in df_m.groupby("month"):
        if len(g) < 30:
            print(f"    {month}: n={len(g)} (skip, too few)")
            continue
        m_ic = spearman_ic(g["pred_ret"].values, g["y_path_ret_4h"].values)
        m_auc = auc_sign(g["pred_ret"].values, g["y_path_ret_4h"].values)
        flag = ""
        if abs(m_ic) < abs(ic) * 0.5:
            flag = "  ⚠ IC < 50% of overall"
        elif m_ic * ic < 0:
            flag = "  ⚠ SIGN FLIP"
        print(f"    {month}: n={len(g):4d}  IC={m_ic:+.4f}  AUC={m_auc:.4f}{flag}")

    # Last-month detail (most recent regime)
    last_month = sorted(df_m["month"].unique())[-1]
    last = df_m[df_m["month"] == last_month]
    if len(last) >= 30:
        last_pred = last["pred_ret"].values
        last_y = last["y_path_ret_4h"].values
        last_neg = (last_pred < 0).sum()
        last_pos = (last_pred > 0).sum()
        last_y_neg = (last_y < 0).sum()
        last_y_pos = (last_y > 0).sum()
        print(f"\n  Last month ({last_month}) prediction balance:")
        print(f"    pred_ret < 0: {last_neg} ({100*last_neg/len(last):.1f}%)   "
              f"pred_ret > 0: {last_pos} ({100*last_pos/len(last):.1f}%)")
        print(f"    y_actual<0:   {last_y_neg} ({100*last_y_neg/len(last):.1f}%)   "
              f"y_actual>0:   {last_y_pos} ({100*last_y_pos/len(last):.1f}%)")

    return {
        "ic": ic, "ic_ci": (ic_lo, ic_hi),
        "auc": auc, "auc_ci": (auc_lo, auc_hi),
    }


def main():
    res_dir = PROJECT_ROOT / "research" / "results" / "dual_model"
    candidates = [
        ("MSE objective (current production)",
         res_dir / "direction_reg_oos_mse.parquet"),
        ("Huber objective",
         res_dir / "direction_reg_oos_huber.parquet"),
    ]

    for label, path in candidates:
        if path.exists():
            evaluate(path, label)
        else:
            print(f"[skip] {path} not found")

    # Magnitude — only if available
    mag_path = res_dir / "magnitude_oos_phase1_voladj.parquet"
    if mag_path.exists():
        df = pd.read_parquet(mag_path)
        print(f"\n{'='*70}")
        print(f"  MAGNITUDE  n={len(df)}  cols={list(df.columns)[:6]}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
