"""
Test whether swapping magnitude target from |ret_4h| to vol-normalized
|ret_4h|/realized_vol recovers Mar/Apr OOS IC.

Runs TWO walk-forward trainings with identical features & hyperparams,
only the target differs:
    A. baseline    : target = y_abs_return        (current production)
    B. vol_adj     : target = y_vol_adj_abs       (candidate fix)

Reports monthly IC vs both:
    - native target  (fair "did model learn" comparison)
    - raw |ret_4h|   (practical: denormalized pred back to raw magnitude)

Usage:
    python research/mag_target_voladj_test.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_magnitude_labels import build_magnitude_labels

import xgboost as xgb

FEATS_PATH = Path("indicator/model_artifacts/dual_model/magnitude_feature_cols.json")
OUT = Path("research/results/mag_target_voladj_test.json")

XGB_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbosity": 0,
    "early_stopping_rounds": 30,
}

MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


def walk_forward_train(df: pd.DataFrame, features: list[str], target_col: str):
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    all_oos = []
    for fold_i, (tr, te) in enumerate(splits):
        train_df = df.iloc[tr]
        test_df = df.iloc[te]
        tr_mask = train_df[target_col].notna()
        te_mask = test_df[target_col].notna()
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue
        X_tr = train_df.loc[tr_mask, features].fillna(0)
        y_tr = train_df.loc[tr_mask, target_col].values
        X_te = test_df.loc[te_mask, features].fillna(0)
        y_te = test_df.loc[te_mask, target_col].values

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = model.predict(X_te)

        oos = pd.DataFrame({
            "y_pred_native": y_pred,
            "y_true_native": y_te,
            "y_abs_return": test_df.loc[te_mask, "y_abs_return"].values,
            "realized_vol": test_df.loc[te_mask, "realized_vol_20b"].values
                if "realized_vol_20b" in test_df.columns else np.nan,
            "fold": fold_i,
        }, index=test_df.loc[te_mask].index)
        all_oos.append(oos)
    return pd.concat(all_oos)


def monthly_ic(df: pd.DataFrame, pred_col: str, target_col: str) -> dict:
    months = df.index.strftime("%Y-%m")
    out = {}
    for m in MONTHS:
        mask = months == m
        if mask.sum() < 30:
            out[m] = {"n": int(mask.sum()), "ic": None}
            continue
        ok = mask & df[pred_col].notna() & df[target_col].notna()
        if ok.sum() < 30:
            out[m] = {"n": int(ok.sum()), "ic": None}
            continue
        ic, _ = spearmanr(df.loc[ok, pred_col].values, df.loc[ok, target_col].values)
        out[m] = {"n": int(ok.sum()), "ic": float(ic)}
    # Overall
    ok = df[pred_col].notna() & df[target_col].notna()
    ic_all, _ = spearmanr(df.loc[ok, pred_col].values, df.loc[ok, target_col].values)
    out["_overall"] = {"n": int(ok.sum()), "ic": float(ic_all)}
    return out


def print_month_row(label: str, monthly: dict):
    cells = []
    for m in MONTHS:
        v = monthly[m]["ic"]
        cells.append(f"{v:+.3f}" if v is not None else "  nan")
    ov = monthly["_overall"]["ic"]
    print(f"  {label:<22} " + "  ".join(f"{c:>6}" for c in cells) + f"    overall={ov:+.3f}")


def main():
    feats = json.loads(FEATS_PATH.read_text())
    df = load_and_cache_data()
    labels = build_magnitude_labels(df)
    for col in labels.columns:
        df[col] = labels[col]

    # Keep only features present
    features = [f for f in feats if f in df.columns]
    print(f"Features: {len(features)} / {len(feats)} requested")
    print(f"Bars    : {len(df)}")

    # === A. Baseline: target = y_abs_return ===
    print("\n[A] Training baseline (target=y_abs_return) ...")
    oos_a = walk_forward_train(df, features, "y_abs_return")
    print(f"    OOS predictions: {len(oos_a)}")

    # === B. Vol-adjusted: target = y_vol_adj_abs ===
    print("[B] Training vol_adj    (target=y_vol_adj_abs) ...")
    oos_b = walk_forward_train(df, features, "y_vol_adj_abs")
    print(f"    OOS predictions: {len(oos_b)}")

    # Denormalize B: raw_pred = pred_vol_adj * realized_vol
    oos_b["y_pred_raw"] = oos_b["y_pred_native"] * oos_b["realized_vol"]

    # === Monthly IC ===
    print()
    print("=" * 88)
    print("MONTHLY OOS IC  — native target (what the model directly learns)")
    print("=" * 88)
    print(f"  {'model':<22} " + "  ".join(f"{m[-5:]:>6}" for m in MONTHS) + "    overall")

    ic_a_native = monthly_ic(oos_a, "y_pred_native", "y_true_native")
    ic_b_native = monthly_ic(oos_b, "y_pred_native", "y_true_native")
    print_month_row("A baseline", ic_a_native)
    print_month_row("B vol_adj", ic_b_native)

    print()
    print("=" * 88)
    print("MONTHLY OOS IC  — vs raw |ret_4h| (practical: rank quality on raw magnitude)")
    print("=" * 88)
    print(f"  {'model':<22} " + "  ".join(f"{m[-5:]:>6}" for m in MONTHS) + "    overall")

    ic_a_raw = monthly_ic(oos_a, "y_pred_native", "y_abs_return")
    ic_b_raw = monthly_ic(oos_b, "y_pred_raw", "y_abs_return")
    print_month_row("A baseline", ic_a_raw)
    print_month_row("B vol_adj (denorm)", ic_b_raw)
    # Also show B's native pred vs raw (no denorm) for comparison
    ic_b_native_vs_raw = monthly_ic(oos_b, "y_pred_native", "y_abs_return")
    print_month_row("B vol_adj (no denorm)", ic_b_native_vs_raw)

    # === Verdict ===
    print()
    print("=" * 88)
    print("VERDICT — Mar/Apr recovery check (raw |ret_4h| IC)")
    print("=" * 88)
    for m in ["2026-03", "2026-04"]:
        a = ic_a_raw[m]["ic"]
        b = ic_b_raw[m]["ic"]
        c = ic_b_native_vs_raw[m]["ic"]
        if a is None or b is None:
            continue
        delta = b - a
        sign = "+" if delta > 0 else ""
        print(f"  {m}: baseline={a:+.3f}  vol_adj_denorm={b:+.3f}  "
              f"vol_adj_native={c:+.3f}  delta={sign}{delta:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_features": len(features),
        "baseline_native": ic_a_native,
        "voladj_native": ic_b_native,
        "baseline_vs_raw": ic_a_raw,
        "voladj_denorm_vs_raw": ic_b_raw,
        "voladj_native_vs_raw": ic_b_native_vs_raw,
    }, indent=2, default=str))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
