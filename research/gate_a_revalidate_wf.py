"""
Gate A re-validation over a LONGER history — clean walk-forward of the
PRODUCTION direction-regressor pipeline (the one that actually generates the
live tracked_signals Strong tier), with the production rolling-percentile
Strong decode.

Why this exists (2026-06-14):
  Live tracked_signals Strong for the CURRENT model (model_version =
  direction_reg_config.trained_at = 2026-05-01T07:42:43) is only n=28,
  WR 53.6%, CI[35.8, 70.5] — too few to assert Gate A.  The 739/59.5% that
  was recorded as "Gate A PASSED" mixed 3+ older model architectures.
  This script estimates the CURRENT architecture's Strong WR over the full
  ~7-month history via a clean walk-forward, to get statistical power.

Faithfulness to production:
  - Same model: XGBRegressor on y_path_ret_{H}h (TWAP path return) with the
    SAME 137 FULL_DIRECTION features regardless of horizon, same hyperparams.
  - Same decode: 500-bar trailing rolling percentile; top 2.5% each tail =
    Strong (matches inference.py).  Rolling-percentile is self-normalizing so
    it is robust to per-fold prediction-scale drift.
  - Clean: per fold trains ONLY on prior data (purge ≥ horizon + embargo 4)
    and uses NO early-stopping-on-test (the train_direction_reg_4h.py
    leakage).  A leaky variant is run side-by-side to quantify the inflation.

Multi-horizon (Path 1, docs/path1_multi_horizon_plan.md):
  --horizon 4  (default) — production 4h target, reproduces existing Gate A
  --horizon 1            — 1h target, tests cross-time-scale orthogonality
  --horizon 24           — 1d target, tests trend-horizon orthogonality
  Purge scales with horizon (must be ≥ horizon to prevent label leakage).
  Output parquet is suffixed with horizon to avoid overwriting.

NOT a substitute for live Gate A — it estimates the architecture's edge, not
the exact deployed weights.  But it is the highest-power clean evidence
available now.

Run:
  python research/gate_a_revalidate_wf.py                  # 4h (default)
  python research/gate_a_revalidate_wf.py --horizon 1      # 1h orthogonality
  python research/gate_a_revalidate_wf.py --horizon 24     # 1d orthogonality
"""
from __future__ import annotations

import argparse
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available

ROLL = 500          # production rolling-percentile window (inference.py)
STRONG_TAIL = 0.025  # top/bottom 2.5% each tail = Strong (inference.py)

BASE_PARAMS = dict(
    objective="reg:squarederror", eval_metric="mae",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)


def walk_forward(df, features, *, leaky: bool, horizon: int, target_col: str):
    """Per-fold XGBRegressor on the horizon-specific target. leaky=True
    reproduces the train_direction_reg_4h.py early-stopping-on-test-fold
    leakage; leaky=False is the clean version (fixed n_estimators, no
    eval_set).

    purge scales with horizon (must be ≥ horizon to prevent forward-label
    leakage of bars whose target overlaps the test fold).
    """
    purge = max(horizon, 4)   # ≥ horizon; never below 4 (legacy floor)
    embargo = 4               # serial-correlation guard, horizon-independent
    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=purge, embargo=embargo,
    )
    rows = []
    params = dict(BASE_PARAMS)
    if leaky:
        params["early_stopping_rounds"] = 30
    t0 = time.time()
    for fi, (tr, te) in enumerate(splits):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        m_tr = tr_df[target_col].notna()
        m_te = te_df[target_col].notna()
        Xtr = tr_df.loc[m_tr, features].fillna(0)
        ytr = tr_df.loc[m_tr, target_col].values.astype(float)
        Xte = te_df.loc[m_te, features].fillna(0)
        yte = te_df.loc[m_te, target_col].values.astype(float)
        if len(ytr) < 50 or len(yte) < 5:
            continue
        model = xgb.XGBRegressor(**params)
        if leaky:
            model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        else:
            model.fit(Xtr, ytr)
        rows.append(pd.DataFrame(
            {"pred_ret": model.predict(Xte), "y": yte, "fold": fi},
            index=te_df.loc[m_te].index))
    oos = pd.concat(rows).sort_index()
    print(f"    {'leaky' if leaky else 'clean'}: {len(oos)} OOS bars, "
          f"{oos['fold'].nunique()} folds, purge={purge} embargo={embargo}, "
          f"{time.time()-t0:.0f}s")
    return oos


def decode_strong(oos):
    """Production rolling-percentile decode → Strong UP / DOWN. Trailing only."""
    s = oos["pred_ret"]
    rank = s.rolling(ROLL, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True)
    oos = oos.copy()
    oos["rank"] = rank
    oos["strong"] = "none"
    oos.loc[oos["rank"] >= 1 - STRONG_TAIL, "strong"] = "UP"
    oos.loc[oos["rank"] <= STRONG_TAIL, "strong"] = "DOWN"
    return oos


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(wins, n_boot=2000):
    """Percentile bootstrap CI of the win rate. wins = 0/1 array."""
    wins = np.asarray(wins, dtype=float)
    n = len(wins)
    if n == 0:
        return (float("nan"), float("nan"))
    # Deterministic seed (no Math.random equivalent needed): fixed RNG
    rng = np.random.default_rng(42)
    means = [wins[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate(oos, label):
    st = oos[oos["strong"].isin(["UP", "DOWN"])].copy()
    win = ((st["strong"] == "UP") & (st["y"] > 0)) | \
          ((st["strong"] == "DOWN") & (st["y"] < 0))
    st["win"] = win.astype(int)
    n = len(st)
    k = int(st["win"].sum())
    wr = k / n if n else float("nan")
    wlo, whi = wilson_ci(k, n)
    blo, bhi = bootstrap_ci(st["win"].values)
    print(f"\n  === {label} ===")
    print(f"    Strong n={n}  fire_rate={n/len(oos)*100:.1f}%  WR={wr*100:.1f}%")
    print(f"    Wilson 95% CI   [{wlo*100:.1f}%, {whi*100:.1f}%]")
    print(f"    Bootstrap 95% CI[{blo*100:.1f}%, {bhi*100:.1f}%]")
    print(f"    Gate A bar: CI lower bound > 52% ?  "
          f"{'PASS' if blo > 0.52 else 'FAIL'} (lower={blo*100:.1f}%)")
    # per-fold sanity (mistake.md 2026-06-02: never trust pooled alone)
    fold_wr = st.groupby("fold")["win"].agg(["mean", "count"])
    fold_wr = fold_wr[fold_wr["count"] >= 3]
    if len(fold_wr):
        fp = float((fold_wr["mean"] > 0.5).mean())
        print(f"    per-fold (>=3 Strong): n_folds={len(fold_wr)}  "
              f"mean_WR={fold_wr['mean'].mean()*100:.1f}%  "
              f"frac_folds>50%={fp*100:.0f}%")
    return dict(n=n, wr=wr, boot_lo=blo, boot_hi=bhi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4,
                    choices=[1, 4, 24],
                    help="Forward horizon in 1h bars (1 / 4 / 24). "
                         "Default 4 reproduces production behaviour.")
    args = ap.parse_args()
    horizon = args.horizon
    target_col = f"y_path_ret_{horizon}h"

    print("=" * 70)
    print(f"  GATE A RE-VALIDATION — clean direction-regressor walk-forward")
    print(f"  HORIZON: {horizon}h  (target column: {target_col})")
    print("=" * 70)
    df = load_and_cache_data()
    labels = build_direction_reg_labels(df, horizon_bars=horizon)
    df = df.copy()
    df[target_col] = labels[target_col]
    feats = filter_available(FULL_DIRECTION, list(df.columns))
    print(f"  Loaded {len(df)} bars, {len(feats)} direction features")
    span = f"{df.index.min()} ~ {df.index.max()}" if hasattr(df.index, "min") else "?"
    print(f"  History span: {span}")

    print("\n  Walk-forward (this is the slow part)...")
    clean = decode_strong(walk_forward(df, feats, leaky=False,
                                       horizon=horizon, target_col=target_col))
    leaky = decode_strong(walk_forward(df, feats, leaky=True,
                                       horizon=horizon, target_col=target_col))

    r_clean = evaluate(clean, "CLEAN WF (no early-stop-on-test) — the honest number")
    r_leaky = evaluate(leaky, "LEAKY WF (early-stop-on-test) — reproduces inflation")

    print("\n" + "=" * 70)
    if horizon == 4:
        print("  CONTRAST vs live Gate A")
        print("=" * 70)
        print(f"  live v7-only (n=28)          : WR 53.6%  CI[35.8, 70.5]")
    else:
        print(f"  {horizon}h horizon — no live tracked_signals to contrast (Path 1)")
        print("=" * 70)
    print(f"  clean WF (n={r_clean['n']})            : WR {r_clean['wr']*100:.1f}%  "
          f"CI[{r_clean['boot_lo']*100:.1f}, {r_clean['boot_hi']*100:.1f}]")
    print(f"  leaky WF (n={r_leaky['n']})            : WR {r_leaky['wr']*100:.1f}%  "
          f"CI[{r_leaky['boot_lo']*100:.1f}, {r_leaky['boot_hi']*100:.1f}]")
    print(f"  early-stopping inflation     : "
          f"{(r_leaky['wr']-r_clean['wr'])*100:+.1f} pp")

    # Horizon-suffixed output path so 1h / 4h / 24h don't overwrite each other.
    out_name = (f"gate_a_revalidate_clean_oos_{horizon}h.parquet"
                if horizon != 4
                else "gate_a_revalidate_clean_oos.parquet")
    out_path = PROJECT_ROOT / "research" / "results" / out_name
    clean.to_parquet(out_path)
    print(f"\n  Saved OOS → {out_path}")


if __name__ == "__main__":
    main()
