"""
Regime-specific Direction sub-models vs global Direction model.

Hypothesis:
    Training separate Direction classifiers per regime (bull/bear/chop) may
    capture regime-local feature→direction relationships that the global
    model averages out.

Setup:
    - 4h horizon, vol-adaptive deadzone labels (k=0.5) — same as production.
    - Walk-forward with purge=embargo=4 bars.
    - At each split:
        (a) Global model: train on all non-NaN train samples.
        (b) Regime models: train 3 independent models, each only on that
            regime's bars in the train window.
    - Predict each test bar twice:
        - global_prob (from global model)
        - regime_prob (from the model of that bar's regime)
    - Evaluate per-regime AUC for both paths.

Regime definition (matches feature_builder_live.py):
    vol_24h_pct_rank = log_ret(1h).rolling(24).std().expanding(72).rank(pct)
    ret_24h = close.pct_change(24)
    BULL  = vol_pct > 0.6 AND ret_24h > 0.005
    BEAR  = vol_pct > 0.6 AND ret_24h < -0.005
    CHOP  = everything else

Gate: any regime sub-model that beats global AUC by >= 0.02 AND has
     n_train_per_split >= 50 is a candidate for Phase 2 validation
     (bootstrap CI + time stability).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits

ABLATION = Path("research/results/ablation_study.json")
OUT = Path("research/results/regime_specific_direction.json")

HORIZON = 4
K_DEADZONE = 0.5
PURGE = 4
EMBARGO = 4
MIN_TRAIN_PER_REGIME = 50

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
}


def get_feats(df: pd.DataFrame) -> list[str]:
    abl = json.loads(ABLATION.read_text())
    base = abl.get("baseline", {}).get("features", [])
    keep = abl.get("combined", {}).get("keep_features", [])
    seen, out = set(), []
    for f in base + keep:
        if f in df.columns and f not in seen:
            out.append(f)
            seen.add(f)
    return out


def compute_regimes(df: pd.DataFrame) -> np.ndarray:
    """Return regime label array: 0=CHOP, 1=BULL, 2=BEAR."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=72).rank(pct=True)
    ret_24h = df["close"].pct_change(24)
    bull = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    r = np.zeros(len(df), dtype=int)
    r[bull] = 1
    r[bear] = 2
    return r


def build_labels(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].values.astype(float)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_1h = log_ret.rolling(20, min_periods=10).std().values
    n = len(close)
    y = np.full(n, np.nan)
    for i in range(n - HORIZON):
        if np.isnan(vol_1h[i]) or vol_1h[i] <= 0:
            continue
        r = close[i + HORIZON] / close[i] - 1
        th = K_DEADZONE * vol_1h[i] * np.sqrt(HORIZON)
        if r > th:
            y[i] = 1.0
        elif r < -th:
            y[i] = 0.0
    return y


def fit_dir(X: np.ndarray, y: np.ndarray, sw: np.ndarray | None = None) -> xgb.XGBClassifier:
    up_rate = y.mean()
    p = DIR_PARAMS.copy()
    p["scale_pos_weight"] = (1 - up_rate) / up_rate if 0 < up_rate < 1 else 1.0
    m = xgb.XGBClassifier(**p)
    m.fit(X, y, sample_weight=sw, verbose=False)
    return m


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    feats = get_feats(df)
    y_all = build_labels(df)
    regimes = compute_regimes(df)
    n = len(df)

    print(f"Rows: {n}, Features: {len(feats)}")
    # Regime distribution on labeled bars
    labeled = ~np.isnan(y_all)
    print(f"Regime mix (labeled only): "
          f"CHOP={((regimes == 0) & labeled).sum()}, "
          f"BULL={((regimes == 1) & labeled).sum()}, "
          f"BEAR={((regimes == 2) & labeled).sum()}")

    splits = walk_forward_splits(
        n, initial_train=288, test_size=48, step=48,
        purge=PURGE, embargo=EMBARGO,
    )

    # Collect per-bar (global_prob, regime_prob, label, regime) for OOS
    g_probs, r_probs, labels, regs = [], [], [], []
    skipped_regime_train = {0: 0, 1: 0, 2: 0}

    for tr_idx, te_idx in splits:
        tr_mask = ~np.isnan(y_all[tr_idx])
        te_mask = ~np.isnan(y_all[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue

        X_tr = df.iloc[tr_idx][feats].fillna(0).values
        y_tr = y_all[tr_idx]
        r_tr = regimes[tr_idx]
        X_te = df.iloc[te_idx][feats].fillna(0).values
        y_te = y_all[te_idx]
        r_te = regimes[te_idx]

        # Global model
        X_tr_g = X_tr[tr_mask]
        y_tr_g = y_tr[tr_mask].astype(int)
        if len(np.unique(y_tr_g)) < 2:
            continue
        m_global = fit_dir(X_tr_g, y_tr_g)

        # Per-regime models
        regime_models = {}
        for rid in (0, 1, 2):
            mask_r = tr_mask & (r_tr == rid)
            if mask_r.sum() < MIN_TRAIN_PER_REGIME:
                skipped_regime_train[rid] += 1
                continue
            y_r = y_tr[mask_r].astype(int)
            if len(np.unique(y_r)) < 2:
                skipped_regime_train[rid] += 1
                continue
            X_r = X_tr[mask_r]
            regime_models[rid] = fit_dir(X_r, y_r)

        # Predict for test bars with valid labels
        test_good = np.where(te_mask)[0]
        for j in test_good:
            gp = float(m_global.predict_proba(X_te[j:j + 1])[:, 1][0])
            rid = int(r_te[j])
            if rid in regime_models:
                rp = float(regime_models[rid].predict_proba(X_te[j:j + 1])[:, 1][0])
            else:
                rp = float("nan")  # No regime model available at this split
            g_probs.append(gp)
            r_probs.append(rp)
            labels.append(int(y_te[j]))
            regs.append(rid)

    g_probs = np.array(g_probs)
    r_probs = np.array(r_probs)
    labels = np.array(labels)
    regs = np.array(regs)
    print(f"\nOOS total: {len(labels)}")
    print(f"Splits skipped per regime (insufficient train): "
          f"CHOP={skipped_regime_train[0]}, BULL={skipped_regime_train[1]}, BEAR={skipped_regime_train[2]}")

    # Overall AUCs
    auc_global_all = float(roc_auc_score(labels, g_probs))
    r_mask = ~np.isnan(r_probs)
    auc_regime_all = float(roc_auc_score(labels[r_mask], r_probs[r_mask])) if r_mask.sum() > 50 else float("nan")

    print("\n" + "=" * 70)
    print(f"{'Regime':<10}{'n':>8}{'AUC global':>14}{'AUC regime':>14}{'Δ':>10}{'regime n':>12}")
    print("-" * 70)
    results_per_regime = {}
    for rid, name in [(0, "CHOP"), (1, "BULL"), (2, "BEAR")]:
        m = regs == rid
        n_r = int(m.sum())
        if n_r < 30 or len(np.unique(labels[m])) < 2:
            print(f"{name:<10}{n_r:>8}{'—':>14}{'—':>14}{'—':>10}{'—':>12}")
            continue
        a_g = float(roc_auc_score(labels[m], g_probs[m]))
        mr = m & ~np.isnan(r_probs)
        n_r_r = int(mr.sum())
        if n_r_r >= 30 and len(np.unique(labels[mr])) >= 2:
            a_r = float(roc_auc_score(labels[mr], r_probs[mr]))
            delta = a_r - a_g
        else:
            a_r = float("nan")
            delta = float("nan")
        print(f"{name:<10}{n_r:>8}{a_g:>14.4f}{a_r:>14.4f}{delta:>+10.4f}{n_r_r:>12}")
        results_per_regime[name] = {
            "n_oos": n_r,
            "auc_global": a_g,
            "auc_regime": a_r,
            "delta": delta,
            "n_regime_oos": n_r_r,
        }

    print("-" * 70)
    print(f"{'ALL':<10}{len(labels):>8}{auc_global_all:>14.4f}{auc_regime_all:>14.4f}"
          f"{(auc_regime_all - auc_global_all):>+10.4f}{int(r_mask.sum()):>12}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    winners = [n for n, r in results_per_regime.items()
               if not np.isnan(r["delta"]) and r["delta"] >= 0.02]
    if winners:
        print(f"  Candidate(s) for Phase 2 validation: {winners}")
        print("  → Run bootstrap CI + time stability on these regimes only.")
    else:
        print("  No regime beats global by >= 0.02 AUC.")
        print("  → Regime-specific sub-models do not break the ceiling.")
        print("  → Direction structural ceiling is confirmed across all partitioning strategies.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_oos": int(len(labels)),
        "auc_global_all": auc_global_all,
        "auc_regime_all": auc_regime_all,
        "per_regime": results_per_regime,
        "skipped_splits_per_regime": skipped_regime_train,
        "horizon": HORIZON,
        "k_deadzone": K_DEADZONE,
        "n_features": len(feats),
    }, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    run()
