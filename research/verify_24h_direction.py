"""
Two validation experiments for migrating Direction model to 24h horizon.

Exp 1 — Bootstrap CI on 24h AUC
    Walk-forward OOS predictions, then bootstrap (n=1000) on (prob, label)
    pairs. Report 95% CI of AUC. Gate: CI lower bound > 0.55.

Exp 2 — Time stability (first half vs second half)
    Split the 4000 bars into two halves chronologically. Run walk-forward
    on each half independently. Report per-half AUC. Gate: abs(diff) < 0.04.

Pass both → safe to migrate.
Fail either → revert to 4h + only change signal-generation logic (path 1).
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
OUT = Path("research/results/verify_24h_direction.json")

HORIZON = 24
K_DEADZONE = 0.5
PURGE = 24
EMBARGO = 24
N_BOOT = 1000
RNG_SEED = 42

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}


def get_feats(df: pd.DataFrame) -> list[str]:
    abl = json.loads(ABLATION.read_text())
    base = abl.get("baseline", {}).get("features", [])
    keep = abl.get("combined", {}).get("keep_features", [])
    seen: set[str] = set()
    out = []
    for f in base + keep:
        if f in df.columns and f not in seen:
            out.append(f)
            seen.add(f)
    return out


def make_weights(df: pd.DataFrame) -> np.ndarray:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h_s = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    w = np.ones(len(df))
    bull = ((vol_pct > 0.6) & (ret_24h_s > 0.005)).fillna(False).values
    bear = ((vol_pct > 0.6) & (ret_24h_s < -0.005)).fillna(False).values
    w[bull] = 4.0
    w[bear] = 2.0
    return w


def build_labels(df: pd.DataFrame, horizon: int, k: float) -> np.ndarray:
    close = df["close"].values.astype(float)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_1h = log_ret.rolling(20, min_periods=10).std().values

    n = len(close)
    y = np.full(n, np.nan)
    for i in range(n - horizon):
        if np.isnan(vol_1h[i]) or vol_1h[i] <= 0:
            continue
        r = close[i + horizon] / close[i] - 1
        th = k * vol_1h[i] * np.sqrt(horizon)
        if r > th:
            y[i] = 1.0
        elif r < -th:
            y[i] = 0.0
    return y


def run_wf(df_slice: pd.DataFrame, feats: list[str], y: np.ndarray,
           weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward on a slice. Returns (probs, labels)."""
    splits = walk_forward_splits(
        len(df_slice), initial_train=288, test_size=48, step=48,
        purge=PURGE, embargo=EMBARGO,
    )
    probs, labels = [], []
    for tr_idx, te_idx in splits:
        tr_mask = ~np.isnan(y[tr_idx])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue
        X_tr = df_slice.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_idx][tr_mask].astype(int)
        X_te = df_slice.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y[te_idx][te_mask].astype(int)
        sw = weights[tr_idx][tr_mask]
        if len(np.unique(y_tr)) < 2:
            continue

        up_rate = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up_rate) / up_rate if 0 < up_rate < 1 else 1.0
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        pr = m.predict_proba(X_te)[:, 1]
        probs.extend(pr.tolist())
        labels.extend(y_te.tolist())
    return np.array(probs), np.array(labels)


def bootstrap_ci(probs: np.ndarray, labels: np.ndarray,
                 n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(probs)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l_boot = labels[idx]
        if len(np.unique(l_boot)) < 2:
            continue
        aucs.append(roc_auc_score(l_boot, probs[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    feats = get_feats(df)
    print(f"Features: {len(feats)}")
    print(f"Rows: {len(df)}")

    weights_full = make_weights(df)
    y_full = build_labels(df, horizon=HORIZON, k=K_DEADZONE)

    # ── Exp 1: Bootstrap CI on full window ────────────────────────
    print("\n[Exp 1] Full-window walk-forward + bootstrap CI")
    probs_full, labels_full = run_wf(df, feats, y_full, weights_full)
    point_auc = float(roc_auc_score(labels_full, probs_full))
    boot_mean, ci_lo, ci_hi = bootstrap_ci(probs_full, labels_full, N_BOOT, RNG_SEED)
    print(f"  n_oos = {len(probs_full)}")
    print(f"  point AUC    = {point_auc:.4f}")
    print(f"  bootstrap mean = {boot_mean:.4f}")
    print(f"  95% CI       = [{ci_lo:.4f}, {ci_hi:.4f}]")
    gate1 = ci_lo > 0.55
    print(f"  Gate (CI lo > 0.55): {'PASS' if gate1 else 'FAIL'}")

    # ── Exp 2: First half vs Second half ──────────────────────────
    print("\n[Exp 2] Time-stability: first half vs second half")
    mid = len(df) // 2
    df1 = df.iloc[:mid].copy()
    df2 = df.iloc[mid:].copy().reset_index(drop=False).set_index(df.index[mid:])
    # Reuse same weight + label slices
    y1 = y_full[:mid]
    y2 = y_full[mid:]
    w1 = weights_full[:mid]
    w2 = weights_full[mid:]

    probs_1, labels_1 = run_wf(df1, feats, y1, w1)
    probs_2, labels_2 = run_wf(df2, feats, y2, w2)

    if len(probs_1) >= 50 and len(np.unique(labels_1)) == 2:
        auc_1 = float(roc_auc_score(labels_1, probs_1))
    else:
        auc_1 = float("nan")
    if len(probs_2) >= 50 and len(np.unique(labels_2)) == 2:
        auc_2 = float(roc_auc_score(labels_2, probs_2))
    else:
        auc_2 = float("nan")

    print(f"  first half  ({df.index[0].date()} → {df.index[mid-1].date()}): "
          f"n={len(probs_1)} AUC={auc_1:.4f}")
    print(f"  second half ({df.index[mid].date()} → {df.index[-1].date()}): "
          f"n={len(probs_2)} AUC={auc_2:.4f}")
    diff = abs(auc_1 - auc_2) if not (np.isnan(auc_1) or np.isnan(auc_2)) else float("inf")
    print(f"  abs diff = {diff:.4f}")
    gate2 = diff < 0.04
    print(f"  Gate (diff < 0.04): {'PASS' if gate2 else 'FAIL'}")

    # Save results
    results = {
        "horizon": HORIZON,
        "k": K_DEADZONE,
        "n_features": len(feats),
        "exp1_bootstrap": {
            "n_oos": int(len(probs_full)),
            "point_auc": point_auc,
            "boot_mean": boot_mean,
            "ci_95_low": ci_lo,
            "ci_95_high": ci_hi,
            "gate_pass": gate1,
        },
        "exp2_time_stability": {
            "first_half_auc": auc_1,
            "second_half_auc": auc_2,
            "abs_diff": diff,
            "first_n": int(len(probs_1)),
            "second_n": int(len(probs_2)),
            "gate_pass": gate2,
        },
        "verdict": "PROCEED_MIGRATE_24H" if (gate1 and gate2) else "REVERT_PATH_1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if gate1 and gate2:
        print("  Both gates PASS — safe to migrate Direction to 24h")
    else:
        print("  At least one gate FAILED")
        if not gate1:
            print(f"    Exp 1: CI lower bound {ci_lo:.4f} is NOT > 0.55")
        if not gate2:
            print(f"    Exp 2: first/second half diff {diff:.4f} >= 0.04")
        print("  Recommendation: revert to 4h + only modify signal-generation logic")


if __name__ == "__main__":
    run()
