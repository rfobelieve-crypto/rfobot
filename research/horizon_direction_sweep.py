"""
Horizon sweep: does direction become predictable at different horizons?

Tests direction AUC across horizons [1, 2, 4, 8, 12, 24, 48] hours using
the same 35-feature ablation set. Each horizon gets its own purge+embargo
matched to the label horizon (purge = embargo = horizon bars).

Label construction: endpoint binary, k * realized_vol_24h * sqrt(horizon)
deadzone, so each horizon has a comparable "meaningful-move" threshold.

Goal: identify whether direction signal exists at some horizon the current
system is not using. Answers: "should we change horizon, or is 4h not the
problem?"
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits

ABLATION = Path("research/results/ablation_study.json")
OUT = Path("research/results/horizon_direction_sweep.json")

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}

HORIZONS = [1, 2, 4, 8, 12, 24, 48]
K_DEADZONE = 0.5  # matches production


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
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    w = np.ones(len(df))
    bull = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    w[bull] = 4.0
    w[bear] = 2.0
    return w


def build_labels(df: pd.DataFrame, horizon: int, k: float = K_DEADZONE) -> tuple[np.ndarray, np.ndarray]:
    """Vol-adaptive binary labels at given horizon.

    Threshold = k * realized_vol_1h * sqrt(horizon), so the difficulty of
    a meaningful move is roughly constant across horizons.
    """
    close = df["close"].values.astype(float)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_1h = log_ret.rolling(20, min_periods=10).std().values

    n = len(close)
    y = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for i in range(n - horizon):
        if np.isnan(vol_1h[i]) or vol_1h[i] <= 0:
            continue
        r = close[i + horizon] / close[i] - 1
        th = k * vol_1h[i] * np.sqrt(horizon)
        ret[i] = r
        if r > th:
            y[i] = 1.0
        elif r < -th:
            y[i] = 0.0
    return y, ret


def wf_eval(df: pd.DataFrame, feats: list[str], y: np.ndarray,
            end_ret: np.ndarray, weights: np.ndarray,
            purge: int, embargo: int) -> dict:
    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=purge, embargo=embargo,
    )
    probs, labels, ends = [], [], []

    for tr_idx, te_idx in splits:
        tr_mask = ~np.isnan(y[tr_idx])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue
        X_tr = df.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_idx][tr_mask].astype(int)
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
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

        te_global = np.array(te_idx)[te_mask]
        probs.extend(pr.tolist())
        labels.extend(y_te.tolist())
        ends.extend(end_ret[te_global].tolist())

    probs = np.array(probs)
    labels = np.array(labels)
    ends = np.array(ends)
    if len(probs) < 50 or len(np.unique(labels)) < 2:
        return {"n_oos": int(len(probs)), "auc": float("nan"), "acc": float("nan"),
                "ic_endpoint": float("nan"), "strong_n": 0, "strong_winrate": float("nan")}

    auc = float(roc_auc_score(labels, probs))
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    ic_end, _ = spearmanr(probs, ends)

    strong_up = probs >= 0.65
    strong_dn = probs <= 0.35
    strong_n = int(strong_up.sum() + strong_dn.sum())
    wins = int((ends[strong_up] > 0).sum() + (ends[strong_dn] < 0).sum())
    wr = wins / strong_n if strong_n > 0 else float("nan")

    return {
        "n_oos": int(len(probs)),
        "auc": auc,
        "acc": acc,
        "ic_endpoint": float(ic_end),
        "strong_n": strong_n,
        "strong_winrate": float(wr),
    }


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    feats = get_feats(df)
    print(f"Features: {len(feats)}")
    print(f"Rows: {len(df)}")

    weights = make_weights(df)

    results = []
    print(f"\n{'h':>4} {'UP%':>7} {'DN%':>7} {'dead%':>7} {'AUC':>8} {'Acc':>8} {'IC':>10} {'StrN':>6} {'StrWR':>7}")
    print("-" * 75)

    for h in HORIZONS:
        y, ret = build_labels(df, horizon=h, k=K_DEADZONE)
        n_total = int(np.isfinite(ret).sum())
        n_up = int(np.nansum(y == 1))
        n_dn = int(np.nansum(y == 0))
        n_dead = n_total - n_up - n_dn
        up_pct = 100 * n_up / n_total if n_total else 0
        dn_pct = 100 * n_dn / n_total if n_total else 0
        dead_pct = 100 * n_dead / n_total if n_total else 0

        # Purge/embargo grow with horizon to prevent label leakage
        purge = h
        embargo = h
        m = wf_eval(df, feats, y, ret, weights, purge=purge, embargo=embargo)

        print(f"{h:>4} {up_pct:>6.1f}% {dn_pct:>6.1f}% {dead_pct:>6.1f}% "
              f"{m['auc']:>8.4f} {m['acc']:>7.1%} {m['ic_endpoint']:>+10.4f} "
              f"{m['strong_n']:>6d} {m['strong_winrate']:>6.1%}")
        results.append({
            "horizon_h": h,
            "up_pct": up_pct, "down_pct": dn_pct, "dead_pct": dead_pct,
            **m,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"horizons": results, "k_deadzone": K_DEADZONE, "n_features": len(feats)}, indent=2))
    print(f"\nSaved: {OUT}")

    # Verdict
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    best = max(results, key=lambda r: r["auc"] if not np.isnan(r["auc"]) else 0)
    print(f"  Best horizon: {best['horizon_h']}h  AUC={best['auc']:.4f} StrWR={best['strong_winrate']:.1%}")
    curr_4h = next(r for r in results if r["horizon_h"] == 4)
    delta = best["auc"] - curr_4h["auc"]
    print(f"  Delta vs 4h baseline (AUC {curr_4h['auc']:.4f}): {delta:+.4f}")
    if best["horizon_h"] != 4 and delta >= 0.02:
        print(f"  → Horizon {best['horizon_h']}h is meaningfully better. Consider migrating.")
    elif all(r["auc"] < 0.56 for r in results if not np.isnan(r["auc"])):
        print("  → ALL horizons are weak (AUC < 0.56). Direction is structurally hard at every scale tested.")
        print("     Recommendation: system re-positioning (path 1 or 3).")
    else:
        print("  → Mixed. See table.")


if __name__ == "__main__":
    run()
