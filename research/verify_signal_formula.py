"""
Compare OLD vs NEW confidence formula — Strong signal winrate.

OLD formula (current production):
    dir_score = (|P-0.5|/0.5)^0.6 * 80
    mag_bonus = mag_score/100 * 20
    confidence = dir_score + mag_bonus
    Strong tier: confidence >= 80

NEW formula (proposed Path A):
    mag_primary = mag_score/100 * 85
    dir_bonus = if |P-0.5|>=0.15: ((|P-0.5|-0.10)/0.40)*15 else 0
    confidence = mag_primary + dir_bonus
    Strong tier: confidence >= 85 AND |P-0.5| >= 0.15 AND mag_score >= 85

Gate metric: Strong-signal directional winrate on realized 4h return sign.
Also report: Strong count, Moderate count, rank-IC of confidence vs |ret_4h|.
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
from research.dual_model.magnitude_features_v2 import EXPANDED_WITH_FRAGILITY, filter_available

ABLATION = Path("research/results/ablation_study.json")
OUT = Path("research/results/verify_signal_formula.json")

HORIZON = 4
PURGE = 4
EMBARGO = 4

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
}
MAG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
}


def get_dir_feats(df: pd.DataFrame) -> list[str]:
    abl = json.loads(ABLATION.read_text())
    base = abl.get("baseline", {}).get("features", [])
    keep = abl.get("combined", {}).get("keep_features", [])
    seen, out = set(), []
    for f in base + keep:
        if f in df.columns and f not in seen:
            out.append(f)
            seen.add(f)
    return out


def build_dir_labels(df: pd.DataFrame, horizon: int = HORIZON, k: float = 0.5) -> np.ndarray:
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


def run_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with dir_prob, mag_pred, realized_ret_4h per OOS bar."""
    dir_feats = get_dir_feats(df)
    mag_feats = filter_available(EXPANDED_WITH_FRAGILITY, list(df.columns))
    print(f"  Dir features: {len(dir_feats)}, Mag features: {len(mag_feats)}")

    y_dir = build_dir_labels(df)
    # Mag target: |return_4h|
    ret_4h = (df["close"].shift(-HORIZON) / df["close"] - 1).values
    y_mag = np.abs(ret_4h)
    weights = make_weights(df)

    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=PURGE, embargo=EMBARGO,
    )

    records = []
    for tr_idx, te_idx in splits:
        # Direction training (only non-NaN labels)
        tr_mask_d = ~np.isnan(y_dir[tr_idx])
        tr_mask_m = ~np.isnan(y_mag[tr_idx])
        te_mask_m = ~np.isnan(y_mag[te_idx]) & ~np.isnan(ret_4h[te_idx])
        if tr_mask_d.sum() < 50 or te_mask_m.sum() < 5:
            continue
        if len(np.unique(y_dir[tr_idx][tr_mask_d].astype(int))) < 2:
            continue

        # Train direction
        X_tr_d = df.iloc[tr_idx][dir_feats].fillna(0).values[tr_mask_d]
        y_tr_d = y_dir[tr_idx][tr_mask_d].astype(int)
        sw_d = weights[tr_idx][tr_mask_d]
        up_rate = y_tr_d.mean()
        p_d = DIR_PARAMS.copy()
        p_d["scale_pos_weight"] = (1 - up_rate) / up_rate if 0 < up_rate < 1 else 1.0
        m_d = xgb.XGBClassifier(**p_d)
        m_d.fit(X_tr_d, y_tr_d, sample_weight=sw_d, verbose=False)

        # Train magnitude
        X_tr_m = df.iloc[tr_idx][mag_feats].fillna(0).values[tr_mask_m]
        y_tr_m = y_mag[tr_idx][tr_mask_m]
        m_m = xgb.XGBRegressor(**MAG_PARAMS)
        m_m.fit(X_tr_m, y_tr_m, verbose=False)

        # Predict on test
        X_te_d = df.iloc[te_idx][dir_feats].fillna(0).values[te_mask_m]
        X_te_m = df.iloc[te_idx][mag_feats].fillna(0).values[te_mask_m]
        if len(X_te_d) == 0:
            continue
        dir_prob = m_d.predict_proba(X_te_d)[:, 1]
        mag_pred = m_m.predict(X_te_m)
        actual = ret_4h[te_idx][te_mask_m]
        for dp, mp, ar in zip(dir_prob, mag_pred, actual):
            records.append({"dir_prob": float(dp), "mag_pred": float(mp), "ret_4h": float(ar)})

    return pd.DataFrame(records)


def compute_mag_score(mag_pred: np.ndarray) -> np.ndarray:
    """Expanding percentile rank × 100 (similar to inference.py _compute_mag_score)."""
    n = len(mag_pred)
    out = np.full(n, 50.0)
    for i in range(n):
        window = mag_pred[max(0, i - 500):i + 1]
        if len(window) >= 30:
            out[i] = (window < mag_pred[i]).mean() * 100
    return out


def formula_old(dir_prob: np.ndarray, mag_score: np.ndarray) -> np.ndarray:
    dir_dist = np.abs(dir_prob - 0.5)
    dir_s = (dir_dist / 0.5) ** 0.6 * 80
    mag_b = (mag_score / 100) * 20
    return np.clip(dir_s + mag_b, 0, 100)


def formula_new(dir_prob: np.ndarray, mag_score: np.ndarray) -> np.ndarray:
    mag_p = (mag_score / 100) * 85
    dir_dist = np.abs(dir_prob - 0.5)
    dir_b = np.where(
        dir_dist >= 0.15,
        np.clip(((dir_dist - 0.10) / 0.40) * 15, 0, 15),
        0.0,
    )
    return np.clip(mag_p + dir_b, 0, 100)


def evaluate(name: str, conf: np.ndarray, dir_prob: np.ndarray, mag_score: np.ndarray,
             ret: np.ndarray, strong_th: float, strict_new: bool = False) -> dict:
    # Determine predicted direction (UP if dir_prob > 0.5 else DOWN)
    pred_up = dir_prob > 0.5
    actual_up = ret > 0

    # Strong mask
    if strict_new:
        strong = (conf >= strong_th) & (np.abs(dir_prob - 0.5) >= 0.15) & (mag_score >= 85)
    else:
        strong = conf >= strong_th

    mod_mask = (conf >= 65) & (conf < strong_th)

    strong_n = int(strong.sum())
    mod_n = int(mod_mask.sum())
    if strong_n > 0:
        strong_wr = float((pred_up[strong] == actual_up[strong]).mean())
        strong_mean_abs_ret = float(np.abs(ret[strong]).mean())
    else:
        strong_wr = float("nan")
        strong_mean_abs_ret = float("nan")
    if mod_n > 0:
        mod_wr = float((pred_up[mod_mask] == actual_up[mod_mask]).mean())
    else:
        mod_wr = float("nan")

    # Rank-IC: confidence vs |return|
    ic_abs, _ = spearmanr(conf, np.abs(ret))

    return {
        "name": name,
        "strong_n": strong_n,
        "strong_winrate": strong_wr,
        "strong_mean_abs_ret": strong_mean_abs_ret,
        "moderate_n": mod_n,
        "moderate_winrate": mod_wr,
        "ic_conf_vs_absret": float(ic_abs),
    }


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Rows: {len(df)}")

    print("Running walk-forward...")
    oos = run_walkforward(df)
    print(f"OOS bars: {len(oos)}")
    if len(oos) < 100:
        print("Too few OOS samples, abort.")
        return

    dir_prob = oos["dir_prob"].values
    mag_pred = oos["mag_pred"].values
    ret = oos["ret_4h"].values
    mag_score = compute_mag_score(mag_pred)

    conf_old = formula_old(dir_prob, mag_score)
    conf_new = formula_new(dir_prob, mag_score)

    print("\n" + "=" * 72)
    print("OLD formula (current production, Strong >= 80)")
    print("=" * 72)
    r_old = evaluate("OLD", conf_old, dir_prob, mag_score, ret, strong_th=80.0, strict_new=False)
    for k, v in r_old.items():
        print(f"  {k:25s} {v}")

    print("\n" + "=" * 72)
    print("NEW formula (Path A, Strong >= 85 + dir>=0.15 + mag>=85)")
    print("=" * 72)
    r_new = evaluate("NEW", conf_new, dir_prob, mag_score, ret, strong_th=85.0, strict_new=True)
    for k, v in r_new.items():
        print(f"  {k:25s} {v}")

    # Alt: NEW formula with only confidence threshold (no double-gate)
    print("\n" + "=" * 72)
    print("NEW-lite (Path A, Strong >= 85, no extra gates)")
    print("=" * 72)
    r_new_lite = evaluate("NEW_lite", conf_new, dir_prob, mag_score, ret, strong_th=85.0, strict_new=False)
    for k, v in r_new_lite.items():
        print(f"  {k:25s} {v}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    delta_wr = (r_new["strong_winrate"] - r_old["strong_winrate"]) if not np.isnan(r_new["strong_winrate"]) else float("nan")
    print(f"  Strong winrate  OLD={r_old['strong_winrate']:.4f}  NEW={r_new['strong_winrate']:.4f}  Δ={delta_wr:+.4f}")
    print(f"  Strong count    OLD={r_old['strong_n']:>4d}  NEW={r_new['strong_n']:>4d}  NEW_lite={r_new_lite['strong_n']:>4d}")
    print(f"  Conf↔|ret| IC   OLD={r_old['ic_conf_vs_absret']:+.4f}  NEW={r_new_lite['ic_conf_vs_absret']:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"old": r_old, "new_strict": r_new, "new_lite": r_new_lite,
                               "n_oos": int(len(oos))}, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    run()
