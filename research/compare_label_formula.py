"""
Apples-to-apples comparison: endpoint label vs TWAP (path) label.

Both configs use:
  - Same 35 features (ablation baseline + KEEP)
  - Same walk-forward (post-fix purge=4, embargo=4)
  - Same XGBoost params, NO early stopping (honest methodology)
  - Same IC denominator: realized endpoint return close[t+4]/close[t]-1

Only the LABEL formula differs:
  - endpoint: y_dir thresholded on close[t+4]/close[t]-1
  - path:     y_dir thresholded on mean(close[t+1..t+4])/close[t]-1

Reports delta in AUC, accuracy, IC, and strong-signal winrate.
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
from research.dual_model.build_direction_labels import build_direction_labels

ABLATION = Path("research/results/ablation_study.json")
OUT = Path("research/results/compare_label_formula.json")

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}


def wf(df: pd.DataFrame, feats: list[str], y: np.ndarray,
       end_ret: np.ndarray, weights: np.ndarray) -> dict:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
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

        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0
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

    auc = float(roc_auc_score(labels, probs))
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    ic_end, _ = spearmanr(probs, ends)

    # Strong signal winrate: prob>=0.65 → expect UP (endpoint > 0),
    # prob<=0.35 → expect DOWN (endpoint < 0)
    strong_up = probs >= 0.65
    strong_dn = probs <= 0.35
    strong_n = int(strong_up.sum() + strong_dn.sum())
    wins = int((ends[strong_up] > 0).sum() + (ends[strong_dn] < 0).sum())
    wr = wins / strong_n if strong_n > 0 else 0.0

    return {
        "n_oos": int(len(probs)),
        "n_train_labels": int(np.isfinite(labels).sum()),
        "auc": auc,
        "acc": acc,
        "ic_endpoint": float(ic_end),
        "strong_n": strong_n,
        "strong_winrate": float(wr),
    }


def main() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)

    abl = json.loads(ABLATION.read_text())
    base = abl.get("baseline", {}).get("features", [])
    keep = abl.get("combined", {}).get("keep_features", [])
    seen: set[str] = set()
    feats = [f for f in base + keep if f in df.columns and not (f in seen or seen.add(f))]
    print(f"Features: {len(feats)}")
    print(f"Rows: {len(df)}")

    # Regime weights
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    weights = np.ones(len(df))
    bull_mask = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear_mask = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    weights[bull_mask] = 4.0
    weights[bear_mask] = 2.0

    # Build both label flavours from the same file. Current build_direction_labels
    # exposes both return_4h (endpoint) and path_return_4h (TWAP). For the
    # endpoint config, threshold on return_4h using the same k; for path, use
    # path_return_4h (which is already what build_direction_labels produces).
    lab = build_direction_labels(df, k=0.5)  # y_dir is ALREADY TWAP now
    end_ret = lab["return_4h"].values.astype(float)
    path_ret = lab["path_return_4h"].values.astype(float)
    vol = lab["vol"].values.astype(float)

    configs = [
        # (name, k, label_array)
        ("endpoint_k0.50",
         _labelize(end_ret,  k=0.50, vol=vol)),
        ("path_k0.30",
         _labelize(path_ret, k=0.30, vol=vol)),
        ("path_k0.35",
         _labelize(path_ret, k=0.35, vol=vol)),
        ("path_k0.40",
         _labelize(path_ret, k=0.40, vol=vol)),
    ]

    results = {}
    for name, y in configs:
        n_up = int(np.nansum(y == 1))
        n_down = int(np.nansum(y == 0))
        n_total = len(df) - 4
        up_pct = 100 * n_up / n_total
        dn_pct = 100 * n_down / n_total
        dead_pct = 100 - up_pct - dn_pct
        print(f"\n=== {name} ===")
        print(f"  UP={up_pct:.1f}% DOWN={dn_pct:.1f}% dead={dead_pct:.1f}%")
        m = wf(df, feats, y, end_ret, weights)
        print(f"  AUC={m['auc']:.4f} Acc={m['acc']:.1%} IC_end={m['ic_endpoint']:+.4f} "
              f"Strong n={m['strong_n']} wr={m['strong_winrate']:.1%}")
        results[name] = {"up_pct": up_pct, "down_pct": dn_pct, "dead_pct": dead_pct, **m}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")

    # Diff summary
    print("\n" + "=" * 90)
    print(f"{'config':<20} {'UP%':>6} {'AUC':>8} {'Acc':>7} {'IC_end':>10} {'StrN':>6} {'StrWR':>7}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<20} {r['up_pct']:>5.1f}% {r['auc']:>8.4f} {r['acc']:>6.1%} "
              f"{r['ic_endpoint']:>+10.4f} {r['strong_n']:>6d} {r['strong_winrate']:>6.1%}")


def _labelize(ret: np.ndarray, k: float, vol: np.ndarray) -> np.ndarray:
    y = np.full(len(ret), np.nan)
    th = k * vol
    y[ret > th] = 1.0
    y[ret < -th] = 0.0
    return y


if __name__ == "__main__":
    main()
