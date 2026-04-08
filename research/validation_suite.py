"""
Validation Suite — Monte Carlo permutation test + IC stability analysis.

1. Monte Carlo: shuffle labels N times, compute null distribution, report p-value
2. IC Stability: rolling IC with multiple windows, regime breakdown, sign consistency

Usage:
    python research/validation_suite.py
"""
import sys, os, warnings, logging, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# MKL fix
_orig = np.polyfit
def safe_polyfit(*a, **k):
    try: return _orig(*a, **k)
    except: return [0.0, 0.0]
np.polyfit = safe_polyfit


def monte_carlo_test(y_true, y_pred, metric_fn, n_permutations=1000, seed=42):
    """
    Permutation test: shuffle y_true N times, compute metric under null.

    Returns:
        actual: real metric value
        null_mean: mean under null
        null_std: std under null
        p_value: fraction of null >= actual
    """
    rng = np.random.RandomState(seed)
    actual = metric_fn(y_true, y_pred)

    null_dist = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_true)
        try:
            null_dist.append(metric_fn(y_shuffled, y_pred))
        except Exception:
            continue

    null_dist = np.array(null_dist)
    p_value = (null_dist >= actual).mean()

    return {
        "actual": round(float(actual), 4),
        "null_mean": round(float(null_dist.mean()), 4),
        "null_std": round(float(null_dist.std()), 4),
        "p_value": round(float(p_value), 4),
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
    }


def ic_stability_analysis(y_true, y_pred, windows=[50, 100, 200]):
    """
    Rolling IC with multiple window sizes + sign consistency.

    Returns:
        per-window: mean IC, std, ICIR, positive ratio, sign changes
    """
    results = {}
    for w in windows:
        ics = []
        for i in range(w, len(y_true)):
            try:
                ic, _ = spearmanr(y_pred[i-w:i], y_true[i-w:i])
                if not np.isnan(ic):
                    ics.append(ic)
            except Exception:
                continue

        if len(ics) < 5:
            results[f"w{w}"] = {"mean": None, "note": "insufficient data"}
            continue

        ics = np.array(ics)
        mean_ic = ics.mean()
        std_ic = ics.std()
        icir = mean_ic / std_ic if std_ic > 0 else 0
        pos_ratio = (ics > 0).mean()

        # Sign changes
        signs = np.sign(ics)
        sign_changes = np.sum(signs[1:] != signs[:-1]) / len(signs)

        results[f"w{w}"] = {
            "mean_ic": round(float(mean_ic), 4),
            "std_ic": round(float(std_ic), 4),
            "icir": round(float(icir), 3),
            "positive_ratio": round(float(pos_ratio), 3),
            "sign_flip_rate": round(float(sign_changes), 3),
            "n_windows": len(ics),
        }

    return results


def regime_ic_breakdown(y_true, y_pred, regimes):
    """IC breakdown by regime."""
    results = {}
    for regime in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        mask = regimes == regime
        if mask.sum() < 20:
            continue
        try:
            ic, p = spearmanr(y_pred[mask], y_true[mask])
            results[regime] = {
                "ic": round(float(ic), 4),
                "p_value": round(float(p), 4),
                "n_bars": int(mask.sum()),
            }
        except Exception:
            pass
    return results


def main():
    from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
    from research.dual_model.build_direction_labels import build_direction_labels
    from research.dual_model.build_magnitude_labels import build_magnitude_labels
    import xgboost as xgb

    # Load pruned features
    pruning = json.loads(Path("research/results/auto_pruning.json").read_text())
    dir_feats = pruning["direction"]["features"]
    mag_feats = pruning["magnitude"]["features"]

    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)

    dir_labels = build_direction_labels(df, k=0.5)
    df["y_dir"] = dir_labels["y_dir"]
    df["dir_return_4h"] = dir_labels["return_4h"]
    mag_labels = build_magnitude_labels(df)
    df["y_vol_adj_abs"] = mag_labels["y_vol_adj_abs"]
    df["y_abs_return"] = mag_labels["y_abs_return"]

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)

    DIR_PARAMS = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
        "verbosity": 0, "early_stopping_rounds": 30,
    }
    MAG_PARAMS = {
        "objective": "reg:squarederror", "eval_metric": "mae",
        "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
        "verbosity": 0, "early_stopping_rounds": 30,
    }

    # ── Collect OOS predictions ──
    print("Running walk-forward for OOS predictions...")
    dir_probs, dir_true, dir_rets, dir_regimes = [], [], [], []
    mag_preds, mag_true = [], []

    # Compute regime for each bar
    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    ret_24h = close.pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    regime_arr = np.full(len(df), "CHOPPY", dtype=object)
    regime_arr[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
    regime_arr[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"

    for tr_idx, te_idx in splits:
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]

        # Direction
        tm, ttm = tr["y_dir"].notna(), te["y_dir"].notna()
        X_t = tr.loc[tm, dir_feats].fillna(0)
        y_t = tr.loc[tm, "y_dir"].values.astype(int)
        X_e = te.loc[ttm, dir_feats].fillna(0)
        y_e = te.loc[ttm, "y_dir"].values.astype(int)
        r_e = te.loc[ttm, "dir_return_4h"].values
        reg_e = regime_arr[te_idx][ttm.values]

        if len(y_t) >= 50 and len(y_e) >= 5:
            up = y_t.mean()
            p = DIR_PARAMS.copy()
            p["scale_pos_weight"] = (1-up)/up if up > 0 else 1.0
            m = xgb.XGBClassifier(**p)
            m.fit(X_t, y_t, eval_set=[(X_e, y_e)], verbose=False)
            dir_probs.extend(m.predict_proba(X_e)[:,1])
            dir_true.extend(y_e)
            dir_rets.extend(r_e)
            dir_regimes.extend(reg_e)

        # Magnitude
        tm2, ttm2 = tr["y_vol_adj_abs"].notna(), te["y_vol_adj_abs"].notna()
        X_t2 = tr.loc[tm2, mag_feats].fillna(0)
        y_t2 = tr.loc[tm2, "y_vol_adj_abs"].values
        X_e2 = te.loc[ttm2, mag_feats].fillna(0)
        y_e2 = te.loc[ttm2, "y_vol_adj_abs"].values

        if len(y_t2) >= 50 and len(y_e2) >= 5:
            m2 = xgb.XGBRegressor(**MAG_PARAMS)
            m2.fit(X_t2, y_t2, eval_set=[(X_e2, y_e2)], verbose=False)
            mag_preds.extend(m2.predict(X_e2))
            mag_true.extend(y_e2)

    dp = np.array(dir_probs)
    dt = np.array(dir_true)
    dr = np.array(dir_rets)
    d_reg = np.array(dir_regimes)
    mp = np.array(mag_preds)
    mt = np.array(mag_true)

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # MONTE CARLO PERMUTATION TEST
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MONTE CARLO PERMUTATION TEST (N=1000)")
    print("=" * 60)

    print("\nDirection AUC...")
    mc_dir_auc = monte_carlo_test(dt, dp, roc_auc_score, n_permutations=1000)
    print(f"  AUC = {mc_dir_auc['actual']}  (null: {mc_dir_auc['null_mean']} +/- {mc_dir_auc['null_std']})")
    print(f"  p-value = {mc_dir_auc['p_value']}  {'*** SIGNIFICANT' if mc_dir_auc['significant_1pct'] else '** sig' if mc_dir_auc['significant_5pct'] else 'NOT significant'}")

    print("\nDirection IC...")
    mc_dir_ic = monte_carlo_test(dr, dp, lambda y, p: spearmanr(p, y)[0], n_permutations=1000)
    print(f"  IC = {mc_dir_ic['actual']}  (null: {mc_dir_ic['null_mean']} +/- {mc_dir_ic['null_std']})")
    print(f"  p-value = {mc_dir_ic['p_value']}  {'*** SIGNIFICANT' if mc_dir_ic['significant_1pct'] else 'NOT significant'}")

    print("\nMagnitude IC...")
    mc_mag_ic = monte_carlo_test(mt, mp, lambda y, p: spearmanr(p, y)[0], n_permutations=1000)
    print(f"  IC = {mc_mag_ic['actual']}  (null: {mc_mag_ic['null_mean']} +/- {mc_mag_ic['null_std']})")
    print(f"  p-value = {mc_mag_ic['p_value']}  {'*** SIGNIFICANT' if mc_mag_ic['significant_1pct'] else 'NOT significant'}")

    results["monte_carlo"] = {
        "dir_auc": mc_dir_auc, "dir_ic": mc_dir_ic, "mag_ic": mc_mag_ic,
    }

    # ═══════════════════════════════════════════════════════════════
    # IC STABILITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("IC STABILITY (rolling windows)")
    print("=" * 60)

    print("\nDirection IC stability:")
    dir_stability = ic_stability_analysis(dr, dp, windows=[50, 100, 200])
    for w, s in dir_stability.items():
        if s.get("mean_ic") is not None:
            print(f"  {w}: IC={s['mean_ic']:+.4f} +/- {s['std_ic']:.4f}, ICIR={s['icir']:+.3f}, positive={s['positive_ratio']:.0%}, flips={s['sign_flip_rate']:.0%}")

    print("\nMagnitude IC stability:")
    mag_stability = ic_stability_analysis(mt, mp, windows=[50, 100, 200])
    for w, s in mag_stability.items():
        if s.get("mean_ic") is not None:
            print(f"  {w}: IC={s['mean_ic']:+.4f} +/- {s['std_ic']:.4f}, ICIR={s['icir']:+.3f}, positive={s['positive_ratio']:.0%}, flips={s['sign_flip_rate']:.0%}")

    results["ic_stability"] = {"direction": dir_stability, "magnitude": mag_stability}

    # ═══════════════════════════════════════════════════════════════
    # REGIME BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("REGIME IC BREAKDOWN")
    print("=" * 60)

    print("\nDirection:")
    dir_regime = regime_ic_breakdown(dr, dp, d_reg)
    for regime, r in dir_regime.items():
        sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else ""
        print(f"  {regime:<20} IC={r['ic']:+.4f} p={r['p_value']:.4f} n={r['n_bars']} {sig}")

    results["regime_breakdown"] = {"direction": dir_regime}

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\n  Direction Model (pruned {len(dir_feats)} features):")
    print(f"    AUC = {mc_dir_auc['actual']}  p-value = {mc_dir_auc['p_value']}")
    print(f"    IC  = {mc_dir_ic['actual']}  p-value = {mc_dir_ic['p_value']}")
    d100 = dir_stability.get("w100", {})
    print(f"    IC stability (w100): ICIR={d100.get('icir','?')}, positive={d100.get('positive_ratio','?')}")

    print(f"\n  Magnitude Model (pruned {len(mag_feats)} features):")
    print(f"    IC  = {mc_mag_ic['actual']}  p-value = {mc_mag_ic['p_value']}")
    m100 = mag_stability.get("w100", {})
    print(f"    IC stability (w100): ICIR={m100.get('icir','?')}, positive={m100.get('positive_ratio','?')}")

    verdict = "PASS" if mc_dir_auc["significant_5pct"] and mc_mag_ic["significant_1pct"] else "REVIEW"
    print(f"\n  Overall verdict: {verdict}")

    # Save
    out = Path("research/results/validation_suite.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
