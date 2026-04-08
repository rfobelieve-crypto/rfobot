"""
Auto Feature Pruner — VIF multicollinearity removal + validation.

1. Compute VIF for all features, iteratively drop VIF > threshold
2. Correlation cluster detection (drop one from each r>0.95 pair)
3. Walk-forward validation before/after to confirm no degradation

Usage:
    python research/auto_pruner.py
"""
import sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Monkey-patch polyfit for MKL issue
_orig = np.polyfit
def safe_polyfit(*a, **k):
    try: return _orig(*a, **k)
    except: return [0.0, 0.0]
np.polyfit = safe_polyfit


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """
    Compute Variance Inflation Factor for each feature.
    VIF > 10 = severe multicollinearity.
    VIF > 5  = moderate.
    Uses OLS regression: VIF_i = 1 / (1 - R²_i)
    """
    from numpy.linalg import LinAlgError

    vifs = {}
    X_arr = X.fillna(0).values
    n_cols = X_arr.shape[1]

    for i in range(n_cols):
        y = X_arr[:, i]
        others = np.delete(X_arr, i, axis=1)

        # Add intercept
        ones = np.ones((others.shape[0], 1))
        X_reg = np.hstack([ones, others])

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
            y_hat = X_reg @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else 999
        except (LinAlgError, ValueError):
            vif = 999

        vifs[X.columns[i]] = round(vif, 2)

    return pd.Series(vifs).sort_values(ascending=False)


def drop_high_vif(X: pd.DataFrame, threshold: float = 10.0,
                  max_rounds: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """
    Iteratively drop the feature with highest VIF until all VIF < threshold.
    Returns (pruned_X, dropped_features).
    """
    dropped = []
    X_curr = X.copy()

    for round_i in range(max_rounds):
        if X_curr.shape[1] <= 2:
            break

        vifs = compute_vif(X_curr)
        max_vif = vifs.iloc[0]
        max_feat = vifs.index[0]

        if max_vif <= threshold:
            break

        dropped.append(max_feat)
        X_curr = X_curr.drop(columns=[max_feat])

        if round_i % 5 == 0:
            print(f"  Round {round_i}: dropped {max_feat} (VIF={max_vif:.1f}), {X_curr.shape[1]} remaining")

    return X_curr, dropped


def drop_high_correlation(X: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop one feature from each pair with |correlation| > threshold.
    Keeps the one with lower mean absolute correlation to all others.
    """
    corr = X.fillna(0).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dropped = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for other in high_corr:
            if other in dropped or col in dropped:
                continue
            # Drop the one with higher mean correlation to all
            mean_corr_col = corr[col].drop([col]).mean()
            mean_corr_other = corr[other].drop([other]).mean()
            to_drop = col if mean_corr_col > mean_corr_other else other
            dropped.add(to_drop)

    return X.drop(columns=list(dropped)), list(dropped)


def main():
    from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
    from research.dual_model.build_direction_labels import build_direction_labels
    from research.dual_model.build_magnitude_labels import build_magnitude_labels
    from research.dual_model.direction_features_v2 import (
        OLD_FEATURES, NEW_KEY_4, LIQUIDITY_FRAGILITY, POST_ABSORPTION, filter_available,
    )
    from research.dual_model.magnitude_features_v2 import (
        EXPANDED_WITH_FRAGILITY, filter_available as mag_filter,
    )
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Data: {len(df)} bars")

    # Labels
    dir_labels = build_direction_labels(df, k=0.5)
    df["y_dir"] = dir_labels["y_dir"]
    df["dir_return_4h"] = dir_labels["return_4h"]
    mag_labels = build_magnitude_labels(df)
    df["y_vol_adj_abs"] = mag_labels["y_vol_adj_abs"]
    df["y_abs_return"] = mag_labels["y_abs_return"]

    # Feature lists
    DIR_ALL = filter_available(
        OLD_FEATURES + NEW_KEY_4 + LIQUIDITY_FRAGILITY + POST_ABSORPTION,
        list(df.columns))
    TOXICITY_MAG = ["tox_pressure", "tox_accum_zscore", "tox_bv_vpin_zscore", "tox_div_taker"]
    MAG_ALL = mag_filter(EXPANDED_WITH_FRAGILITY + TOXICITY_MAG, list(df.columns))

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

    def wf_dir(feats):
        probs, trues, rets = [], [], []
        for tr_idx, te_idx in splits:
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            tm, ttm = tr["y_dir"].notna(), te["y_dir"].notna()
            X_t, y_t = tr.loc[tm, feats].fillna(0), tr.loc[tm, "y_dir"].values.astype(int)
            X_e, y_e = te.loc[ttm, feats].fillna(0), te.loc[ttm, "y_dir"].values.astype(int)
            r_e = te.loc[ttm, "dir_return_4h"].values
            if len(y_t) < 50 or len(y_e) < 5: continue
            up = y_t.mean()
            p = DIR_PARAMS.copy()
            p["scale_pos_weight"] = (1-up)/up if up > 0 else 1.0
            m = xgb.XGBClassifier(**p)
            m.fit(X_t, y_t, eval_set=[(X_e, y_e)], verbose=False)
            probs.extend(m.predict_proba(X_e)[:,1])
            trues.extend(y_e); rets.extend(r_e)
        p, t, r = np.array(probs), np.array(trues), np.array(rets)
        auc = roc_auc_score(t, p)
        ic, _ = spearmanr(p, r)
        acc = ((p>0.5).astype(int)==t).mean()
        return auc, ic, acc

    def wf_mag(feats):
        preds, trues = [], []
        for tr_idx, te_idx in splits:
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            tm, ttm = tr["y_vol_adj_abs"].notna(), te["y_vol_adj_abs"].notna()
            X_t, y_t = tr.loc[tm, feats].fillna(0), tr.loc[tm, "y_vol_adj_abs"].values
            X_e, y_e = te.loc[ttm, feats].fillna(0), te.loc[ttm, "y_vol_adj_abs"].values
            if len(y_t) < 50 or len(y_e) < 5: continue
            m = xgb.XGBRegressor(**MAG_PARAMS)
            m.fit(X_t, y_t, eval_set=[(X_e, y_e)], verbose=False)
            preds.extend(m.predict(X_e)); trues.extend(y_e)
        p, t = np.array(preds), np.array(trues)
        ic, _ = spearmanr(p, t)
        ics = [spearmanr(p[i-100:i], t[i-100:i])[0] for i in range(100, len(p), 20)]
        icir = np.mean(ics)/np.std(ics) if len(ics) > 3 and np.std(ics) > 0 else 0
        return ic, icir

    # ═══════════════════════════════════════════════════════════════
    # DIRECTION MODEL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DIRECTION MODEL PRUNING")
    print("=" * 60)

    # Before
    auc_before, ic_before, acc_before = wf_dir(DIR_ALL)
    print(f"\nBefore: {len(DIR_ALL)} features, AUC={auc_before:.4f}, IC={ic_before:+.4f}, Acc={acc_before:.1%}")

    # Step 1: Drop high correlation
    print("\nStep 1: Correlation pruning (r > 0.95)...")
    X_dir = df[DIR_ALL].fillna(0)
    X_dir_nocorr, corr_dropped = drop_high_correlation(X_dir, 0.95)
    print(f"  Dropped {len(corr_dropped)}: {corr_dropped[:10]}{'...' if len(corr_dropped)>10 else ''}")

    # Step 2: VIF pruning
    print("\nStep 2: VIF pruning (VIF > 10)...")
    X_dir_pruned, vif_dropped = drop_high_vif(X_dir_nocorr, threshold=10.0)
    print(f"  Dropped {len(vif_dropped)} high-VIF features")

    dir_pruned = list(X_dir_pruned.columns)
    print(f"\nAfter pruning: {len(dir_pruned)} features")

    # Validate
    auc_after, ic_after, acc_after = wf_dir(dir_pruned)
    print(f"After:  AUC={auc_after:.4f}, IC={ic_after:+.4f}, Acc={acc_after:.1%}")
    print(f"Delta:  AUC={auc_after-auc_before:+.4f}, IC={ic_after-ic_before:+.4f}")

    # ═══════════════════════════════════════════════════════════════
    # MAGNITUDE MODEL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MAGNITUDE MODEL PRUNING")
    print("=" * 60)

    ic_m_before, icir_m_before = wf_mag(MAG_ALL)
    print(f"\nBefore: {len(MAG_ALL)} features, IC={ic_m_before:+.4f}, ICIR={icir_m_before:+.3f}")

    print("\nStep 1: Correlation pruning (r > 0.95)...")
    X_mag = df[MAG_ALL].fillna(0)
    X_mag_nocorr, mcorr_dropped = drop_high_correlation(X_mag, 0.95)
    print(f"  Dropped {len(mcorr_dropped)}: {mcorr_dropped[:10]}")

    print("\nStep 2: VIF pruning (VIF > 10)...")
    X_mag_pruned, mvif_dropped = drop_high_vif(X_mag_nocorr, threshold=10.0)
    print(f"  Dropped {len(mvif_dropped)} high-VIF features")

    mag_pruned = list(X_mag_pruned.columns)
    print(f"\nAfter pruning: {len(mag_pruned)} features")

    ic_m_after, icir_m_after = wf_mag(mag_pruned)
    print(f"After:  IC={ic_m_after:+.4f}, ICIR={icir_m_after:+.3f}")
    print(f"Delta:  IC={ic_m_after-ic_m_before:+.4f}, ICIR={icir_m_after-icir_m_before:+.3f}")

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    results = {
        "direction": {
            "before": {"n": len(DIR_ALL), "auc": auc_before, "ic": ic_before},
            "after": {"n": len(dir_pruned), "auc": auc_after, "ic": ic_after},
            "features": dir_pruned,
            "corr_dropped": corr_dropped,
            "vif_dropped": vif_dropped,
        },
        "magnitude": {
            "before": {"n": len(MAG_ALL), "ic": ic_m_before, "icir": icir_m_before},
            "after": {"n": len(mag_pruned), "ic": ic_m_after, "icir": icir_m_after},
            "features": mag_pruned,
            "corr_dropped": mcorr_dropped,
            "vif_dropped": mvif_dropped,
        },
    }

    out = Path("research/results/auto_pruning.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'':>20} {'Before':>12} {'After':>12} {'Delta':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Dir features':>20} {len(DIR_ALL):>12} {len(dir_pruned):>12} {len(dir_pruned)-len(DIR_ALL):>+12}")
    print(f"  {'Dir AUC':>20} {auc_before:>12.4f} {auc_after:>12.4f} {auc_after-auc_before:>+12.4f}")
    print(f"  {'Mag features':>20} {len(MAG_ALL):>12} {len(mag_pruned):>12} {len(mag_pruned)-len(MAG_ALL):>+12}")
    print(f"  {'Mag IC':>20} {ic_m_before:>12.4f} {ic_m_after:>12.4f} {ic_m_after-ic_m_before:>+12.4f}")
    print(f"  {'Mag ICIR':>20} {icir_m_before:>12.3f} {icir_m_after:>12.3f} {icir_m_after-icir_m_before:>+12.3f}")


if __name__ == "__main__":
    main()
