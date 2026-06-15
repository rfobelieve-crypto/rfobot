"""
3-Phase Feature Selection for Direction + Magnitude models.

Phase 1: XGB gain-based coarse filter (cut < 0.5% / 0.4%)
Phase 2: Permutation importance on survivors (cut <= 0 contribution)
Phase 3: Walk-forward retrain comparison (before vs after)
"""
import sys, os, json, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
_orig = np.polyfit
def safe_polyfit(*a, **k):
    try: return _orig(*a, **k)
    except: return [0.0, 0.0]
np.polyfit = safe_polyfit

import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from pathlib import Path

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_direction_labels import build_direction_labels
from research.dual_model.build_magnitude_labels import build_magnitude_labels
from research.dual_model.direction_features_v2 import (
    OLD_FEATURES, NEW_KEY_4, LIQUIDITY_FRAGILITY, POST_ABSORPTION, filter_available,
)
from research.dual_model.magnitude_features_v2 import (
    EXPANDED_WITH_FRAGILITY, filter_available as mag_filter,
)

# ── Load data ──
print("Loading data...")
df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=6.0)
n = len(df)
print(f"Data: {n} bars")

splits = walk_forward_splits(n, initial_train=288, test_size=48, step=48)

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
    list(df.columns)
)
TOXICITY_MAG = ["tox_pressure", "tox_accum_zscore", "tox_bv_vpin_zscore", "tox_div_taker"]
MAG_ALL = mag_filter(EXPANDED_WITH_FRAGILITY + TOXICITY_MAG, list(df.columns))

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

# Train/val split (70/30 temporal)
split_idx = int(n * 0.7)
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

# ═════════════════════════════════════════════════════════════════
# PHASE 1: Gain-based coarse filter
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: XGB Gain Coarse Filter")
print("=" * 70)


def gain_filter(feats, target, model_type, threshold_pct, label):
    """Train on 70%, get gain importance, filter by threshold."""
    if model_type == "dir":
        tm = train_df[target].notna()
        vm = val_df[target].notna()
        X_tr = train_df.loc[tm, feats].fillna(0)
        y_tr = train_df.loc[tm, target].values.astype(int)
        X_val = val_df.loc[vm, feats].fillna(0)
        y_val = val_df.loc[vm, target].values.astype(int)
        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        tm = train_df[target].notna()
        vm = val_df[target].notna()
        X_tr = train_df.loc[tm, feats].fillna(0)
        y_tr = train_df.loc[tm, target].values
        X_val = val_df.loc[vm, feats].fillna(0)
        y_val = val_df.loc[vm, target].values
        m = xgb.XGBRegressor(**MAG_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    imp_pct = imp / imp.sum() * 100
    keep = imp_pct[imp_pct >= threshold_pct].index.tolist()
    cut = imp_pct[imp_pct < threshold_pct].index.tolist()

    print(f"\n  {label}: {len(feats)} -> {len(keep)} (cut {len(cut)}, threshold={threshold_pct}%)")
    for feat in keep:
        print(f"    {feat:<45} {imp_pct[feat]:.2f}%")
    print(f"  Cut: {cut[:5]}{'...' if len(cut)>5 else ''}")
    return keep, m


dir_keep, dir_m = gain_filter(DIR_ALL, "y_dir", "dir", 0.5, "Direction")
mag_keep, mag_m = gain_filter(MAG_ALL, "y_vol_adj_abs", "mag", 0.4, "Magnitude")

# ═════════════════════════════════════════════════════════════════
# PHASE 2: Permutation Importance
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Permutation Importance (validation set)")
print("=" * 70)


def perm_filter(feats, target, model_type, label):
    """Retrain on survivors, run permutation importance on val set."""
    if model_type == "dir":
        tm = train_df[target].notna()
        vm = val_df[target].notna()
        X_tr = train_df.loc[tm, feats].fillna(0)
        y_tr = train_df.loc[tm, target].values.astype(int)
        X_val = val_df.loc[vm, feats].fillna(0)
        y_val = val_df.loc[vm, target].values.astype(int)
        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scoring = "roc_auc"
    else:
        tm = train_df[target].notna()
        vm = val_df[target].notna()
        X_tr = train_df.loc[tm, feats].fillna(0)
        y_tr = train_df.loc[tm, target].values
        X_val = val_df.loc[vm, feats].fillna(0)
        y_val = val_df.loc[vm, target].values
        m = xgb.XGBRegressor(**MAG_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scoring = "neg_mean_absolute_error"

    perm = permutation_importance(m, X_val, y_val, n_repeats=10,
                                   random_state=42, scoring=scoring)
    perm_df = pd.DataFrame({
        "feature": feats,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mean", ascending=False)

    keep = perm_df[perm_df["perm_mean"] > 0]["feature"].tolist()
    cut = perm_df[perm_df["perm_mean"] <= 0]["feature"].tolist()

    print(f"\n  {label}: {len(feats)} -> {len(keep)} (cut {len(cut)} zero/negative)")
    for _, row in perm_df.iterrows():
        mark = "KEEP" if row["perm_mean"] > 0 else "CUT "
        print(f"    [{mark}] {row['feature']:<42} {row['perm_mean']:+.4f} +/- {row['perm_std']:.4f}")

    return keep


dir_final = perm_filter(dir_keep, "y_dir", "dir", "Direction")
mag_final = perm_filter(mag_keep, "y_vol_adj_abs", "mag", "Magnitude")

# ═════════════════════════════════════════════════════════════════
# PHASE 3: Walk-Forward Retrain Comparison
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Walk-Forward Retrain Comparison")
print("=" * 70)


def wf_direction(feats, label):
    probs, trues, rets = [], [], []
    for train_idx, test_idx in splits:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        tm, ttm = tr["y_dir"].notna(), te["y_dir"].notna()
        X_t = tr.loc[tm, feats].fillna(0)
        y_t = tr.loc[tm, "y_dir"].values.astype(int)
        X_e = te.loc[ttm, feats].fillna(0)
        y_e = te.loc[ttm, "y_dir"].values.astype(int)
        r_e = te.loc[ttm, "dir_return_4h"].values
        if len(y_t) < 50 or len(y_e) < 5:
            continue
        up = y_t.mean()
        pp = DIR_PARAMS.copy()
        pp["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0
        m = xgb.XGBClassifier(**pp)
        m.fit(X_t, y_t, eval_set=[(X_e, y_e)], verbose=False)
        probs.extend(m.predict_proba(X_e)[:, 1])
        trues.extend(y_e)
        rets.extend(r_e)
    p, t, r = np.array(probs), np.array(trues), np.array(rets)
    auc = roc_auc_score(t, p)
    acc = ((p > 0.5).astype(int) == t).mean()
    ic, _ = spearmanr(p, r)
    d = max(1, len(p) // 10)
    spread = (r[np.argsort(p)[-d:]].mean() - r[np.argsort(p)[:d]].mean()) * 100
    print(f"  {label:<40} AUC={auc:.4f}  Acc={acc:.1%}  IC={ic:+.4f}  Spread={spread:.3f}%  #F={len(feats)}")
    return auc


def wf_magnitude(feats, label):
    preds, trues, abss = [], [], []
    for train_idx, test_idx in splits:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        tm, ttm = tr["y_vol_adj_abs"].notna(), te["y_vol_adj_abs"].notna()
        X_t = tr.loc[tm, feats].fillna(0)
        y_t = tr.loc[tm, "y_vol_adj_abs"].values
        X_e = te.loc[ttm, feats].fillna(0)
        y_e = te.loc[ttm, "y_vol_adj_abs"].values
        a_e = te.loc[ttm, "y_abs_return"].values
        if len(y_t) < 50 or len(y_e) < 5:
            continue
        m = xgb.XGBRegressor(**MAG_PARAMS)
        m.fit(X_t, y_t, eval_set=[(X_e, y_e)], verbose=False)
        preds.extend(m.predict(X_e))
        trues.extend(y_e)
        abss.extend(a_e)
    p, t, a = np.array(preds), np.array(trues), np.array(abss)
    ic, _ = spearmanr(p, t)
    ics = [spearmanr(p[i - 100:i], t[i - 100:i])[0] for i in range(100, len(p), 20)]
    icir = np.mean(ics) / np.std(ics) if len(ics) > 3 and np.std(ics) > 0 else 0
    d = max(1, len(p) // 10)
    ratio = a[np.argsort(p)[-d:]].mean() / max(a[np.argsort(p)[:d]].mean(), 1e-10)
    print(f"  {label:<40} IC={ic:+.4f}  ICIR={icir:+.3f}  Top/Bot={ratio:.2f}x  #F={len(feats)}")
    return ic, icir


print("\n--- Direction ---")
wf_direction(DIR_ALL, f"Before ({len(DIR_ALL)} features)")
wf_direction(dir_keep, f"Phase1 gain ({len(dir_keep)} features)")
wf_direction(dir_final, f"Phase2 perm ({len(dir_final)} features)")

print("\n--- Magnitude ---")
wf_magnitude(MAG_ALL, f"Before ({len(MAG_ALL)} features)")
wf_magnitude(mag_keep, f"Phase1 gain ({len(mag_keep)} features)")
wf_magnitude(mag_final, f"Phase2 perm ({len(mag_final)} features)")

# Save final feature lists
results = {
    "direction_before": len(DIR_ALL),
    "direction_after": len(dir_final),
    "direction_features": dir_final,
    "magnitude_before": len(MAG_ALL),
    "magnitude_after": len(mag_final),
    "magnitude_features": mag_final,
}
Path("research/results/feature_selection.json").write_text(json.dumps(results, indent=2))
print(f"\nFinal feature lists saved to research/results/feature_selection.json")
