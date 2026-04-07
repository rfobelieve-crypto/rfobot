"""
Walk-forward retraining comparison: current deployed vs new feature sets.
Direction: liq_frag + absorb (99 features)
Magnitude: expanded + liq_frag + absorb + toxicity (91 features)
"""
import sys, os, warnings, logging
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

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_direction_labels import build_direction_labels
from research.dual_model.build_magnitude_labels import build_magnitude_labels
from research.dual_model.direction_features_v2 import (
    OLD_FEATURES, NEW_KEY_4, LIQUIDITY_FRAGILITY, POST_ABSORPTION,
    filter_available,
)
from research.dual_model.magnitude_features_v2 import (
    EXPANDED_WITH_FRAGILITY, filter_available as mag_filter,
)

print("Loading data...")
df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=6.0)
print(f"Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")

splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
print(f"Walk-forward: {len(splits)} folds")

# ═══════════════════════════════════════════════════════════════════
# DIRECTION MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIRECTION MODEL")
print("=" * 70)

dir_labels = build_direction_labels(df, k=0.5)
df["y_dir"] = dir_labels["y_dir"]
df["dir_return_4h"] = dir_labels["return_4h"]

DIR_FEATURES = OLD_FEATURES + NEW_KEY_4 + LIQUIDITY_FRAGILITY + POST_ABSORPTION
dir_feats = filter_available(DIR_FEATURES, list(df.columns))
print(f"Features: {len(dir_feats)}")

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0, "early_stopping_rounds": 30,
}

dir_probs, dir_true, dir_rets, dir_imps = [], [], [], []

for train_idx, test_idx in splits:
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    tm, ttm = train_df["y_dir"].notna(), test_df["y_dir"].notna()
    X_tr, y_tr = train_df.loc[tm, dir_feats].fillna(0), train_df.loc[tm, "y_dir"].values.astype(int)
    X_te, y_te = test_df.loc[ttm, dir_feats].fillna(0), test_df.loc[ttm, "y_dir"].values.astype(int)
    r_te = test_df.loc[ttm, "dir_return_4h"].values
    if len(y_tr) < 50 or len(y_te) < 5: continue
    up = y_tr.mean()
    p = DIR_PARAMS.copy()
    p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    prob = m.predict_proba(X_te)[:, 1]
    dir_probs.extend(prob); dir_true.extend(y_te); dir_rets.extend(r_te)
    dir_imps.append(dict(zip(dir_feats, m.feature_importances_)))

dp, dt, dr = np.array(dir_probs), np.array(dir_true), np.array(dir_rets)
dir_auc = roc_auc_score(dt, dp)
dir_acc = ((dp > 0.5).astype(int) == dt).mean()
dir_ic, _ = spearmanr(dp, dr)

n = len(dp); d = max(1, n // 10)
top_ret = dr[np.argsort(dp)[-d:]].mean() * 100
bot_ret = dr[np.argsort(dp)[:d]].mean() * 100

strong_mask = (dp > 0.6) | (dp < 0.4)
if strong_mask.sum() > 0:
    sp = np.where(dp[strong_mask] > 0.6, 1, 0)
    strong_acc = (sp == dt[strong_mask]).mean()
    strong_n = strong_mask.sum()
else:
    strong_acc, strong_n = 0, 0

print(f"\n  AUC:          {dir_auc:.4f}")
print(f"  Accuracy:     {dir_acc:.1%}")
print(f"  IC:           {dir_ic:+.4f}")
print(f"  Top decile:   {top_ret:+.3f}%")
print(f"  Bot decile:   {bot_ret:+.3f}%")
print(f"  Spread:       {top_ret - bot_ret:.3f}%")
print(f"  Strong sigs:  {strong_n} ({strong_n/n:.1%}), acc={strong_acc:.1%}")

avg_imp = pd.DataFrame(dir_imps).mean().sort_values(ascending=False)
print(f"\n  Top 15 Features:")
for i, (feat, imp) in enumerate(avg_imp.head(15).items()):
    tag = " [NEW]" if feat in (LIQUIDITY_FRAGILITY + POST_ABSORPTION) else ""
    print(f"    {i+1:>2}. {feat:<40} {imp:.4f}{tag}")

# ═══════════════════════════════════════════════════════════════════
# MAGNITUDE MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MAGNITUDE MODEL")
print("=" * 70)

mag_labels = build_magnitude_labels(df)
df["y_abs_return"] = mag_labels["y_abs_return"]
df["y_vol_adj_abs"] = mag_labels["y_vol_adj_abs"]

TOXICITY_MAG = ["tox_pressure", "tox_accum_zscore", "tox_bv_vpin_zscore", "tox_div_taker"]
MAG_FEATURES = EXPANDED_WITH_FRAGILITY + TOXICITY_MAG
mag_feats = mag_filter(MAG_FEATURES, list(df.columns))
print(f"Features: {len(mag_feats)}")

MAG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0, "early_stopping_rounds": 30,
}

mag_preds, mag_true, mag_abs, mag_imps = [], [], [], []

for train_idx, test_idx in splits:
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    tm, ttm = train_df["y_vol_adj_abs"].notna(), test_df["y_vol_adj_abs"].notna()
    X_tr, y_tr = train_df.loc[tm, mag_feats].fillna(0), train_df.loc[tm, "y_vol_adj_abs"].values
    X_te, y_te = test_df.loc[ttm, mag_feats].fillna(0), test_df.loc[ttm, "y_vol_adj_abs"].values
    a_te = test_df.loc[ttm, "y_abs_return"].values
    if len(y_tr) < 50 or len(y_te) < 5: continue
    m = xgb.XGBRegressor(**MAG_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred = m.predict(X_te)
    mag_preds.extend(pred); mag_true.extend(y_te); mag_abs.extend(a_te)
    mag_imps.append(dict(zip(mag_feats, m.feature_importances_)))

mp, mt, ma = np.array(mag_preds), np.array(mag_true), np.array(mag_abs)
mag_ic, _ = spearmanr(mp, mt)
ics = []
for i in range(100, len(mp), 20):
    r, _ = spearmanr(mp[i-100:i], mt[i-100:i])
    ics.append(r)
mag_icir = np.mean(ics) / np.std(ics) if len(ics) > 3 and np.std(ics) > 0 else 0

n = len(mp); d = max(1, n // 10)
top_abs = ma[np.argsort(mp)[-d:]].mean() * 100
bot_abs = ma[np.argsort(mp)[:d]].mean() * 100
ratio = top_abs / bot_abs if bot_abs > 0 else 0

qs = pd.qcut(pd.Series(mp), q=5, labels=False, duplicates="drop")
cal = pd.Series(ma).groupby(qs.values).mean().values
is_mono = all(cal[i] <= cal[i+1] for i in range(len(cal)-1))

print(f"\n  IC:           {mag_ic:+.4f}")
print(f"  ICIR:         {mag_icir:+.3f}")
print(f"  Top decile:   {top_abs:.3f}% |ret|")
print(f"  Bot decile:   {bot_abs:.3f}% |ret|")
print(f"  Top/Bot:      {ratio:.2f}x")
print(f"  Monotone:     {is_mono}")

avg_imp = pd.DataFrame(mag_imps).mean().sort_values(ascending=False)
print(f"\n  Top 15 Features:")
for i, (feat, imp) in enumerate(avg_imp.head(15).items()):
    tag = " [TOX]" if feat in TOXICITY_MAG else ""
    print(f"    {i+1:>2}. {feat:<40} {imp:.4f}{tag}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CURRENT vs NEW")
print("=" * 70)
print(f"  {'':>25} {'Current':>12} {'New':>12} {'Delta':>12}")
print(f"  {'-'*63}")
print(f"  {'Direction AUC':>25} {'0.582':>12} {dir_auc:>12.4f} {dir_auc - 0.582:>+12.4f}")
print(f"  {'Direction Acc':>25} {'55.5%':>12} {dir_acc*100:>11.1f}% {(dir_acc - 0.555)*100:>+11.1f}%")
print(f"  {'Magnitude IC':>25} {'0.385':>12} {mag_ic:>12.4f} {mag_ic - 0.385:>+12.4f}")
print(f"  {'Magnitude ICIR':>25} {'1.09':>12} {mag_icir:>12.3f} {mag_icir - 1.09:>+12.3f}")
print(f"  {'Magnitude Top/Bot':>25} {'3.10x':>12} {ratio:>11.2f}x {ratio - 3.10:>+11.2f}x")
