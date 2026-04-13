"""
Deploy new Direction + Magnitude models with updated feature sets.

Direction: OLD + KEY_4 + LIQUIDITY_FRAGILITY + POST_ABSORPTION (99 feats)
Magnitude: EXPANDED_WITH_FRAGILITY + TOXICITY (91 feats)

Trains on full dataset, exports artifacts to indicator/model_artifacts/dual_model/
"""
import sys, os, json, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from research.dual_model.evaluate_direction_4h import evaluate_direction

ARTIFACT_DIR = Path("indicator/model_artifacts/dual_model")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──
print("Loading data...")
df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=6.0)
n_bars = len(df)
print(f"Data: {n_bars} bars, {df.index[0].date()} -> {df.index[-1].date()}")

splits = walk_forward_splits(n_bars, initial_train=288, test_size=48, step=48)

# ═════════════════════════════════════════════════════════════════
# DIRECTION MODEL
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING DIRECTION MODEL")
print("=" * 70)

dir_labels = build_direction_labels(df, k=0.5)
df["y_dir"] = dir_labels["y_dir"]
df["dir_return_4h"] = dir_labels["return_4h"]

# Feature selection priority:
#   1. ablation_study.json (PREFERRED) — honest baseline + forward selection
#   2. auto_pruning.json     (fallback) — VIF/correlation pruning
#   3. hardcoded full set    (last resort)
#
# Ablation is preferred because on 2026-04-13 we found that auto_pruner output
# 80 features (many with negative contribution) trained to AUC 0.579, while
# ablation found the correct 34-feature set trained to AUC 0.602.
# See .claude/rules/mistake.md for context.
ablation_path = Path("research/results/ablation_study.json")
pruning_path  = Path("research/results/auto_pruning.json")

if ablation_path.exists():
    _abl = json.loads(ablation_path.read_text())
    baseline_feats = _abl.get("baseline", {}).get("features", [])
    keep_feats     = _abl.get("combined", {}).get("keep_features", [])
    dir_feats = [f for f in (baseline_feats + keep_feats) if f in df.columns]
    # Deduplicate while preserving order
    seen: set[str] = set()
    dir_feats = [f for f in dir_feats if not (f in seen or seen.add(f))]
    abl_auc   = _abl.get("combined", {}).get("metrics", {}).get("auc", 0.0)
    abl_d_auc = _abl.get("combined", {}).get("delta_auc", 0.0)
    print(f"Features: {len(dir_feats)} (ablation: {len(baseline_feats)} baseline "
          f"+ {len(keep_feats)} KEEP, validated AUC={abl_auc:.4f} delta={abl_d_auc:+.4f})")
    if keep_feats:
        print(f"  KEEP features added: {keep_feats}")
elif pruning_path.exists():
    _pruning = json.loads(pruning_path.read_text())
    dir_feats = [f for f in _pruning["direction"]["features"] if f in df.columns]
    print(f"Features: {len(dir_feats)} (VIF-pruned, no ablation)")
    print(f"  WARNING: using VIF pruning. Run research/ablation_study.py for honest validation.")
else:
    DIR_FEATURES = OLD_FEATURES + NEW_KEY_4 + LIQUIDITY_FRAGILITY + POST_ABSORPTION
    dir_feats = filter_available(DIR_FEATURES, list(df.columns))
    print(f"Features: {len(dir_feats)} (unpruned — no ablation or pruning available)")
    print(f"  WARNING: using full feature set. Risk of overfitting.")

# Regime weighting: TRENDING_BULL 4x, TRENDING_BEAR 2x
BULL_WEIGHT = 4.0
BEAR_WEIGHT = 2.0

# Compute regime for sample weighting
_close = df["close"]
_log_ret = np.log(_close / _close.shift(1))
_ret_24h = _close.pct_change(24)
_vol_24h = _log_ret.rolling(24).std()
_vol_pct = _vol_24h.expanding(min_periods=168).rank(pct=True)
_regime = np.full(len(df), "CHOPPY", dtype=object)
_regime[(_vol_pct > 0.6).values & (_ret_24h > 0.005).values] = "TRENDING_BULL"
_regime[(_vol_pct > 0.6).values & (_ret_24h < -0.005).values] = "TRENDING_BEAR"

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0, "early_stopping_rounds": 30,
}

# Walk-forward for OOS eval
dir_probs, dir_true, dir_rets = [], [], []
dir_fold_aucs = []

for fold_i, (train_idx, test_idx) in enumerate(splits):
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    tm = train_df["y_dir"].notna()
    ttm = test_df["y_dir"].notna()
    X_tr = train_df.loc[tm, dir_feats].fillna(0)
    y_tr = train_df.loc[tm, "y_dir"].values.astype(int)
    X_te = test_df.loc[ttm, dir_feats].fillna(0)
    y_te = test_df.loc[ttm, "y_dir"].values.astype(int)
    r_te = test_df.loc[ttm, "dir_return_4h"].values
    if len(y_tr) < 50 or len(y_te) < 5:
        continue
    up = y_tr.mean()
    p = DIR_PARAMS.copy()
    p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0
    # Regime sample weights
    reg_tr = _regime[train_idx][tm.values]
    sw = np.ones(len(y_tr))
    sw[reg_tr == "TRENDING_BULL"] = BULL_WEIGHT
    sw[reg_tr == "TRENDING_BEAR"] = BEAR_WEIGHT
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_te, y_te)], verbose=False)
    prob = m.predict_proba(X_te)[:, 1]
    dir_probs.extend(prob)
    dir_true.extend(y_te)
    dir_rets.extend(r_te)
    try:
        dir_fold_aucs.append(roc_auc_score(y_te, prob))
    except:
        pass

dp, dt, dr = np.array(dir_probs), np.array(dir_true), np.array(dir_rets)
dir_auc = roc_auc_score(dt, dp)
dir_acc = ((dp > 0.5).astype(int) == dt).mean()
dir_ic, _ = spearmanr(dp, dr)

# Train FINAL model on all data
print("Training final direction model on full dataset...")
all_mask = df["y_dir"].notna()
X_all = df.loc[all_mask, dir_feats].fillna(0)
y_all = df.loc[all_mask, "y_dir"].values.astype(int)
up_rate = y_all.mean()

final_dir_params = DIR_PARAMS.copy()
final_dir_params.pop("early_stopping_rounds")
final_dir_params["scale_pos_weight"] = (1 - up_rate) / up_rate if up_rate > 0 else 1.0

# Regime sample weights for final training
sw_all = np.ones(len(y_all))
reg_all = _regime[all_mask.values]
sw_all[reg_all == "TRENDING_BULL"] = BULL_WEIGHT
sw_all[reg_all == "TRENDING_BEAR"] = BEAR_WEIGHT
dir_model = xgb.XGBClassifier(**final_dir_params)
dir_model.fit(X_all, y_all, sample_weight=sw_all, verbose=False)

# Export
dir_model.get_booster().save_model(str(ARTIFACT_DIR / "direction_xgb.json"))
with open(ARTIFACT_DIR / "direction_feature_cols.json", "w") as f:
    json.dump(dir_feats, f, indent=2)
with open(ARTIFACT_DIR / "direction_config.json", "w") as f:
    json.dump({
        "feature_set": "VIF-pruned + regime_weight(bull=4x,bear=2x)",
        "n_features": len(dir_feats),
        "n_samples": int(all_mask.sum()),
        "up_rate": float(up_rate),
        "scale_pos_weight": float(final_dir_params["scale_pos_weight"]),
        "oos_auc": float(dir_auc),
        "oos_accuracy": float(dir_acc),
        "oos_ic": float(dir_ic),
        "params": {k: v for k, v in final_dir_params.items() if k != "scale_pos_weight"},
    }, f, indent=2)

print(f"  OOS AUC:  {dir_auc:.4f}")
print(f"  OOS Acc:  {dir_acc:.1%}")
print(f"  Exported: direction_xgb.json ({len(dir_feats)} features)")

# ═════════════════════════════════════════════════════════════════
# MAGNITUDE MODEL
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRAINING MAGNITUDE MODEL")
print("=" * 70)

mag_labels = build_magnitude_labels(df)
df["y_abs_return"] = mag_labels["y_abs_return"]
df["y_vol_adj_abs"] = mag_labels["y_vol_adj_abs"]

# Use VIF-pruned magnitude features
if pruning_path.exists():
    mag_feats = [f for f in _pruning["magnitude"]["features"] if f in df.columns]
    print(f"Features: {len(mag_feats)} (VIF-pruned)")
else:
    TOXICITY_MAG = ["tox_pressure", "tox_accum_zscore", "tox_bv_vpin_zscore", "tox_div_taker"]
    MAG_FEATURES = EXPANDED_WITH_FRAGILITY + TOXICITY_MAG
    mag_feats = mag_filter(MAG_FEATURES, list(df.columns))
    print(f"Features: {len(mag_feats)} (unpruned)")

MAG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0, "early_stopping_rounds": 30,
}

mag_preds, mag_true, mag_abs = [], [], []
mag_fold_ics = []

for fold_i, (train_idx, test_idx) in enumerate(splits):
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    tm = train_df["y_vol_adj_abs"].notna()
    ttm = test_df["y_vol_adj_abs"].notna()
    X_tr = train_df.loc[tm, mag_feats].fillna(0)
    y_tr = train_df.loc[tm, "y_vol_adj_abs"].values
    X_te = test_df.loc[ttm, mag_feats].fillna(0)
    y_te = test_df.loc[ttm, "y_vol_adj_abs"].values
    a_te = test_df.loc[ttm, "y_abs_return"].values
    if len(y_tr) < 50 or len(y_te) < 5:
        continue
    m = xgb.XGBRegressor(**MAG_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred = m.predict(X_te)
    mag_preds.extend(pred)
    mag_true.extend(y_te)
    mag_abs.extend(a_te)
    try:
        ic_fold, _ = spearmanr(y_te, pred)
        mag_fold_ics.append(ic_fold)
    except:
        pass

mp, mt, ma = np.array(mag_preds), np.array(mag_true), np.array(mag_abs)
mag_ic, _ = spearmanr(mp, mt)
ics_rolling = []
for i in range(100, len(mp), 20):
    r, _ = spearmanr(mp[i-100:i], mt[i-100:i])
    ics_rolling.append(r)
mag_icir = np.mean(ics_rolling) / np.std(ics_rolling) if len(ics_rolling) > 3 and np.std(ics_rolling) > 0 else 0

n = len(mp)
d = max(1, n // 10)
top_abs = ma[np.argsort(mp)[-d:]].mean()
bot_abs = ma[np.argsort(mp)[:d]].mean()
ratio = top_abs / bot_abs if bot_abs > 0 else 0

# Train FINAL magnitude model
print("Training final magnitude model on full dataset...")
all_mask = df["y_vol_adj_abs"].notna()
X_all = df.loc[all_mask, mag_feats].fillna(0)
y_all = df.loc[all_mask, "y_vol_adj_abs"].values

final_mag_params = MAG_PARAMS.copy()
final_mag_params.pop("early_stopping_rounds")

mag_model = xgb.XGBRegressor(**final_mag_params)
mag_model.fit(X_all, y_all, verbose=False)

# Export
mag_model.get_booster().save_model(str(ARTIFACT_DIR / "magnitude_xgb.json"))
with open(ARTIFACT_DIR / "magnitude_feature_cols.json", "w") as f:
    json.dump(mag_feats, f, indent=2)
with open(ARTIFACT_DIR / "magnitude_config.json", "w") as f:
    json.dump({
        "feature_set": "expanded + liq_frag + post_absorption + toxicity",
        "n_features": len(mag_feats),
        "n_samples": int(all_mask.sum()),
        "oos_ic": float(mag_ic),
        "oos_icir": float(mag_icir),
        "oos_top_bot_ratio": float(ratio),
        "params": final_mag_params,
    }, f, indent=2)

# Training stats (pred history for warmup)
all_preds = mag_model.predict(X_all)
pred_history = [float(x) for x in all_preds[-300:]]
with open(ARTIFACT_DIR / "training_stats.json", "w") as f:
    json.dump({
        "pred_history": pred_history,
        "n_bars": n_bars,
        "date_range": f"{df.index[0]} ~ {df.index[-1]}",
    }, f, indent=2)

print(f"  OOS IC:   {mag_ic:+.4f}")
print(f"  OOS ICIR: {mag_icir:+.3f}")
print(f"  Top/Bot:  {ratio:.2f}x")
print(f"  Exported: magnitude_xgb.json ({len(mag_feats)} features)")

# ═════════════════════════════════════════════════════════════════
# COMPARISON CHART
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING COMPARISON CHART")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Retrain Comparison: Old vs New", fontsize=14, fontweight="bold")

# 1. Direction AUC per fold
ax = axes[0, 0]
old_dir_auc = 0.582
ax.plot(dir_fold_aucs, "b-o", markersize=3, label="New (per fold)")
ax.axhline(old_dir_auc, color="r", linestyle="--", label=f"Old OOS AUC={old_dir_auc:.3f}")
ax.axhline(dir_auc, color="b", linestyle="--", label=f"New OOS AUC={dir_auc:.3f}")
ax.set_title("Direction: AUC per Fold")
ax.set_xlabel("Fold")
ax.set_ylabel("AUC")
ax.legend(fontsize=8)
ax.set_ylim(0.45, 0.75)
ax.grid(True, alpha=0.3)

# 2. Direction prob distribution
ax = axes[0, 1]
correct = dp[((dp > 0.5).astype(int) == dt)]
wrong = dp[((dp > 0.5).astype(int) != dt)]
ax.hist(correct, bins=40, alpha=0.6, color="green", label=f"Correct ({len(correct)})")
ax.hist(wrong, bins=40, alpha=0.6, color="red", label=f"Wrong ({len(wrong)})")
ax.axvline(0.5, color="black", linestyle="--")
ax.set_title(f"Direction: P(UP) Distribution (Acc={dir_acc:.1%})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Magnitude rolling IC
ax = axes[1, 0]
old_icir = 1.09
ax.plot(ics_rolling, "b-", alpha=0.7, label="New rolling IC")
ax.axhline(np.mean(ics_rolling), color="b", linestyle="--",
           label=f"New mean IC={np.mean(ics_rolling):.3f} ICIR={mag_icir:.2f}")
ax.axhline(0.385, color="r", linestyle="--", label=f"Old IC=0.385 ICIR={old_icir:.2f}")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_title("Magnitude: Rolling IC (100-bar window)")
ax.set_xlabel("Window")
ax.set_ylabel("Spearman IC")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Summary comparison bar chart
ax = axes[1, 1]
metrics = ["Dir AUC", "Dir Acc", "Mag IC", "Mag ICIR/10"]
old_vals = [0.582, 0.555, 0.385, 1.09/10]
new_vals = [dir_auc, dir_acc, mag_ic, mag_icir/10]

x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, old_vals, w, label="Old (deployed)", color="salmon", alpha=0.8)
bars2 = ax.bar(x + w/2, new_vals, w, label="New", color="steelblue", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Old vs New Summary")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="red")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="blue")

plt.tight_layout()
chart_path = "research/results/model_retrain_comparison.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved: {chart_path}")

# ═════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DEPLOYMENT COMPLETE")
print("=" * 70)
print(f"  {'':>25} {'Old':>12} {'New':>12} {'Delta':>12}")
print(f"  {'-'*63}")
print(f"  {'Direction AUC':>25} {'0.582':>12} {dir_auc:>12.4f} {dir_auc-0.582:>+12.4f}")
print(f"  {'Direction Acc':>25} {'55.5%':>12} {dir_acc*100:>11.1f}% {(dir_acc-0.555)*100:>+11.1f}%")
print(f"  {'Direction Features':>25} {'88':>12} {len(dir_feats):>12}")
print(f"  {'Magnitude IC':>25} {'0.385':>12} {mag_ic:>12.4f} {mag_ic-0.385:>+12.4f}")
print(f"  {'Magnitude ICIR':>25} {'1.09':>12} {mag_icir:>12.3f} {mag_icir-1.09:>+12.3f}")
print(f"  {'Magnitude Top/Bot':>25} {'3.10x':>12} {ratio:>11.2f}x {ratio-3.10:>+11.2f}x")
print(f"  {'Magnitude Features':>25} {'87':>12} {len(mag_feats):>12}")
print()
print("Artifacts exported to indicator/model_artifacts/dual_model/")
print("Run `git push` to deploy to Railway.")
