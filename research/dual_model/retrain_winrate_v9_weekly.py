"""
Weekly auto-retrain pipeline for v9 binary win-rate classifier.

Production schedule: Railway cron, every Sunday 02:00 UTC.

What this script does:
  1. Refresh raw parquet (Binance klines + Coinglass) via ensure_fresh()
  2. Rebuild features_all.parquet cache (force_refresh=True)
  3. Rebuild winrate labels (TP=0.5% / SL=0.3% / H=8h) from latest klines
  4. Train PRODUCTION v9 long_win + short_win XGBClassifiers on ALL data
     (no walk-forward — that is evaluation-only; this is the actual model
     that will be used at inference time)
  5. Backup previous v9 artifacts to history/
  6. Save new artifacts to indicator/model_artifacts/dual_model/v9_winrate/
  7. Update training_stats.json with v9 retrain timestamp (read-then-merge)
  8. Print sanity metrics + acceptance gates

Acceptance gates (script exits non-zero on failure so cron alerts fire):
  - Holdout AUC (short side) > 0.51  [absolute floor — any positive edge]
  - Holdout AUC (short side) > previous AUC - 0.03  [no drastic regression]
  - n_bars >= 1500  [minimum sample]
  - Features build OK (cache populated)

If any gate fails, new artifacts are NOT promoted; previous model stays live.
"""
from __future__ import annotations

import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data
from research.dual_model.build_winrate_labels import build_winrate_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──
ARTIFACTS_DIR = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model" / "v9_winrate"
)
HISTORY_DIR = ARTIFACTS_DIR / "history"
FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)
TRAINING_STATS_PATH = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "training_stats.json"
)
KLINES_PATH = (
    PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
)
LABEL_CACHE_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"

# ── Hyperparameters (same as research/dual_model/train_winrate_v9.py) ──
HORIZON = 8
TP_DIST = 0.005
SL_DIST = 0.003

BASE_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=30,
    objective="binary:logistic",
    eval_metric="auc",
)

# Final production model: use fixed n_estimators (no early-stopping) so the
# model is trained on 100% of data.  Picked to match the median best_iter
# observed in walk-forward CV.  Using last 15% as val for early stopping
# made the model underfit (predictions collapsed to ~0.5) because that
# window happened to coincide with the 4月 v9 failure period.
FINAL_N_ESTIMATORS = 300

# ── Acceptance gates ──
MIN_BARS = 1500
MIN_AUC_SHORT = 0.51
MAX_AUC_DROP = 0.03


def refresh_features(limit: int = 4000) -> pd.DataFrame:
    """Force-refresh raw parquet + features_all cache.

    Calls ensure_fresh() FIRST so backfill completes before feature build,
    then force-rebuilds the cache from the now-fresh raw parquets.
    """
    logger.info("Refreshing raw parquet (backfill if stale)...")
    try:
        from research.backfill_all_parquet import ensure_fresh
        ensure_fresh(max_age_hours=4)
    except Exception as exc:
        logger.warning("Backfill check raised (continuing): %s", exc)

    logger.info("Rebuilding features cache (force_refresh=True, limit=%d)...", limit)
    features = load_and_cache_data(
        limit=limit, force_refresh=True, max_stale_hours=4,
    )
    if features.index.tz is not None:
        features.index = features.index.tz_convert("UTC").tz_localize(None)
    logger.info("Features ready: %d bars x %d cols, range %s ~ %s",
                len(features), len(features.columns),
                features.index.min(), features.index.max())
    return features


def refresh_labels() -> pd.DataFrame:
    """Rebuild winrate labels from latest klines parquet."""
    logger.info("Refreshing labels (TP=%.1f%% SL=%.1f%% H=%d)...",
                TP_DIST * 100, SL_DIST * 100, HORIZON)
    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]]
    klines = klines.dropna()
    klines = klines[~klines.index.duplicated(keep="last")].sort_index()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    labels = build_winrate_labels(klines, HORIZON)
    labels = labels.dropna(subset=["y_long_win", "y_short_win"])

    out_path = (
        LABEL_CACHE_DIR
        / f"labels_winrate_TP{int(TP_DIST*10000)}_SL{int(SL_DIST*10000)}_H{HORIZON}.parquet"
    )
    labels.to_parquet(out_path)
    logger.info("Labels: %d rows -> %s", len(labels), out_path)
    return labels


def _train_single(X_tr, y_tr, X_val, y_val, seed=42):
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    params = dict(BASE_PARAMS, scale_pos_weight=spw, random_state=seed)
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else float("nan")
    return model, auc, spw


def evaluate_rolling_cv(X: np.ndarray, y: np.ndarray, side: str,
                          n_folds: int = 3, val_fraction: float = 0.05) -> list[float]:
    """Rolling time-series CV — last n_folds non-overlapping windows.

    Each fold: train on data up to fold_start, validate on next 5% of data.
    val_fraction=0.05 -> ~200 bars per fold (~8 days) which matches the
    retrain cadence we expect (weekly).  AUC on each fold tells us whether
    the model can predict the NEXT week, repeated 3x.
    """
    n = len(X)
    val_size = max(int(n * val_fraction), 50)
    fold_aucs = []
    for k in range(n_folds, 0, -1):
        val_end = n - (k - 1) * val_size
        val_start = val_end - val_size
        if val_start < int(n * 0.5):
            logger.warning("[%s] CV fold %d skipped — would need to train on <50%% data", side, k)
            continue
        X_tr = X[:val_start]
        y_tr = y[:val_start]
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        _, auc, _ = _train_single(X_tr, y_tr, X_val, y_val, seed=42 + k)
        fold_aucs.append(auc)
        logger.info("[%s] CV fold (val=%d-%d, n_tr=%d, n_val=%d) AUC=%.4f",
                    side, val_start, val_end, len(X_tr), len(X_val), auc)
    return fold_aucs


def train_production_model(X: np.ndarray, y: np.ndarray, side: str):
    """Train PRODUCTION v9 classifier.

    1. Rolling 3-fold CV → robust AUC estimate (median across folds).
       This is the metric that gates deployment.
    2. Final production model: train on 100% of data with FIXED
       n_estimators=300 (no early-stopping val set).  Early-stopping on
       last 15% caused predictions to collapse to ~0.5 when that window
       happened to land on a regime where v9 was weak.
    """
    cv_aucs = evaluate_rolling_cv(X, y, side, n_folds=3, val_fraction=0.05)
    median_auc = float(np.nanmedian(cv_aucs)) if cv_aucs else float("nan")
    logger.info("[%s] CV AUCs: %s  median=%.4f",
                 side, [f"{a:.4f}" for a in cv_aucs], median_auc)

    # Final model — 100% of data, fixed n_estimators
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    params = dict(BASE_PARAMS, scale_pos_weight=spw, n_estimators=FINAL_N_ESTIMATORS)
    params.pop("early_stopping_rounds", None)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    # Self-eval on training data (informational only — NOT a holdout)
    p_train = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, p_train) if len(np.unique(y)) > 1 else float("nan")
    logger.info("[%s] final model: n=%d (100%% fit) train_auc=%.4f scale_pos_weight=%.2f n_est=%d",
                 side, len(X), train_auc, spw, FINAL_N_ESTIMATORS)
    return model, median_auc, train_auc, cv_aucs


def read_previous_meta() -> dict | None:
    meta_path = ARTIFACTS_DIR / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read previous meta.json: %s", exc)
        return None


def backup_previous_artifacts(ts_str: str):
    if not ARTIFACTS_DIR.exists():
        return
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("long_xgb.json", "short_xgb.json", "feature_cols.json", "meta.json"):
        src = ARTIFACTS_DIR / fname
        if src.exists():
            dst = HISTORY_DIR / f"{src.stem}_{ts_str}{src.suffix}"
            shutil.copy2(src, dst)
            logger.info("Backed up %s -> %s", src.name, dst.name)


def update_training_stats(meta: dict):
    """Read-then-merge update to shared training_stats.json (mistake log §2026-04-19)."""
    if TRAINING_STATS_PATH.exists():
        with open(TRAINING_STATS_PATH) as f:
            stats = json.load(f)
    else:
        stats = {}
    stats["v9_retrain"] = {
        "version": meta["version"],
        "trained_at_utc": meta["trained_at_utc"],
        "n_bars": meta["n_bars"],
        "train_auc_long": meta["train_auc_long"],
        "train_auc_short": meta["train_auc_short"],
        "median_auc_long": meta["median_auc_long"],
        "median_auc_short": meta["median_auc_short"],
        "feature_cols_n": meta["feature_cols_n"],
        "horizon_bars": meta["horizon_bars"],
        "tp_dist": meta["tp_dist"],
        "sl_dist": meta["sl_dist"],
    }
    with open(TRAINING_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Updated training_stats.json with v9_retrain key")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and report metrics but do NOT write artifacts")
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()

    start = datetime.now(timezone.utc)
    ts_str = start.strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 80)
    logger.info("v9 weekly retrain started %s (%s)",
                start.isoformat(), "DRY RUN" if args.dry_run else "DEPLOY")
    logger.info("=" * 80)

    # 1) Refresh features + labels
    features = refresh_features(limit=args.limit)
    labels = refresh_labels()

    # tz-align both
    if features.index.tz is not None:
        features.index = features.index.tz_convert("UTC").tz_localize(None)
    if labels.index.tz is not None:
        labels.index = labels.index.tz_convert("UTC").tz_localize(None)
    df = features.join(
        labels[["y_long_win", "y_short_win"]], how="inner"
    ).dropna(subset=["y_long_win", "y_short_win"])

    n_bars = len(df)
    logger.info("Joined dataset: n=%d bars", n_bars)
    if n_bars < MIN_BARS:
        logger.error("ACCEPTANCE GATE FAIL: n_bars=%d < %d", n_bars, MIN_BARS)
        sys.exit(1)

    # 2) Feature columns
    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("Using %d features", len(feature_cols))

    X = df[feature_cols].values.astype(np.float32)
    y_long = df["y_long_win"].values.astype(int)
    y_short = df["y_short_win"].values.astype(int)

    # 3) Train production models (long + short)
    long_model, median_auc_long, train_auc_long, cv_long = train_production_model(X, y_long, "LONG")
    short_model, median_auc_short, train_auc_short, cv_short = train_production_model(X, y_short, "SHORT")

    # Distribution sanity — production model should produce spread in P,
    # not collapse to 0.5.  Top-1% prob is what threshold 0.65 needs to clear.
    p_long_all = long_model.predict_proba(X)[:, 1]
    p_short_all = short_model.predict_proba(X)[:, 1]
    p_long_top1 = float(np.quantile(p_long_all, 0.99))
    p_short_top1 = float(np.quantile(p_short_all, 0.99))
    logger.info("Distribution: P_long  range [%.3f, %.3f] top1=%.3f >=0.65: %d",
                 p_long_all.min(), p_long_all.max(), p_long_top1,
                 int((p_long_all >= 0.65).sum()))
    logger.info("Distribution: P_short range [%.3f, %.3f] top1=%.3f >=0.65: %d",
                 p_short_all.min(), p_short_all.max(), p_short_top1,
                 int((p_short_all >= 0.65).sum()))

    # 4) Acceptance gates (use rolling CV median, not single fold)
    logger.info("-" * 80)
    logger.info("ACCEPTANCE GATES")
    logger.info("-" * 80)
    gates_pass = True

    if median_auc_short < MIN_AUC_SHORT:
        logger.error("GATE FAIL: short CV median AUC %.4f < %.4f", median_auc_short, MIN_AUC_SHORT)
        gates_pass = False
    else:
        logger.info("GATE PASS: short CV median AUC %.4f >= %.4f",
                     median_auc_short, MIN_AUC_SHORT)

    prev = read_previous_meta()
    if prev is not None and "median_auc_short" in prev:
        prev_auc = prev["median_auc_short"]
        drop = prev_auc - median_auc_short
        if drop > MAX_AUC_DROP:
            logger.error("GATE FAIL: short CV median AUC dropped %.4f (prev %.4f -> new %.4f) > max %.4f",
                         drop, prev_auc, median_auc_short, MAX_AUC_DROP)
            gates_pass = False
        else:
            logger.info("GATE PASS: short CV median AUC delta %+.4f (prev %.4f -> new %.4f) within tol %.4f",
                        -drop, prev_auc, median_auc_short, MAX_AUC_DROP)
    else:
        logger.info("No previous model found — first retrain, skipping regression gate")

    if not gates_pass:
        logger.error("Acceptance gates FAILED; not deploying. Previous model stays live.")
        sys.exit(2)

    # 5) Calibration table on training data (informational)
    logger.info("Calibration on training data (SHORT) — in-sample, but useful "
                "for picking threshold spread:")
    logger.info("  %-10s %-10s %-10s", "threshold", "n_signals", "in_sample_WR")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = p_short_all >= thr
        if mask.sum() >= 5:
            wr = y_short[mask].mean()
            logger.info("  %-10.2f %-10d %-9.2f%%", thr, int(mask.sum()), wr * 100)
        else:
            logger.info("  %-10.2f %-10d (too few)", thr, int(mask.sum()))

    # 6) Persist artifacts
    if args.dry_run:
        logger.info("DRY RUN complete — no artifacts written.")
        return 0

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    backup_previous_artifacts(ts_str)

    long_model.save_model(str(ARTIFACTS_DIR / "long_xgb.json"))
    short_model.save_model(str(ARTIFACTS_DIR / "short_xgb.json"))
    with open(ARTIFACTS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    end = datetime.now(timezone.utc)
    meta = {
        "version": f"v9_{ts_str}",
        "trained_at_utc": end.isoformat(),
        "duration_seconds": (end - start).total_seconds(),
        "n_bars": n_bars,
        "feature_cols_n": len(feature_cols),
        "horizon_bars": HORIZON,
        "tp_dist": TP_DIST,
        "sl_dist": SL_DIST,
        "train_auc_long": float(train_auc_long),
        "train_auc_short": float(train_auc_short),
        "median_auc_long": float(median_auc_long),
        "median_auc_short": float(median_auc_short),
        "cv_aucs_long": [float(a) for a in cv_long],
        "cv_aucs_short": [float(a) for a in cv_short],
        "p_long_top1_pct": p_long_top1,
        "p_short_top1_pct": p_short_top1,
        "n_estimators_final": FINAL_N_ESTIMATORS,
        "model_params": {k: v for k, v in BASE_PARAMS.items()
                          if k not in ("random_state",)},
        "data_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
    }
    with open(ARTIFACTS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    update_training_stats(meta)

    logger.info("=" * 80)
    logger.info("RETRAIN SUCCEEDED — artifacts written to %s", ARTIFACTS_DIR)
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
