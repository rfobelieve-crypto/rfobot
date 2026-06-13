"""
Research: Lorentzian KNN vs XGBoost on this project's orderflow features.

Purpose
-------
Test whether the Lorentzian distance metric (from jdehorty's LDC indicator)
wins XGBoost when applied to *our* 136 orderflow/CG/cross-market features
and the 4h TWAP path-return label.

This is a research-only benchmark. It does NOT modify any production code
under indicator/. All outputs go to research/results/lorentzian_bench*.json.

Method
------
For each walk-forward fold (77 folds, initial_train=288, test_size=48,
step=48, purge=4, embargo=4 — same as production training split), we train
three models on the fold's training window and predict its test window:

  1. XGBoost baseline        — same hyperparams as production dual_direction
  2. Lorentzian KNN (K=8)    — metric = Σ log(1 + |xᵢ - yᵢ|)
  3. Euclidean KNN (K=8)     — metric = √Σ (xᵢ - yᵢ)²   (control group)

KNN follows LDC's chronological stride (every 4th train bar as a candidate
neighbor) to avoid correlated-sample clustering.

Features are z-score normalized per fold using the train set only — KNN is
scale-sensitive; XGBoost is scale-invariant (fed raw).

Outputs
-------
- Per model: Spearman/Pearson IC, bootstrap CI, top-5%/10% bidirectional
  precision, monthly IC stability.
- Signal overlap (Jaccard, top-5% bars) between every model pair — high
  overlap ⇒ same information; low overlap ⇒ potential ensemble value.
- Top-10 XGB feature importances (for context).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "dual_model"))
from shared_data import walk_forward_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CACHE = Path("research/dual_model/.cache/features_all.parquet")
FEATURE_COLS_FILE = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")
RESULTS_FILE = Path("research/results/lorentzian_bench.json")

HORIZON = 4
K = 8
KNN_STRIDE = 4
N_FOLDS_LIMIT: int | None = None  # full 77 folds


def build_labels(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    fwd_mean = sum(close.shift(-i) for i in range(1, HORIZON + 1)) / HORIZON
    return fwd_mean / close - 1


def knn_predict(train_X: np.ndarray, train_y: np.ndarray,
                test_X: np.ndarray, k: int, metric: str,
                stride: int = KNN_STRIDE) -> np.ndarray:
    sub_idx = np.arange(0, len(train_X), stride)
    X_sub = train_X[sub_idx]
    y_sub = train_y[sub_idx]

    preds = np.zeros(len(test_X))
    for i, x in enumerate(test_X):
        diff = X_sub - x[None, :]
        if metric == "lorentzian":
            d = np.log1p(np.abs(diff)).sum(axis=1)
        else:
            d = np.sqrt((diff ** 2).sum(axis=1))
        kk = min(k, len(d) - 1)
        k_idx = np.argpartition(d, kk)[:kk]
        preds[i] = y_sub[k_idx].mean()
    return preds


def xgb_predict(train_X: np.ndarray, train_y: np.ndarray,
                test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(train_X, train_y)
    return model.predict(test_X), model.feature_importances_


def compute_ic(pred: np.ndarray, actual: np.ndarray) -> dict:
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() < 30:
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(mask.sum())}
    sp, _ = spearmanr(pred[mask], actual[mask])
    pe, _ = pearsonr(pred[mask], actual[mask])
    return {"spearman": float(sp), "pearson": float(pe), "n": int(mask.sum())}


def bootstrap_ic(pred: np.ndarray, actual: np.ndarray, n_boot: int = 500) -> dict:
    mask = ~(np.isnan(pred) | np.isnan(actual))
    p, a = pred[mask], actual[mask]
    n = len(p)
    if n < 30:
        return {"ic": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(p[idx], a[idx])
        boots[b] = r
    return {
        "ic": float(np.mean(boots)),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
    }


def top_k_precision(pred: np.ndarray, actual: np.ndarray, k_frac: float) -> dict:
    mask = ~(np.isnan(pred) | np.isnan(actual))
    p, a = pred[mask], actual[mask]
    n = len(p)
    k = int(n * k_frac)
    if k < 4:
        return {"precision": float("nan"), "k": 0, "n": n}
    k_side = k // 2
    up_idx = np.argsort(p)[-k_side:]
    dn_idx = np.argsort(p)[:k_side]
    correct = (a[up_idx] > 0).sum() + (a[dn_idx] < 0).sum()
    return {"precision": float(correct / (2 * k_side)), "k": int(2 * k_side), "n": n}


def signal_overlap(pred_a: np.ndarray, pred_b: np.ndarray, k_frac: float = 0.05) -> float:
    mask = ~(np.isnan(pred_a) | np.isnan(pred_b))
    pa, pb = pred_a[mask], pred_b[mask]
    n = len(pa)
    k = int(n * k_frac)
    if k < 4:
        return float("nan")
    k_side = k // 2
    top_a = set(np.argsort(pa)[-k_side:]) | set(np.argsort(pa)[:k_side])
    top_b = set(np.argsort(pb)[-k_side:]) | set(np.argsort(pb)[:k_side])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return float(inter / union) if union else float("nan")


def main():
    t0 = time.time()
    logger.info("Loading features from %s", FEATURE_CACHE)
    df = pd.read_parquet(FEATURE_CACHE)
    logger.info("Cache: %d rows × %d cols, %s ~ %s",
                len(df), df.shape[1], df.index[0], df.index[-1])

    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Missing %d features in cache (first 5: %s) — dropping them",
                       len(missing), missing[:5])
        feature_cols = [c for c in feature_cols if c in df.columns]

    df["y"] = build_labels(df)
    df = df.iloc[:-HORIZON].copy()

    # Match production: feature NaN → 0 (same as indicator/inference.py line 398).
    # This keeps full history available; early bars will have many zero-filled
    # features (cross-market / FNG etc. added mid-history). KNN is more
    # sensitive to this than XGB, but it's the fair production comparison.
    X_full = df[feature_cols].fillna(0).values.astype(float)
    y = df["y"].values.astype(float)
    valid = ~np.isnan(y)
    df = df.loc[valid].copy()
    X_full = X_full[valid]
    y = y[valid]
    logger.info("After label NaN filter: n=%d features=%d (NaN→0), %s ~ %s",
                len(df), len(feature_cols), df.index[0], df.index[-1])

    splits = walk_forward_splits(len(df))
    if N_FOLDS_LIMIT:
        splits = splits[:N_FOLDS_LIMIT]
    logger.info("Walk-forward folds: %d", len(splits))

    n = len(df)
    oos = {k: np.full(n, np.nan) for k in ("xgb", "knn_lor", "knn_euc")}
    feature_imp_sum = np.zeros(len(feature_cols))
    n_imp = 0

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X_full[train_idx], y[train_idx]
        X_te = X_full[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        xgb_pred, imp = xgb_predict(X_tr, y_tr, X_te)
        oos["xgb"][test_idx] = xgb_pred
        feature_imp_sum += imp
        n_imp += 1

        oos["knn_lor"][test_idx] = knn_predict(X_tr_s, y_tr, X_te_s, K, "lorentzian")
        oos["knn_euc"][test_idx] = knn_predict(X_tr_s, y_tr, X_te_s, K, "euclidean")

        if (fold_i + 1) % 10 == 0 or fold_i == len(splits) - 1:
            elapsed = time.time() - t0
            logger.info("  fold %d/%d done (%.1fs elapsed)", fold_i + 1, len(splits), elapsed)

    valid_mask = ~np.isnan(oos["xgb"])
    y_oos = y[valid_mask]
    dates = df.index[valid_mask]
    logger.info("OOS coverage: %d samples (%s ~ %s)",
                len(y_oos), dates[0], dates[-1])

    results = {
        "n_oos": int(len(y_oos)),
        "n_folds": len(splits),
        "n_features": len(feature_cols),
        "horizon_bars": HORIZON,
        "knn_k": K,
        "knn_stride": KNN_STRIDE,
        "date_range": [str(dates[0]), str(dates[-1])],
        "models": {},
    }

    for name, pred_full in oos.items():
        pred = pred_full[valid_mask]
        ic = compute_ic(pred, y_oos)
        boot = bootstrap_ic(pred, y_oos)
        prec5 = top_k_precision(pred, y_oos, 0.05)
        prec10 = top_k_precision(pred, y_oos, 0.10)
        results["models"][name] = {
            "ic_spearman": ic["spearman"],
            "ic_pearson": ic["pearson"],
            "ic_boot_mean": boot["ic"],
            "ic_boot_ci95": [boot["ci_lo"], boot["ci_hi"]],
            "top5_precision": prec5["precision"],
            "top5_k": prec5["k"],
            "top10_precision": prec10["precision"],
            "top10_k": prec10["k"],
        }
        logger.info(
            "%-8s  IC=%.4f CI[%.3f,%.3f]  top5=%.3f (k=%d)  top10=%.3f",
            name, ic["spearman"], boot["ci_lo"], boot["ci_hi"],
            prec5["precision"], prec5["k"], prec10["precision"],
        )

    results["overlap_top5_jaccard"] = {
        "xgb_vs_knn_lor": signal_overlap(
            oos["xgb"][valid_mask], oos["knn_lor"][valid_mask]
        ),
        "xgb_vs_knn_euc": signal_overlap(
            oos["xgb"][valid_mask], oos["knn_euc"][valid_mask]
        ),
        "knn_lor_vs_knn_euc": signal_overlap(
            oos["knn_lor"][valid_mask], oos["knn_euc"][valid_mask]
        ),
    }

    monthly_ic = {}
    periods = dates.to_period("M")
    for month in periods.unique():
        m_mask = periods == month
        if m_mask.sum() < 30:
            continue
        row = {}
        for name, pred_full in oos.items():
            pred_m = pred_full[valid_mask][m_mask]
            y_m = y_oos[m_mask]
            ic_m = compute_ic(pred_m, y_m)
            row[name] = ic_m["spearman"]
        row["_n"] = int(m_mask.sum())
        monthly_ic[str(month)] = row
    results["monthly_ic"] = monthly_ic

    avg_imp = feature_imp_sum / n_imp
    top10_idx = np.argsort(avg_imp)[::-1][:10]
    results["top10_xgb_features"] = [
        {"name": feature_cols[i], "importance": float(avg_imp[i])}
        for i in top10_idx
    ]

    results["elapsed_seconds"] = round(time.time() - t0, 1)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results → %s (%.1fs total)", RESULTS_FILE, results["elapsed_seconds"])


if __name__ == "__main__":
    main()
