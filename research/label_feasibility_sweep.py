"""
Research: label feasibility sweep — what target does this orderflow
feature set actually predict best?

Motivation
----------
Production model targets `y_path_ret_4h` (signed 4h TWAP return). Its
walk-forward OOS Spearman IC ≈ 0.18 and AUC ≈ 0.60 suggest it is
already at the structural ceiling for BTC 4h direction prediction
(mistake log 2026-04-14). But orderflow-derived features are arguably
better suited to targets *other than* signed direction — magnitude,
volatility, liquidation cascades. This sweep answers empirically which
target the existing 136-feature set predicts best.

Method
------
- Use the production direction feature list (136 features) loaded from
  indicator/model_artifacts/dual_model/direction_feature_cols.json.
- Evaluate 6 candidate labels against the same feature set, under
  identical walk-forward splits (77 folds, initial_train=288,
  test_size=48, step=48, purge=4, embargo=4) with the same XGBoost
  hyperparameters used elsewhere in this repo (n_estimators=250,
  max_depth=5, lr=0.05, subsample=0.8, colsample_bytree=0.8).
- For each label, train per-fold and collect OOS predictions. Compute:
    * regression (for continuous labels): Spearman IC, Pearson IC,
      bootstrap 95% CI, top-5% bidirectional precision (hit-rate
      relative to sign of realized signed return)
    * classification (for binary labels): AUC with bootstrap CI,
      top-5% precision (rank-based)
    * Monthly stability: IC/AUC per calendar month
- Labels evaluated:
    1. path_ret_4h      : current production target
                          y[t] = mean(close[t+1..t+4]) / close[t] - 1
    2. abs_ret_4h       : magnitude, |endpoint return|
                          y[t] = |close[t+4] / close[t] - 1|
    3. realized_vol_4h  : realized volatility
                          y[t] = sqrt(sum(log_ret[t+1..t+4]^2))
    4. liq_surge_4h     : binary — forward 4h aggregate liquidation
                          exceeds its trailing 90th-percentile over a
                          500-bar window. Uses cg_liq_total.
    5. path_ret_1h      : short-horizon direction
                          y[t] = close[t+1] / close[t] - 1
    6. breakout_dir_4h  : CONDITIONAL signed return — only labeled when
                          |y_path_ret_4h| > 1% (vol surge gate), else
                          NaN. Measures "given a big move occurs, can
                          we predict its direction".

All labels use strictly trailing features (no feature→label leakage).
Purge+embargo prevents horizon leakage across fold boundaries.

Outputs: research/results/label_feasibility.json
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "dual_model"))
from shared_data import walk_forward_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CACHE = Path("research/dual_model/.cache/features_all.parquet")
FEATURE_COLS_FILE = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")
RESULTS_FILE = Path("research/results/label_feasibility.json")

HORIZON = 4
N_FOLDS_LIMIT: int | None = None  # None = full 77 folds; set for sanity runs

XGB_PARAMS = dict(
    n_estimators=250, max_depth=5, learning_rate=0.05,
    reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbosity=0,
)


# ── Label constructors ──

def label_path_ret_4h(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    fwd_mean = sum(close.shift(-i) for i in range(1, HORIZON + 1)) / HORIZON
    return fwd_mean / close - 1


def label_abs_ret_4h(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    return (close.shift(-HORIZON) / close - 1).abs()


def label_realized_vol_4h(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    # sum of squared forward returns, sqrt → realized vol over 4 bars
    sq = log_ret ** 2
    fwd_sum = sum(sq.shift(-i) for i in range(1, HORIZON + 1))
    return np.sqrt(fwd_sum)


def label_liq_surge_4h(df: pd.DataFrame) -> pd.Series:
    """Binary: forward 4h aggregate liquidation > its trailing 500-bar
    90th percentile. Trailing percentile avoids look-ahead."""
    if "cg_liq_total" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    liq = df["cg_liq_total"].astype(float).fillna(0)
    fwd_liq = sum(liq.shift(-i) for i in range(1, HORIZON + 1))
    # Trailing 500-bar 90th percentile of hourly liq × 4 as threshold
    trailing_p90 = liq.rolling(500, min_periods=100).quantile(0.90) * HORIZON
    return (fwd_liq > trailing_p90).astype(float)


def label_path_ret_1h(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    return close.shift(-1) / close - 1


def label_breakout_dir_4h(df: pd.DataFrame) -> pd.Series:
    """Conditional label: signed 4h endpoint return, but only defined
    when |return| > 1%. Else NaN (row excluded from training/eval).
    Measures 'given a meaningful move, can we predict direction'."""
    close = df["close"].astype(float)
    ret = close.shift(-HORIZON) / close - 1
    y = ret.where(ret.abs() > 0.01)
    return y


LABEL_SPECS = {
    "path_ret_4h":     {"fn": label_path_ret_4h,     "task": "regression"},
    "abs_ret_4h":      {"fn": label_abs_ret_4h,      "task": "regression"},
    "realized_vol_4h": {"fn": label_realized_vol_4h, "task": "regression"},
    "liq_surge_4h":    {"fn": label_liq_surge_4h,    "task": "classification"},
    "path_ret_1h":     {"fn": label_path_ret_1h,     "task": "regression"},
    "breakout_dir_4h": {"fn": label_breakout_dir_4h, "task": "regression"},
}


# ── Evaluation primitives ──

def bootstrap_stat(pred: np.ndarray, actual: np.ndarray, stat_fn,
                   n_boot: int = 500) -> dict:
    n = len(pred)
    if n < 30:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots[b] = stat_fn(pred[idx], actual[idx])
        except Exception:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) < 20:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "mean": float(np.mean(boots)),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
    }


def top_k_precision_binary(pred: np.ndarray, actual: np.ndarray,
                           k_frac: float) -> dict:
    """Rank-based: take top k_frac by pred probability, precision = base rate on those."""
    n = len(pred)
    k = max(int(n * k_frac), 4)
    top_idx = np.argsort(pred)[-k:]
    precision = float(actual[top_idx].mean())
    return {"precision": precision, "k": k, "n": n}


def top_k_precision_directional(pred: np.ndarray, actual: np.ndarray,
                                k_frac: float) -> dict:
    """Bidirectional: top k/2 on each tail, check sign match on actual."""
    n = len(pred)
    k = max(int(n * k_frac), 4)
    k_side = k // 2
    up_idx = np.argsort(pred)[-k_side:]
    dn_idx = np.argsort(pred)[:k_side]
    correct = (actual[up_idx] > 0).sum() + (actual[dn_idx] < 0).sum()
    return {"precision": float(correct / (2 * k_side)), "k": int(2 * k_side), "n": n}


def _spearman(p, a):
    r, _ = spearmanr(p, a)
    return r if r == r else 0.0  # NaN → 0


def _pearson(p, a):
    r, _ = pearsonr(p, a)
    return r if r == r else 0.0


def _auc(p, a):
    try:
        return roc_auc_score(a, p)
    except Exception:
        return 0.5


def train_and_predict(train_X, train_y, test_X, task: str):
    import xgboost as xgb
    if task == "classification":
        # If label is all one class in train fold, return constant pred
        if len(np.unique(train_y)) < 2:
            return np.full(len(test_X), float(train_y.mean()))
        model = xgb.XGBClassifier(**XGB_PARAMS, use_label_encoder=False,
                                  eval_metric="logloss")
        model.fit(train_X, train_y.astype(int))
        return model.predict_proba(test_X)[:, 1]
    else:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train_X, train_y)
        return model.predict(test_X)


def evaluate_label(label_name: str, y_full: np.ndarray, X_full: np.ndarray,
                   dates: pd.DatetimeIndex, task: str,
                   splits) -> dict:
    t0 = time.time()
    oos_pred = np.full(len(y_full), np.nan)

    valid_label_mask = ~np.isnan(y_full)
    total_valid = int(valid_label_mask.sum())
    if total_valid < 500:
        logger.warning("%s: only %d valid labels — skipping", label_name, total_valid)
        return {"label": label_name, "task": task, "error": "insufficient labels",
                "n_valid_labels": total_valid}

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        # Filter to rows where label is valid
        tr_mask = valid_label_mask[tr_idx]
        te_mask = valid_label_mask[te_idx]
        tr_i = np.array(tr_idx)[tr_mask]
        te_i = np.array(te_idx)[te_mask]
        if len(tr_i) < 50 or len(te_i) < 5:
            continue
        X_tr = X_full[tr_i]
        y_tr = y_full[tr_i]
        X_te = X_full[te_i]
        pred = train_and_predict(X_tr, y_tr, X_te, task)
        oos_pred[te_i] = pred

        if (fold_i + 1) % 20 == 0:
            logger.info("  %s fold %d/%d (%.1fs)", label_name, fold_i + 1,
                        len(splits), time.time() - t0)

    # Collect OOS
    oos_mask = ~np.isnan(oos_pred) & valid_label_mask
    pred = oos_pred[oos_mask]
    actual = y_full[oos_mask]
    dates_oos = dates[oos_mask]

    n = len(pred)
    if n < 100:
        return {"label": label_name, "task": task, "error": "insufficient OOS",
                "n_oos": n}

    result = {"label": label_name, "task": task, "n_oos": n,
              "n_valid_labels": total_valid}

    if task == "regression":
        ic_sp = _spearman(pred, actual)
        ic_pe = _pearson(pred, actual)
        boot_sp = bootstrap_stat(pred, actual, _spearman)
        # Top-5% precision only meaningful when label is signed
        is_signed = bool((actual < 0).any() and (actual > 0).any())
        if is_signed:
            prec5 = top_k_precision_directional(pred, actual, 0.05)
            prec10 = top_k_precision_directional(pred, actual, 0.10)
        else:
            # For non-signed labels (magnitude, vol), precision isn't
            # directly meaningful — report mag top-5% rank concordance
            # via pred-actual rank correlation at extreme quantile
            k = max(int(n * 0.05), 4)
            top_pred_idx = np.argsort(pred)[-k:]
            rank_overlap = float(np.mean(
                (actual[top_pred_idx] >= np.quantile(actual, 0.95)).astype(float)
            ))
            prec5 = {"rank_overlap_top5": rank_overlap, "k": k, "n": n}
            prec10 = {"rank_overlap_top10": float(np.mean(
                (actual[np.argsort(pred)[-max(int(n*0.1), 4):]] >= np.quantile(actual, 0.9)).astype(float)
            ))}
        result.update({
            "ic_spearman": ic_sp,
            "ic_pearson": ic_pe,
            "ic_boot_mean": boot_sp["mean"],
            "ic_boot_ci95": [boot_sp["ci_lo"], boot_sp["ci_hi"]],
            "top5": prec5,
            "top10": prec10,
            "is_signed": is_signed,
        })
    else:
        base_rate = float(actual.mean())
        auc = _auc(pred, actual)
        boot_auc = bootstrap_stat(pred, actual, _auc)
        prec5 = top_k_precision_binary(pred, actual, 0.05)
        prec10 = top_k_precision_binary(pred, actual, 0.10)
        result.update({
            "auc": auc,
            "auc_boot_mean": boot_auc["mean"],
            "auc_boot_ci95": [boot_auc["ci_lo"], boot_auc["ci_hi"]],
            "base_rate": base_rate,
            "top5": {**prec5, "lift": float(prec5["precision"] / max(base_rate, 1e-6))},
            "top10": {**prec10, "lift": float(prec10["precision"] / max(base_rate, 1e-6))},
        })

    # Monthly stability
    month_stability = {}
    periods = dates_oos.to_period("M") if hasattr(dates_oos, "to_period") else pd.DatetimeIndex(dates_oos).to_period("M")
    for m in periods.unique():
        m_mask = (periods == m)
        if m_mask.sum() < 30:
            continue
        p = pred[m_mask]; a = actual[m_mask]
        if task == "regression":
            month_stability[str(m)] = {"n": int(m_mask.sum()), "ic": _spearman(p, a)}
        else:
            month_stability[str(m)] = {"n": int(m_mask.sum()), "auc": _auc(p, a),
                                       "base_rate": float(a.mean())}
    result["monthly"] = month_stability
    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def main():
    logger.info("Loading features from %s", FEATURE_CACHE)
    df = pd.read_parquet(FEATURE_CACHE)
    logger.info("Cache: %d rows × %d cols, %s ~ %s",
                len(df), df.shape[1], df.index[0], df.index[-1])

    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("Using %d production features", len(feature_cols))

    X_full = df[feature_cols].fillna(0).values.astype(float)
    dates = df.index

    # Build all labels up front
    labels = {}
    for name, spec in LABEL_SPECS.items():
        y = spec["fn"](df).values.astype(float)
        n_valid = int(np.sum(~np.isnan(y)))
        logger.info("Label %-18s: %d / %d valid (%.1f%%)",
                    name, n_valid, len(y), 100 * n_valid / len(y))
        labels[name] = {"y": y, "task": spec["task"]}

    splits = walk_forward_splits(len(df))
    if N_FOLDS_LIMIT:
        splits = splits[:N_FOLDS_LIMIT]
    logger.info("Walk-forward folds: %d", len(splits))

    results = {}
    for name, info in labels.items():
        logger.info("=== Evaluating: %s (%s) ===", name, info["task"])
        r = evaluate_label(name, info["y"], X_full, dates, info["task"], splits)
        results[name] = r
        if info["task"] == "regression":
            logger.info(
                "%-18s  IC=%.4f CI[%.3f,%.3f]  n_oos=%d",
                name,
                r.get("ic_spearman", float("nan")),
                *(r.get("ic_boot_ci95", [float("nan"), float("nan")])),
                r.get("n_oos", 0),
            )
        else:
            logger.info(
                "%-18s  AUC=%.4f CI[%.3f,%.3f]  base=%.3f  top5_lift=%.2fx  n_oos=%d",
                name,
                r.get("auc", float("nan")),
                *(r.get("auc_boot_ci95", [float("nan"), float("nan")])),
                r.get("base_rate", float("nan")),
                r.get("top5", {}).get("lift", float("nan")),
                r.get("n_oos", 0),
            )

    out = {
        "n_features": len(feature_cols),
        "horizon_bars": HORIZON,
        "n_folds": len(splits),
        "xgb_params": XGB_PARAMS,
        "date_range": [str(df.index[0]), str(df.index[-1])],
        "labels": results,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Results → %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
