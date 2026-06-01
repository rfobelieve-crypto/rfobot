"""A/B walk-forward: V7 baseline vs V7 + 7 liquidity-proxy features.

Uses the exact same XGB hyperparams and 77-fold WF split as
train_direction_reg_4h.py.  Only difference: feature set.

Compares OOS:
  - Spearman IC overall + per-month
  - Sign-AUC (treating sign(pred_ret) as binary classifier on sign(y))
  - Strong WR @ thr=0.008
  - MAE / RMSE

Output:
  research/results/dual_model/liq_features_ab_metrics.csv
  research/results/dual_model/liq_features_ab_oos_new.parquet

Does NOT touch production model_artifacts.  Read-only research artefact.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, walk_forward_splits, RESULTS_DIR,
)
from research.dual_model.build_direction_reg_labels import (
    build_direction_reg_labels,
)
from research.dual_model.direction_features_v2 import (
    FULL_DIRECTION, filter_available,
)
from shared.db import get_db_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_PARAMS = {
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 400,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbosity": 0,
    "early_stopping_rounds": 30,
    "objective": "reg:squarederror", "eval_metric": "mae",
}


# ── Liquidity proxy features ──────────────────────────────────────────


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 7 features to df in place.

    Trailing-only; uses df's own OHLC + volume + DB joins for liq/depth.
    Returns the modified df (same object).
    """
    # A: swing distances
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    rh_168 = high.rolling(168).max()
    rl_168 = low.rolling(168).min()
    rh_24 = high.rolling(24).max()
    rl_24 = low.rolling(24).min()
    rl_4 = low.rolling(4).min()

    df["liq_A_swing_high_dist_168h"] = (rh_168 - close) / close
    df["liq_A_swing_low_dist_168h"] = (close - rl_168) / close
    df["liq_A_swing_high_dist_24h"] = (rh_24 - close) / close
    df["liq_A_swing_low_dist_24h"] = (close - rl_24) / close
    df["liq_A_swing_low_dist_4h"] = (close - rl_4) / close
    df["liq_A_swing_high_dist_4h"] = (high.rolling(4).max() - close) / close

    # F: bear sweep magnitude
    past_high_24 = high.shift(1).rolling(24).max()
    bear_sweep = (high > past_high_24) & (close < past_high_24)
    df["liq_F_bear_sweep_mag"] = np.where(
        bear_sweep, (high - past_high_24) / close, 0.0,
    )

    # E: liq z-score (4h vs 7d baseline)
    liq_h = _load_liq_hourly(df.index)
    if liq_h is not None and not liq_h.empty:
        liq_4h = liq_h.rolling(4).sum() / 4
        base_mu = liq_h.rolling(168).mean()
        base_sd = liq_h.rolling(168).std()
        df["liq_E_liq_z_4h_vs_7d"] = (liq_4h - base_mu) / (base_sd + 1)
    else:
        df["liq_E_liq_z_4h_vs_7d"] = np.nan

    # G: depth_imbalance mean
    depth_h = _load_depth_hourly(df.index)
    if depth_h is not None and not depth_h.empty:
        df["liq_G_depth_imb_4h_mean"] = depth_h.rolling(4).mean()
    else:
        df["liq_G_depth_imb_4h_mean"] = np.nan

    return df


def _load_liq_hourly(target_index: pd.DatetimeIndex) -> pd.Series:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT window_start, liq_total_usd FROM liquidation_1m
                WHERE canonical_symbol='BTC-USD'
                ORDER BY window_start ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    df = df.set_index("ts")["liq_total_usd"].astype(float)
    h = df.resample("1h", label="right", closed="right").sum()
    return h.reindex(target_index, method="nearest",
                      tolerance=pd.Timedelta("30min"))


def _load_depth_hourly(target_index: pd.DatetimeIndex) -> pd.Series:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, depth_imbalance FROM indicator_depth_snapshots
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df = df.set_index("dt")["depth_imbalance"].astype(float)
    return df.reindex(target_index, method="nearest",
                      tolerance=pd.Timedelta("30min"))


LIQ_FEATURES = [
    "liq_A_swing_high_dist_168h",
    "liq_A_swing_low_dist_168h",
    "liq_A_swing_high_dist_24h",
    "liq_A_swing_low_dist_24h",
    "liq_A_swing_low_dist_4h",
    "liq_A_swing_high_dist_4h",
    "liq_F_bear_sweep_mag",
    "liq_E_liq_z_4h_vs_7d",
    "liq_G_depth_imb_4h_mean",
]


# ── Training (mirrors train_direction_reg_4h.py) ──────────────────────


def train_walk_forward(df: pd.DataFrame, feature_cols: list[str],
                        label_col: str = "y_path_ret_4h",
                        label="") -> pd.DataFrame:
    """Returns OOS predictions DataFrame [pred_ret, y_path_ret_4h, fold]."""
    splits = walk_forward_splits(len(df))
    all_oos = []
    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = df.iloc[tr_idx][feature_cols], df.iloc[tr_idx][label_col]
        X_te, y_te = df.iloc[te_idx][feature_cols], df.iloc[te_idx][label_col]
        # Drop rows with NaN target
        tr_mask = y_tr.notna()
        te_mask = y_te.notna()
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_te, y_te = X_te[te_mask], y_te[te_mask]
        if len(X_tr) < 50 or len(X_te) < 5:
            continue
        # Internal validation split for early stopping
        val_n = max(20, len(X_tr) // 10)
        X_train, X_val = X_tr.iloc[:-val_n], X_tr.iloc[-val_n:]
        y_train, y_val = y_tr.iloc[:-val_n], y_tr.iloc[-val_n:]

        model = xgb.XGBRegressor(**BASE_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_te)
        all_oos.append(pd.DataFrame({
            "pred_ret": preds,
            "y_path_ret_4h": y_te.values,
            "fold": fold_idx,
        }, index=X_te.index))
        if fold_idx % 10 == 0:
            logger.info("%s  fold %d/%d  train=%d  test=%d",
                        label, fold_idx, len(splits), len(X_tr), len(X_te))
    if not all_oos:
        return pd.DataFrame()
    return pd.concat(all_oos)


def metrics(oos: pd.DataFrame, label: str) -> dict:
    """Spearman IC + sign AUC + Strong WR @ thr=0.008 + MAE/RMSE."""
    df = oos.dropna()
    if df.empty:
        return {"label": label, "n": 0}
    ic, _ = spearmanr(df["pred_ret"], df["y_path_ret_4h"])
    sign_y = (df["y_path_ret_4h"] > 0).astype(int)
    try:
        auc = roc_auc_score(sign_y, df["pred_ret"])
    except Exception:
        auc = np.nan
    thr = 0.008
    strong_mask = df["pred_ret"].abs() >= thr
    n_strong = int(strong_mask.sum())
    if n_strong > 0:
        strong = df[strong_mask].copy()
        correct = (np.sign(strong["pred_ret"]) ==
                   np.sign(strong["y_path_ret_4h"])).sum()
        wr = correct / n_strong
    else:
        wr = np.nan
    mae = (df["pred_ret"] - df["y_path_ret_4h"]).abs().mean()
    rmse = np.sqrt(((df["pred_ret"] - df["y_path_ret_4h"]) ** 2).mean())
    return {
        "label": label, "n": len(df),
        "ic": float(ic),
        "sign_auc": float(auc) if not np.isnan(auc) else None,
        "strong_n": n_strong,
        "strong_wr": float(wr) if not np.isnan(wr) else None,
        "mae": float(mae), "rmse": float(rmse),
    }


def main() -> int:
    logger.info("Loading features…")
    df = load_and_cache_data(limit=4000)
    logger.info("Loaded: %d bars × %d cols", *df.shape)

    logger.info("Adding liquidity features…")
    df = add_liquidity_features(df)
    logger.info("Now: %d cols", df.shape[1])
    logger.info("Non-NaN coverage of new features:")
    for c in LIQ_FEATURES:
        logger.info("  %-32s  %5d / %5d (%.0f%%)", c,
                    df[c].notna().sum(), len(df),
                    df[c].notna().mean() * 100)

    logger.info("Building labels…")
    labels = build_direction_reg_labels(df)
    df = df.join(labels[["y_path_ret_4h"]], how="left")
    logger.info("Label coverage: %d / %d", df["y_path_ret_4h"].notna().sum(),
                len(df))

    # Baseline feature set: existing FULL_DIRECTION (V7) intersected with
    # what's actually available in df
    base_feats = filter_available(df, FULL_DIRECTION)
    new_feats = base_feats + LIQ_FEATURES
    logger.info("Baseline: %d features  |  +liq: %d features",
                len(base_feats), len(new_feats))

    logger.info("Training BASELINE (V7) — 77-fold walk-forward…")
    oos_base = train_walk_forward(df, base_feats, label="BASE")
    logger.info("BASE OOS: %d rows", len(oos_base))

    logger.info("Training NEW (V7 + liq) — 77-fold walk-forward…")
    oos_new = train_walk_forward(df, new_feats, label="NEW ")
    logger.info("NEW  OOS: %d rows", len(oos_new))

    out_path = RESULTS_DIR / "liq_features_ab_oos_new.parquet"
    oos_new.to_parquet(out_path)
    logger.info("Wrote %s", out_path)

    m_base = metrics(oos_base, "BASELINE (V7)")
    m_new = metrics(oos_new, "NEW (V7 + 9 liq)")

    print()
    print("=" * 78)
    print(f"{'Metric':16s}  {'BASE':>15s}  {'NEW':>15s}  {'Δ':>10s}")
    print("=" * 78)
    keys = ["n", "ic", "sign_auc", "strong_n", "strong_wr", "mae", "rmse"]
    for k in keys:
        a = m_base.get(k)
        b = m_new.get(k)
        if a is None or b is None:
            continue
        if isinstance(a, int):
            d_str = f"{b - a:+d}"
        else:
            d_str = f"{b - a:+.4f}"
        print(f"{k:16s}  {a:>15}  {b:>15}  {d_str:>10s}")
    print("=" * 78)

    df_metrics = pd.DataFrame([m_base, m_new])
    df_metrics.to_csv(RESULTS_DIR / "liq_features_ab_metrics.csv", index=False)
    logger.info("Wrote metrics → %s", RESULTS_DIR / "liq_features_ab_metrics.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
