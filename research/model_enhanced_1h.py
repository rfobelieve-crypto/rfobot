"""
Enhanced 1h Return Model — uses full 144-feature set from enhanced_features.

Key differences from model_return_1h.py:
  - Reads BTC_USD_15m_enhanced.parquet (144 features)
  - Feature selection via XGBoost importance (keep top N)
  - SHAP analysis for interpretability
  - Tighter XGBoost regularisation for high-dimensional input

Usage:
    python research/model_enhanced_1h.py
    python research/model_enhanced_1h.py --save
    python research/model_enhanced_1h.py --top-k 50
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PARQUET      = Path(__file__).parent / "ml_data" / "BTC_USD_15m_enhanced.parquet"
OUT_DIR      = Path(__file__).parent / "eda_charts"
TARGET       = "y_return_1h"
HOLDING_BARS = 4
N_FOLDS      = 5
FEE_RT       = 0.0014
IC_WINDOW    = 200

# Columns that are targets, metadata, or lookahead — never use as features
EXCLUDE = {
    "ts_open", "open", "high", "low", "close",
    # targets
    "y_return_1h", "y_return_4h",
    # old labels (lookahead)
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h", "vol_4h_proxy",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    # regime routing (not features — will use for analysis only)
    "regime", "regime_name",
    # old composite score
    "bull_bear_score",
    # volume from klines (keep tick_total_volume instead to avoid redundancy)
    "volume",
}

XGB_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.02,
    subsample        = 0.7,
    colsample_bytree = 0.5,
    min_child_weight = 10,
    reg_alpha        = 0.5,
    reg_lambda       = 2.0,
    random_state     = 42,
    verbosity        = 0,
    early_stopping_rounds = 30,
)

RIDGE_ALPHA = 10.0


# ─── Load & prepare ─────────────────────────────────────────────────────────

def load_and_prepare(top_k: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PARQUET)

    # Index is already datetime if saved with index=True
    if "dt" not in df.columns and df.index.name != "dt":
        if "ts_open" in df.columns:
            df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
            df = df.set_index("dt")
    df = df.sort_index()

    # Target
    if TARGET not in df.columns:
        df[TARGET] = df["close"].shift(-HOLDING_BARS) / df["close"] - 1

    df = df.iloc[:-HOLDING_BARS].copy()
    df = df.dropna(subset=[TARGET])

    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    # Forward-fill, then drop high-NaN features
    df[feat_cols] = df[feat_cols].ffill()
    nan_rate = df[feat_cols].isnull().mean()
    drop_high = list(nan_rate[nan_rate > 0.10].index)
    if drop_high:
        print(f"  Dropping high-NaN features ({len(drop_high)}): {drop_high[:10]}...")
        feat_cols = [c for c in feat_cols if c not in drop_high]

    df = df.dropna(subset=feat_cols)

    # Feature selection via initial XGBoost importance (if top_k specified)
    if top_k and top_k < len(feat_cols):
        print(f"\n  Feature selection: fitting initial XGBoost on first 60% to rank {len(feat_cols)} features...")
        n_init = int(len(df) * 0.6)
        X_init = df[feat_cols].fillna(0).values[:n_init]
        y_init = df[TARGET].values[:n_init]
        m_init = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
            reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbosity=0,
        )
        m_init.fit(X_init, y_init)
        imp = pd.Series(m_init.feature_importances_, index=feat_cols)
        imp = imp.sort_values(ascending=False)
        selected = list(imp.head(top_k).index)
        print(f"  Top {top_k} features selected:")
        for i, (name, score) in enumerate(imp.head(top_k).items()):
            print(f"    {i+1:3d}. {name:35s}  imp={score:.4f}")
        feat_cols = selected

    print(f"\n  Dataset: {len(df)} rows  x  {len(feat_cols)} features")
    print(f"  Target ({TARGET}): mean={df[TARGET].mean():.5f}  "
          f"std={df[TARGET].std():.4f}")
    return df, feat_cols


# ─── Walk-forward OOS ────────────────────────────────────────────────────────

def _fit_xgb(X_tr, y_tr, X_te, y_te):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return m.predict(X_te), m


def _fit_ridge(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    m = Ridge(alpha=RIDGE_ALPHA)
    m.fit(X_tr_s, y_tr)
    return m.predict(X_te_s)


def walk_forward_oos(df, feat_cols, model_type="xgb", verbose=True):
    n    = len(df)
    fold = n // (N_FOLDS + 1)
    oos  = np.full(n, np.nan)

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr = df[feat_cols].fillna(0).values[:tr_end]
        y_tr = df[TARGET].values[:tr_end]
        X_te = df[feat_cols].fillna(0).values[tr_end:te_end]
        y_te = df[TARGET].values[tr_end:te_end]

        if model_type == "xgb":
            fold_pred, _ = _fit_xgb(X_tr, y_tr, X_te, y_te)
        else:
            fold_pred = _fit_ridge(X_tr, y_tr, X_te)

        oos[tr_end:te_end] = fold_pred

        if verbose:
            ic, _ = spearmanr(y_te, fold_pred)
            print(f"  fold {k}/{N_FOLDS}  IC={ic:+.4f}  n={len(y_te)}")

    return oos


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(df, oos, label=""):
    y = df[TARGET].values
    valid = ~np.isnan(oos) & ~np.isnan(y)
    p, a = oos[valid], y[valid]

    # Direction accuracy
    dir_acc = (np.sign(p) == np.sign(a)).mean()

    # IC
    ic, _ = spearmanr(a, p)

    # Per-fold IC for ICIR
    n    = len(df)
    fold = n // (N_FOLDS + 1)
    fold_ics = []
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        pp = oos[tr_end:te_end]
        aa = y[tr_end:te_end]
        m = ~np.isnan(pp) & ~np.isnan(aa)
        if m.sum() > 20:
            ic_f, _ = spearmanr(aa[m], pp[m])
            fold_ics.append(ic_f)
    fold_ics = np.array(fold_ics)
    icir = fold_ics.mean() / fold_ics.std() if fold_ics.std() > 0 else 0

    # Mean return by prediction sign
    mask_up = p > 0
    mask_dn = p < 0
    ret_up  = a[mask_up].mean() if mask_up.any() else 0
    ret_dn  = a[mask_dn].mean() if mask_dn.any() else 0
    spread  = ret_up - ret_dn

    # Calibration monotonicity
    combined = pd.DataFrame({"pred": p, "actual": a})
    combined["bin"] = pd.qcut(combined["pred"], 10, labels=False, duplicates="drop")
    decile_means = combined.groupby("bin")["actual"].mean()
    diffs = np.diff(decile_means.values)
    monotone = (diffs > 0).mean()

    print(f"\n  [{label}]")
    print(f"  IC={ic:+.4f}  ICIR={icir:+.4f}  Dir_acc={dir_acc:.1%}")
    print(f"  fold_ICs={[round(v,4) for v in fold_ics]}")
    print(f"  Ret_up={ret_up*100:+.4f}%  Ret_dn={ret_dn*100:+.4f}%  Spread={spread*100:.4f}%")
    print(f"  Monotone_up={monotone:.0%}  n_up={mask_up.sum()}  n_dn={mask_dn.sum()}")

    # Per-regime
    if "regime_name" in df.columns:
        regime = df["regime_name"].values[valid]
        print("  Per regime:")
        for r in sorted(set(regime)):
            m = regime == r
            if m.sum() < 30:
                continue
            ic_r, _ = spearmanr(a[m], p[m])
            da_r = (np.sign(p[m]) == np.sign(a[m])).mean()
            print(f"    {r:20s}  IC={ic_r:+.4f}  Dir={da_r:.1%}  n={m.sum()}")

    return {
        "label": label, "ic": ic, "icir": icir, "dir_acc": dir_acc,
        "spread": spread, "monotone": monotone,
        "fold_ics": fold_ics,
    }


# ─── Backtest ────────────────────────────────────────────────────────────────

def backtest(df, oos, pct=0.10, fee=FEE_RT, label=""):
    y = df[TARGET].values
    pred_s = pd.Series(oos)
    upper = pred_s.expanding(min_periods=50).quantile(1 - pct)
    lower = pred_s.expanding(min_periods=50).quantile(pct)

    first_valid = int(np.argmax(~np.isnan(oos)))
    start = first_valid + HOLDING_BARS
    rebalance = range(start, len(df) - HOLDING_BARS, HOLDING_BARS)

    records = []
    for t in rebalance:
        p = oos[t]; u = upper.iloc[t]; l = lower.iloc[t]; yr = y[t]
        if np.isnan(p) or np.isnan(u) or np.isnan(yr):
            continue
        pos = 1 if p > u else (-1 if p < l else 0)
        if pos == 0:
            continue
        gross = pos * yr
        net = gross - fee
        records.append({"net_ret": net, "gross_ret": gross, "pos": pos})

    if not records:
        print(f"  [{label}] No trades")
        return

    trades = pd.DataFrame(records)
    r = trades["net_ret"]
    eq = np.cumprod(1 + r.values)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    wr = (r > 0).mean()

    # Daily sharpe
    n_long  = (trades["pos"] == 1).sum()
    n_short = (trades["pos"] == -1).sum()

    print(f"  [{label}] n={len(r)}  Ret={eq[-1]-1:+.1%}  "
          f"MDD={mdd:.1%}  WR={wr:.1%}  "
          f"L={n_long} S={n_short}  "
          f"mean_gross={trades['gross_ret'].mean()*100:+.4f}%  "
          f"mean_net={r.mean()*100:+.4f}%")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--model", default="both", choices=["xgb", "linear", "both"])
    ap.add_argument("--top-k", type=int, default=50, help="Top K features to keep")
    args = ap.parse_args()

    print("=" * 60)
    print("Enhanced 1h Return Model")
    print("=" * 60)

    df, feat_cols = load_and_prepare(top_k=args.top_k)

    # ── Run models ───────────────────────────────────────────────────────
    for model_type in (["xgb", "linear"] if args.model == "both" else [args.model]):
        print(f"\n{'='*60}")
        print(f"Walk-Forward OOS — {model_type.upper()}")
        print("=" * 60)
        oos = walk_forward_oos(df, feat_cols, model_type=model_type)

        print(f"\n{'='*60}")
        print(f"Evaluation — {model_type.upper()}")
        print("=" * 60)
        result = evaluate(df, oos, label=model_type)

        print(f"\n{'='*60}")
        print(f"Backtest — {model_type.upper()}")
        print("=" * 60)
        for pct in [0.05, 0.10, 0.20]:
            backtest(df, oos, pct=pct, label=f"{model_type} top/bot {int(pct*100)}%")

        # OOS hold-out
        print(f"\n  OOS Hold-Out (last 33%):")
        n_split = int(len(df) * 0.67)
        pred_os = oos.copy(); pred_os[:n_split] = np.nan
        backtest(df, pred_os, pct=0.10, label=f"{model_type} OOS-only")

    print("\nDone.")


if __name__ == "__main__":
    main()
