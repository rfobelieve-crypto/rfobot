"""
4h Close-to-Close Return Pipeline.

Fixes the core issues from the backtest audit:
  - Clean target: y_return_4h = close[t+16] / close[t] - 1  (no ambiguity)
  - No TP/SL -> no path-based ambiguity
  - Non-overlapping 4h rebalancing -> correct trade independence
  - Daily Sharpe -> correct annualization
  - Rolling expanding percentile threshold -> no lookahead on signal

Pipeline:
  Step 1  Build y_return_4h label
  Step 2  Feature preparation (order flow only, no price levels)
  Step 3  Walk-forward OOS predictions (Linear + XGBoost, regime-routed)
  Step 4  IC / ICIR evaluation per fold and per regime
  Step 5  Simple ranking backtest (long top N%, short bottom N%)
  Step 6  Performance metrics
  Step 7  Threshold sweep / regime breakdown / calibration

Usage:
    python research/model_return_4h.py
    python research/model_return_4h.py --save
    python research/model_return_4h.py --model xgb --save
    python research/model_return_4h.py --model linear --save
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────

PARQUET      = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
MODEL_DIR    = Path(__file__).parent / "models" / "return_4h"
OUT_DIR      = Path(__file__).parent / "eda_charts"
TARGET       = "y_return_4h"
HOLDING_BARS = 16           # 4h = 16 x 15m
N_FOLDS      = 5
FEE_RT       = 0.0014       # 0.07% x 2 sides
IC_WINDOW    = 200          # rolling IC window (bars)

# Features to exclude (lookahead labels, OI NaN, price levels, routing cols)
EXCLUDE = {
    "ts_open",
    # target
    "y_return_4h",
    # old range labels (all lookahead)
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h", "vol_4h_proxy",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    # forward returns (all lookahead)
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    # OI (all NaN in dataset)
    "oi", "oi_delta", "oi_accel", "oi_divergence",
    "cvd_x_oi_delta", "cvd_oi_ratio",
    # price levels (scale-dependent, not order flow)
    "open", "high", "low", "close",
    # regime routing columns (not features)
    "regime", "regime_name",
    # existing score (contains regime info)
    "bull_bear_score",
}

XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 3,       # shallower: returns are noisier than range metrics
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    min_child_weight = 20,      # more regularisation for noisy target
    reg_alpha        = 0.1,
    reg_lambda       = 2.0,
    random_state     = 42,
    verbosity        = 0,
    early_stopping_rounds = 30,
)

RIDGE_ALPHA = 1.0   # L2 regularisation for linear baseline

MIN_REGIME_TRAIN = 200
MIN_REGIME_TEST  = 5


# ─── Step 1 & 2: Load and prepare ─────────────────────────────────────────────

def load_and_prepare() -> tuple[pd.DataFrame, list[str]]:
    """
    Load parquet, build y_return_4h, prepare feature columns.

    y_return_4h = close[t + 16] / close[t] - 1
    Shift is negative (future), so last 16 rows will be NaN -> dropped.
    All features are past-only order flow signals.
    """
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()

    # ── Target: strict close-to-close 4h return ───────────────────────────────
    # shift(-16): close 16 bars in the future relative to bar t
    # this is computed BEFORE dropping any rows to get correct alignment
    df[TARGET] = df["close"].shift(-HOLDING_BARS) / df["close"] - 1

    # Drop last HOLDING_BARS rows (NaN target) + rows without valid target
    df = df.iloc[:-HOLDING_BARS].copy()
    df = df.dropna(subset=[TARGET])

    # ── Feature set (order flow only) ────────────────────────────────────────
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    # Forward-fill any gaps in features (e.g. funding_rate between settlements)
    df[feat_cols] = df[feat_cols].ffill()

    # Drop features with >5% NaN after ffill
    nan_rate  = df[feat_cols].isnull().mean()
    drop_high = list(nan_rate[nan_rate > 0.05].index)
    if drop_high:
        print(f"  Dropping high-NaN features: {drop_high}")
        feat_cols = [c for c in feat_cols if c not in drop_high]

    df = df.dropna(subset=feat_cols)

    print(f"  Dataset: {len(df)} rows  x  {len(feat_cols)} features")
    print(f"  Target ({TARGET}): mean={df[TARGET].mean():.5f}  "
          f"std={df[TARGET].std():.4f}  "
          f"range=[{df[TARGET].min():.3f}, {df[TARGET].max():.3f}]")
    return df, feat_cols


# ─── Step 3: Walk-forward OOS predictions ─────────────────────────────────────

def _fit_xgb(X_tr, y_tr, X_te, y_te) -> np.ndarray:
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return m.predict(X_te), m


def _fit_ridge(X_tr, y_tr, X_te) -> np.ndarray:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    m = Ridge(alpha=RIDGE_ALPHA)
    m.fit(X_tr_s, y_tr)
    return m.predict(X_te_s)


def walk_forward_oos(df: pd.DataFrame, feat_cols: list[str],
                      model_type: str = "xgb",
                      verbose: bool = True) -> np.ndarray:
    """
    Walk-forward OOS with regime routing.

    Fold k: train on bars 0..tr_end, test on tr_end..te_end.
    Global model trained first; per-regime models override predictions
    for bars of that regime in the test window.

    Returns np.array of OOS predictions aligned to df.index.
    NaN for the first fold's training window (no prediction available).
    """
    n    = len(df)
    fold = n // (N_FOLDS + 1)
    oos  = np.full(n, np.nan)

    regimes = sorted(df["regime_name"].dropna().unique()) \
              if "regime_name" in df.columns else []

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr = df[feat_cols].fillna(0).values[:tr_end]
        y_tr = df[TARGET].values[:tr_end]
        X_te = df[feat_cols].fillna(0).values[tr_end:te_end]
        y_te = df[TARGET].values[tr_end:te_end]

        # ── Global model ──────────────────────────────────────────────────────
        if model_type == "xgb":
            fold_pred, _ = _fit_xgb(X_tr, y_tr, X_te, y_te)
        else:
            fold_pred = _fit_ridge(X_tr, y_tr, X_te)

        ic_g, _ = spearmanr(y_te, fold_pred)

        # ── Regime override ───────────────────────────────────────────────────
        regime_ics = {}
        df_tr = df.iloc[:tr_end]
        df_te = df.iloc[tr_end:te_end]

        for regime in regimes:
            m_tr = (df_tr["regime_name"] == regime).values
            m_te = (df_te["regime_name"] == regime).values
            if m_tr.sum() < MIN_REGIME_TRAIN or m_te.sum() < MIN_REGIME_TEST:
                continue
            if model_type == "xgb":
                reg_pred, _ = _fit_xgb(X_tr[m_tr], y_tr[m_tr],
                                        X_te[m_te], y_te[m_te])
            else:
                reg_pred = _fit_ridge(X_tr[m_tr], y_tr[m_tr], X_te[m_te])
            fold_pred[m_te] = reg_pred
            ic_r, _ = spearmanr(y_te[m_te], reg_pred)
            regime_ics[regime] = round(ic_r, 4)

        oos[tr_end:te_end] = fold_pred

        if verbose:
            ic_oos, _ = spearmanr(y_te, fold_pred)
            r_str = "  ".join(f"{r}:{v:+.3f}" for r, v in regime_ics.items())
            print(f"  fold {k}/{N_FOLDS}  IC_global={ic_g:+.4f}  "
                  f"IC_oos={ic_oos:+.4f}  n={len(y_te)}  [{r_str}]")

    return oos


# ─── Step 4: IC / ICIR evaluation ─────────────────────────────────────────────

def compute_ic_analysis(oos_pred: np.ndarray,
                         df: pd.DataFrame) -> dict:
    """
    Per-fold IC and ICIR.  Also computes per-regime and global.
    """
    n    = len(df)
    fold = n // (N_FOLDS + 1)
    y    = df[TARGET].values

    fold_ics = []
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        p = oos_pred[tr_end:te_end]
        a = y[tr_end:te_end]
        mask = ~np.isnan(p) & ~np.isnan(a)
        if mask.sum() < 20:
            continue
        ic, _ = spearmanr(a[mask], p[mask])
        fold_ics.append(ic)

    fold_ics = np.array(fold_ics)
    global_ic   = fold_ics.mean()    if len(fold_ics) > 0 else np.nan
    global_icir = (fold_ics.mean() / fold_ics.std()
                   if fold_ics.std() > 0 else 0.0)

    # Per-regime IC
    regime_results = {}
    if "regime_name" in df.columns:
        for regime in sorted(df["regime_name"].dropna().unique()):
            mask_r = (df["regime_name"] == regime).values
            p_r = oos_pred[mask_r]
            a_r = y[mask_r]
            valid = ~np.isnan(p_r) & ~np.isnan(a_r)
            if valid.sum() < 20:
                continue
            ic_r, _ = spearmanr(a_r[valid], p_r[valid])

            # ICIR per regime (using fold structure within regime)
            regime_ics = []
            for k in range(1, N_FOLDS + 1):
                tr_end = fold * k
                te_end = min(fold * (k + 1), n)
                r_te = (df.iloc[tr_end:te_end]["regime_name"] == regime).values
                p_te = oos_pred[tr_end:te_end][r_te]
                a_te = y[tr_end:te_end][r_te]
                m = ~np.isnan(p_te) & ~np.isnan(a_te)
                if m.sum() < 10:
                    continue
                ic_f, _ = spearmanr(a_te[m], p_te[m])
                regime_ics.append(ic_f)

            regime_ics = np.array(regime_ics)
            regime_results[regime] = {
                "mean_ic":  round(float(regime_ics.mean()), 4) if len(regime_ics) else np.nan,
                "icir":     round(float(regime_ics.mean() / regime_ics.std()), 4)
                            if len(regime_ics) > 1 and regime_ics.std() > 0 else np.nan,
                "n_bars":   int(mask_r.sum()),
                "fold_ics": [round(v, 4) for v in regime_ics],
            }

    return {
        "global_mean_ic":  round(float(global_ic),   4),
        "global_icir":     round(float(global_icir), 4),
        "fold_ics":        [round(v, 4) for v in fold_ics],
        "per_regime":      regime_results,
    }


def compute_rolling_ic(oos_pred: np.ndarray, y: np.ndarray,
                        window: int = IC_WINDOW) -> pd.Series:
    """Rolling Spearman IC using expanding then rolling window."""
    n    = len(oos_pred)
    ics  = np.full(n, np.nan)
    for i in range(window, n):
        start = max(0, i - window)
        p = oos_pred[start:i]
        a = y[start:i]
        mask = ~np.isnan(p) & ~np.isnan(a)
        if mask.sum() < 20:
            continue
        ic, _ = spearmanr(a[mask], p[mask])
        ics[i] = ic
    return pd.Series(ics)


# ─── Step 5: Simple ranking backtest ──────────────────────────────────────────

def backtest_ranking(df: pd.DataFrame,
                      oos_pred: np.ndarray,
                      pct_threshold: float = 0.10,
                      fee_rt: float = FEE_RT) -> pd.DataFrame:
    """
    Non-overlapping 4h rebalancing with ranking signal.

    At each rebalance bar (every HOLDING_BARS bars):
      - Compute expanding percentile of past predictions (no lookahead)
      - Long  if pred > expanding_upper (top pct_threshold)
      - Short if pred < expanding_lower (bottom pct_threshold)
      - Flat otherwise

    P&L = position x y_return_4h - fee_rt (when position != 0)
    """
    y_ret  = df[TARGET].values
    index  = df.index
    regime = df["regime_name"].values if "regime_name" in df.columns else \
             np.full(len(df), "unknown")

    # Expanding percentile threshold (past-only, no lookahead)
    pred_s = pd.Series(oos_pred)
    upper  = pred_s.expanding(min_periods=50).quantile(1 - pct_threshold)
    lower  = pred_s.expanding(min_periods=50).quantile(pct_threshold)

    # Non-overlapping rebalance indices
    first_valid = int(np.argmax(~np.isnan(oos_pred)))   # first OOS bar
    start       = first_valid + HOLDING_BARS             # give one HOLDING_BARS gap
    rebalance   = range(start, len(df) - HOLDING_BARS, HOLDING_BARS)

    records = []
    for t in rebalance:
        p = oos_pred[t]
        u = upper.iloc[t]
        l = lower.iloc[t]
        y = y_ret[t]

        if np.isnan(p) or np.isnan(u) or np.isnan(y):
            continue

        pos = 0
        if   p > u: pos =  1   # long
        elif p < l: pos = -1   # short
        else:
            continue            # flat: no trade, no cost

        gross = pos * y
        net   = gross - fee_rt

        records.append({
            "entry_dt":     index[t],
            "position":     pos,
            "pred_return":  round(p, 6),
            "actual_return":round(y, 6),
            "upper_thr":    round(u, 6),
            "lower_thr":    round(l, 6),
            "gross_ret":    round(gross, 6),
            "net_ret":      round(net,   6),
            "regime":       regime[t],
        })

    trades = pd.DataFrame(records).set_index("entry_dt")
    trades.index = pd.to_datetime(trades.index)
    return trades


# ─── Step 6: Performance metrics ──────────────────────────────────────────────

def daily_sharpe(trades: pd.DataFrame) -> float:
    """
    Assign trade net_ret to its entry date.
    Fill 0 for days with no trades.
    Sharpe = mean(daily) / std(daily) * sqrt(252).
    """
    if trades.empty:
        return 0.0
    daily = trades["net_ret"].groupby(trades.index.normalize()).sum()
    full  = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0.0)
    mu, sg = daily.mean(), daily.std()
    return mu / sg * np.sqrt(252) if sg > 0 else 0.0


def calc_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if trades.empty:
        return {"label": label, "n": 0}

    r  = trades["net_ret"].dropna()
    g  = trades["gross_ret"].dropna()
    eq = np.cumprod(1 + r.values)
    peak = np.maximum.accumulate(eq)
    mdd  = ((eq - peak) / peak).min()
    wr   = (r > 0).mean()
    sh   = daily_sharpe(trades)

    longs  = trades[trades["position"] ==  1]["net_ret"]
    shorts = trades[trades["position"] == -1]["net_ret"]

    m = {
        "label":         label,
        "n_trades":      len(r),
        "total_return":  round(eq[-1] - 1, 4),
        "mean_gross":    round(g.mean(),   5),
        "mean_net":      round(r.mean(),   5),
        "sharpe_daily":  round(sh,         3),
        "max_dd":        round(mdd,        4),
        "win_rate":      round(wr,         4),
        "avg_win":       round(r[r>0].mean(), 5) if (r>0).any() else np.nan,
        "avg_loss":      round(r[r<0].mean(), 5) if (r<0).any() else np.nan,
        "n_long":        len(longs),
        "n_short":       len(shorts),
        "long_mean":     round(longs.mean(),  5) if len(longs)  > 0 else np.nan,
        "short_mean":    round(shorts.mean(), 5) if len(shorts) > 0 else np.nan,
    }

    tag = f"[{label:35s}]" if label else " " * 37
    print(f"  {tag} n={m['n_trades']:3d}  "
          f"Ret={m['total_return']:+.1%}  "
          f"Sharpe(daily)={m['sharpe_daily']:+.3f}  "
          f"MDD={m['max_dd']:.1%}  WR={m['win_rate']:.1%}  "
          f"L={m['n_long']} S={m['n_short']}")
    return m


# ─── Step 7: Threshold sweep / regime breakdown / calibration ─────────────────

def threshold_sweep(df: pd.DataFrame, oos_pred: np.ndarray,
                     pcts: tuple = (0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Run backtest at multiple signal thresholds."""
    rows = []
    for pct in pcts:
        t = backtest_ranking(df, oos_pred, pct_threshold=pct)
        m = calc_metrics(t, label=f"top/bot {int(pct*100)}%")
        if m.get("n_trades", 0) > 0:
            m["threshold"] = f"{int(pct*100)}%"
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("threshold")[
        ["n_trades", "mean_gross", "mean_net", "sharpe_daily",
         "max_dd", "win_rate", "n_long", "n_short"]
    ]


def regime_breakdown(df: pd.DataFrame, oos_pred: np.ndarray,
                      pct_threshold: float = 0.10) -> pd.DataFrame:
    """Run backtest per regime (only trade bars in that regime)."""
    rows = []
    for regime in sorted(df["regime_name"].dropna().unique()):
        # Mask predictions to this regime
        masked = oos_pred.copy()
        is_regime = (df["regime_name"] == regime).values
        masked[~is_regime] = np.nan

        t = backtest_ranking(df, masked, pct_threshold=pct_threshold)
        m = calc_metrics(t, label=regime)
        if m.get("n_trades", 0) > 0:
            m["regime"] = regime
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("regime")[
        ["n_trades", "mean_gross", "mean_net", "sharpe_daily",
         "max_dd", "win_rate"]
    ]


def calibration_analysis(oos_pred: np.ndarray,
                           df: pd.DataFrame,
                           n_bins: int = 10) -> pd.DataFrame:
    """
    Divide OOS predictions into deciles.
    Compute mean actual 4h return per decile.
    Monotone positive trend = model has calibrated directional signal.
    """
    valid = ~np.isnan(oos_pred)
    pred_v = oos_pred[valid]
    y_v    = df[TARGET].values[valid]

    combined = pd.DataFrame({"pred": pred_v, "actual": y_v})
    combined["bin"] = pd.qcut(combined["pred"], n_bins, labels=False)

    calib = combined.groupby("bin")["actual"].agg(["mean", "std", "count"])
    calib.columns = ["mean_actual", "std_actual", "n"]
    calib["mean_pred"] = combined.groupby("bin")["pred"].mean()
    calib["se"] = calib["std_actual"] / np.sqrt(calib["n"])

    return calib


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_ic_timeseries(rolling_ic: pd.Series, df: pd.DataFrame,
                        label: str, save: bool = False):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index[rolling_ic.notna()],
            rolling_ic.dropna().values,
            lw=1, color="#1976d2", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.05,  color="green", lw=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-0.05, color="red",   lw=0.5, linestyle="--", alpha=0.5)
    ax.fill_between(df.index[rolling_ic.notna()],
                    rolling_ic.dropna().values, 0,
                    where=rolling_ic.dropna().values > 0, alpha=0.15, color="green")
    ax.fill_between(df.index[rolling_ic.notna()],
                    rolling_ic.dropna().values, 0,
                    where=rolling_ic.dropna().values < 0, alpha=0.15, color="red")
    ax.set_ylabel("Rolling IC (Spearman)")
    ax.set_title(f"Rolling {IC_WINDOW}-bar IC — {label}")
    plt.tight_layout()
    _save(fig, f"return4h_ic_{label.replace(' ','_')}.png", save)


def plot_equity_and_regime(trades: pd.DataFrame, label: str,
                            save: bool = False):
    if trades.empty:
        return
    r  = trades["net_ret"].dropna()
    eq = np.cumprod(1 + r.values)
    dt = trades.index[trades["net_ret"].notna()]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Equity
    ax = axes[0]
    ax.plot(dt, eq, lw=1.5, color="#1976d2")
    ax.axhline(1.0, color="black", lw=0.5)
    peak = np.maximum.accumulate(eq)
    ax.fill_between(dt, eq, peak, where=eq < peak,
                    alpha=0.25, color="#ef5350", label="Drawdown")
    ax.set_ylabel("Equity")
    ax.set_title(f"Equity Curve — {label}")
    ax.legend(fontsize=8)

    # Per-trade P&L coloured by regime
    ax = axes[1]
    rc = {"TRENDING_BULL": "#26a69a", "TRENDING_BEAR": "#ef5350",
          "CHOPPY": "#ff9800", "unknown": "#9e9e9e"}
    for reg, grp in trades.groupby("regime"):
        ax.scatter(grp.index, grp["net_ret"], s=8, alpha=0.6,
                   color=rc.get(reg, "#9e9e9e"), label=reg)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Net return per trade")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Per-trade return by regime")

    plt.tight_layout()
    _save(fig, f"return4h_equity_{label.replace(' ','_')}.png", save)


def plot_calibration(calib: pd.DataFrame, label: str, save: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean actual return per decile
    ax = axes[0]
    colors = ["#ef5350" if v < 0 else "#26a69a" for v in calib["mean_actual"]]
    ax.bar(calib.index, calib["mean_actual"] * 100, color=colors, alpha=0.85)
    ax.errorbar(calib.index, calib["mean_actual"] * 100,
                yerr=calib["se"] * 100, fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Prediction decile (0=lowest, 9=highest)")
    ax.set_ylabel("Mean actual 4h return (%)")
    ax.set_title(f"Calibration — {label}")
    ax.set_xticks(calib.index)

    # Pred vs actual scatter (sample 2000 bars)
    ax = axes[1]
    valid_mask = ~np.isnan(calib["mean_pred"].values)
    ax.scatter(calib["mean_pred"][valid_mask] * 100,
               calib["mean_actual"][valid_mask] * 100,
               s=60, alpha=0.8, color="#1976d2")
    ax.axhline(0, color="black", lw=0.3)
    ax.axvline(0, color="black", lw=0.3)
    mn = min(calib["mean_pred"].min(), calib["mean_actual"].min()) * 100
    mx = max(calib["mean_pred"].max(), calib["mean_actual"].max()) * 100
    ax.plot([mn, mx], [mn, mx], "r--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted return (%)")
    ax.set_ylabel("Mean actual return (%)")
    ax.set_title("Predicted vs Actual (decile means)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, f"return4h_calibration_{label.replace(' ','_')}.png", save)


def plot_comparison(xgb_trades, linear_trades,
                    xgb_ic, linear_ic, save: bool = False):
    """Side-by-side summary: XGBoost vs Linear."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # IC comparison
    ax = axes[0]
    regimes = ["global"] + list(xgb_ic["per_regime"].keys())
    xgb_ics = [xgb_ic["global_mean_ic"]] + \
               [xgb_ic["per_regime"].get(r, {}).get("mean_ic", np.nan) for r in regimes[1:]]
    lin_ics = [linear_ic["global_mean_ic"]] + \
               [linear_ic["per_regime"].get(r, {}).get("mean_ic", np.nan) for r in regimes[1:]]
    x = np.arange(len(regimes))
    ax.bar(x - 0.2, xgb_ics, 0.35, label="XGBoost", color="#1976d2", alpha=0.8)
    ax.bar(x + 0.2, lin_ics,  0.35, label="Linear",  color="#ff9800", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_title("Mean IC: XGBoost vs Linear"); ax.legend(fontsize=8)
    ax.set_ylabel("Mean IC (Spearman)")

    # Equity curves
    ax = axes[1]
    for trades, label, color in [
        (xgb_trades,    "XGBoost", "#1976d2"),
        (linear_trades, "Linear",  "#ff9800"),
    ]:
        if not trades.empty:
            r  = trades["net_ret"].dropna()
            eq = np.cumprod(1 + r.values)
            ax.plot(range(len(eq)), eq, lw=1.5, label=label, color=color, alpha=0.9)
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Equity")
    ax.set_title("Equity: XGBoost vs Linear"); ax.legend(fontsize=8)

    # Sharpe / MDD comparison
    ax = axes[2]
    labels = ["Sharpe(daily)", "Win Rate", "|Max DD|"]
    xgb_m = calc_metrics(xgb_trades, label="xgb_quiet") if not xgb_trades.empty else {}
    lin_m = calc_metrics(linear_trades, label="lin_quiet") if not linear_trades.empty else {}
    vals_x = [xgb_m.get("sharpe_daily", 0), xgb_m.get("win_rate", 0),
               abs(xgb_m.get("max_dd", 0))]
    vals_l = [lin_m.get("sharpe_daily", 0), lin_m.get("win_rate", 0),
               abs(lin_m.get("max_dd", 0))]
    x2 = np.arange(3)
    ax.bar(x2 - 0.2, vals_x, 0.35, label="XGBoost", color="#1976d2", alpha=0.8)
    ax.bar(x2 + 0.2, vals_l, 0.35, label="Linear",  color="#ff9800", alpha=0.8)
    ax.set_xticks(x2); ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Key Metrics Comparison"); ax.legend(fontsize=8)

    plt.suptitle("XGBoost vs Linear Regression — 4h Return Prediction", fontsize=12)
    plt.tight_layout()
    _save(fig, "return4h_model_comparison.png", save)


def _save(fig, name, save):
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / name
        fig.savefig(p, bbox_inches="tight")
        print(f"  [saved] {p.name}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save",  action="store_true")
    ap.add_argument("--model", default="both",
                    choices=["xgb", "linear", "both"])
    args = ap.parse_args()

    # ── Load ────────────────────────────────────────────────────────────────
    print("="*60)
    print("STEP 1-2: Load and Prepare")
    print("="*60)
    df, feat_cols = load_and_prepare()

    print(f"\nFeatures used ({len(feat_cols)}):")
    print("  " + ", ".join(feat_cols))

    if "regime_name" in df.columns:
        print(f"\nRegime distribution:")
        print(df["regime_name"].value_counts().to_string())

    # ── OOS predictions ─────────────────────────────────────────────────────
    oos_preds = {}

    if args.model in ("xgb", "both"):
        print("\n" + "="*60)
        print("STEP 3: Walk-Forward OOS — XGBoost")
        print("="*60)
        oos_preds["xgb"] = walk_forward_oos(df, feat_cols, model_type="xgb")

    if args.model in ("linear", "both"):
        print("\n" + "="*60)
        print("STEP 3: Walk-Forward OOS — Linear (Ridge)")
        print("="*60)
        oos_preds["linear"] = walk_forward_oos(df, feat_cols, model_type="linear")

    # ── IC evaluation ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: IC / ICIR Evaluation")
    print("="*60)
    ic_results = {}
    for mname, pred in oos_preds.items():
        print(f"\n  [{mname}]")
        ic = compute_ic_analysis(pred, df)
        ic_results[mname] = ic

        print(f"  Global: mean_IC={ic['global_mean_ic']:+.4f}  "
              f"ICIR={ic['global_icir']:+.4f}  "
              f"fold_ICs={ic['fold_ics']}")
        print("  Per regime:")
        for regime, rv in ic["per_regime"].items():
            print(f"    {regime:20s}  mean_IC={rv['mean_ic']:+.4f}  "
                  f"ICIR={rv.get('icir', 'nan'):>7}  "
                  f"n={rv['n_bars']}  folds={rv['fold_ics']}")

    # ── Rolling IC time series ───────────────────────────────────────────────
    rolling_ics = {}
    for mname, pred in oos_preds.items():
        rolling_ics[mname] = compute_rolling_ic(pred, df[TARGET].values)

    # ── Calibration ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7b: Calibration Analysis")
    print("="*60)
    calib_results = {}
    for mname, pred in oos_preds.items():
        calib = calibration_analysis(pred, df)
        calib_results[mname] = calib
        print(f"\n  [{mname}] Decile mean actual returns (%):")
        print(f"  " + "  ".join(f"D{int(i)}:{v*100:+.4f}"
              for i, v in calib["mean_actual"].items()))
        # Monotonicity check
        diffs = np.diff(calib["mean_actual"].values)
        pct_pos = (diffs > 0).mean()
        print(f"  Monotone up: {pct_pos:.0%} of adjacent deciles  "
              f"({'GOOD' if pct_pos > 0.6 else 'WEAK'})")

    # ── Backtest ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5-6: Backtest (non-overlapping 4h, top/bot 10%)")
    print("="*60)
    backtest_trades = {}
    for mname, pred in oos_preds.items():
        print(f"\n  [{mname}]")
        t = backtest_ranking(df, pred, pct_threshold=0.10)
        backtest_trades[mname] = t
        calc_metrics(t, label=f"{mname} base")

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7a: Threshold Sweep")
    print("="*60)
    for mname, pred in oos_preds.items():
        print(f"\n  [{mname}]")
        sweep = threshold_sweep(df, pred)
        if not sweep.empty:
            print(sweep.round(5).to_string())

    # ── Regime breakdown ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7c: Regime Breakdown (10% threshold)")
    print("="*60)
    for mname, pred in oos_preds.items():
        print(f"\n  [{mname}]")
        rbd = regime_breakdown(df, pred, pct_threshold=0.10)
        if not rbd.empty:
            print(rbd.round(5).to_string())

    # ── Final OOS split test (last 33%) ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7d: OOS Hold-Out Test (last 33%)")
    print("="*60)
    n_split = int(len(df) * 0.67)
    split_dt = df.index[n_split]
    print(f"  Split: {split_dt.date()}  "
          f"Train={n_split} bars  Test={len(df)-n_split} bars")

    for mname, pred in oos_preds.items():
        print(f"\n  [{mname}]")
        # IS
        pred_is = pred.copy(); pred_is[n_split:] = np.nan
        t_is = backtest_ranking(df, pred_is, pct_threshold=0.10)
        calc_metrics(t_is, label=f"{mname} in-sample")

        # OOS
        pred_os = pred.copy(); pred_os[:n_split] = np.nan
        t_os = backtest_ranking(df, pred_os, pct_threshold=0.10)
        calc_metrics(t_os, label=f"{mname} out-of-sample")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nPlotting...")
    for mname, pred in oos_preds.items():
        plot_ic_timeseries(rolling_ics[mname], df, label=mname, save=args.save)
        plot_equity_and_regime(backtest_trades[mname], label=mname, save=args.save)
        plot_calibration(calib_results[mname], label=mname, save=args.save)

    if args.model == "both" and "xgb" in oos_preds and "linear" in oos_preds:
        plot_comparison(
            backtest_trades["xgb"], backtest_trades["linear"],
            ic_results["xgb"],     ic_results["linear"],
            save=args.save,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
