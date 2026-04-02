"""
Path-Based TP/SL Backtest — Regime-Conditional Order Flow Model.

Why path-based:
  The model predicts 4h HIGH/LOW range asymmetry (up_move_vol_adj / down_move_vol_adj),
  not close-to-close return. A TP/SL simulation that uses future_high_4h / future_low_4h
  correctly evaluates range-prediction signals.

Signal:
  regime == TRENDING_BEAR AND pred_down_move in top N%
  → open SHORT at bar close

Vol-adj → price conversion:
  pred values are in ATR-normalised units (vol_adj = price_move / atr).
  atr = (high - low).rolling(14).mean()   [past-only, no lookahead]
  pred_pct = pred_vol_adj * atr / close   [fraction of close]

TP / SL for SHORT:
  tp_price = close * (1 - tp_mult * pred_down_pct)
  sl_price = close * (1 + sl_mult * pred_up_pct)
  Minimum distance enforced: max(pred-based, 0.5 * atr_pct)

Exit logic (path-based, vectorized):
  future_low_4h  <= tp_price → TP hit
  future_high_4h >= sl_price → SL hit
  both hit within window     → TP assumed first (conservative)
  neither hit                → close at close[t + HOLDING_BARS]

Costs (round-trip):
  mode = "taker"        → 0.07% × 2 = 0.14%
  mode = "maker_taker"  → (0.02 + 0.07)% × 2 = 0.18%  [maker entry, taker exit]
  mode = "maker"        → 0.02% × 2 = 0.04%

Bonus sweeps:
  - threshold_sweep: top 10% / 20% / 30%
  - tpsl_sweep: tp_mult × sl_mult in [0.3, 0.5, 0.7]²

Usage:
    python research/backtest_pathbased.py
    python research/backtest_pathbased.py --save
    python research/backtest_pathbased.py --mode maker --save
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────

PARQUET      = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
OUT_DIR      = Path(__file__).parent / "eda_charts"
HOLDING_BARS = 16           # 4h = 16 × 15m bars
N_FOLDS      = 5

FEES = {
    "taker":       (0.0007, 0.0007),   # (entry, exit) per side
    "maker_taker": (0.0002, 0.0007),
    "maker":       (0.0002, 0.0002),
}

ALL_TARGETS = ["up_move_vol_adj", "down_move_vol_adj"]

EXCLUDE_BASE = {
    "ts_open",
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    "oi", "oi_delta", "oi_accel", "oi_divergence",
    "cvd_x_oi_delta", "cvd_oi_ratio",
    # keep: open, high, low, close (needed for simulation, not features)
    "vol_4h_proxy",
    "regime", "regime_name",
}

XGB_PARAMS = dict(
    n_estimators          = 500,
    max_depth             = 4,
    learning_rate         = 0.03,
    subsample             = 0.8,
    colsample_bytree      = 0.7,
    min_child_weight      = 15,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    random_state          = 42,
    verbosity             = 0,
    early_stopping_rounds = 30,
)

MIN_REGIME_TR = 200
MIN_REGIME_TE = 5


# ─── Load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, list[str]]:
    """
    Load parquet. Keep OHLC and future labels for simulation.
    feat_cols excludes OHLC, labels, regime, OI, etc.
    """
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()

    df = df.dropna(subset=ALL_TARGETS)

    # Feature set: everything not in EXCLUDE_BASE, AND not OHLC (OHLC used for sim only)
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE_BASE
                 and c not in {"open", "high", "low", "close"}]

    df[feat_cols] = df[feat_cols].ffill()

    nan_rate  = df[feat_cols].isnull().mean()
    drop_cols = list(nan_rate[nan_rate > 0.05].index)
    if drop_cols:
        feat_cols = [c for c in feat_cols if c not in drop_cols]

    df = df.dropna(subset=feat_cols)
    print(f"  {len(df)} rows × {len(feat_cols)} features")
    return df, feat_cols


# ─── OOS prediction generation (regime-routed walk-forward) ────────────────────

def _generate_oos_routed(df: pd.DataFrame, feat_cols: list[str],
                          target: str) -> np.ndarray:
    """
    Walk-forward OOS with regime routing (no lookahead).
    Global XGBoost is trained first, then per-regime models override
    predictions for bars belonging to that regime's test window.
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

        df_tr = df.iloc[:tr_end]
        df_te = df.iloc[tr_end:te_end]
        X_tr  = df_tr[feat_cols].fillna(0).values
        y_tr  = df_tr[target].values
        X_te  = df_te[feat_cols].fillna(0).values
        y_te  = df_te[target].values

        # Global model (baseline for all test bars)
        gm = xgb.XGBRegressor(**XGB_PARAMS)
        gm.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        fold_pred = gm.predict(X_te)

        # Regime-specific override
        for regime in regimes:
            m_tr = (df_tr["regime_name"] == regime).values
            m_te = (df_te["regime_name"] == regime).values
            if m_tr.sum() < MIN_REGIME_TR or m_te.sum() < MIN_REGIME_TE:
                continue
            rm = xgb.XGBRegressor(**XGB_PARAMS)
            rm.fit(
                X_tr[m_tr], y_tr[m_tr],
                eval_set=[(X_te[m_te], y_te[m_te])],
                verbose=False,
            )
            fold_pred[m_te] = rm.predict(X_te[m_te])

        oos[tr_end:te_end] = fold_pred

    return oos


def generate_predictions(df: pd.DataFrame,
                          feat_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """
    Generate OOS pred_down and pred_up as pandas Series aligned to df.index.
    """
    print("  Generating OOS predictions: down_move_vol_adj ...")
    raw_down = _generate_oos_routed(df, feat_cols, "down_move_vol_adj")
    print("  Generating OOS predictions: up_move_vol_adj ...")
    raw_up   = _generate_oos_routed(df, feat_cols, "up_move_vol_adj")

    pred_down = pd.Series(raw_down, index=df.index, name="pred_down")
    pred_up   = pd.Series(raw_up,   index=df.index, name="pred_up")

    # Clip negative predictions to 0 (magnitude should be non-negative)
    pred_down = pred_down.clip(lower=0)
    pred_up   = pred_up.clip(lower=0)

    return pred_down, pred_up


# ─── ATR (past-only, no lookahead) ─────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    14-bar rolling mean of (high - low).
    Fully past-looking.  Matches the vol_4h_proxy scale used in label generation.
    """
    return (df["high"] - df["low"]).rolling(window, min_periods=5).mean()


# ─── Core vectorized simulation ────────────────────────────────────────────────

def simulate_short_trades(
    df: pd.DataFrame,
    pred_down: pd.Series,
    pred_up: pd.Series,
    atr: pd.Series,
    top_pct: float = 0.20,
    tp_mult: float = 0.50,
    sl_mult: float = 0.50,
    fee_rt: float  = 0.0014,
) -> pd.DataFrame:
    """
    Fully vectorised path-based simulation for SHORT trades.

    Entry filter:
        regime == TRENDING_BEAR
        AND pred_down in top `top_pct` of all TRENDING_BEAR bars

    TP/SL (SHORT):
        pred_down_pct = pred_down * atr / close   [vol_adj → price fraction]
        tp_price = close * (1 - max(tp_mult * pred_down_pct, 0.5 * atr_pct))
        sl_price = close * (1 + max(sl_mult * pred_up_pct,   0.5 * atr_pct))

    Exit (vectorised):
        TP hit  → future_low_4h  <= tp_price
        SL hit  → future_high_4h >= sl_price
        Both    → TP assumed first (conservative)
        Neither → close at close[t + HOLDING_BARS]

    Returns:
        trades DataFrame with one row per trade.
    """
    close = df["close"]
    atr_p = atr.reindex(df.index)
    atr_pct = atr_p / close   # ATR as fraction of close

    # ── Vol-adj → price fraction ──────────────────────────────────────────────
    pred_down_pct = (pred_down * atr_pct).reindex(df.index)
    pred_up_pct   = (pred_up   * atr_pct).reindex(df.index)

    # ── Entry filter ──────────────────────────────────────────────────────────
    bear_mask  = df["regime_name"] == "TRENDING_BEAR"
    bear_down  = pred_down[bear_mask]

    threshold  = bear_down.quantile(1 - top_pct)   # e.g. 80th pct for top 20%
    entry_mask = bear_mask & (pred_down >= threshold)

    # Drop rows where ATR or prediction is NaN, or no forward close available
    fwd_close = close.shift(-HOLDING_BARS)
    valid      = (
        entry_mask
        & atr_p.notna()
        & pred_down.notna()
        & fwd_close.notna()
        & df["future_high_4h"].notna()
        & df["future_low_4h"].notna()
    )
    t = df[valid].copy()

    if t.empty:
        return pd.DataFrame()

    # ── TP / SL price levels ──────────────────────────────────────────────────
    # Minimum distance: 0.5 × atr as fraction
    atr_min = atr_pct[valid] * 0.5

    tp_dist = pd.concat([tp_mult * pred_down_pct[valid],
                         atr_min], axis=1).max(axis=1)
    sl_dist = pd.concat([sl_mult * pred_up_pct[valid],
                         atr_min], axis=1).max(axis=1)

    t["tp_price"]   = t["close"] * (1 - tp_dist)
    t["sl_price"]   = t["close"] * (1 + sl_dist)
    t["close_fwd"]  = fwd_close[valid]
    t["pred_down"]  = pred_down[valid]
    t["pred_up"]    = pred_up[valid]
    t["tp_dist_pct"]= tp_dist
    t["sl_dist_pct"]= sl_dist

    # ── Path-based exit determination ─────────────────────────────────────────
    tp_hit = t["future_low_4h"]  <= t["tp_price"]    # short TP: price fell to TP
    sl_hit = t["future_high_4h"] >= t["sl_price"]    # short SL: price rose to SL

    t["tp_hit"] = tp_hit
    t["sl_hit"] = sl_hit

    tp_only  = tp_hit & ~sl_hit
    sl_only  = sl_hit & ~tp_hit
    both_hit = tp_hit &  sl_hit
    neither  = ~tp_hit & ~sl_hit

    t["exit_reason"] = np.select(
        [tp_only, sl_only, both_hit, neither],
        ["TP",    "SL",    "TP(both)", "time"],
        default="unknown",
    )

    # ── Gross return (SHORT: gain when price falls) ───────────────────────────
    # gross_ret = (entry - exit) / entry
    gross = pd.Series(np.nan, index=t.index)
    gross[tp_only]   = (t["close"] - t["tp_price"])[tp_only]   / t["close"][tp_only]
    gross[sl_only]   = (t["close"] - t["sl_price"])[sl_only]   / t["close"][sl_only]
    gross[both_hit]  = (t["close"] - t["tp_price"])[both_hit]  / t["close"][both_hit]
    gross[neither]   = (t["close"] - t["close_fwd"])[neither]  / t["close"][neither]
    t["gross_ret"] = gross

    # ── Net return ────────────────────────────────────────────────────────────
    t["fee_rt"]    = fee_rt
    t["net_ret"]   = t["gross_ret"] - fee_rt

    # Keep useful metadata
    t["regime"] = df["regime_name"][valid]

    cols = [
        "close", "tp_price", "sl_price", "close_fwd",
        "future_high_4h", "future_low_4h",
        "pred_down", "pred_up",
        "tp_dist_pct", "sl_dist_pct",
        "tp_hit", "sl_hit", "exit_reason",
        "gross_ret", "fee_rt", "net_ret",
        "regime",
    ]
    return t[[c for c in cols if c in t.columns]]


# ─── Performance metrics ───────────────────────────────────────────────────────

def calc_metrics(trades: pd.DataFrame,
                 label: str = "",
                 periods_per_year: int = 365 * 6) -> dict:
    """
    Compute and print performance metrics.
    periods_per_year uses 4h as base (non-overlapping, same as HOLDING_BARS).
    """
    if trades.empty or trades["net_ret"].isna().all():
        print(f"  [{label}] No trades.")
        return {}

    r = trades["net_ret"].dropna()
    g = trades["gross_ret"].dropna()
    n = len(r)

    equity    = np.cumprod(1 + r.values)
    total_ret = equity[-1] - 1
    mean_r    = r.mean()
    std_r     = r.std()
    sharpe    = mean_r / std_r * np.sqrt(periods_per_year) if std_r > 0 else 0.0

    peak  = np.maximum.accumulate(equity)
    dd    = (equity - peak) / peak
    mdd   = dd.min()

    wins  = r[r > 0]
    losses= r[r < 0]
    wr    = (r > 0).mean()

    # Exit breakdown
    exit_counts = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades else {}

    result = {
        "n_trades":      n,
        "total_return":  round(total_ret, 4),
        "mean_gross":    round(g.mean(), 5),
        "mean_net":      round(mean_r,   5),
        "sharpe":        round(sharpe,   3),
        "max_drawdown":  round(mdd,      4),
        "win_rate":      round(wr,       4),
        "avg_win":       round(wins.mean(),   5) if len(wins)   > 0 else np.nan,
        "avg_loss":      round(losses.mean(), 5) if len(losses) > 0 else np.nan,
        "n_tp":          exit_counts.get("TP", 0) + exit_counts.get("TP(both)", 0),
        "n_sl":          exit_counts.get("SL", 0),
        "n_time":        exit_counts.get("time", 0),
    }

    tag = f"[{label}] " if label else ""
    print(f"  {tag}n={n:3d}  Return={total_ret:+.1%}  Sharpe={sharpe:+.3f}  "
          f"MDD={mdd:.1%}  WR={wr:.1%}  "
          f"AvgW={result['avg_win']:+.4%}  AvgL={result['avg_loss']:+.4%}  "
          f"TP={result['n_tp']} SL={result['n_sl']} Time={result['n_time']}")

    return result


# ─── Threshold sweep ───────────────────────────────────────────────────────────

def threshold_sweep(
    df: pd.DataFrame,
    pred_down: pd.Series,
    pred_up: pd.Series,
    atr: pd.Series,
    tp_mult: float = 0.5,
    sl_mult: float = 0.5,
    fee_rt:  float = 0.0014,
    top_pcts: tuple = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """
    Run the simulation at multiple top-percentile thresholds.
    Returns a summary DataFrame.
    """
    rows = []
    for pct in top_pcts:
        trades = simulate_short_trades(
            df, pred_down, pred_up, atr,
            top_pct=pct, tp_mult=tp_mult, sl_mult=sl_mult, fee_rt=fee_rt,
        )
        m = calc_metrics(trades, label=f"top {int(pct*100)}%")
        if m:
            m["top_pct"] = f"top {int(pct*100)}%"
            rows.append(m)
    return pd.DataFrame(rows).set_index("top_pct") if rows else pd.DataFrame()


# ─── TP/SL multiplier sweep ────────────────────────────────────────────────────

def tpsl_sweep(
    df: pd.DataFrame,
    pred_down: pd.Series,
    pred_up: pd.Series,
    atr: pd.Series,
    top_pct: float = 0.20,
    fee_rt:  float = 0.0014,
    mults: tuple = (0.3, 0.5, 0.7),
) -> pd.DataFrame:
    """
    Grid search over tp_mult × sl_mult combinations.
    Returns a pivot table: tp_mult (rows) × sl_mult (cols) with Sharpe values.
    """
    results = {}
    for tp_m, sl_m in product(mults, mults):
        trades = simulate_short_trades(
            df, pred_down, pred_up, atr,
            top_pct=top_pct, tp_mult=tp_m, sl_mult=sl_m, fee_rt=fee_rt,
        )
        m = calc_metrics(trades, label=f"TP={tp_m} SL={sl_m}")
        results[(tp_m, sl_m)] = {
            "sharpe":    m.get("sharpe",       np.nan),
            "total_ret": m.get("total_return", np.nan),
            "win_rate":  m.get("win_rate",     np.nan),
            "n":         m.get("n_trades",     0),
        }

    rows = []
    for (tp_m, sl_m), v in results.items():
        rows.append({"tp_mult": tp_m, "sl_mult": sl_m, **v})
    df_res = pd.DataFrame(rows)

    # Sharpe pivot
    sharpe_pivot = df_res.pivot(index="tp_mult", columns="sl_mult", values="sharpe")
    return df_res, sharpe_pivot


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_equity_and_exits(trades: pd.DataFrame,
                           title: str = "",
                           save: bool = False):
    if trades.empty:
        return

    r      = trades["net_ret"].dropna()
    equity = np.cumprod(1 + r.values)
    dts    = trades.index[trades["net_ret"].notna()]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    # ── Equity curve ──
    ax = axes[0]
    ax.plot(dts, equity, lw=1.5, color="#1976d2")
    ax.axhline(1.0, color="black", lw=0.5)
    peak = np.maximum.accumulate(equity)
    ax.fill_between(dts, equity, peak, where=equity < peak,
                    alpha=0.3, color="#ef5350", label="Drawdown")
    ax.set_ylabel("Equity")
    ax.set_title(f"{title}  |  Equity Curve (SHORT TRENDING_BEAR)")
    ax.legend(fontsize=8)

    # ── Per-trade P&L coloured by exit reason ──
    ax = axes[1]
    exit_colors = {"TP": "#26a69a", "TP(both)": "#80cbc4",
                   "SL": "#ef5350", "time": "#ff9800"}
    for reason, grp in trades.groupby("exit_reason"):
        c = exit_colors.get(reason, "#9e9e9e")
        ax.scatter(grp.index, grp["net_ret"],
                   s=12, alpha=0.7, color=c, label=reason)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Net return per trade")
    ax.legend(fontsize=8, ncol=4)
    ax.set_title("Per-trade net return by exit type")

    # ── Score vs gross return ──
    ax = axes[2]
    sc = ax.scatter(trades["pred_down"], trades["gross_ret"],
                    s=8, alpha=0.4, c=trades["tp_hit"].astype(int),
                    cmap="RdYlGn")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("pred_down_move (vol-adj)")
    ax.set_ylabel("Gross return")
    ax.set_title("Signal strength vs gross return  (green=TP hit, red=SL/time)")
    plt.colorbar(sc, ax=ax, label="TP hit (1=yes)")

    plt.tight_layout()
    _save(fig, "backtest_path_equity.png", save)


def plot_threshold_comparison(summary: pd.DataFrame, save: bool = False):
    if summary.empty:
        return

    metrics = ["sharpe", "total_return", "win_rate", "mean_net"]
    titles  = ["Sharpe", "Total Return", "Win Rate", "Mean Net Return"]
    colors  = ["#1976d2", "#26a69a", "#ff9800", "#e91e63"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, col, title, c in zip(axes, metrics, titles, colors):
        if col not in summary.columns:
            continue
        vals = summary[col]
        ax.bar(summary.index, vals, color=c, alpha=0.85)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title)
        ax.set_xticklabels(summary.index, rotation=15, ha="right")
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + (abs(v) * 0.02 + 0.001) * (1 if v >= 0 else -1),
                        f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle("Threshold Sweep: top 10% / 20% / 30% pred_down (TRENDING_BEAR)",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, "backtest_path_threshold.png", save)


def plot_tpsl_heatmap(sharpe_pivot: pd.DataFrame, save: bool = False):
    if sharpe_pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    data = sharpe_pivot.values.astype(float)
    im   = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                     vmin=-2, vmax=2)
    ax.set_xticks(range(len(sharpe_pivot.columns)))
    ax.set_xticklabels([f"SL={v}" for v in sharpe_pivot.columns])
    ax.set_yticks(range(len(sharpe_pivot.index)))
    ax.set_yticklabels([f"TP={v}" for v in sharpe_pivot.index])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="black" if abs(v) < 1.5 else "white")

    plt.colorbar(im, ax=ax, label="Sharpe")
    ax.set_title("TP/SL Multiplier Sweep — Sharpe (TRENDING_BEAR short, top 20%)")
    plt.tight_layout()
    _save(fig, "backtest_path_tpsl_heatmap.png", save)


def _save(fig, name: str, save: bool):
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
    ap.add_argument("--save",    action="store_true")
    ap.add_argument("--mode",    default="taker",
                    choices=list(FEES.keys()),
                    help="Fee mode: taker / maker_taker / maker")
    ap.add_argument("--top-pct", type=float, default=0.20,
                    help="Signal percentile cutoff (default 0.20 = top 20%%)")
    ap.add_argument("--tp-mult", type=float, default=0.50)
    ap.add_argument("--sl-mult", type=float, default=0.50)
    args = ap.parse_args()

    fee_entry, fee_exit = FEES[args.mode]
    fee_rt = fee_entry + fee_exit     # round-trip (per side each, so total = entry+exit)
    print(f"Fee mode: {args.mode}  entry={fee_entry:.3%}  exit={fee_exit:.3%}  "
          f"RT={fee_rt:.3%}")

    # ── Load ────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    df, feat_cols = load()

    if "regime_name" not in df.columns:
        print("WARNING: regime_name missing — run regime_detection.py first")
        return

    print("\nRegime distribution (full dataset):")
    print(df["regime_name"].value_counts().to_string())

    bear_n = (df["regime_name"] == "TRENDING_BEAR").sum()
    print(f"\nTRENDING_BEAR bars: {bear_n} ({bear_n/len(df):.1%} of total)")

    # ── Generate OOS predictions ─────────────────────────────────────────────
    print("\nGenerating OOS predictions (walk-forward, regime-routed)...")
    pred_down, pred_up = generate_predictions(df, feat_cols)

    # OOS IC check
    valid = pred_down.notna()
    ic_d, _ = spearmanr(df.loc[valid, "down_move_vol_adj"], pred_down[valid])
    ic_u, _ = spearmanr(df.loc[valid, "up_move_vol_adj"],   pred_up[valid])
    print(f"  OOS IC: down={ic_d:+.4f}  up={ic_u:+.4f}")

    # ── ATR (past-only) ──────────────────────────────────────────────────────
    atr = compute_atr(df)
    print(f"\nATR stats (USD): mean={atr.mean():.1f}  "
          f"median={atr.median():.1f}  max={atr.max():.1f}")

    # ── Primary backtest ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PRIMARY BACKTEST  |  top {int(args.top_pct*100)}%  "
          f"TP={args.tp_mult}  SL={args.sl_mult}  mode={args.mode}")
    print(f"{'='*60}")

    trades = simulate_short_trades(
        df, pred_down, pred_up, atr,
        top_pct = args.top_pct,
        tp_mult = args.tp_mult,
        sl_mult = args.sl_mult,
        fee_rt  = fee_rt,
    )

    print(f"\nTotal trade candidates (TRENDING_BEAR): {bear_n}")
    print(f"Trades executed (top {int(args.top_pct*100)}%): {len(trades)}")
    if not trades.empty:
        print(f"Exit breakdown: {trades['exit_reason'].value_counts().to_dict()}")

    print()
    perf = calc_metrics(trades, label="PRIMARY")

    # ── Fee sensitivity ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FEE SENSITIVITY")
    print(f"{'='*60}")
    for mode_name, (fe, fx) in FEES.items():
        rt = fe + fx
        t  = trades.copy()
        t["net_ret"] = t["gross_ret"] - rt
        calc_metrics(t, label=mode_name)

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"THRESHOLD SWEEP  (TP={args.tp_mult}  SL={args.sl_mult}  mode={args.mode})")
    print(f"{'='*60}")
    thresh_summary = threshold_sweep(
        df, pred_down, pred_up, atr,
        tp_mult = args.tp_mult,
        sl_mult = args.sl_mult,
        fee_rt  = fee_rt,
    )
    if not thresh_summary.empty:
        cols = ["n_trades", "mean_gross", "mean_net", "sharpe",
                "max_drawdown", "win_rate", "avg_win", "avg_loss",
                "n_tp", "n_sl", "n_time"]
        print(thresh_summary[[c for c in cols if c in thresh_summary.columns]]
              .round(4).to_string())

    # ── TP/SL multiplier sweep ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TP/SL MULT SWEEP  (top {int(args.top_pct*100)}%  mode={args.mode})")
    print(f"{'='*60}")
    sweep_df, sharpe_pivot = tpsl_sweep(
        df, pred_down, pred_up, atr,
        top_pct = args.top_pct,
        fee_rt  = fee_rt,
    )
    print("\nSharpe pivot (rows=TP_mult, cols=SL_mult):")
    print(sharpe_pivot.round(3).to_string())
    print("\nReturn pivot:")
    ret_pivot = sweep_df.pivot(index="tp_mult", columns="sl_mult", values="total_ret")
    print(ret_pivot.round(3).to_string())
    print("\nWin-rate pivot:")
    wr_pivot = sweep_df.pivot(index="tp_mult", columns="sl_mult", values="win_rate")
    print(wr_pivot.round(3).to_string())

    # ── Plots ────────────────────────────────────────────────────────────────
    if not trades.empty:
        title = f"TP={args.tp_mult} SL={args.sl_mult} top={int(args.top_pct*100)}% [{args.mode}]"
        print("\nPlotting...")
        plot_equity_and_exits(trades, title=title, save=args.save)
        plot_threshold_comparison(thresh_summary, save=args.save)
        plot_tpsl_heatmap(sharpe_pivot, save=args.save)

    print("\nDone.")


if __name__ == "__main__":
    main()
