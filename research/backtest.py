"""
Walk-Forward Backtest — Regime-Conditional Order Flow Model.

Signal pipeline:
  1. Regenerate OOS predictions via walk-forward CV with regime routing
       pred_up   = up_move_vol_adj  (regime-routed XGBoost)
       pred_down = down_move_vol_adj (regime-routed XGBoost)
       direction_score = pred_up - pred_down
  2. Strategy: non-overlapping 4h trades
       Long  when score > deadband
       Short when score < -deadband
       Flat  otherwise
  3. Costs:
       Taker fee  : 0.05% per side
       Slippage   : 0.02% per side
       Funding    : actual 8h rate from dataset, at each settlement within hold

Caveat:
  regime_name labels come from full-data HMM, not incremental fit.
  In production, regime would need a rolling/expanding HMM refit.

Usage:
    python research/backtest.py
    python research/backtest.py --save
    python research/backtest.py --deadband 0.05  (default 0)
    python research/backtest.py --long-only
    python research/backtest.py --short-only
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────

PARQUET      = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
OUT_DIR      = Path(__file__).parent / "eda_charts"
HOLDING_BARS = 16          # 4h hold = 16 × 15m bars
TAKER_FEE    = 0.0005      # 0.05% per side (Binance USDT-M)
SLIPPAGE     = 0.0002      # 0.02% per side (estimated)
RT_COST      = (TAKER_FEE + SLIPPAGE) * 2   # round-trip cost
N_FOLDS      = 5
SETTLEMENT_MS = 8 * 3600 * 1000   # 8h in milliseconds

ALL_TARGETS = ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]

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
    "open", "high", "low", "close",
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

MIN_REGIME_TR = 200   # min training bars for a regime model to be built
MIN_REGIME_TE = 5     # min test bars to apply regime routing


# ─── Load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()

    df = df.dropna(subset=ALL_TARGETS)

    feat_cols = [c for c in df.columns if c not in EXCLUDE_BASE]
    df[feat_cols] = df[feat_cols].ffill()

    nan_rate  = df[feat_cols].isnull().mean()
    drop_cols = list(nan_rate[nan_rate > 0.05].index)
    if drop_cols:
        feat_cols = [c for c in feat_cols if c not in drop_cols]

    df = df.dropna(subset=feat_cols)
    print(f"  {len(df)} rows x {len(feat_cols)} features")
    return df, feat_cols


# ─── OOS prediction with regime routing ────────────────────────────────────────

def generate_oos_routed(df: pd.DataFrame, feat_cols: list[str],
                         target: str, verbose: bool = True) -> np.ndarray:
    """
    Walk-forward OOS predictions with per-bar regime routing.

    Each fold k:
      - trains global model on bars 0..tr_end
      - trains regime models on same window filtered by regime
      - for test bars: uses regime model if available, else global

    Returns array aligned to df.index (NaN for first fold's training window).
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
        X_te  = df_te[feat_cols].fillna(0).values
        y_te  = df_te[target].values

        # ── global baseline ──
        global_m = xgb.XGBRegressor(**XGB_PARAMS)
        global_m.fit(
            df_tr[feat_cols].fillna(0).values,
            df_tr[target].values,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )
        fold_pred = global_m.predict(X_te)   # start with global preds

        # ── regime override ──
        for regime in regimes:
            mask_tr = (df_tr["regime_name"] == regime).values
            mask_te = (df_te["regime_name"] == regime).values
            if mask_tr.sum() < MIN_REGIME_TR or mask_te.sum() < MIN_REGIME_TE:
                continue
            rm = xgb.XGBRegressor(**XGB_PARAMS)
            rm.fit(
                df_tr[feat_cols].fillna(0).values[mask_tr],
                df_tr[target].values[mask_tr],
                eval_set=[(X_te[mask_te], y_te[mask_te])],
                verbose=False,
            )
            fold_pred[mask_te] = rm.predict(X_te[mask_te])

        oos[tr_end:te_end] = fold_pred
        if verbose:
            ic, _ = spearmanr(y_te, fold_pred)
            print(f"    [{target}] fold {k}/{N_FOLDS}  IC={ic:+.4f}  n={len(y_te)}")

    return oos


# ─── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame,
                    direction_score: np.ndarray,
                    deadband: float = 0.0,
                    long_only: bool = False,
                    short_only: bool = False) -> pd.DataFrame:
    """
    Non-overlapping 4h trades.
    Rebalance every HOLDING_BARS bars.  Signal uses 1-bar lag (no lookahead).
    Entry/exit at bar close.

    Returns trade DataFrame with columns:
      entry_dt, exit_dt, entry_price, exit_price,
      position (+1/-1/0), gross_ret, funding_cost, net_ret, regime
    """
    close  = df["close"].values
    ts_ms  = df["ts_open"].values
    fund   = df["funding_rate"].values
    regime = df["regime_name"].values if "regime_name" in df.columns else \
             np.full(len(df), "unknown")
    index  = df.index

    # Find the first bar with valid OOS prediction
    valid_start = np.argmax(~np.isnan(direction_score))

    # Align rebalance to start of valid OOS window
    start = valid_start + 1   # 1-bar lag on signal
    start = start + (HOLDING_BARS - (start % HOLDING_BARS)) % HOLDING_BARS

    records = []

    t = start
    while t + HOLDING_BARS <= len(df):
        score = direction_score[t - 1]   # 1-bar lag

        if np.isnan(score):
            t += HOLDING_BARS
            continue

        if abs(score) <= deadband:
            t += HOLDING_BARS
            continue

        if score > 0:
            pos = 1   # long
        else:
            pos = -1  # short

        if long_only  and pos < 0:
            t += HOLDING_BARS; continue
        if short_only and pos > 0:
            t += HOLDING_BARS; continue

        entry_p = close[t]
        exit_p  = close[t + HOLDING_BARS]
        gross   = pos * (exit_p - entry_p) / entry_p

        # Funding cost: sum rates at 8h settlement bars within hold window
        fund_cost = 0.0
        for j in range(t + 1, t + HOLDING_BARS + 1):
            if ts_ms[j] % SETTLEMENT_MS == 0:
                # long pays when rate>0, short receives; short pays when rate<0
                fund_cost += pos * fund[j]

        net = gross - RT_COST - fund_cost

        records.append({
            "entry_dt":    index[t],
            "exit_dt":     index[t + HOLDING_BARS],
            "entry_price": entry_p,
            "exit_price":  exit_p,
            "position":    pos,
            "score":       score,
            "gross_ret":   gross,
            "funding_cost": fund_cost,
            "cost_rt":     RT_COST,
            "net_ret":     net,
            "regime":      regime[t - 1],
        })

        t += HOLDING_BARS

    return pd.DataFrame(records)


# ─── Performance metrics ───────────────────────────────────────────────────────

def calc_performance(trades: pd.DataFrame, label: str = "") -> dict:
    if trades.empty:
        print("  No trades generated.")
        return {}

    rets = trades["net_ret"].values
    n    = len(rets)

    # Equity curve (compounded)
    equity = np.cumprod(1 + rets)

    # Sharpe (annualized, non-overlapping 4h periods)
    periods_per_year = 365 * 6   # 6 × 4h per day
    mean_r = rets.mean()
    std_r  = rets.std()
    sharpe = (mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0

    # Max drawdown
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    # Win rate
    win_rate = (rets > 0).mean()

    # Total return
    total_ret = equity[-1] - 1

    # Long / short breakdown
    long_mask  = trades["position"] == 1
    short_mask = trades["position"] == -1
    long_sr  = trades.loc[long_mask,  "net_ret"].mean() if long_mask.any()  else np.nan
    short_sr = trades.loc[short_mask, "net_ret"].mean() if short_mask.any() else np.nan

    result = {
        "n_trades":       n,
        "total_return":   round(total_ret,  4),
        "sharpe":         round(sharpe,     4),
        "max_drawdown":   round(max_dd,     4),
        "win_rate":       round(win_rate,   4),
        "mean_net_ret":   round(mean_r,     5),
        "mean_gross_ret": round(trades["gross_ret"].mean(), 5),
        "mean_fund_cost": round(trades["funding_cost"].mean(), 6),
        "long_mean_ret":  round(long_sr, 5) if not np.isnan(long_sr)  else np.nan,
        "short_mean_ret": round(short_sr, 5) if not np.isnan(short_sr) else np.nan,
        "n_long":         int(long_mask.sum()),
        "n_short":        int(short_mask.sum()),
    }

    tag = f"  [{label}] " if label else "  "
    print(f"{tag}Trades={n:4d}  Return={total_ret:+.1%}  "
          f"Sharpe={sharpe:.3f}  MaxDD={max_dd:.1%}  WinRate={win_rate:.1%}")

    return result


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_equity_curve(trades: pd.DataFrame, df: pd.DataFrame,
                       title: str = "", save: bool = False):
    if trades.empty:
        return

    equity = np.cumprod(1 + trades["net_ret"].values)
    entry_dts = pd.to_datetime(trades["entry_dt"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    # ── Equity curve ──
    ax = axes[0]
    ax.plot(entry_dts, equity, lw=1.5, color="#1976d2", label="Net equity")
    ax.axhline(1.0, color="black", lw=0.5)
    peak  = np.maximum.accumulate(equity)
    ax.fill_between(entry_dts, equity, peak, where=equity < peak,
                    alpha=0.25, color="#ef5350", label="Drawdown")
    ax.set_ylabel("Equity (1 = start)")
    ax.set_title(f"{title}  Equity Curve")
    ax.legend(fontsize=8)

    # ── Rolling Sharpe (20-trade window) ──
    ax = axes[1]
    window = min(20, len(trades) // 3)
    if window >= 5:
        roll_sr = (trades["net_ret"]
                   .rolling(window)
                   .apply(lambda x: x.mean() / x.std() * np.sqrt(365 * 6)
                          if x.std() > 0 else 0))
        ax.plot(entry_dts, roll_sr, lw=1, color="#ff9800")
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(1.5, color="green", lw=0.5, linestyle="--", label="Target 1.5")
        ax.set_ylabel(f"Rolling Sharpe ({window}-trade)")
        ax.legend(fontsize=8)

    # ── Regime breakdown ──
    ax = axes[2]
    regime_colors = {
        "TRENDING_BULL": "#26a69a",
        "TRENDING_BEAR": "#ef5350",
        "CHOPPY":        "#ff9800",
        "unknown":       "#9e9e9e",
    }
    for regime, grp in trades.groupby("regime"):
        c = regime_colors.get(regime, "#9e9e9e")
        ax.scatter(pd.to_datetime(grp["entry_dt"]),
                   grp["net_ret"],
                   s=8, alpha=0.7, color=c, label=regime)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Net return per trade")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Per-trade return coloured by regime")

    plt.tight_layout()
    _save(fig, "backtest_equity.png", save)


def plot_regime_breakdown(trades: pd.DataFrame, save: bool = False):
    """Bar chart of mean net return and Sharpe per regime."""
    if trades.empty:
        return

    periods_per_year = 365 * 6
    rows = []
    for regime, grp in trades.groupby("regime"):
        r   = grp["net_ret"]
        sr  = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else 0
        wr  = (r > 0).mean()
        rows.append({"regime": regime,
                     "mean_ret": r.mean(),
                     "sharpe":   sr,
                     "win_rate": wr,
                     "n_trades": len(grp)})
    rdf = pd.DataFrame(rows).set_index("regime")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors_map = {
        "TRENDING_BULL": "#26a69a",
        "TRENDING_BEAR": "#ef5350",
        "CHOPPY":        "#ff9800",
    }
    bc = [colors_map.get(r, "#9e9e9e") for r in rdf.index]

    for ax, col, title in zip(axes,
                               ["mean_ret", "sharpe", "win_rate"],
                               ["Mean Net Return", "Sharpe", "Win Rate"]):
        ax.bar(rdf.index, rdf[col], color=bc, alpha=0.85)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title)
        ax.set_xticklabels(rdf.index, rotation=15, ha="right")
        for x, val in zip(range(len(rdf)), rdf[col]):
            ax.text(x, val + (0.001 if val >= 0 else -0.003),
                    f"{val:.3f}", ha="center", fontsize=8)

    # trade count annotation
    for ax in axes:
        ax2 = ax.twinx()
        ax2.bar(rdf.index, rdf["n_trades"], alpha=0, color="none")
        for x, (r, row) in enumerate(rdf.iterrows()):
            ax2.text(x, 0, f"n={int(row['n_trades'])}", ha="center",
                     va="bottom", fontsize=7, color="gray")
        ax2.set_yticks([])

    plt.suptitle("Regime Breakdown — Net Return / Sharpe / Win Rate", fontsize=12)
    plt.tight_layout()
    _save(fig, "backtest_regime.png", save)


def plot_monthly_returns(trades: pd.DataFrame, save: bool = False):
    """Monthly return heatmap."""
    if trades.empty:
        return

    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades["entry_dt"]).dt.to_period("M")
    monthly = trades.groupby("month")["net_ret"].sum()

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["#ef5350" if v < 0 else "#26a69a" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=30, ha="right")
    ax.set_ylabel("Monthly Net Return")
    ax.set_title("Monthly Returns (sum of 4h trades)")
    for i, v in enumerate(monthly.values):
        ax.text(i, v + (0.001 if v >= 0 else -0.003),
                f"{v:.1%}", ha="center", fontsize=8)
    plt.tight_layout()
    _save(fig, "backtest_monthly.png", save)


def plot_score_vs_return(trades: pd.DataFrame, save: bool = False):
    """Scatter: direction score vs net return — should show positive correlation."""
    if trades.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ic, _ = spearmanr(trades["score"].abs(), trades["gross_ret"] * trades["position"])
    axes[0].scatter(trades["score"], trades["net_ret"],
                    s=5, alpha=0.4, color="#1976d2")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].axvline(0, color="black", lw=0.5)
    axes[0].set_xlabel("Direction score")
    axes[0].set_ylabel("Net return")
    axes[0].set_title(f"Score vs Net Return  (rank-IC={ic:.3f})")

    # Quintile analysis
    trades["score_q"] = pd.qcut(trades["score"], 5, labels=["Q1\n(short)", "Q2", "Q3", "Q4", "Q5\n(long)"])
    qmeans = trades.groupby("score_q")["net_ret"].mean()
    colors = ["#ef5350" if v < 0 else "#26a69a" for v in qmeans.values]
    axes[1].bar(range(len(qmeans)), qmeans.values, color=colors, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xticks(range(len(qmeans)))
    axes[1].set_xticklabels(qmeans.index, fontsize=9)
    axes[1].set_ylabel("Mean net return")
    axes[1].set_title("Score Quintile Analysis")

    plt.tight_layout()
    _save(fig, "backtest_score_analysis.png", save)


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
    ap.add_argument("--save",       action="store_true")
    ap.add_argument("--deadband",   type=float, default=0.0,
                    help="Min |direction_score| to enter (default 0)")
    ap.add_argument("--long-only",  action="store_true")
    ap.add_argument("--short-only", action="store_true")
    args = ap.parse_args()

    print("Loading data...")
    df, feat_cols = load()

    # ── Generate OOS predictions for up/down separately ───────────────────────
    print("\n[1/2] OOS predictions: up_move_vol_adj")
    pred_up = generate_oos_routed(df, feat_cols, "up_move_vol_adj")

    print("\n[2/2] OOS predictions: down_move_vol_adj")
    pred_down = generate_oos_routed(df, feat_cols, "down_move_vol_adj")

    direction_score = pred_up - pred_down

    valid = ~np.isnan(direction_score)
    ic_up, _   = spearmanr(df["up_move_vol_adj"].values[valid],   pred_up[valid])
    ic_down, _ = spearmanr(df["down_move_vol_adj"].values[valid], pred_down[valid])
    print(f"\n  OOS IC  up={ic_up:+.4f}  down={ic_down:+.4f}")

    # ── Trade simulation ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRADE SIMULATION")
    print("="*60)
    print(f"  Holding: {HOLDING_BARS} bars (4h)  "
          f"Fee+slip: {RT_COST:.2%}  Deadband: {args.deadband}")

    trades = simulate_trades(
        df, direction_score,
        deadband   = args.deadband,
        long_only  = args.long_only,
        short_only = args.short_only,
    )

    if trades.empty:
        print("  No trades generated — try reducing deadband.")
        return

    # ── Full-period performance ───────────────────────────────────────────────
    print("\n[Overall]")
    perf = calc_performance(trades, label="ALL")

    print("\n[By direction]")
    calc_performance(trades[trades["position"] == 1],  label="LONG ")
    calc_performance(trades[trades["position"] == -1], label="SHORT")

    print("\n[By regime]")
    for regime in sorted(trades["regime"].unique()):
        calc_performance(trades[trades["regime"] == regime], label=regime)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in perf.items():
        print(f"  {k:<20} {v}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    title = f"OOF-Routed | deadband={args.deadband}"
    if args.long_only:  title += " | LONG only"
    if args.short_only: title += " | SHORT only"

    print("\nPlotting...")
    plot_equity_curve(trades, df, title=title, save=args.save)
    plot_regime_breakdown(trades, save=args.save)
    plot_monthly_returns(trades, save=args.save)
    plot_score_vs_return(trades, save=args.save)

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
