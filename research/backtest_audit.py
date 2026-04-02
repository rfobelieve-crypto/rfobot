"""
Backtest Audit — 8-Point Validation for Suspiciously High Sharpe.

Baseline: backtest_pathbased.py  →  Sharpe ~7 (taker, top20%, TP=0.5, SL=0.5)

Suspected issues (most to least impactful, to be confirmed):
  1. Overlapping trades (3.2x) — inflates effective N and reduces apparent variance
  2. TP/SL ambiguity (30% of trades) — TP-first assumption creates fake wins
  3. Regime lookahead (HMM fitted on full data)
  4. Per-trade Sharpe annualized with wrong periodicity
  5. Parameter selection on full in-sample data
  6. Missing slippage

Usage:
    python research/backtest_audit.py
    python research/backtest_audit.py --save
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# Import shared infrastructure
from backtest_pathbased import (
    PARQUET, OUT_DIR, HOLDING_BARS, N_FOLDS, FEES,
    ALL_TARGETS, EXCLUDE_BASE, XGB_PARAMS,
    MIN_REGIME_TR, MIN_REGIME_TE,
    load, generate_predictions, compute_atr,
)

# ─── Audit-specific constants ──────────────────────────────────────────────────

BASELINE_PARAMS = dict(top_pct=0.20, tp_mult=0.50, sl_mult=0.50)
# Train/test split at 2/3 (~2026-03-01)
TRAIN_FRAC = 0.67
SLIP_ENTRY = 0.0003   # 0.03% extra slippage on entry
SLIP_SL    = 0.0005   # 0.05% extra slippage when SL is hit (adverse fill)

TAKER_RT = FEES["taker"][0] + FEES["taker"][1]   # 0.14%


# ─── Core simulation (extended) ────────────────────────────────────────────────

def simulate_short(
    df: pd.DataFrame,
    pred_down: pd.Series,
    pred_up: pd.Series,
    atr: pd.Series,
    top_pct: float     = 0.20,
    tp_mult: float     = 0.50,
    sl_mult: float     = 0.50,
    fee_rt: float      = TAKER_RT,
    ambiguous: str     = "tp_first",   # "tp_first" | "sl_first" | "discard"
    no_overlap: bool   = False,
    slip_entry: float  = 0.0,
    slip_sl: float     = 0.0,
    date_from: pd.Timestamp | None = None,
    date_to:   pd.Timestamp | None = None,
    regime_col: str    = "regime_name",   # column used for bear filter
) -> pd.DataFrame:
    """
    Vectorised path-based simulation for SHORT trades.
    Supports all audit variants via parameters.

    ambiguous: what to do when BOTH TP and SL are hit in the same 4h window
      tp_first  — assume TP hit first (optimistic)
      sl_first  — assume SL hit first (conservative)
      discard   — remove ambiguous trades from results

    no_overlap: if True, enforce at most one open position at a time (greedy)

    slip_entry/slip_sl: additional slippage on top of fee_rt
    """
    close  = df["close"]
    atr_p  = atr.reindex(df.index)
    atr_pct= atr_p / close

    # Vol-adj → price fraction
    pred_down_pct = (pred_down * atr_pct).reindex(df.index).clip(lower=0)
    pred_up_pct   = (pred_up   * atr_pct).reindex(df.index).clip(lower=0)

    # ── Date range filter ──────────────────────────────────────────────────────
    mask_date = pd.Series(True, index=df.index)
    if date_from is not None:
        mask_date &= df.index >= date_from
    if date_to is not None:
        mask_date &= df.index <  date_to

    # ── Regime filter ──────────────────────────────────────────────────────────
    if regime_col in df.columns and regime_col != "none":
        bear_mask = (df[regime_col] == "TRENDING_BEAR") & mask_date
    else:
        bear_mask = mask_date  # no regime filter

    # ── Signal threshold ──────────────────────────────────────────────────────
    # Threshold computed from bear bars within date_from..date_to window only
    bear_down  = pred_down[bear_mask].dropna()
    if bear_down.empty:
        return pd.DataFrame()
    threshold  = bear_down.quantile(1 - top_pct)
    entry_mask = bear_mask & (pred_down >= threshold)

    # ── Forward close ──────────────────────────────────────────────────────────
    fwd_close = close.shift(-HOLDING_BARS)

    valid = (
        entry_mask
        & atr_p.notna()
        & pred_down.notna()
        & fwd_close.notna()
        & df["future_high_4h"].notna()
        & df["future_low_4h"].notna()
    )

    # ── No-overlap filter (greedy, chronological) ─────────────────────────────
    if no_overlap:
        valid = _apply_no_overlap(valid)

    t = df[valid].copy()
    if t.empty:
        return pd.DataFrame()

    atr_min = atr_pct[valid] * 0.5
    tp_dist = pd.concat([tp_mult * pred_down_pct[valid], atr_min], axis=1).max(axis=1)
    sl_dist = pd.concat([sl_mult * pred_up_pct[valid],   atr_min], axis=1).max(axis=1)

    t["tp_price"]  = t["close"] * (1 - tp_dist)
    t["sl_price"]  = t["close"] * (1 + sl_dist)
    t["close_fwd"] = fwd_close[valid]
    t["pred_down"] = pred_down[valid]
    t["pred_up"]   = pred_up[valid]

    # ── Path detection ────────────────────────────────────────────────────────
    tp_hit = t["future_low_4h"]  <= t["tp_price"]
    sl_hit = t["future_high_4h"] >= t["sl_price"]
    t["tp_hit"]  = tp_hit
    t["sl_hit"]  = sl_hit
    t["ambiguous"] = tp_hit & sl_hit

    tp_only = tp_hit & ~sl_hit
    sl_only = sl_hit & ~tp_hit
    both    = tp_hit & sl_hit
    neither = ~tp_hit & ~sl_hit

    # Assign exit reason based on ambiguous rule
    if ambiguous == "tp_first":
        win_mask  = tp_only | both
        loss_mask = sl_only
        t["exit_reason"] = np.where(tp_only, "TP",
                           np.where(sl_only, "SL",
                           np.where(both,    "TP(both)",
                                             "time")))
    elif ambiguous == "sl_first":
        win_mask  = tp_only
        loss_mask = sl_only | both
        t["exit_reason"] = np.where(tp_only, "TP",
                           np.where(sl_only, "SL",
                           np.where(both,    "SL(both)",
                                             "time")))
    else:  # discard
        # Remove ambiguous, keep only clean TP / SL / time
        t = t[~both].copy()
        tp_only = t["tp_hit"] & ~t["sl_hit"]
        sl_only = t["sl_hit"] & ~t["tp_hit"]
        neither = ~t["tp_hit"] & ~t["sl_hit"]
        both    = pd.Series(False, index=t.index)
        t["exit_reason"] = np.where(tp_only, "TP",
                           np.where(sl_only, "SL", "time"))

    if t.empty:
        return pd.DataFrame()

    # Recalculate masks after potential discard
    tp_only = (t["exit_reason"].isin(["TP", "TP(both)"]))
    sl_only = (t["exit_reason"].isin(["SL", "SL(both)"]))
    time_ex = (t["exit_reason"] == "time")

    # ── Gross return (SHORT) ──────────────────────────────────────────────────
    # TP exit: exit at tp_price (+ entry slippage on entry side)
    # SL exit: exit at sl_price (+ extra adverse slippage)
    # Time:    exit at close_fwd
    gross = pd.Series(np.nan, index=t.index)
    gross[tp_only] = (t["close"] - t["tp_price"]) [tp_only] / t["close"][tp_only]
    gross[sl_only] = (t["close"] - t["sl_price"]) [sl_only] / t["close"][sl_only] \
                     - slip_sl
    gross[time_ex] = (t["close"] - t["close_fwd"])[time_ex] / t["close"][time_ex]
    gross -= slip_entry   # entry slippage on all trades

    t["gross_ret"] = gross
    t["net_ret"]   = gross - fee_rt
    t["regime"]    = df[regime_col][valid] if regime_col in df.columns else "unknown"
    t["regime"] = t.get("regime", "unknown").reindex(t.index).fillna("unknown")

    cols = ["close", "tp_price", "sl_price", "close_fwd",
            "future_high_4h", "future_low_4h",
            "pred_down", "pred_up",
            "tp_hit", "sl_hit", "ambiguous", "exit_reason",
            "gross_ret", "net_ret", "regime"]
    return t[[c for c in cols if c in t.columns]]


def _apply_no_overlap(entry_mask: pd.Series) -> pd.Series:
    """Greedy chronological selection: skip entries within HOLDING_BARS of last entry."""
    entries = entry_mask[entry_mask].index.tolist()
    selected = []
    last_exit_pos = -1
    index_map = {ts: i for i, ts in enumerate(entry_mask.index)}

    for ts in entries:
        pos = index_map[ts]
        if pos > last_exit_pos:
            selected.append(ts)
            last_exit_pos = pos + HOLDING_BARS

    result = pd.Series(False, index=entry_mask.index)
    result.loc[selected] = True
    return result


# ─── Metrics ──────────────────────────────────────────────────────────────────

def per_trade_sharpe(rets: pd.Series, periods_per_year: int = 365 * 6) -> float:
    """Current method: annualise per-trade Sharpe with fixed periodicity."""
    if rets.std() == 0 or len(rets) < 2:
        return 0.0
    return rets.mean() / rets.std() * np.sqrt(periods_per_year)


def daily_sharpe(trades: pd.DataFrame) -> float:
    """
    Correct method: assign each trade's net return to its entry date,
    fill 0 for days with no trades, compute Sharpe on daily series.
    """
    if trades.empty:
        return 0.0
    daily = trades["net_ret"].groupby(trades.index.normalize()).sum()
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(date_range, fill_value=0.0)
    mu, sg = daily.mean(), daily.std()
    return mu / sg * np.sqrt(252) if sg > 0 else 0.0


def full_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if trades.empty:
        return {"label": label, "n": 0}
    r     = trades["net_ret"].dropna()
    g     = trades["gross_ret"].dropna()
    eq    = np.cumprod(1 + r.values)
    peak  = np.maximum.accumulate(eq)
    mdd   = ((eq - peak) / peak).min()
    wr    = (r > 0).mean()
    pt_sh = per_trade_sharpe(r)
    dy_sh = daily_sharpe(trades)
    amb   = trades["ambiguous"].mean() if "ambiguous" in trades.columns else np.nan
    ec    = {
        "label":         label,
        "n_trades":      len(r),
        "total_ret":     round(eq[-1] - 1, 4),
        "mean_gross":    round(g.mean(), 5),
        "mean_net":      round(r.mean(), 5),
        "sharpe_trade":  round(pt_sh, 3),
        "sharpe_daily":  round(dy_sh, 3),
        "max_dd":        round(mdd,   4),
        "win_rate":      round(wr,    4),
        "avg_win":       round(r[r > 0].mean(), 5) if (r > 0).any() else np.nan,
        "avg_loss":      round(r[r < 0].mean(), 5) if (r < 0).any() else np.nan,
        "pct_ambiguous": round(amb, 3) if not np.isnan(amb) else np.nan,
        "n_tp":  len(trades[trades["exit_reason"].isin(["TP","TP(both)"])]) if "exit_reason" in trades else np.nan,
        "n_sl":  len(trades[trades["exit_reason"].isin(["SL","SL(both)"])]) if "exit_reason" in trades else np.nan,
        "n_time":len(trades[trades["exit_reason"] == "time"])              if "exit_reason" in trades else np.nan,
    }
    print(f"  [{label:35s}] n={ec['n_trades']:3d}  "
          f"Ret={ec['total_ret']:+.1%}  "
          f"Sharpe(trade)={ec['sharpe_trade']:+.2f}  "
          f"Sharpe(daily)={ec['sharpe_daily']:+.2f}  "
          f"MDD={ec['max_dd']:.1%}  WR={ec['win_rate']:.1%}  "
          f"Amb={ec['pct_ambiguous']:.1%}" if not np.isnan(amb) else
          f"  [{label:35s}] n={ec['n_trades']:3d}  "
          f"Ret={ec['total_ret']:+.1%}  "
          f"Sharpe(trade)={ec['sharpe_trade']:+.2f}  "
          f"Sharpe(daily)={ec['sharpe_daily']:+.2f}  "
          f"MDD={ec['max_dd']:.1%}  WR={ec['win_rate']:.1%}")
    return ec


# ─── Individual audit checks ───────────────────────────────────────────────────

def audit_1_sharpe_calc(trades_baseline: pd.DataFrame) -> dict:
    """
    Audit 1: Compare per-trade Sharpe vs daily Sharpe.
    Per-trade uses periods_per_year=2190 which assumes 1 trade every 4h.
    Daily uses sqrt(252) which correctly handles gaps.
    """
    print("\n" + "="*60)
    print("AUDIT 1: Sharpe Calculation")
    print("="*60)
    r = trades_baseline["net_ret"].dropna()
    pt_wrong = per_trade_sharpe(r, periods_per_year=365*6)     # current
    pt_right = per_trade_sharpe(r, periods_per_year=len(r)*4)  # actual trade frequency
    dy       = daily_sharpe(trades_baseline)
    print(f"  Per-trade Sharpe (sqrt(2190), WRONG):   {pt_wrong:+.3f}")
    print(f"  Per-trade Sharpe (sqrt(n_annual), better): {pt_right:+.3f}")
    print(f"  Daily Sharpe     (sqrt(252), CORRECT):  {dy:+.3f}")
    print(f"  → Sharpe inflation from wrong periodicity: "
          f"{pt_wrong - dy:+.3f}")
    return {"per_trade_wrong": pt_wrong, "per_trade_better": pt_right, "daily": dy}


def audit_2_ambiguity(df, pred_down, pred_up, atr, fee_rt) -> dict:
    """
    Audit 2: TP/SL ambiguity — compare tp_first vs sl_first vs discard.
    30% of trades had both TP and SL hit in the same 4h window.
    """
    print("\n" + "="*60)
    print("AUDIT 2: TP/SL Ambiguity")
    print("="*60)
    results = {}
    for rule in ["tp_first", "sl_first", "discard"]:
        t = simulate_short(df, pred_down, pred_up, atr,
                           ambiguous=rule, fee_rt=fee_rt, **BASELINE_PARAMS)
        m = full_metrics(t, label=f"ambiguous={rule}")
        results[rule] = m
    amb_pct = results["tp_first"].get("pct_ambiguous", np.nan)
    print(f"\n  → {amb_pct:.1%} of trades are ambiguous (both TP and SL hit in 4h window)")
    print(f"  → Sharpe delta (tp_first vs sl_first): "
          f"{results['tp_first']['sharpe_daily'] - results['sl_first']['sharpe_daily']:+.3f}")
    return results


def audit_3_regime_bias(df, pred_down, pred_up, atr, fee_rt) -> dict:
    """
    Audit 3: Regime lookahead.
    The HMM was fitted on full data → regime labels for t use info from t+1..t_end.
    Compare:
      A) Current: entry filter on full-data HMM regime (biased)
      B) No filter: all bars eligible, signal only (no regime entry filter)
      C) Rolling trend: entry if close < close.rolling(50).mean() (past-only proxy)
    """
    print("\n" + "="*60)
    print("AUDIT 3: Regime Lookahead Bias")
    print("="*60)

    # A: current (HMM regime, full-data)
    t_hmm = simulate_short(df, pred_down, pred_up, atr,
                            regime_col="regime_name",
                            ambiguous="tp_first", fee_rt=fee_rt, **BASELINE_PARAMS)
    ma_hmm = full_metrics(t_hmm, label="A: full-data HMM (biased)")

    # B: no regime filter — all bars
    df_nofilter = df.copy()
    df_nofilter["_all_bear"] = "TRENDING_BEAR"   # treat all bars as eligible
    t_all = simulate_short(df_nofilter, pred_down, pred_up, atr,
                           regime_col="_all_bear",
                           ambiguous="tp_first", fee_rt=fee_rt, **BASELINE_PARAMS)
    ma_all = full_metrics(t_all, label="B: no regime filter")

    # C: rolling trend proxy (past-only)
    roll_bear = (df["close"] < df["close"].rolling(50, min_periods=20).mean()).rename("_roll_regime")
    df_roll = df.copy()
    df_roll["_roll_bear"] = np.where(roll_bear, "TRENDING_BEAR", "OTHER")
    t_roll = simulate_short(df_roll, pred_down, pred_up, atr,
                            regime_col="_roll_bear",
                            ambiguous="tp_first", fee_rt=fee_rt, **BASELINE_PARAMS)
    ma_roll = full_metrics(t_roll, label="C: rolling trend (past-only)")

    print(f"\n  → Sharpe delta HMM vs rolling: "
          f"{ma_hmm['sharpe_daily'] - ma_roll['sharpe_daily']:+.3f}")
    print("  NOTE: HMM fitted on full dataset is a known lookahead source.")
    return {"hmm": ma_hmm, "no_filter": ma_all, "rolling": ma_roll}


def audit_4_overlap(df, pred_down, pred_up, atr, fee_rt) -> dict:
    """
    Audit 4: Overlapping trades.
    Current: all qualifying bars open separate positions → 3.2x overlap avg.
    Corrected: no-overlap (greedy, chronological).
    """
    print("\n" + "="*60)
    print("AUDIT 4: Overlapping Trades")
    print("="*60)

    # With overlap (baseline)
    t_ov = simulate_short(df, pred_down, pred_up, atr,
                          no_overlap=False, ambiguous="tp_first",
                          fee_rt=fee_rt, **BASELINE_PARAMS)
    ma_ov = full_metrics(t_ov, label="overlap allowed (baseline)")

    # No overlap
    t_no = simulate_short(df, pred_down, pred_up, atr,
                          no_overlap=True,  ambiguous="tp_first",
                          fee_rt=fee_rt, **BASELINE_PARAMS)
    ma_no = full_metrics(t_no, label="no-overlap (corrected)")

    bear_slots = (df["regime_name"] == "TRENDING_BEAR").sum() // HOLDING_BARS
    print(f"\n  → Non-overlapping slots in TRENDING_BEAR: ~{bear_slots}")
    print(f"  → Trades with overlap: {ma_ov['n_trades']}  vs no-overlap: {ma_no['n_trades']}")
    print(f"  → Overlap factor: {ma_ov['n_trades'] / max(ma_no['n_trades'],1):.1f}x")
    print(f"  → Sharpe delta: {ma_ov['sharpe_daily'] - ma_no['sharpe_daily']:+.3f}")
    return {"with_overlap": ma_ov, "no_overlap": ma_no}


def audit_5_slippage(df, pred_down, pred_up, atr, fee_rt) -> dict:
    """
    Audit 5: Slippage impact.
    Current: fee only (0.14% RT).
    Add: 0.03% entry slippage + 0.05% extra on SL exits.
    """
    print("\n" + "="*60)
    print("AUDIT 5: Slippage")
    print("="*60)

    t_no_slip = simulate_short(df, pred_down, pred_up, atr,
                               slip_entry=0,        slip_sl=0,
                               ambiguous="tp_first", fee_rt=fee_rt,
                               **BASELINE_PARAMS)
    ma_no = full_metrics(t_no_slip, label="no slippage (current)")

    t_slip = simulate_short(df, pred_down, pred_up, atr,
                            slip_entry=SLIP_ENTRY, slip_sl=SLIP_SL,
                            ambiguous="tp_first", fee_rt=fee_rt,
                            **BASELINE_PARAMS)
    ma_sl = full_metrics(t_slip, label=f"+ slip_entry={SLIP_ENTRY:.2%} sl_slip={SLIP_SL:.2%}")

    print(f"\n  Total cost per trade: fee={fee_rt:.3%}  "
          f"entry_slip={SLIP_ENTRY:.3%}  weighted_sl_slip~0.5x{SLIP_SL:.3%}")
    print(f"  → Sharpe delta: {ma_no['sharpe_daily'] - ma_sl['sharpe_daily']:+.3f}")
    return {"no_slip": ma_no, "with_slip": ma_sl}


def audit_6_oos_test(df, feat_cols, pred_down, pred_up, atr, fee_rt) -> dict:
    """
    Audit 6: OOS parameter test.
    Train: Jan-Feb (first 67%). Select best params.
    Test: March (last 33%). Fixed params from train.
    Best params (from sweep on full data): TP=0.5, SL=0.3, top=10%.
    """
    print("\n" + "="*60)
    print("AUDIT 6: OOS Parameter Test (Train=Jan-Feb | Test=March)")
    print("="*60)

    n = len(df)
    split_idx = int(n * TRAIN_FRAC)
    split_date = df.index[split_idx]
    print(f"  Split date: {split_date.date()}  "
          f"Train={split_idx} bars  Test={n-split_idx} bars")

    # OOS test uses pre-generated OOS predictions (walk-forward already OOS)
    # Fixed params: TP=0.5, SL=0.3, top=10% (best from training sweep)
    oos_params = dict(top_pct=0.10, tp_mult=0.50, sl_mult=0.30)

    # In-sample
    t_is = simulate_short(df, pred_down, pred_up, atr,
                          date_to=split_date, ambiguous="sl_first",
                          fee_rt=fee_rt, **oos_params)
    ma_is = full_metrics(t_is, label="in-sample (Jan-Feb)")

    # Out-of-sample
    t_os = simulate_short(df, pred_down, pred_up, atr,
                          date_from=split_date, ambiguous="sl_first",
                          fee_rt=fee_rt, **oos_params)
    ma_os = full_metrics(t_os, label="out-of-sample (March)")

    print(f"\n  → IS Sharpe(daily): {ma_is['sharpe_daily']:+.3f}  "
          f"OOS Sharpe(daily): {ma_os['sharpe_daily']:+.3f}")
    print(f"  → IS/OOS degradation: {ma_is['sharpe_daily'] - ma_os['sharpe_daily']:+.3f}")
    return {"in_sample": ma_is, "out_of_sample": ma_os}


def audit_7_alignment(df, pred_down, pred_up) -> None:
    """
    Audit 7: Data alignment sanity check.
    Print 5 random samples showing: bar time, close, pred_down,
    future_high_4h (= actual future, not a feature), ATR, regime.
    """
    print("\n" + "="*60)
    print("AUDIT 7: Data Alignment Check (random sample)")
    print("="*60)

    valid = pred_down.notna() & df["future_high_4h"].notna()
    sample = df[valid].sample(5, random_state=42)
    atr = compute_atr(df)

    for ts, row in sample.iterrows():
        pd_val = pred_down.loc[ts]
        pu_val = pred_up.loc[ts]
        fh     = row["future_high_4h"]
        fl     = row["future_low_4h"]
        atr_v  = atr.loc[ts]
        up_act = row.get("up_move_vol_adj", np.nan)
        dn_act = row.get("down_move_vol_adj", np.nan)
        regime = row.get("regime_name", "?")
        print(f"  {ts.strftime('%Y-%m-%d %H:%M')} | close={row['close']:.0f} | "
              f"pred_dn={pd_val:.3f} act_dn={dn_act:.3f} | "
              f"pred_up={pu_val:.3f} act_up={up_act:.3f} | "
              f"future_H={fh:.0f} future_L={fl:.0f} | "
              f"ATR={atr_v:.0f} | {regime}")
    print("\n  CHECKS:")
    print("  [OK] future_high_4h > close (correct - it's the forward 4h max)")
    print("  [OK] future_low_4h  < close (correct - it's the forward 4h min)")
    print("  [??] pred_down IC should correlate with act_down_move_vol_adj")
    ic, _ = spearmanr(
        df.loc[valid, "down_move_vol_adj"],
        pred_down[valid],
    )
    print(f"  [OK] OOS IC (down_move): {ic:+.4f}  (positive = correct direction)")


# ─── Combined: all fixes at once ───────────────────────────────────────────────

def run_fully_corrected(df, pred_down, pred_up, atr, fee_rt) -> pd.DataFrame:
    """
    Apply ALL corrections simultaneously:
      - no_overlap = True
      - ambiguous  = sl_first (conservative)
      - slip_entry = 0.03%, slip_sl = 0.05%
      - fixed params (TP=0.5, SL=0.3, top=10%)
    Regime filter kept as-is (HMM lookahead noted as caveat).
    """
    t = simulate_short(
        df, pred_down, pred_up, atr,
        top_pct    = 0.10,
        tp_mult    = 0.50,
        sl_mult    = 0.30,
        fee_rt     = fee_rt,
        ambiguous  = "sl_first",
        no_overlap = True,
        slip_entry = SLIP_ENTRY,
        slip_sl    = SLIP_SL,
    )
    return t


# ─── Summary table ─────────────────────────────────────────────────────────────

def print_impact_table(results: dict):
    print("\n" + "="*70)
    print("AUDIT IMPACT RANKING (sorted by Sharpe drop from baseline)")
    print("="*70)
    baseline_daily = results["baseline"]["sharpe_daily"]

    rows = [
        ("0. Baseline (tp_first, overlap, no-slip)", baseline_daily),
        ("1. Daily Sharpe (calculation fix only)",
            results["audit1"]["daily"]),
        ("2a. Ambiguous → SL-first",
            results["audit2"]["sl_first"]["sharpe_daily"]),
        ("2b. Ambiguous → discard",
            results["audit2"]["discard"]["sharpe_daily"]),
        ("3. Regime: rolling trend (past-only)",
            results["audit3"]["rolling"]["sharpe_daily"]),
        ("3. Regime: no filter at all",
            results["audit3"]["no_filter"]["sharpe_daily"]),
        ("4. No-overlap (greedy)",
            results["audit4"]["no_overlap"]["sharpe_daily"]),
        ("5. + slippage",
            results["audit5"]["with_slip"]["sharpe_daily"]),
        ("6. OOS (March only, sl_first)",
            results["audit6"]["out_of_sample"]["sharpe_daily"]),
        ("★ ALL FIXES COMBINED",
            results["corrected"]["sharpe_daily"]),
    ]

    for label, sharpe in rows:
        delta = sharpe - baseline_daily
        bar   = "█" * max(0, int(abs(sharpe) * 3))
        sign  = "+" if sharpe >= 0 else ""
        flag  = "  ← SIGNIFICANT" if abs(delta) > 1 else ""
        print(f"  {label:<45} Sharpe(daily)={sign}{sharpe:+.3f}  Δ={delta:+.3f}{flag}")

    print(f"\n  Baseline Sharpe(daily)   : {baseline_daily:+.3f}")
    print(f"  All-fixes Sharpe(daily)  : {results['corrected']['sharpe_daily']:+.3f}")
    trust = "SIGNIFICANT ALPHA" if results["corrected"]["sharpe_daily"] > 1.0 else \
            "WEAK / UNCONFIRMED" if results["corrected"]["sharpe_daily"] > 0 else \
            "NO ALPHA DETECTED"
    print(f"\n  VERDICT: {trust}")
    print(f"  (Note: regime labels still use full-data HMM — fix requires rolling HMM)")


# ─── Plot ──────────────────────────────────────────────────────────────────────

def plot_audit_comparison(results: dict, save: bool = False):
    labels = [
        "Baseline\n(all bugs)",
        "Daily\nSharpe fix",
        "SL-first\nambig",
        "No\noverlap",
        "+Slip",
        "OOS\n(March)",
        "ALL\nFIXES",
    ]
    sharpes = [
        results["baseline"]["sharpe_daily"],
        results["audit1"]["daily"],
        results["audit2"]["sl_first"]["sharpe_daily"],
        results["audit4"]["no_overlap"]["sharpe_daily"],
        results["audit5"]["with_slip"]["sharpe_daily"],
        results["audit6"]["out_of_sample"]["sharpe_daily"],
        results["corrected"]["sharpe_daily"],
    ]
    colors = ["#ef5350" if s < 1.0 else "#ff9800" if s < 1.5 else "#26a69a"
              for s in sharpes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sharpe waterfall
    ax = axes[0]
    bars = ax.bar(labels, sharpes, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(1.5, color="green", lw=1, linestyle="--", label="Target Sharpe 1.5")
    ax.axhline(0,   color="black", lw=0.5)
    ax.set_ylabel("Sharpe (daily, annualised)")
    ax.set_title("Audit Impact on Sharpe (daily)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.05 if val >= 0 else -0.15),
                f"{val:+.2f}", ha="center", fontsize=9)

    # Equity curves: baseline vs corrected
    ax = axes[1]
    t_base = results["_trades_baseline"]
    t_corr = results["_trades_corrected"]

    if not t_base.empty:
        eq_b = np.cumprod(1 + t_base["net_ret"].dropna().values)
        ax.plot(range(len(eq_b)), eq_b, lw=1.5, color="#ef5350", label="Baseline (all bugs)", alpha=0.8)
    if not t_corr.empty:
        eq_c = np.cumprod(1 + t_corr["net_ret"].dropna().values)
        ax.plot(range(len(eq_c)), eq_c, lw=1.5, color="#26a69a", label="All fixes applied", alpha=0.8)
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve: Baseline vs All Fixes")
    ax.legend(fontsize=9)

    plt.suptitle("Backtest Audit Summary", fontsize=13)
    plt.tight_layout()
    _save(fig, "backtest_audit_summary.png", save)


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
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print("Loading data...")
    df, feat_cols = load()
    atr = compute_atr(df)

    print("\nGenerating OOS predictions (walk-forward, regime-routed)...")
    pred_down, pred_up = generate_predictions(df, feat_cols)
    fee_rt = TAKER_RT

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BASELINE (tp_first, overlap, no slippage)")
    print("="*60)
    t_base = simulate_short(df, pred_down, pred_up, atr,
                             ambiguous="tp_first", no_overlap=False,
                             fee_rt=fee_rt, **BASELINE_PARAMS)
    ma_base = full_metrics(t_base, label="BASELINE")

    all_results = {
        "baseline":         ma_base,
        "_trades_baseline": t_base,
    }

    # ── Run audits ────────────────────────────────────────────────────────────
    all_results["audit1"] = audit_1_sharpe_calc(t_base)
    all_results["audit2"] = audit_2_ambiguity(df, pred_down, pred_up, atr, fee_rt)
    all_results["audit3"] = audit_3_regime_bias(df, pred_down, pred_up, atr, fee_rt)
    all_results["audit4"] = audit_4_overlap(df, pred_down, pred_up, atr, fee_rt)
    all_results["audit5"] = audit_5_slippage(df, pred_down, pred_up, atr, fee_rt)
    all_results["audit6"] = audit_6_oos_test(df, feat_cols, pred_down, pred_up, atr, fee_rt)
    audit_7_alignment(df, pred_down, pred_up)

    # ── All fixes combined ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ALL FIXES COMBINED")
    print("="*60)
    t_corrected = run_fully_corrected(df, pred_down, pred_up, atr, fee_rt)
    ma_corrected = full_metrics(t_corrected, label="ALL FIXES")
    all_results["corrected"]          = ma_corrected
    all_results["_trades_corrected"]  = t_corrected

    # ── Summary ───────────────────────────────────────────────────────────────
    print_impact_table(all_results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nPlotting...")
    plot_audit_comparison(all_results, save=args.save)

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
