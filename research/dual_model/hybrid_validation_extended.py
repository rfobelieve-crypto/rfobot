"""
Extended validation: retrain frequency sensitivity + multiple holdout windows.

Test A: Retrain frequency sensitivity
    Train v9 once at fixed point (60% mark), apply to rest of data WITHOUT
    retraining.  Compute hybrid cohort WR in 1-week buckets to see when
    edge degrades.  Production retrain frequency should be < (degradation
    time).

Test C: Multiple holdout windows
    5 different chronological 80/20 splits.  For each:
      - Train on first 80%
      - Apply (no retrain) to last 20%
      - Compute hybrid WR
    If most/all holdouts fail, edge requires frequent retrain (confirmed).
    If only April fails, April was regime-specific outlier.

Output decides:
  All holdouts pass → 1 holdout fail was outlier, ready for testnet
  Some holdouts pass → need retrain pipeline, frequency = degradation threshold
  All fail → edge dependent on continuous retrain, big production challenge
"""
from __future__ import annotations
import sys
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
LABEL_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "labels_winrate_TP50_SL30_H8.parquet"
FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)
LDC_FULL = PROJECT_ROOT / "research" / "results" / "ldc_signals_full.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

TP_DIST = 0.005
SL_DIST = 0.003
MAKER_COST = 0.0005
HORIZON = 8
V9_THRESHOLD = 0.65
NOTIONAL = 300.0
MAKER_BE_WR = 0.414

BASE_PARAMS = dict(
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
    early_stopping_rounds=30,
    objective="binary:logistic", eval_metric="auc",
)


def load_all():
    f = pd.read_parquet(FEATURES_CACHE)
    l = pd.read_parquet(LABEL_CACHE)
    if f.index.tz is not None:
        f.index = f.index.tz_convert("UTC").tz_localize(None)
    if l.index.tz is not None:
        l.index = l.index.tz_convert("UTC").tz_localize(None)
    df = f.join(l[["y_short_win"]], how="inner").dropna(subset=["y_short_win"])
    with open(FEATURE_COLS_FILE) as fh:
        feature_cols = json.load(fh)
    feature_cols = [c for c in feature_cols if c in df.columns]
    ldc = pd.read_parquet(LDC_FULL)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)
    klines = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    return df, feature_cols, ldc, klines


def train_once(X_tr, y_tr, seed=42):
    cut = int(len(X_tr) * 0.85)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    params = dict(BASE_PARAMS, random_state=seed,
                   scale_pos_weight=(n_neg / max(n_pos, 1)))
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr[:cut], y_tr[:cut],
              eval_set=[(X_tr[cut:], y_tr[cut:])], verbose=False)
    return model


def ohlc_short_pnl(ts_list, klines, notional, cost):
    rows = []
    klines_idx = klines.index
    for ts in ts_list:
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in klines_idx:
            continue
        entry = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) < 1:
            continue
        future = klines.loc[future_idx]
        ep, bh, reason = _find_exit_for_signal(
            entry_price=entry, direction="DOWN",
            sl_dist=SL_DIST, rr=TP_DIST / SL_DIST,
            bars=future, timeout_bars=HORIZON,
        )
        gross_pct = -(ep / entry - 1.0)
        net_dollar = (gross_pct - cost) * notional
        rows.append({
            "ts": ts, "gross_pct": gross_pct,
            "net_dollar": net_dollar, "win": int(gross_pct > 0),
        })
    return pd.DataFrame(rows)


# ── Test C: Multiple holdout windows ───────────────────────────────────────

def test_c_multi_holdouts(df, feature_cols, ldc, klines):
    """5 chronological 80/20 splits at different cut points."""
    print(f"\n{'='*100}")
    print(f"TEST C: Multiple holdout windows")
    print(f"{'='*100}")
    print(f"  {'split':<25} {'train':<25} {'holdout':<25} {'AUC':>6} {'n':>4} {'WR':>7} {'PnL$':>8}")
    print("  " + "-" * 100)

    # 5 different split points (40%, 50%, 60%, 70%, 80%) — vary holdout location
    results = []
    for split_pct in [0.40, 0.50, 0.60, 0.70, 0.80]:
        split_idx = int(len(df) * split_pct)
        # Holdout = next 20% of total length, capped at end
        holdout_end = min(split_idx + int(len(df) * 0.20), len(df))
        train_df = df.iloc[:split_idx]
        holdout_df = df.iloc[split_idx:holdout_end]
        if len(holdout_df) < 50:
            continue

        X_tr = train_df[feature_cols].values.astype(np.float32)
        y_tr = train_df["y_short_win"].values.astype(int)
        X_ho = holdout_df[feature_cols].values.astype(np.float32)
        y_ho = holdout_df["y_short_win"].values.astype(int)

        model = train_once(X_tr, y_tr)
        p_ho = model.predict_proba(X_ho)[:, 1]
        auc = roc_auc_score(y_ho, p_ho)

        holdout_df = holdout_df.copy()
        holdout_df["p_short"] = p_ho
        holdout_df["ldc_signal"] = ldc["signal"].reindex(holdout_df.index).fillna(0).astype(int)
        hybrid = holdout_df[(holdout_df["p_short"] >= V9_THRESHOLD)
                              & (holdout_df["ldc_signal"] == -1)]

        if len(hybrid) >= 3:
            trades = ohlc_short_pnl(hybrid.index, klines, NOTIONAL, MAKER_COST)
            wins = int(trades["win"].sum())
            n = len(trades)
            wr = wins / n if n > 0 else 0
            pnl = trades["net_dollar"].sum() if n > 0 else 0
            ci_str = ""
            if n >= 5:
                ci = binomtest(wins, n).proportion_ci(0.95)
                ci_str = f" CI[{ci.low*100:.0f},{ci.high*100:.0f}]"
            print(f"  {split_pct*100:>3.0f}%  "
                  f"{train_df.index[0].date()}~{train_df.index[-1].date()}  "
                  f"{holdout_df.index[0].date()}~{holdout_df.index[-1].date()}  "
                  f"{auc:>5.3f}  {n:>4} {wr*100:>5.1f}%{ci_str:<14} ${pnl:>+6.2f}")
            results.append({
                "split_pct": split_pct, "auc": auc,
                "n": n, "wr": wr, "pnl": pnl,
                "holdout_start": holdout_df.index[0],
                "holdout_end": holdout_df.index[-1],
            })
        else:
            print(f"  {split_pct*100:>3.0f}%  "
                  f"{train_df.index[0].date()}~{train_df.index[-1].date()}  "
                  f"{holdout_df.index[0].date()}~{holdout_df.index[-1].date()}  "
                  f"{auc:>5.3f}  n={len(hybrid)} too few")

    # Aggregate verdict
    if results:
        n_above_be = sum(1 for r in results
                          if r["n"] >= 5 and r["wr"] > MAKER_BE_WR)
        n_with_data = sum(1 for r in results if r["n"] >= 5)
        n_profit = sum(1 for r in results if r["pnl"] > 0)
        print(f"\n  Splits with WR > maker BE (41%): {n_above_be}/{n_with_data}")
        print(f"  Splits profitable:                 {n_profit}/{len(results)}")
        if n_above_be >= n_with_data * 0.6:
            print(f"  TEST C verdict: PASS  (>= 60% of splits show edge)")
        elif n_above_be >= n_with_data * 0.3:
            print(f"  TEST C verdict: MIXED  (30-60% of splits show edge, time-dependent)")
        else:
            print(f"  TEST C verdict: FAIL  (< 30% of splits show edge, edge unstable)")
    return results


# ── Test A: Retrain frequency sensitivity ──────────────────────────────────

def test_a_decay_curve(df, feature_cols, ldc, klines):
    """Train once at 60% mark, see WR degradation over next 40% data.
    Bucket the holdout by week."""
    print(f"\n{'='*100}")
    print(f"TEST A: Edge decay curve — train once, observe WR by week")
    print(f"{'='*100}")

    split_idx = int(len(df) * 0.60)
    train_df = df.iloc[:split_idx]
    forward_df = df.iloc[split_idx:]
    print(f"  Train period:    {train_df.index[0].date()} ~ {train_df.index[-1].date()}")
    print(f"  Forward period:  {forward_df.index[0].date()} ~ {forward_df.index[-1].date()}")
    print(f"  Train bars: {len(train_df)}, Forward bars: {len(forward_df)}")

    X_tr = train_df[feature_cols].values.astype(np.float32)
    y_tr = train_df["y_short_win"].values.astype(int)
    X_fw = forward_df[feature_cols].values.astype(np.float32)

    model = train_once(X_tr, y_tr)
    p_fw = model.predict_proba(X_fw)[:, 1]

    forward_df = forward_df.copy()
    forward_df["p_short"] = p_fw
    forward_df["ldc_signal"] = ldc["signal"].reindex(forward_df.index).fillna(0).astype(int)
    hybrid = forward_df[(forward_df["p_short"] >= V9_THRESHOLD)
                          & (forward_df["ldc_signal"] == -1)]
    trades = ohlc_short_pnl(hybrid.index, klines, NOTIONAL, MAKER_COST)

    if trades.empty:
        print("  No hybrid signals in forward period.")
        return

    # Bucket by week
    trades["ts"] = pd.to_datetime(trades["ts"])
    trades["week_offset"] = ((trades["ts"] - trades["ts"].min()).dt.total_seconds()
                              / (7 * 86400)).astype(int)
    print(f"\n  Forward week-by-week WR (model NOT retrained):")
    print(f"  {'week#':>5} {'date_range':<25} {'n':>4} {'wins':>5} {'WR':>7} {'PnL$':>8}")
    print("  " + "-" * 70)

    for wk, g in trades.groupby("week_offset"):
        n = len(g)
        wins = int(g["win"].sum())
        wr = wins / n if n > 0 else 0
        pnl = g["net_dollar"].sum()
        start = g["ts"].min().date()
        end = g["ts"].max().date()
        ci_str = ""
        if n >= 5:
            ci = binomtest(wins, n).proportion_ci(0.95)
            ci_str = f" CI[{ci.low*100:.0f},{ci.high*100:.0f}]"
        print(f"  {wk:>5} {str(start):<10} ~ {str(end):<10}  "
              f"{n:>4} {wins:>5} {wr*100:>5.1f}%{ci_str:<14} ${pnl:>+6.2f}")

    # Look for degradation: when does WR drop below BE?
    weekly = trades.groupby("week_offset").agg(
        n=("win", "count"), wr=("win", "mean")
    )
    weekly_with_data = weekly[weekly["n"] >= 3]
    if len(weekly_with_data) >= 2:
        print(f"\n  Decay summary:")
        print(f"  Avg WR weeks 0-3 (early):  "
              f"{weekly_with_data[weekly_with_data.index <= 3]['wr'].mean()*100:.1f}%")
        print(f"  Avg WR weeks 4+ (late):    "
              f"{weekly_with_data[weekly_with_data.index >= 4]['wr'].mean()*100:.1f}%")
        # First week WR drops below BE?
        below_be_weeks = weekly_with_data[weekly_with_data["wr"] < MAKER_BE_WR]
        if not below_be_weeks.empty:
            print(f"  First week WR drops below {MAKER_BE_WR*100:.0f}% BE: week {below_be_weeks.index[0]}")
        else:
            print(f"  No week below {MAKER_BE_WR*100:.0f}% BE — edge stable")


def main():
    df, feature_cols, ldc, klines = load_all()
    logger.info("Data: %d bars, %d features", len(df), len(feature_cols))

    print(f"\n{'#'*100}")
    print(f"EXTENDED VALIDATION: retrain frequency + multiple holdouts")
    print(f"{'#'*100}")

    test_c_multi_holdouts(df, feature_cols, ldc, klines)
    test_a_decay_curve(df, feature_cols, ldc, klines)


if __name__ == "__main__":
    main()
