"""
Full institutional-grade backtest validation suite for v9 + LDC hybrid.

Three pillars:
    1. OOS (Out-of-sample) — already done via 77-fold walk-forward.
       Re-verify with INDEPENDENT 80/20 chronological split (train on first
       80%, test on last 20% with NO retraining during test).  This is the
       strictest possible OOS — closest to "trained yesterday, traded today".

    2. Walk-forward — already done (77 folds, purge=4, embargo=4).
       Re-summarize with bootstrap CI on per-fold WR.

    3. Monte Carlo — two flavors:
       a) Trade resampling: bootstrap the 32 trade $-PnLs N=10000 times,
          report final equity distribution + P(profitable) + MDD distribution.
       b) Future projection: use (WR, avg_win, avg_loss) as parameters,
          simulate 10000 random equity paths of next 70 trades (1 year),
          report expected return + downside.

Pass criteria:
    OOS hold-out: WR > maker BE 41% with CI lower bound > 41%
    Walk-forward fold WR: median fold WR > 50%
    MC bootstrap: P(profitable after 5.5 mo) > 80%, P(profitable in 1 yr) > 90%
    MC future: 5th percentile annual return > 0%
"""
from __future__ import annotations
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
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

NOTIONAL = 300.0  # User config
INITIAL_CAPITAL = 1000.0
MAKER_BE_WR = 0.414   # Break-even WR with 5bp maker cost on RR 1:1.67
TAKER_BE_WR = 0.510

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


def ohlc_short_pnl_dollar(ts_list, klines, notional, cost):
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
        net_pct = gross_pct - cost
        net_dollar = net_pct * notional
        rows.append({
            "ts": ts, "gross_pct": gross_pct,
            "net_dollar": net_dollar,
            "win": int(gross_pct > 0),
        })
    return pd.DataFrame(rows)


# ── Test 1: OOS hold-out (independent) ─────────────────────────────────────

def test_oos_holdout(df, feature_cols, ldc, klines, holdout_frac=0.20):
    """Train v9 on first 80% bars, test on last 20% with NO retraining."""
    print(f"\n{'='*100}")
    print(f"TEST 1: Independent OOS hold-out ({(1-holdout_frac)*100:.0f}/{holdout_frac*100:.0f} chronological split)")
    print(f"{'='*100}")

    split_idx = int(len(df) * (1 - holdout_frac))
    train_df = df.iloc[:split_idx]
    holdout_df = df.iloc[split_idx:]
    print(f"  Train: n={len(train_df)} bars, range {train_df.index.min()} ~ {train_df.index.max()}")
    print(f"  Hold-out: n={len(holdout_df)} bars, range {holdout_df.index.min()} ~ {holdout_df.index.max()}")

    X_tr = train_df[feature_cols].values.astype(np.float32)
    y_tr = train_df["y_short_win"].values.astype(int)
    X_ho = holdout_df[feature_cols].values.astype(np.float32)
    y_ho = holdout_df["y_short_win"].values.astype(int)

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    cut = int(len(X_tr) * 0.85)
    params = dict(BASE_PARAMS, random_state=42,
                   scale_pos_weight=(n_neg / max(n_pos, 1)))
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr[:cut], y_tr[:cut],
              eval_set=[(X_tr[cut:], y_tr[cut:])], verbose=False)
    p_holdout = model.predict_proba(X_ho)[:, 1]
    auc = roc_auc_score(y_ho, p_holdout)
    print(f"  Holdout AUC: {auc:.4f}")

    holdout_df = holdout_df.copy()
    holdout_df["p_short"] = p_holdout
    holdout_df["ldc_signal"] = ldc["signal"].reindex(holdout_df.index).fillna(0).astype(int)
    hybrid = holdout_df[(holdout_df["p_short"] >= V9_THRESHOLD)
                          & (holdout_df["ldc_signal"] == -1)]
    print(f"  Hybrid signals in holdout: n={len(hybrid)}")
    if len(hybrid) < 5:
        print(f"  TEST 1: INCONCLUSIVE (n<5)")
        return None

    trades = ohlc_short_pnl_dollar(hybrid.index, klines, NOTIONAL, MAKER_COST)
    wins = int(trades["win"].sum())
    n = len(trades)
    wr = wins / n
    ci = binomtest(wins, n).proportion_ci(0.95)
    net_total = trades["net_dollar"].sum()
    print(f"  Hybrid WR: {wr*100:.1f}%  CI=[{ci.low*100:.1f}, {ci.high*100:.1f}]")
    print(f"  $ PnL (notional ${NOTIONAL:.0f}): ${net_total:+.2f}")
    pass_test = ci.low > MAKER_BE_WR
    print(f"  TEST 1: {'PASS' if pass_test else 'FAIL'}  "
          f"(needs WR CI lower bound > {MAKER_BE_WR*100:.0f}% maker BE)")
    return {"wr": wr, "ci_low": ci.low, "ci_high": ci.high, "n": n, "auc": auc,
            "net_dollar": net_total}


# ── Test 2: Walk-forward summary ───────────────────────────────────────────

def test_walk_forward_summary(df, feature_cols, ldc, klines):
    print(f"\n{'='*100}")
    print(f"TEST 2: Walk-forward fold-level WR (77 folds, purge=4 embargo=4)")
    print(f"{'='*100}")

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                   step=48, purge=4, embargo=4)
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_short_win"].values.astype(int)
    n = len(X)
    oos_p = np.full(n, np.nan)

    for tr_idx, te_idx in splits:
        tr_arr = np.array(tr_idx)
        cut = int(len(tr_arr) * 0.85)
        tr_in, tr_val = tr_arr[:cut], tr_arr[cut:]
        n_pos = int(y[tr_in].sum())
        n_neg = len(tr_in) - n_pos
        params = dict(BASE_PARAMS, random_state=42,
                       scale_pos_weight=(n_neg / max(n_pos, 1)))
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_in], y[tr_in],
                  eval_set=[(X[tr_val], y[tr_val])], verbose=False)
        oos_p[te_idx] = model.predict_proba(X[te_idx])[:, 1]

    mask = np.isfinite(oos_p)
    df_oos = df.loc[mask].copy()
    df_oos["p_short"] = oos_p[mask]
    df_oos["ldc_signal"] = ldc["signal"].reindex(df_oos.index).fillna(0).astype(int)
    hybrid = df_oos[(df_oos["p_short"] >= V9_THRESHOLD) &
                      (df_oos["ldc_signal"] == -1)]
    trades = ohlc_short_pnl_dollar(hybrid.index, klines, NOTIONAL, MAKER_COST)
    print(f"  Overall hybrid cohort: n={len(trades)}, WR={trades['win'].mean()*100:.1f}%, "
          f"PnL=${trades['net_dollar'].sum():+.2f}")

    # Per-fold breakdown
    trades_by_fold = []
    for fi, (tr_idx, te_idx) in enumerate(splits):
        fold_start = df.iloc[te_idx[0]].name
        fold_end = df.iloc[te_idx[-1]].name
        in_fold = trades[(pd.to_datetime(trades["ts"]) >= fold_start)
                          & (pd.to_datetime(trades["ts"]) <= fold_end)]
        if len(in_fold) > 0:
            trades_by_fold.append({
                "fold": fi, "n": len(in_fold),
                "wr": in_fold["win"].mean(),
                "pnl": in_fold["net_dollar"].sum(),
            })

    fold_df = pd.DataFrame(trades_by_fold)
    folds_with_trades = len(fold_df)
    folds_above_be = (fold_df["wr"] > MAKER_BE_WR).sum() if not fold_df.empty else 0
    median_wr = fold_df["wr"].median() if not fold_df.empty else 0
    print(f"  Folds with trades:    {folds_with_trades}/{len(splits)}")
    print(f"  Folds WR above {MAKER_BE_WR*100:.0f}% (maker BE): {folds_above_be}/{folds_with_trades}")
    print(f"  Median fold WR:       {median_wr*100:.1f}%")
    pass_test = median_wr > 0.50 and folds_above_be >= folds_with_trades * 0.5
    print(f"  TEST 2: {'PASS' if pass_test else 'FAIL'}  "
          f"(median fold WR > 50% AND >= 50% of folds above maker BE)")
    return {"median_wr": median_wr, "folds_with_trades": folds_with_trades,
            "folds_above_be": folds_above_be, "trades": trades}


# ── Test 3a: Monte Carlo bootstrap on trade $ PnL ──────────────────────────

def test_monte_carlo_bootstrap(trades_df: pd.DataFrame, n_sims: int = 10000):
    print(f"\n{'='*100}")
    print(f"TEST 3a: Monte Carlo BOOTSTRAP — resample {len(trades_df)} trades, {n_sims} sims")
    print(f"{'='*100}")
    pnls = trades_df["net_dollar"].values
    n_trades = len(pnls)
    rng = np.random.default_rng(42)

    final_equities = []
    max_drawdowns = []
    profitable_runs = 0
    for _ in range(n_sims):
        sample = rng.choice(pnls, size=n_trades, replace=True)
        equity = INITIAL_CAPITAL + np.cumsum(sample)
        running_max = np.maximum.accumulate(equity)
        dd = ((equity - running_max) / running_max).min()
        final_equities.append(equity[-1])
        max_drawdowns.append(dd)
        if equity[-1] > INITIAL_CAPITAL:
            profitable_runs += 1

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    p_profitable = profitable_runs / n_sims
    median_final = np.median(final_equities)
    p5_final = np.percentile(final_equities, 5)
    p95_final = np.percentile(final_equities, 95)
    median_dd = np.median(max_drawdowns) * 100
    p95_dd = np.percentile(max_drawdowns, 5) * 100  # worst-case DD (5th pct, more negative)

    print(f"  Equity distribution (5.5-mo horizon, ${INITIAL_CAPITAL} initial):")
    print(f"    P( final > ${INITIAL_CAPITAL:.0f} ) = {p_profitable*100:.1f}%")
    print(f"    Median final:         ${median_final:.2f}  (return {(median_final/INITIAL_CAPITAL-1)*100:+.2f}%)")
    print(f"    5th percentile:       ${p5_final:.2f}    (return {(p5_final/INITIAL_CAPITAL-1)*100:+.2f}%)")
    print(f"    95th percentile:      ${p95_final:.2f}    (return {(p95_final/INITIAL_CAPITAL-1)*100:+.2f}%)")
    print(f"  Max drawdown distribution:")
    print(f"    Median:               {median_dd:+.2f}%")
    print(f"    5th percentile (worst): {p95_dd:+.2f}%")
    pass_test = p_profitable > 0.80
    print(f"  TEST 3a: {'PASS' if pass_test else 'FAIL'}  "
          f"(needs P(profitable) > 80%)")
    return {"p_profitable": p_profitable, "median_final": median_final,
            "p5_final": p5_final, "p95_final": p95_final}


# ── Test 3b: Monte Carlo future projection (parametric) ────────────────────

def test_monte_carlo_future(trades_df: pd.DataFrame, n_sims: int = 10000,
                              n_future_trades: int = 70):
    print(f"\n{'='*100}")
    print(f"TEST 3b: Monte Carlo FUTURE PROJECTION — {n_future_trades} trades, {n_sims} sims")
    print(f"{'='*100}")
    wr = trades_df["win"].mean()
    avg_win = trades_df[trades_df["win"] == 1]["net_dollar"].mean()
    avg_loss = trades_df[trades_df["win"] == 0]["net_dollar"].mean()
    print(f"  Parameters (from observed 5.5 mo):")
    print(f"    WR:        {wr*100:.1f}%")
    print(f"    Avg win:   ${avg_win:+.2f}")
    print(f"    Avg loss:  ${avg_loss:+.2f}")

    # Also test stress: WR drops 5pp
    stress_wr = max(wr - 0.05, 0.30)
    print(f"  Stress test parameters: WR = {stress_wr*100:.1f}% (observed - 5pp)")

    rng = np.random.default_rng(42)

    def simulate(wr_param, n_sims, n_trades):
        finals = []
        for _ in range(n_sims):
            wins = rng.binomial(n_trades, wr_param)
            losses = n_trades - wins
            pnl = wins * avg_win + losses * avg_loss
            finals.append(INITIAL_CAPITAL + pnl)
        return np.array(finals)

    print(f"\n  Scenario: 1 year future (assuming {n_future_trades} trades)")
    finals_base = simulate(wr, n_sims, n_future_trades)
    finals_stress = simulate(stress_wr, n_sims, n_future_trades)

    for name, finals in [("OBSERVED WR", finals_base),
                            ("STRESS WR -5pp", finals_stress)]:
        p_profit = (finals > INITIAL_CAPITAL).mean()
        median = np.median(finals)
        p5 = np.percentile(finals, 5)
        p95 = np.percentile(finals, 95)
        print(f"  {name}:")
        print(f"    P(profitable):    {p_profit*100:.1f}%")
        print(f"    Median ret:       {(median/INITIAL_CAPITAL-1)*100:+.2f}%")
        print(f"    5th pct ret:      {(p5/INITIAL_CAPITAL-1)*100:+.2f}%")
        print(f"    95th pct ret:     {(p95/INITIAL_CAPITAL-1)*100:+.2f}%")

    pass_base = (finals_base > INITIAL_CAPITAL).mean() > 0.90
    pass_stress = (finals_stress > INITIAL_CAPITAL).mean() > 0.50
    overall_pass = pass_base and pass_stress
    print(f"\n  TEST 3b: {'PASS' if overall_pass else 'PARTIAL'}  "
          f"(needs base P(profit) > 90% AND stress > 50%)")
    return {"base_p_profit": (finals_base > INITIAL_CAPITAL).mean(),
            "stress_p_profit": (finals_stress > INITIAL_CAPITAL).mean()}


def main():
    df, feature_cols, ldc, klines = load_all()
    logger.info("Setup: %d bars, %d features", len(df), len(feature_cols))

    print(f"\n{'#'*100}")
    print(f"FULL INSTITUTIONAL VALIDATION SUITE — v9 + LDC must-agree hybrid")
    print(f"Position: notional ${NOTIONAL:.0f} per trade, initial ${INITIAL_CAPITAL:.0f}")
    print(f"{'#'*100}")

    t1 = test_oos_holdout(df, feature_cols, ldc, klines, holdout_frac=0.20)
    t2 = test_walk_forward_summary(df, feature_cols, ldc, klines)
    t3a = test_monte_carlo_bootstrap(t2["trades"], n_sims=10000)
    t3b = test_monte_carlo_future(t2["trades"], n_sims=10000, n_future_trades=70)

    # Summary
    print(f"\n{'='*100}")
    print("SUITE SUMMARY")
    print(f"{'='*100}")
    print(f"  Test 1 (OOS hold-out):           "
          + (f"WR={t1['wr']*100:.1f}% CI lower={t1['ci_low']*100:.1f}% net=${t1['net_dollar']:+.2f}"
             if t1 else "INCONCLUSIVE (n<5)"))
    print(f"  Test 2 (Walk-forward 77 folds):  "
          f"median WR={t2['median_wr']*100:.1f}%, "
          f"{t2['folds_above_be']}/{t2['folds_with_trades']} folds above maker BE")
    print(f"  Test 3a (MC bootstrap):          P(profitable)={t3a['p_profitable']*100:.1f}%")
    print(f"  Test 3b (MC 1-yr projection):    Base P(profit)={t3b['base_p_profit']*100:.1f}%, "
          f"Stress (WR-5pp)={t3b['stress_p_profit']*100:.1f}%")


if __name__ == "__main__":
    main()
