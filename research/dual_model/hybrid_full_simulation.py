"""
Full end-to-end simulation of the v9+LDC hybrid strategy with user-specified
trading config — to verify edge translates into real $ profit before any
exchange integration.

User config (2026-05-12):
    Initial capital:        $1000 USDT
    Position notional:      $300 per trade
    Leverage:               20x  →  $15 margin used per trade
    Max concurrent:         1
    SL (price move):        0.3% (short: price rises 0.3% from entry)
    TP (price move):        0.5% (short: price falls 0.5% from entry)
    Maker cost:             5 bps round-trip
    Risk manager:           defaults from risk_manager.RiskConfig.from_stage('paper')

Stress test variants:
    A. Base case (50% maker fill, 5bp cost)
    B. Pessimistic cost (8bp instead of 5bp — real-world fill less efficient)
    C. WR degradation (assume future WR drops 5pp from observed)
    D. Fill rate 80% (skip 20% of signals at random)
"""
from __future__ import annotations
import sys
import json
import logging
from datetime import timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
from research.paper_trading_tpsl import _find_exit_for_signal
from indicator.risk_manager import RiskManager, RiskConfig

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

# User's trading config
INITIAL_CAPITAL = 1000.0
NOTIONAL_PER_TRADE = 300.0
LEVERAGE = 20.0
MARGIN_PER_TRADE = NOTIONAL_PER_TRADE / LEVERAGE   # $15

TP_DIST = 0.005    # 0.5% price move
SL_DIST = 0.003    # 0.3% price move
HORIZON = 8

V9_THRESHOLD = 0.65

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


def train_v9_seed42(df, feature_cols):
    """Train v9 seed=42 to get OOS predictions for hybrid cohort."""
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
        model.fit(X[tr_in], y[tr_in], eval_set=[(X[tr_val], y[tr_val])], verbose=False)
        oos_p[te_idx] = model.predict_proba(X[te_idx])[:, 1]
    return oos_p


def get_hybrid_signals(df, oos_p, ldc):
    """Return DataFrame of bars where v9 + LDC agree on SHORT."""
    mask = np.isfinite(oos_p)
    out = df.loc[mask].copy()
    out["p_short"] = oos_p[mask]
    out["ldc_signal"] = ldc["signal"].reindex(out.index).fillna(0).astype(int)
    signals = out[(out["p_short"] >= V9_THRESHOLD) & (out["ldc_signal"] == -1)].copy()
    return signals.sort_index()


def run_simulation(signals: pd.DataFrame, klines: pd.DataFrame,
                    cost_bps: float, fill_rate: float = 1.0,
                    seed: int = 42, label: str = "") -> dict:
    """Walk through signals chronologically, apply risk manager, simulate trades."""
    rm = RiskManager(RiskConfig(
        initial_equity=INITIAL_CAPITAL,
        max_risk_per_trade=0.005,    # 0.5% of equity — but we use fixed notional
        max_concurrent_positions=1,
        daily_loss_cap=0.03,
        consecutive_loss_threshold=7,
        max_drawdown=0.20,
        stage="paper",
    ))

    cost = cost_bps / 10000.0
    trades = []
    rng = np.random.default_rng(seed)
    klines_idx = klines.index

    for ts, sig_row in signals.iterrows():
        ts_naive = pd.Timestamp(ts)
        if ts_naive.tz is not None:
            ts_naive = ts_naive.tz_convert("UTC").tz_localize(None)
        ts_aware = ts_naive.tz_localize("UTC")

        # Fill rate simulation
        if rng.random() > fill_rate:
            continue  # signal skipped due to limit not filled

        # Risk gate
        allowed, reason = rm.can_open_position(ts_aware, signal_id=ts_naive.isoformat())
        if not allowed:
            trades.append({"ts": ts_naive, "blocked": True, "reason": reason})
            continue

        if ts_naive not in klines_idx:
            continue
        entry_price = float(klines.loc[ts_naive, "close"])
        future_idx = klines_idx[klines_idx > ts_naive]
        if len(future_idx) < 1:
            continue
        future = klines.loc[future_idx]

        exit_price, bars_held, reason_exit = _find_exit_for_signal(
            entry_price=entry_price, direction="DOWN",
            sl_dist=SL_DIST, rr=TP_DIST / SL_DIST,
            bars=future, timeout_bars=HORIZON,
        )

        # PnL in dollars
        gross_pct = -(exit_price / entry_price - 1.0)  # SHORT: gain when price drops
        gross_dollars = gross_pct * NOTIONAL_PER_TRADE
        cost_dollars = cost * NOTIONAL_PER_TRADE
        net_dollars = gross_dollars - cost_dollars

        # Open + close with risk_manager (size = notional, not %)
        pos_id = f"hybrid_{ts_naive.isoformat()}"
        rm.record_position_open(ts_aware, pos_id, "DOWN", entry_price, NOTIONAL_PER_TRADE)
        # When closing, convert dollars back to fraction of notional
        pnl_pct_notional = net_dollars / NOTIONAL_PER_TRADE
        exit_ts = future_idx[bars_held - 1] if bars_held > 0 else future_idx[0]
        exit_ts_aware = exit_ts.tz_localize("UTC") if exit_ts.tz is None else exit_ts
        rm.record_position_close(exit_ts_aware, pos_id,
                                  pnl_pct=pnl_pct_notional, exit_reason=reason_exit)

        trades.append({
            "ts": ts_naive, "blocked": False,
            "entry": entry_price, "exit": exit_price,
            "gross_pct": gross_pct, "net_dollars": net_dollars,
            "win": int(gross_pct > 0),
            "exit_reason": reason_exit, "bars_held": bars_held,
            "equity_after": rm.state.equity,
            "drawdown_pct": rm.state.current_drawdown * 100,
        })

    return {
        "label": label,
        "trades": pd.DataFrame(trades),
        "rm": rm,
    }


def report(sim_result: dict):
    label = sim_result["label"]
    rm = sim_result["rm"]
    trades_df = sim_result["trades"]

    print(f"\n{'='*100}")
    print(f"SCENARIO: {label}")
    print(f"{'='*100}")
    print(f"  Initial equity:   ${INITIAL_CAPITAL:.2f}")
    print(f"  Final equity:     ${rm.state.equity:.2f}")
    print(f"  Total return:     {(rm.state.equity/INITIAL_CAPITAL - 1)*100:+.2f}%")
    print(f"  Total $:          ${rm.state.equity - INITIAL_CAPITAL:+.2f}")
    print(f"  High-water mark:  ${rm.state.high_water_mark:.2f}")

    filled = trades_df[~trades_df["blocked"]] if "blocked" in trades_df.columns else trades_df
    blocked = trades_df[trades_df["blocked"]] if "blocked" in trades_df.columns else pd.DataFrame()
    if not filled.empty:
        print(f"  Total signals:    {len(trades_df)}")
        print(f"  Blocked by risk:  {len(blocked)}")
        print(f"  Filled trades:    {len(filled)}")
        wr = filled["win"].mean()
        wins = int(filled["win"].sum())
        losses = len(filled) - wins
        avg_win = filled[filled["win"] == 1]["net_dollars"].mean() if wins > 0 else 0
        avg_loss = filled[filled["win"] == 0]["net_dollars"].mean() if losses > 0 else 0
        total_net = filled["net_dollars"].sum()
        print(f"  Win rate:         {wr*100:.1f}% ({wins}W / {losses}L)")
        print(f"  Avg win:          ${avg_win:+.2f}")
        print(f"  Avg loss:         ${avg_loss:+.2f}")
        print(f"  Total $ PnL:      ${total_net:+.2f}")

        # Equity curve stats
        eq = filled["equity_after"].values
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max
        max_dd_pct = dd.min() * 100
        max_dd_dollars = (eq - running_max).min()
        print(f"  Max drawdown:     {max_dd_pct:.2f}% (${max_dd_dollars:+.2f})")

        # Monthly breakdown
        filled["month"] = pd.to_datetime(filled["ts"]).dt.to_period("M")
        monthly = filled.groupby("month").agg(
            n=("ts", "count"),
            wins=("win", "sum"),
            wr=("win", "mean"),
            net=("net_dollars", "sum"),
        )
        monthly["wr_pct"] = monthly["wr"] * 100
        print(f"\n  Per-month PnL:")
        print(f"  {'month':<10} {'n':>4} {'wins':>5} {'WR':>7} {'PnL$':>9}")
        for m, row in monthly.iterrows():
            print(f"  {str(m):<10} {int(row['n']):>4} {int(row['wins']):>5} "
                  f"{row['wr_pct']:>5.1f}% {row['net']:>+8.2f}")

    # Risk events
    risk_events = [e for e in rm.events if e.event_type in ("halt_set", "gate_block")]
    halts = [e for e in risk_events if e.event_type == "halt_set"]
    blocks = [e for e in risk_events if e.event_type == "gate_block"]
    print(f"\n  Risk events:")
    print(f"    Halts triggered: {len(halts)}")
    for h in halts[:5]:
        print(f"      - {h.ts}: {h.rule or h.reason}")
    print(f"    Gate blocks:     {len(blocks)}")


def main():
    logger.info("Loading data...")
    df, feature_cols, ldc, klines = load_all()
    logger.info("Training v9 seed=42 (~5 min)...")
    oos_p = train_v9_seed42(df, feature_cols)
    signals = get_hybrid_signals(df, oos_p, ldc)
    logger.info("Hybrid signals: n=%d, range %s ~ %s",
                 len(signals), signals.index.min(), signals.index.max())

    print(f"\n{'#'*100}")
    print(f"User config: ${INITIAL_CAPITAL:.0f} capital, ${NOTIONAL_PER_TRADE:.0f}/trade, "
          f"{LEVERAGE:.0f}x leverage (margin=${MARGIN_PER_TRADE:.2f})")
    print(f"TP={TP_DIST*100:.1f}% / SL={SL_DIST*100:.1f}% / horizon={HORIZON}h")
    print(f"Trading window: {signals.index.min()} ~ {signals.index.max()}")
    print(f"{'#'*100}")

    # A. Base case: 5bp cost, 100% fill
    sim_a = run_simulation(signals, klines, cost_bps=5, fill_rate=1.0,
                             label="A. BASE CASE — 5bp cost, 100% fill")
    report(sim_a)

    # B. Pessimistic cost: 8bp
    sim_b = run_simulation(signals, klines, cost_bps=8, fill_rate=1.0,
                             label="B. PESSIMISTIC COST — 8bp (real maker fill imperfect)")
    report(sim_b)

    # C. Pessimistic fill: 80% fill rate (skip 20% randomly)
    sim_c = run_simulation(signals, klines, cost_bps=5, fill_rate=0.8, seed=42,
                             label="C. PESSIMISTIC FILL — 80% fill rate (5bp cost)")
    report(sim_c)

    # D. Combined pessimistic: 8bp + 80% fill
    sim_d = run_simulation(signals, klines, cost_bps=8, fill_rate=0.8, seed=42,
                             label="D. COMBINED PESSIMISTIC — 8bp cost + 80% fill")
    report(sim_d)

    # Summary
    print(f"\n{'='*100}")
    print("STRESS-TEST SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Scenario':<50} {'Final $':>12} {'Return %':>10} {'MDD %':>8}")
    print("-" * 80)
    for sim in [sim_a, sim_b, sim_c, sim_d]:
        rm = sim["rm"]
        ret = (rm.state.equity / INITIAL_CAPITAL - 1) * 100
        # Recompute MDD from trade equity curve
        filled = sim["trades"][~sim["trades"]["blocked"]] if "blocked" in sim["trades"].columns else sim["trades"]
        if not filled.empty:
            eq = filled["equity_after"].values
            running_max = np.maximum.accumulate(eq)
            mdd = ((eq - running_max) / running_max).min() * 100
        else:
            mdd = 0
        print(f"  {sim['label']:<50} ${rm.state.equity:>+10.2f} {ret:>+8.2f}% {mdd:>+7.2f}%")


if __name__ == "__main__":
    main()
