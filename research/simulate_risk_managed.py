"""
Risk-managed simulation: v9 H=8 SHORT @ P>=0.65 + maker order + RiskManager gates.

Pipeline:
    For each v9 SHORT signal in chronological order:
        1. Ask RiskManager: can_open_position()? If blocked, record skip reason.
        2. If allowed, compute position size from RiskManager.position_size_notional()
        3. Simulate maker fill (limit at signal close, wait 1h)
           - If unfilled: log, no position
        4. Walk forward 8h: TP / SL / timeout
        5. Record close to RiskManager → updates equity, DD, consecutive losses
        6. Risk manager may halt for consecutive loss / daily cap / DD trigger

Outputs:
    - Equity curve
    - Trade log (entry, exit, pnl, blocked reason if blocked)
    - Risk event timeline (every halt, daily reset, gate decision)
    - Comparison: with vs without risk rules
"""
from __future__ import annotations
import sys
import argparse
import logging
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicator.risk_manager import RiskManager, RiskConfig
from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"
KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

TP_DIST = 0.005
SL_DIST = 0.003
MAKER_COST = 0.0005  # 5 bps round-trip


def maker_fill_short(klines: pd.DataFrame, ts: pd.Timestamp,
                      limit_price: float, wait_hours: int) -> tuple:
    """Return (fill_bar, fill_price) or (None, None) if not filled."""
    klines_idx = klines.index
    wait_idx = klines_idx[(klines_idx > ts)
                            & (klines_idx <= ts + pd.Timedelta(hours=wait_hours))]
    for fb in wait_idx:
        if klines.loc[fb, "high"] >= limit_price:
            return fb, limit_price
    return None, None


def simulate_short_with_risk(signals: pd.DataFrame, klines: pd.DataFrame,
                              rm: RiskManager, horizon_hours: int = 8,
                              wait_hours: int = 1) -> pd.DataFrame:
    """Walk signals chronologically, apply risk gates, simulate maker fill + TP/SL."""
    rows = []
    klines_idx = klines.index

    for ts, _ in signals.iterrows():
        ts_naive = pd.Timestamp(ts)
        if ts_naive.tz is not None:
            ts_naive = ts_naive.tz_convert("UTC").tz_localize(None)
        if ts_naive not in klines_idx:
            continue
        ts_aware = ts_naive.tz_localize("UTC")

        # Risk gate
        allowed, reason = rm.can_open_position(ts_aware, signal_id=ts_naive.isoformat())
        if not allowed:
            rows.append({"ts": ts_naive, "filled": False, "blocked": True,
                          "block_reason": reason, "net_ret": 0.0})
            continue

        close_at_signal = float(klines.loc[ts_naive, "close"])
        limit_price = close_at_signal  # offset=0
        fill_bar, fill_price = maker_fill_short(klines, ts_naive,
                                                  limit_price, wait_hours)
        if fill_bar is None:
            rows.append({"ts": ts_naive, "filled": False, "blocked": False,
                          "block_reason": "unfilled", "net_ret": 0.0})
            continue

        # Open position
        notional = rm.position_size_notional(SL_DIST)
        position_id = f"v9s_{ts_naive.isoformat()}"
        rm.record_position_open(ts_aware, position_id, "DOWN",
                                  fill_price, notional)

        # Walk forward TP/SL from fill_bar
        post_idx = klines_idx[klines_idx > fill_bar]
        if len(post_idx) == 0:
            continue
        post = klines.loc[post_idx]
        exit_price, bars_held, exit_reason = _find_exit_for_signal(
            entry_price=fill_price, direction="DOWN",
            sl_dist=SL_DIST, rr=TP_DIST / SL_DIST,
            bars=post, timeout_bars=horizon_hours,
        )

        # Compute PnL
        gross_pct = -(exit_price / fill_price - 1.0)
        net_pct = gross_pct - MAKER_COST

        # Close position in RM
        exit_ts = post.index[bars_held - 1] if bars_held > 0 else fill_bar
        exit_ts_aware = exit_ts.tz_localize("UTC") if exit_ts.tz is None else exit_ts
        rm.record_position_close(exit_ts_aware, position_id,
                                  pnl_pct=net_pct, exit_reason=exit_reason)

        rows.append({
            "ts": ts_naive, "fill_bar": fill_bar,
            "fill_price": fill_price, "exit_price": exit_price,
            "filled": True, "blocked": False,
            "bars_held": bars_held, "exit_reason": exit_reason,
            "gross_ret": gross_pct, "net_ret": net_pct,
            "win": int(gross_pct > 0),
            "equity_after": rm.state.equity,
            "drawdown_after": rm.state.current_drawdown,
        })

    return pd.DataFrame(rows)


def report(df: pd.DataFrame, rm: RiskManager, label: str):
    print(f"\n{'='*100}")
    print(f"{label}")
    print(f"{'='*100}")
    n_total = len(df)
    n_blocked = int(df["blocked"].sum())
    n_filled = int(df["filled"].sum())
    n_unfilled = n_total - n_blocked - n_filled
    print(f"  Total signals:    {n_total}")
    print(f"  Blocked by risk:  {n_blocked}  ({n_blocked/n_total*100:.1f}%)")
    if n_blocked > 0:
        block_reasons = df.loc[df["blocked"], "block_reason"].value_counts()
        for r, c in block_reasons.items():
            print(f"    - {r}: {c}")
    print(f"  Unfilled (no maker fill): {n_unfilled}")
    print(f"  Filled:           {n_filled}")
    if n_filled > 0:
        filled = df[df["filled"]]
        wr = filled["win"].mean()
        net_bps = filled["net_ret"].mean() * 10000
        print(f"  WR (filled):      {wr*100:.1f}%")
        print(f"  Avg net per trade: {net_bps:+.2f} bps")

    print(f"\n--- Final account state ---")
    st = rm.status()
    print(f"  Equity:           ${st['equity']:.2f}  (start ${rm.config.initial_equity:.2f})")
    print(f"  Total return:     {(st['equity']/rm.config.initial_equity - 1)*100:+.2f}%")
    print(f"  High-water mark:  ${st['high_water_mark']:.2f}")
    print(f"  Current DD:       {st['current_drawdown_pct']:.2f}%")
    print(f"  Trades:           {st['n_trades_total']} (W:{rm.state.n_wins} L:{rm.state.n_losses})")
    print(f"  Winrate:          {st['winrate']*100:.1f}%")
    print(f"  Halted?:          {st['halted']} {st['halt_reason'] or ''}")

    # Drawdown analysis from equity curve
    if n_filled > 0:
        equity_curve = df.loc[df["filled"], "equity_after"].values
        running_max = np.maximum.accumulate(equity_curve)
        dd_series = (equity_curve - running_max) / running_max
        max_dd = dd_series.min() * 100
        print(f"  Realized max DD:  {max_dd:.2f}%  (from equity curve)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--stage", type=str, default="paper")
    args = parser.parse_args()

    oos_path = RESULTS_DIR / f"direction_v9_winrate_H{args.horizon}_oos.parquet"
    oos = pd.read_parquet(oos_path)
    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    selected = oos[oos["p_short_win"] >= args.threshold].copy()
    selected["ts"] = pd.to_datetime(selected["ts"])
    if selected["ts"].dt.tz is not None:
        selected["ts"] = selected["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    selected = selected.set_index("ts").sort_index()
    logger.info("v9 SHORT @ P>=%.2f: n=%d", args.threshold, len(selected))

    # ── Run A: no risk rules (just maker fill + TP/SL) ──
    rm_no_rules = RiskManager(RiskConfig(
        stage="paper_no_rules",
        initial_equity=1000.0,
        max_risk_per_trade=0.005,
        max_concurrent_positions=999,
        daily_loss_cap=1.0,         # disable
        consecutive_loss_threshold=999,
        max_drawdown=1.0,           # disable
    ))
    df_no_rules = simulate_short_with_risk(selected, klines, rm_no_rules)
    report(df_no_rules, rm_no_rules, "RUN A: NO risk rules — baseline equity curve")

    # ── Run B: full risk rules (stage=paper) ──
    rm_with_rules = RiskManager(RiskConfig.from_stage(args.stage))
    df_with_rules = simulate_short_with_risk(selected, klines, rm_with_rules)
    report(df_with_rules, rm_with_rules,
            f"RUN B: WITH risk rules ({args.stage} stage defaults)")

    # ── Risk events from run B ──
    print(f"\n--- Risk events (run B) ---")
    halt_events = [e for e in rm_with_rules.events if e.event_type == "halt_set"]
    daily_resets = [e for e in rm_with_rules.events if e.event_type == "daily_reset"]
    print(f"  Halt events triggered: {len(halt_events)}")
    for e in halt_events:
        print(f"    - {e.ts}: {e.rule or e.reason} | payload={e.payload}")
    print(f"  Daily resets: {len(daily_resets)}")

    # Save artifacts
    out_dir = PROJECT_ROOT / "research" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_no_rules.to_csv(out_dir / "simulate_risk_norules.csv", index=False)
    df_with_rules.to_csv(out_dir / "simulate_risk_withrules.csv", index=False)
    logger.info("Saved trade logs to research/results/simulate_risk_*.csv")


if __name__ == "__main__":
    main()
