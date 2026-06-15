"""
Path 1 (Step 2): Maker-order simulation on v9 H=8 SHORT @ P>=0.65.

Binance USDT-M perp fee structure (regular user, no VIP):
    Maker: 0.0200%  (2 bps per side)
    Taker: 0.0500%  (5 bps per side)
    Round-trip:  Maker 4 bps / Taker 10 bps
    + slippage:  Maker ~0 (no spread cross) / Taker ~2 bps
    + funding:   ~1 bp avg per 4-8h hold
    Total:       Maker ~5 bps / Taker ~13 bps
    Saving:      ~8 bps per round-trip

Two simulation modes:
    A) Limit at signal-bar close (offset=0):
       Fill if subsequent bar high (short) / low (long) crosses entry close.
       Most permissive — fill rate ~95%+ but entry == taker price.
       Realistic for our needs: still save fee, slip a few bp on entry timing.

    B) Limit at offset > 0 (e.g., +5 bps above close for short):
       Better entry but lower fill rate.  Conservative variant.

Trade lifecycle:
    1. Signal at bar t close → place maker limit order
    2. Wait up to `wait_hours` for fill
    3. If filled at bar t+k: open position at limit_price, start TP/SL clock
       (TP=0.5%, SL=0.3% relative to actual fill price)
    4. Walk forward up to `horizon_hours` bars after fill, look for TP or SL touch
    5. If neither touches, exit at last bar close (timeout)

Input:
    research/results/dual_model/direction_v9_winrate_H8_oos.parquet  (v9 OOS, 3696 rows)
    market_data/raw_data/binance_klines_1h.parquet                   (OHLC)

Reports per config:
    fill_rate, n_filled, avg_net_bps (with 5 bps maker cost),
    cum_pct, MDD, WR.
"""
from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"
KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

TP_DIST = 0.005   # 0.5%
SL_DIST = 0.003   # 0.3%
MAKER_COST_BPS = 5.0    # 4 bps fees + 1 bp funding (no slippage as maker)
MAKER_COST = MAKER_COST_BPS / 10000.0
TAKER_COST_BPS = 13.0
TAKER_COST = TAKER_COST_BPS / 10000.0


def simulate_maker_short(signals: pd.DataFrame, klines: pd.DataFrame,
                          wait_hours: int, offset_bp: float,
                          horizon_hours: int) -> pd.DataFrame:
    """Simulate maker limit short orders for a list of signal timestamps."""
    rows = []
    klines_idx = klines.index
    offset = offset_bp / 10000.0

    for ts, _ in signals.iterrows():
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in klines_idx:
            continue
        close_at_signal = float(klines.loc[ts, "close"])
        # SHORT limit price: sell at close × (1 + offset)
        limit_price = close_at_signal * (1.0 + offset)

        # Find fill within wait_hours after signal bar close
        wait_idx = klines_idx[(klines_idx > ts) &
                                (klines_idx <= ts + pd.Timedelta(hours=wait_hours))]
        fill_bar = None
        for fb in wait_idx:
            if klines.loc[fb, "high"] >= limit_price:
                fill_bar = fb
                break

        if fill_bar is None:
            rows.append({
                "ts": ts, "filled": False,
                "limit_price": limit_price,
                "close_at_signal": close_at_signal,
                "net_ret": 0.0, "gross_ret": 0.0, "win": 0,
                "reason": "unfilled",
            })
            continue

        # Position opened at limit_price (= sell price) at fill_bar
        # Now walk forward up to horizon_hours from fill_bar
        post_idx = klines_idx[klines_idx > fill_bar]
        if len(post_idx) == 0:
            continue
        post = klines.loc[post_idx]

        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=limit_price,
            direction="DOWN",
            sl_dist=SL_DIST,
            rr=TP_DIST / SL_DIST,
            bars=post,
            timeout_bars=horizon_hours,
        )
        # short PnL = (entry - exit) / entry = -(exit/entry - 1)
        gross = -(exit_price / limit_price - 1.0)
        net = gross - MAKER_COST
        rows.append({
            "ts": ts, "filled": True,
            "fill_bar": fill_bar,
            "limit_price": limit_price,
            "close_at_signal": close_at_signal,
            "exit_price": exit_price,
            "bars_held": bars_held,
            "reason": reason,
            "gross_ret": gross,
            "net_ret": net,
            "win": int(gross > 0),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str):
    n_total = len(df)
    n_filled = int(df["filled"].sum())
    if n_filled == 0:
        print(f"  {label:<40} n_total={n_total}  fill_rate=0%  (no fills)")
        return
    fill_rate = n_filled / n_total
    filled = df[df["filled"]].copy()
    net = filled["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    wr = filled["win"].mean()
    avg_net = net.mean() * 10000
    avg_gross = filled["gross_ret"].mean() * 10000
    print(f"  {label:<40} "
          f"n={n_total:>3}/filled={n_filled:>3} fill={fill_rate*100:>5.1f}%  "
          f"WR={wr*100:>5.1f}%  "
          f"gross={avg_gross:>+6.1f}bp  net={avg_net:>+6.1f}bp  "
          f"cum={cum[-1]*100:>+5.1f}%  MDD={mdd:>+5.1f}%  PF={pf:>4.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    oos_path = RESULTS_DIR / f"direction_v9_winrate_H{args.horizon}_oos.parquet"
    oos = pd.read_parquet(oos_path)
    logger.info("Loaded v9 OOS H=%d (n=%d)", args.horizon, len(oos))

    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    # Select v9 SHORT signals at threshold
    selected = oos[oos["p_short_win"] >= args.threshold].copy()
    selected["ts"] = pd.to_datetime(selected["ts"])
    if selected["ts"].dt.tz is not None:
        selected["ts"] = selected["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    selected = selected.set_index("ts").sort_index()
    logger.info("v9 SHORT @ P>=%.2f: n=%d", args.threshold, len(selected))

    print(f"\n{'='*110}")
    print(f"Maker-order simulation — H={args.horizon}h SHORT @ P>={args.threshold}")
    print(f"TP={TP_DIST*100:.1f}% SL={SL_DIST*100:.1f}%  "
          f"maker cost={MAKER_COST_BPS:.0f}bp (taker baseline {TAKER_COST_BPS:.0f}bp, saving "
          f"{TAKER_COST_BPS-MAKER_COST_BPS:.0f}bp)")
    print(f"{'='*110}")
    print(f"  {'config':<40} {'n_total/filled':<18} {'fill%':>6} {'WR%':>6} "
          f"{'gross':>10} {'net_bps':>8} {'cum%':>7} {'MDD%':>7}")
    print("-" * 110)

    # Baseline: taker (force fill at close, no maker)
    # Just re-run the v7 OHLC backtest from before for comparison
    baseline_rows = []
    for ts, _ in selected.iterrows():
        if ts not in klines.index:
            continue
        entry = float(klines.loc[ts, "close"])
        future = klines.loc[klines.index > ts]
        if len(future) == 0:
            continue
        ep, bh, reason = _find_exit_for_signal(
            entry_price=entry, direction="DOWN",
            sl_dist=SL_DIST, rr=TP_DIST/SL_DIST,
            bars=future, timeout_bars=args.horizon,
        )
        gross = -(ep / entry - 1.0)
        net = gross - TAKER_COST
        baseline_rows.append({"filled": True, "gross_ret": gross, "net_ret": net,
                              "win": int(gross > 0)})
    baseline = pd.DataFrame(baseline_rows)
    report(baseline, "TAKER baseline (force fill, 13bp)")

    # Maker simulations: sweep offset + wait_hours
    for offset_bp in [0.0, 2.0, 5.0, 10.0]:
        for wait_h in [1, 2, 4]:
            df = simulate_maker_short(
                selected, klines,
                wait_hours=wait_h,
                offset_bp=offset_bp,
                horizon_hours=args.horizon,
            )
            label = f"MAKER off=+{offset_bp:>4.1f}bp wait={wait_h}h"
            report(df, label)


if __name__ == "__main__":
    main()
