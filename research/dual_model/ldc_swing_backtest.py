"""
LDC swing-trade backtest (Path 2, user 2026-05-12).

Honours jdehorty's intended exit logic — "Use Dynamic Exits: ON".

For each entry signal (start_long_trade / start_short_trade), simulate
holding until ONE of these closes the trade:
  1. Reverse entry signal (start_short_trade closes a long, vice versa).
  2. Dynamic exit:  bearish cross (yhat1 crosses below yhat2) closes a long;
                    bullish cross (yhat1 crosses above yhat2) closes a short.
  3. Max hold cap (safety — default 168h = 7 days).

NO TP/SL.  Pure trend-following swing.

Compare 4 strategies:
  A) Pure LDC swing (jdehorty as-is) — entries + dynamic exits
  B) Pure LDC swing without dynamic exits (only reverse signal closes)
  C) v9-augmented LDC swing (LONG: take only if v9 P_long >= threshold;
                              SHORT: take only if v9 P_short >= threshold)
  D) v9-veto LDC swing (LONG: skip if v9 P_short >= 0.65;
                         SHORT: skip if v9 P_long >= 0.65)
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LDC_FULL = PROJECT_ROOT / "research" / "results" / "ldc_signals_full.parquet"
V9_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_v9_winrate_H8_oos.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

MAKER_COST = 0.0005     # round-trip 5 bps
TAKER_COST = 0.0013     # round-trip 13 bps
MAX_HOLD_HOURS = 168    # safety cap 1 week


def load_data():
    ldc = pd.read_parquet(LDC_FULL)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)

    klines = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    v9 = pd.read_parquet(V9_OOS)
    v9["ts"] = pd.to_datetime(v9["ts"])
    if v9["ts"].dt.tz is not None:
        v9["ts"] = v9["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    v9 = v9.set_index("ts").sort_index()
    return ldc, klines, v9


def simulate_swing(ldc: pd.DataFrame, klines: pd.DataFrame,
                    entry_filter=None,
                    exit_mode: str = "kernel_color",
                    # extra options:
                    min_hold_bars: int = 0,    # don't allow any exit before N bars
                    fixed_hold_bars: int = 0,  # exit exactly at N bars (overrides other rules)
                    cost: float = MAKER_COST,
                    max_hold_hours: int = MAX_HOLD_HOURS,
                    note: str = "") -> pd.DataFrame:
    """
    Walk bar-by-bar. Open at start_*_trade entries.

    Exit conditions (ALL modes always close on reverse signal):

    exit_mode == "kernel_color"  (user's manual rule, 2026-05-12):
        LONG  exit when is_bearish_kernel becomes True (yhat1 turns down)
        SHORT exit when is_bullish_kernel becomes True (yhat1 turns up)
        This is "kernel line color change" — the most aggressive trend-follow
        exit; trims trade to the active trend segment.

    exit_mode == "dynamic_cross":
        LONG  exit on is_bearish_cross (yhat1 crosses below yhat2)
        SHORT exit on is_bullish_cross
        jdehorty's "Use Dynamic Exits ON" original behaviour.

    exit_mode == "none":
        Only reverse signal (start_*_trade) closes; pure swing.

    entry_filter(ts, row, direction) returns False to skip the entry.
    """
    if exit_mode not in ("kernel_color", "dynamic_cross", "none"):
        raise ValueError(f"unknown exit_mode {exit_mode}")
    common = klines.index.intersection(ldc.index)
    df = ldc.loc[common].copy()
    df["close"] = klines.loc[common, "close"]
    df = df.sort_index()

    trades = []
    position = None  # 'LONG' / 'SHORT' / None
    entry_price = None
    entry_ts = None

    ts_list = df.index.tolist()
    for i, ts in enumerate(ts_list):
        row = df.iloc[i]
        close = float(row["close"])

        if position is None:
            # Look for an entry
            wants_long = bool(row["start_long_trade"])
            wants_short = bool(row["start_short_trade"])
            if not (wants_long or wants_short):
                continue
            direction = "LONG" if wants_long else "SHORT"
            if entry_filter is not None and not entry_filter(ts, row, direction):
                continue
            position = direction
            entry_price = close
            entry_ts = ts
            continue

        # Have a position — check exit conditions
        bars_since_entry = int((ts - entry_ts).total_seconds() / 3600)
        exit_reason = None

        # fixed_hold_bars: simple N-bar exit, overrides everything
        if fixed_hold_bars > 0:
            if bars_since_entry >= fixed_hold_bars:
                exit_reason = "fixed_hold"
        else:
            past_min_hold = bars_since_entry >= min_hold_bars
            if position == "LONG":
                if bool(row["start_short_trade"]) and past_min_hold:
                    exit_reason = "reverse"
                elif exit_mode == "kernel_color" and past_min_hold and bool(row["is_bearish_kernel"]):
                    exit_reason = "kernel_flip"
                elif exit_mode == "dynamic_cross" and past_min_hold and bool(row["is_bearish_cross"]):
                    exit_reason = "dynamic"
            else:
                if bool(row["start_long_trade"]) and past_min_hold:
                    exit_reason = "reverse"
                elif exit_mode == "kernel_color" and past_min_hold and bool(row["is_bullish_kernel"]):
                    exit_reason = "kernel_flip"
                elif exit_mode == "dynamic_cross" and past_min_hold and bool(row["is_bullish_cross"]):
                    exit_reason = "dynamic"

        held_hours = (ts - entry_ts).total_seconds() / 3600
        if exit_reason is None and held_hours >= max_hold_hours:
            exit_reason = "timeout"

        if exit_reason is None:
            continue

        # Close current
        if position == "LONG":
            gross = close / entry_price - 1.0
        else:
            gross = -(close / entry_price - 1.0)
        net = gross - cost
        trades.append({
            "entry_ts": entry_ts, "exit_ts": ts,
            "direction": position, "entry_price": entry_price,
            "exit_price": close, "gross_pct": gross,
            "net_pct": net, "win": int(gross > 0),
            "held_hours": int(held_hours), "exit_reason": exit_reason,
        })

        # If reverse signal closed us, immediately open opposite (jdehorty's behavior)
        if exit_reason == "reverse":
            new_dir = "SHORT" if position == "LONG" else "LONG"
            if entry_filter is not None and not entry_filter(ts, row, new_dir):
                position = None
                entry_price = None
                entry_ts = None
            else:
                position = new_dir
                entry_price = close
                entry_ts = ts
        else:
            position = None
            entry_price = None
            entry_ts = None

    # Close any leftover open trade at last bar
    if position is not None:
        last_row = df.iloc[-1]
        last_close = float(last_row["close"])
        last_ts = df.index[-1]
        if position == "LONG":
            gross = last_close / entry_price - 1.0
        else:
            gross = -(last_close / entry_price - 1.0)
        net = gross - cost
        trades.append({
            "entry_ts": entry_ts, "exit_ts": last_ts,
            "direction": position, "entry_price": entry_price,
            "exit_price": last_close, "gross_pct": gross,
            "net_pct": net, "win": int(gross > 0),
            "held_hours": int((last_ts - entry_ts).total_seconds() / 3600),
            "exit_reason": "open_at_end",
        })

    return pd.DataFrame(trades)


def report(df: pd.DataFrame, label: str):
    n = len(df)
    if n == 0:
        print(f"  {label:<60} n=0")
        return None
    wr = df["win"].mean()
    net = df["net_pct"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    avg_net_bps = net.mean() * 10000
    avg_gross_bps = df["gross_pct"].mean() * 10000
    avg_hold = df["held_hours"].mean()
    wins_int = int(df["win"].sum())
    ci = binomtest(wins_int, n).proportion_ci(0.95) if n >= 5 else None
    ci_str = f"[{ci.low*100:.0f},{ci.high*100:.0f}]" if ci else ""
    print(f"  {label:<60} n={n:>3} WR={wr*100:>5.1f}% {ci_str:<10}  "
          f"net={avg_net_bps:>+7.1f}bp gross={avg_gross_bps:>+7.1f}bp  "
          f"cum={cum[-1]*100:>+6.1f}% MDD={mdd:>+5.1f}% PF={pf:>4.2f} "
          f"hold={avg_hold:.0f}h")
    return {"n": n, "wr": wr, "net_bps": avg_net_bps, "gross_bps": avg_gross_bps,
            "cum_pct": cum[-1] * 100, "mdd_pct": mdd, "pf": pf, "hold_h": avg_hold}


def main():
    ldc, klines, v9 = load_data()
    logger.info("LDC: %d bars (%s - %s)", len(ldc), ldc.index.min(), ldc.index.max())
    logger.info("v9 OOS: %d bars (%s - %s)", len(v9), v9.index.min(), v9.index.max())
    logger.info("Klines: %d bars", len(klines))

    # Join v9 P columns to ldc for filtering
    ldc["p_long_win"] = v9["p_long_win"].reindex(ldc.index)
    ldc["p_short_win"] = v9["p_short_win"].reindex(ldc.index)

    print(f"\n{'='*135}")
    print(f"LDC SWING-TRADE BACKTEST (jdehorty intended use, no TP/SL)")
    print(f"Maker cost = {MAKER_COST*10000:.0f}bp round-trip; max hold = {MAX_HOLD_HOURS}h")
    print(f"{'='*135}")
    print(f"  {'strategy':<60} {'n':>5} {'WR':>8} {'CI':<10}  "
          f"{'net':>10} {'gross':>10} {'cum%':>7} {'MDD%':>7} {'PF':>5}")
    print("-" * 135)

    # ── Exit-rule sweep: find which one matches TradingView WR 46.7% ──
    print()
    print(">> Exit-rule sweep on LDC entries (no v9 filter):")
    print()

    configs = [
        ("kernel-flip raw (instant)", dict(exit_mode="kernel_color", min_hold_bars=0)),
        ("kernel-flip min hold 4h", dict(exit_mode="kernel_color", min_hold_bars=4)),
        ("kernel-flip min hold 8h", dict(exit_mode="kernel_color", min_hold_bars=8)),
        ("kernel-flip min hold 12h", dict(exit_mode="kernel_color", min_hold_bars=12)),
        ("dynamic_cross (jdehorty default)", dict(exit_mode="dynamic_cross", min_hold_bars=0)),
        ("dynamic_cross min hold 4h", dict(exit_mode="dynamic_cross", min_hold_bars=4)),
        ("reverse signal only (none)", dict(exit_mode="none", min_hold_bars=0)),
        ("fixed 4h hold", dict(fixed_hold_bars=4)),
        ("fixed 8h hold", dict(fixed_hold_bars=8)),
        ("fixed 16h hold", dict(fixed_hold_bars=16)),
        ("fixed 24h hold", dict(fixed_hold_bars=24)),
    ]

    print(f"  {'exit rule':<38} {'n':>4} {'WR':>6}  {'net bps':>9}  {'cum%':>7}  {'MDD%':>7}  {'PF':>5}  {'hold':>6}")
    print("  " + "-" * 100)
    sweep_results = []
    for label, kw in configs:
        t = simulate_swing(ldc, klines, cost=MAKER_COST, **kw)
        if t.empty:
            print(f"  {label:<38} n=0")
            continue
        wr = t["win"].mean()
        net = t["net_pct"].values
        cum = np.cumsum(net)
        mdd = (cum - np.maximum.accumulate(cum)).min() * 100
        wins = net[net > 0]; losses = net[net < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
        avg_net_bps = net.mean() * 10000
        avg_hold = t["held_hours"].mean()
        print(f"  {label:<38} {len(t):>4} {wr*100:>5.1f}% {avg_net_bps:>+8.1f}bp "
              f"{cum[-1]*100:>+6.1f}% {mdd:>+6.1f}% {pf:>5.2f} {avg_hold:>5.1f}h")
        sweep_results.append((label, len(t), wr, avg_net_bps))

    print()
    print(">> SHORT-only (LDC LONG drags overall — split SHORT side):")
    print(f"  {'exit rule':<38} {'n':>4} {'WR':>6}  {'net bps':>9}  {'cum%':>7}  {'MDD%':>7}  {'PF':>5}  {'hold':>6}")
    print("  " + "-" * 100)
    for label, kw in configs:
        t_full = simulate_swing(ldc, klines, cost=MAKER_COST, **kw)
        if t_full.empty:
            continue
        t = t_full[t_full["direction"] == "SHORT"]
        if t.empty:
            continue
        wr = t["win"].mean()
        net = t["net_pct"].values
        cum = np.cumsum(net)
        mdd = (cum - np.maximum.accumulate(cum)).min() * 100
        wins = net[net > 0]; losses = net[net < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
        avg_net_bps = net.mean() * 10000
        avg_hold = t["held_hours"].mean()
        print(f"  {label:<38} {len(t):>4} {wr*100:>5.1f}% {avg_net_bps:>+8.1f}bp "
              f"{cum[-1]*100:>+6.1f}% {mdd:>+6.1f}% {pf:>5.2f} {avg_hold:>5.1f}h")


if __name__ == "__main__":
    main()
