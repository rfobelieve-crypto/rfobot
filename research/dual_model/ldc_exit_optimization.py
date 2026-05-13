"""
LDC exit-logic sweep — find the best dynamic-exit variant under v9-gated entries.

User observation 2026-05-13: entry has 6 confirmation gates but dynamic exit
has only 1 (single-bar kernel cross).  This asymmetry was amplified when v9
filter was added to entries on 2026-05-13 but not to exits.

Holds entry rule constant (v9 P_side >= 0.55, same as production), sweeps
four exit families:

  A) v9-confirmed cross
       Exit = dynamic_cross AND v9 P_opposite >= threshold
       Restores symmetry with entry; opposite-side v9 must agree before cutting

  B) Multi-bar kernel-relation
       Exit = yhat1 / yhat2 relation persists wrong-side for N consecutive bars
       Anti-noise — single-bar cross alone doesn't trigger

  C) Signal-aligned cross
       Exit = dynamic_cross AND ML prediction signal flipped to opposite side
       Pulls exit to entry-equivalent strictness

  D) Trailing profit lock
       Exit when retracement from peak favourable excursion > X%
       Different exit philosophy — keep winners running, cut on reversal

Also tests A+B and A+C combinations.

NOTE: this is in-sample variant selection.  Best variant chosen here must
still be confirmed by forward Stage 1 paper trading before any production
change.  CLAUDE.md mistake log 2026-04-13 (in-sample vs OOS) applies.
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LDC_FULL = PROJECT_ROOT / "research" / "results" / "ldc_signals_full.parquet"
V9_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_v9_winrate_H8_oos.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

MAKER_COST = 0.0005
MAX_HOLD_HOURS = 168
MIN_HOLD_BARS = 4
V9_ENTRY_THRESHOLD = 0.55


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LDC + v9 OOS, return single merged frame indexed by ts."""
    ldc = pd.read_parquet(LDC_FULL)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)

    klines = pd.read_parquet(KLINES)[["close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    v9 = pd.read_parquet(V9_OOS)
    v9["ts"] = pd.to_datetime(v9["ts"])
    if v9["ts"].dt.tz is not None:
        v9["ts"] = v9["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    v9 = v9.set_index("ts").sort_index()

    common = ldc.index.intersection(klines.index)
    df = ldc.loc[common].copy()
    df["close"] = klines.loc[common, "close"]
    df["p_long_win"] = v9["p_long_win"].reindex(df.index)
    df["p_short_win"] = v9["p_short_win"].reindex(df.index)
    df = df.sort_index()
    return df


# ── Entry filter (kept constant across all sweep variants) ──────────────────

def v9_entry_filter(row, direction: str) -> bool:
    """Production entry rule: v9 P_side >= 0.55, NaN treated as pass."""
    if direction == "LONG":
        p = row.get("p_long_win")
    else:
        p = row.get("p_short_win")
    if p is None or pd.isna(p):
        return True
    return float(p) >= V9_ENTRY_THRESHOLD


# ── Exit rule signatures ─────────────────────────────────────────────────────
# exit_fn(state, row, direction, bars_held) -> (should_exit: bool, reason: str|None)
# `state` is a dict the caller may mutate to track running counters / peaks.


def make_dynamic_cross_exit() -> Callable:
    """Baseline: 1-bar kernel cross (current production)."""
    def fn(state, row, direction, bars_held):
        if bars_held < MIN_HOLD_BARS:
            return False, None
        if direction == "LONG" and bool(row.get("is_bearish_cross")):
            return True, "dynamic"
        if direction == "SHORT" and bool(row.get("is_bullish_cross")):
            return True, "dynamic"
        return False, None
    return fn


def make_v9_confirmed_exit(p_threshold: float) -> Callable:
    """A: cross + v9 P_opposite >= threshold."""
    def fn(state, row, direction, bars_held):
        if bars_held < MIN_HOLD_BARS:
            return False, None
        if direction == "LONG":
            if not bool(row.get("is_bearish_cross")):
                return False, None
            p_opp = row.get("p_short_win")
            if p_opp is None or pd.isna(p_opp):
                return True, "v9_cross_naskip"
            if float(p_opp) >= p_threshold:
                return True, "v9_cross"
            return False, None
        else:
            if not bool(row.get("is_bullish_cross")):
                return False, None
            p_opp = row.get("p_long_win")
            if p_opp is None or pd.isna(p_opp):
                return True, "v9_cross_naskip"
            if float(p_opp) >= p_threshold:
                return True, "v9_cross"
            return False, None
    return fn


def make_multibar_kernel_exit(n_bars: int) -> Callable:
    """B: yhat1/yhat2 relation persists wrong-side for N consecutive bars."""
    def fn(state, row, direction, bars_held):
        if bars_held < MIN_HOLD_BARS:
            state["wrong_side_count"] = 0
            return False, None
        yhat1 = row.get("yhat1")
        yhat2 = row.get("yhat2")
        if yhat1 is None or yhat2 is None or pd.isna(yhat1) or pd.isna(yhat2):
            state["wrong_side_count"] = 0
            return False, None
        # LONG wants yhat1 > yhat2; wrong-side = yhat1 < yhat2
        if direction == "LONG":
            wrong = yhat1 < yhat2
        else:
            wrong = yhat1 > yhat2
        if wrong:
            state["wrong_side_count"] = state.get("wrong_side_count", 0) + 1
        else:
            state["wrong_side_count"] = 0
        if state["wrong_side_count"] >= n_bars:
            return True, f"multibar_{n_bars}"
        return False, None
    return fn


def make_signal_aligned_exit() -> Callable:
    """C: cross + ML signal flipped to opposite."""
    def fn(state, row, direction, bars_held):
        if bars_held < MIN_HOLD_BARS:
            return False, None
        sig = row.get("signal")
        if sig is None or pd.isna(sig):
            return False, None
        if direction == "LONG":
            if bool(row.get("is_bearish_cross")) and int(sig) == -1:
                return True, "signal_cross"
        else:
            if bool(row.get("is_bullish_cross")) and int(sig) == 1:
                return True, "signal_cross"
        return False, None
    return fn


def make_trailing_lock_exit(retrace_pct: float) -> Callable:
    """D: exit when MFE - current return > retrace_pct."""
    def fn(state, row, direction, bars_held):
        if bars_held < MIN_HOLD_BARS:
            return False, None
        # Caller passes current_gross into state["current_gross"] each bar
        cur = state.get("current_gross")
        if cur is None:
            return False, None
        peak = state.get("peak_gross", -1e9)
        if cur > peak:
            state["peak_gross"] = cur
            peak = cur
        # Only arm once we've been in profit (positive peak)
        if peak <= 0:
            return False, None
        retrace = peak - cur
        if retrace >= retrace_pct:
            return True, f"trail_{int(retrace_pct*1000)}"
        return False, None
    return fn


def make_combined_exit(*exits) -> Callable:
    """Any-of: trigger on first sub-exit that fires."""
    def fn(state, row, direction, bars_held):
        for sub in exits:
            should, reason = sub(state, row, direction, bars_held)
            if should:
                return True, reason
        return False, None
    return fn


# ── Simulator ────────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, exit_fn: Callable, entry_fn: Callable,
              cost: float = MAKER_COST,
              max_hold: int = MAX_HOLD_HOURS) -> pd.DataFrame:
    trades = []
    position = None
    entry_price = None
    entry_ts = None
    state: dict = {}

    rows = df.itertuples(index=True)
    last_idx = None
    for tup in rows:
        ts = tup.Index
        last_idx = ts
        row = tup._asdict()
        close = float(row["close"])

        if position is None:
            wants_long = bool(row.get("start_long_trade"))
            wants_short = bool(row.get("start_short_trade"))
            if not (wants_long or wants_short):
                continue
            direction = "LONG" if wants_long else "SHORT"
            if not entry_fn(row, direction):
                continue
            position = direction
            entry_price = close
            entry_ts = ts
            state = {}
            continue

        bars_held = int((ts - entry_ts).total_seconds() / 3600)

        if position == "LONG":
            cur_gross = close / entry_price - 1.0
        else:
            cur_gross = -(close / entry_price - 1.0)
        state["current_gross"] = cur_gross

        exit_reason = None

        # Reverse signal always closes (jdehorty intended behaviour)
        if position == "LONG" and bool(row.get("start_short_trade")) and bars_held >= MIN_HOLD_BARS:
            exit_reason = "reverse"
        elif position == "SHORT" and bool(row.get("start_long_trade")) and bars_held >= MIN_HOLD_BARS:
            exit_reason = "reverse"
        else:
            should, reason = exit_fn(state, row, position, bars_held)
            if should:
                exit_reason = reason

        if exit_reason is None and bars_held >= max_hold:
            exit_reason = "timeout"

        if exit_reason is None:
            continue

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
            "held_hours": bars_held, "exit_reason": exit_reason,
        })

        if exit_reason == "reverse":
            new_dir = "SHORT" if position == "LONG" else "LONG"
            if entry_fn(row, new_dir):
                position = new_dir
                entry_price = close
                entry_ts = ts
                state = {}
            else:
                position = None
                entry_price = entry_ts = None
        else:
            position = None
            entry_price = entry_ts = None

    # Close any leftover open trade at last bar
    if position is not None and last_idx is not None:
        close = float(df.loc[last_idx, "close"])
        if position == "LONG":
            gross = close / entry_price - 1.0
        else:
            gross = -(close / entry_price - 1.0)
        net = gross - cost
        bars_held = int((last_idx - entry_ts).total_seconds() / 3600)
        trades.append({
            "entry_ts": entry_ts, "exit_ts": last_idx,
            "direction": position, "entry_price": entry_price,
            "exit_price": close, "gross_pct": gross,
            "net_pct": net, "win": int(gross > 0),
            "held_hours": bars_held, "exit_reason": "open_at_end",
        })

    return pd.DataFrame(trades)


# ── Reporting ────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    net = df["net_pct"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = float((cum - rmax).min()) * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else float("inf")
    return {
        "n": n,
        "wr": float(df["win"].mean()),
        "avg_net_bps": float(net.mean()) * 10000,
        "avg_gross_bps": float(df["gross_pct"].mean()) * 10000,
        "cum_pct": float(cum[-1]) * 100,
        "mdd_pct": mdd,
        "pf": pf,
        "avg_hold": float(df["held_hours"].mean()),
        "ratio": abs(float(cum[-1]) * 100 / mdd) if mdd < 0 else float("inf"),
    }


def fmt_row(label: str, m: dict) -> str:
    if m["n"] == 0:
        return f"  {label:<42} n=0"
    return (
        f"  {label:<42} n={m['n']:>3} WR={m['wr']*100:>5.1f}% "
        f"net={m['avg_net_bps']:>+7.1f}bp "
        f"cum={m['cum_pct']:>+6.1f}% MDD={m['mdd_pct']:>+6.1f}% "
        f"PF={m['pf']:>4.2f} ratio={m['ratio']:>4.2f} hold={m['avg_hold']:>4.0f}h"
    )


def main():
    df = load_data()
    logger.info("Loaded %d bars, %s → %s", len(df), df.index.min(), df.index.max())
    logger.info("v9 coverage: %d / %d (%.1f%%)",
                 df["p_long_win"].notna().sum(), len(df),
                 100 * df["p_long_win"].notna().mean())

    def entry_fn_with_v9(row, direction):
        return v9_entry_filter(row, direction)

    def entry_fn_no_filter(row, direction):
        return True

    variants = [
        ("BASELINE: dynamic_cross + min_hold_4 (production)",
         make_dynamic_cross_exit()),

        # A: v9-confirmed cross — symmetry with entry
        ("A.1 v9-cross conf 0.40",
         make_v9_confirmed_exit(0.40)),
        ("A.2 v9-cross conf 0.50",
         make_v9_confirmed_exit(0.50)),
        ("A.3 v9-cross conf 0.55 (sym with entry)",
         make_v9_confirmed_exit(0.55)),
        ("A.4 v9-cross conf 0.60",
         make_v9_confirmed_exit(0.60)),

        # B: multi-bar kernel relation
        ("B.1 multibar kernel N=2",
         make_multibar_kernel_exit(2)),
        ("B.2 multibar kernel N=3",
         make_multibar_kernel_exit(3)),
        ("B.3 multibar kernel N=4",
         make_multibar_kernel_exit(4)),

        # C: signal-aligned cross
        ("C.1 cross + signal flip",
         make_signal_aligned_exit()),

        # D: trailing profit lock
        ("D.1 trailing lock 1.0%",
         make_trailing_lock_exit(0.010)),
        ("D.2 trailing lock 1.5%",
         make_trailing_lock_exit(0.015)),
        ("D.3 trailing lock 2.0%",
         make_trailing_lock_exit(0.020)),

        # Combinations — any-of (first to fire wins)
        ("A.3 ∪ B.2 (v9-cross 0.55 OR multibar 3)",
         make_combined_exit(make_v9_confirmed_exit(0.55),
                             make_multibar_kernel_exit(3))),
        ("A.3 ∪ D.2 (v9-cross 0.55 OR trail 1.5%)",
         make_combined_exit(make_v9_confirmed_exit(0.55),
                             make_trailing_lock_exit(0.015))),
        ("A.2 ∪ D.2 (v9-cross 0.50 OR trail 1.5%)",
         make_combined_exit(make_v9_confirmed_exit(0.50),
                             make_trailing_lock_exit(0.015))),
    ]

    for scope_label, entry_fn in [
        ("NO v9 ENTRY FILTER (jdehorty original — for variant comparison, n~120)",
         entry_fn_no_filter),
        ("WITH v9 ENTRY FILTER P>=0.55 (production — small n)",
         entry_fn_with_v9),
    ]:
        print()
        print("=" * 130)
        print(f"SWEEP: {scope_label}")
        print("=" * 130)
        print(f"  {'variant':<42} {'n':>5} {'WR':>8} {'net':>13} "
              f"{'cum%':>9} {'MDD%':>9} {'PF':>6} {'ratio':>7} {'hold':>6}")
        print("-" * 130)

        results = []
        for label, exit_fn in variants:
            trades = simulate(df, exit_fn, entry_fn)
            m = summarize(trades)
            results.append((label, m, trades))
            print(fmt_row(label, m))

        # Ranked tables
        valid = [(l, m) for l, m, _ in results if m["n"] > 0]
        if valid:
            print()
            print(f"  -- ranked by cum/|MDD| (risk-adjusted) --")
            for l, m in sorted(valid, key=lambda x: x[1]["ratio"], reverse=True)[:5]:
                print(fmt_row(l, m))
            print()
            print(f"  -- ranked by cum% (raw return) --")
            for l, m in sorted(valid, key=lambda x: x[1]["cum_pct"], reverse=True)[:5]:
                print(fmt_row(l, m))


if __name__ == "__main__":
    main()
