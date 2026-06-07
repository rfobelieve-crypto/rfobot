"""V7.1 exit variants A/B/C backtest.

Reuses load_data / decode_signals / metrics from v71_v7_sizing_1x.py but
swaps the exit logic per variant to answer: "would changing how we exit
move Sharpe / net bps / MDD enough to be worth deploying?"

Variants tested:
  0. baseline      — 3xATR trail, 72h time cap, opp_signal=any
  1. opp_strong    — opp_signal only fires on Strong reversal
  2. opp_2bar      — opp_signal requires 2 consecutive reverse bars
  3. trail_2x      — tighter trail (2xATR instead of 3xATR)
  4. trail_4x      — looser trail (4xATR)
  5. no_opp        — remove opp_signal entirely (only trail + time_cap)
  6. be_after_1atr — once unrealized > 1xATR, move stop to entry (breakeven)

Each variant runs against the SAME WF-OOS direction predictions, so
diffs are purely from exit mechanics. Compares Sharpe, net bps, MDD, WR,
PF, avg hold time.

Usage:
    python research/exit_variants_backtest.py
    python research/exit_variants_backtest.py --variants baseline opp_strong trail_2x

Caveats — read before deploying any change:
  - Backtest is on the same 5.5-month WF-OOS window v71_v7_sizing_1x uses.
    Live regime (since 2026-04-17) may differ from this window.
  - Costs: TAKER_COST round-trip from v71_v7_sizing_1x (no extra slippage).
  - No leverage scaling — net_pct is per-trade asset return %, not equity.
    Sizing-B effective leverage 2x will scale equity_ret accordingly in live.
  - Sample size (~169 trades on baseline) is small; CIs are wide.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from research.v71_v7_sizing_1x import (
    ATR_PERIOD, INITIAL_CAPITAL, MAX_LEVERAGE, RISK_FRAC, TAKER_COST,
    TIME_CAP_HOURS, TRAIL_MULT,
    load_data, decode_signals, _atr_wilder, _max_drawdown,
)


# ── Exit policy ─────────────────────────────────────────────────────────

@dataclass
class ExitPolicy:
    """Parameterised exit logic. Defaults reproduce baseline behavior."""
    name: str = "baseline"
    trail_mult: float = TRAIL_MULT
    time_cap_hours: int = TIME_CAP_HOURS
    # Opposite signal exit
    opp_enabled: bool = True
    opp_require_strong: bool = False        # only Strong reversal triggers
    opp_require_consecutive: int = 1        # 1 = any reverse bar; 2 = need two in a row
    # Breakeven move
    breakeven_after_atr_mult: float = 0.0   # 0 disables; e.g. 1.0 = BE when unrealized > 1×ATR


def variants_catalog() -> dict[str, ExitPolicy]:
    return {
        "baseline":      ExitPolicy(name="baseline"),
        "opp_strong":    ExitPolicy(name="opp_strong",     opp_require_strong=True),
        "opp_2bar":      ExitPolicy(name="opp_2bar",       opp_require_consecutive=2),
        "trail_2x":      ExitPolicy(name="trail_2x",       trail_mult=2.0),
        "trail_4x":      ExitPolicy(name="trail_4x",       trail_mult=4.0),
        "no_opp":        ExitPolicy(name="no_opp",         opp_enabled=False),
        "be_after_1atr": ExitPolicy(name="be_after_1atr",  breakeven_after_atr_mult=1.0),
    }


# ── Simulator with policy hook ──────────────────────────────────────────

def simulate_with_policy(df: pd.DataFrame,
                         direction: np.ndarray, tier: np.ndarray,
                         policy: ExitPolicy,
                         entry_mode: str = "signal_close") -> pd.DataFrame:
    """One-position-at-a-time v7.1 sim with parameterised exits.

    Logic mirrors v71_v7_sizing_1x.simulate(); divergences are tagged
    `# POLICY:` so they're easy to audit.
    """
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    atr = df["atr"].values
    ts = df.index
    n = len(df)

    equity = INITIAL_CAPITAL
    pos: Optional[dict] = None
    trades: list[dict] = []
    # POLICY: opp_2bar requires tracking how many consecutive reverse bars
    opp_streak = 0

    for i in range(n):
        # ── Case A: manage an open position ──
        if pos is not None:
            bars_held = int((ts[i] - pos["entry_ts"]).total_seconds() / 3600)
            sd = pos["stop_dist"]
            prev_ext = pos["trail_extreme"]
            if pos["side"] == "LONG":
                cur_stop = prev_ext - sd
            else:
                cur_stop = prev_ext + sd

            # POLICY: breakeven floor — once unrealized > N×ATR, refuse
            # to let the stop drift below entry. Implemented by clamping
            # cur_stop AGAINST entry (LONG: stop >= entry; SHORT: stop <= entry).
            if policy.breakeven_after_atr_mult > 0:
                a_at_entry = pos["atr_at_entry"]
                threshold = policy.breakeven_after_atr_mult * a_at_entry
                if pos["side"] == "LONG":
                    unrealized = h[i] - pos["entry_price"]
                    pos["be_armed"] = pos.get("be_armed") or unrealized >= threshold
                    if pos["be_armed"]:
                        cur_stop = max(cur_stop, pos["entry_price"])
                else:  # SHORT
                    unrealized = pos["entry_price"] - lo[i]
                    pos["be_armed"] = pos.get("be_armed") or unrealized >= threshold
                    if pos["be_armed"]:
                        cur_stop = min(cur_stop, pos["entry_price"])

            exit_price = exit_reason = None
            # 1) trailing stop — intrabar, gap-through filled at bar open
            if pos["side"] == "LONG" and lo[i] <= cur_stop:
                exit_price, exit_reason = min(cur_stop, o[i]), "trail_stop"
            elif pos["side"] == "SHORT" and h[i] >= cur_stop:
                exit_price, exit_reason = max(cur_stop, o[i]), "trail_stop"
            # 2) time cap
            if exit_reason is None and bars_held >= policy.time_cap_hours:
                exit_price, exit_reason = c[i], "time_cap"
            # 3) opposite signal — POLICY hooks here
            if exit_reason is None and policy.opp_enabled:
                is_opp = ((pos["side"] == "LONG" and direction[i] == "DOWN") or
                          (pos["side"] == "SHORT" and direction[i] == "UP"))
                if is_opp:
                    opp_streak += 1
                    strong_ok = (not policy.opp_require_strong) or (tier[i] == "Strong")
                    bars_ok = opp_streak >= policy.opp_require_consecutive
                    if strong_ok and bars_ok:
                        exit_price, exit_reason = c[i], "opp_signal"
                else:
                    opp_streak = 0
            # 4) data end — force close
            if exit_reason is None and i == n - 1:
                exit_price, exit_reason = c[i], "data_end"

            if exit_reason is None:
                if pos["side"] == "LONG":
                    pos["trail_extreme"] = max(prev_ext, h[i])
                else:
                    pos["trail_extreme"] = min(prev_ext, lo[i])
                continue

            # Close position
            cost = TAKER_COST  # variants don't add extra slip
            if pos["side"] == "LONG":
                gross = exit_price / pos["entry_price"] - 1.0
            else:
                gross = -(exit_price / pos["entry_price"] - 1.0)
            net = gross - cost
            equity_ret = pos["size_frac"] * net
            equity = max(equity * (1.0 + equity_ret), 0.0)
            trades.append(dict(
                entry_ts=pos["entry_ts"], exit_ts=ts[i], side=pos["side"],
                tier=pos["tier"], entry_price=pos["entry_price"],
                exit_price=exit_price, exit_reason=exit_reason,
                bars_held=bars_held, gross_pct=gross, net_pct=net,
                size_frac=pos["size_frac"], equity_ret_pct=equity_ret * 100.0,
                equity_after=equity, win=int(gross > 0)))
            pos = None
            opp_streak = 0
            continue

        # ── Case B: flat — check for entry ──
        if not df["in_oos"].values[i]:
            continue
        if direction[i] not in ("UP", "DOWN"):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        if entry_mode == "signal_close":
            entry_idx, entry_price = i, c[i]
        else:
            if i + 1 >= n:
                continue
            entry_idx, entry_price = i + 1, o[i + 1]

        stop_dist = policy.trail_mult * a
        stop_pct = stop_dist / entry_price
        size_frac = (min(MAX_LEVERAGE, RISK_FRAC / stop_pct)
                     if stop_pct > 0 else MAX_LEVERAGE)
        pos = dict(
            side="LONG" if direction[i] == "UP" else "SHORT",
            tier=tier[i], entry_ts=ts[entry_idx], entry_price=entry_price,
            stop_dist=stop_dist, trail_extreme=entry_price,
            size_frac=size_frac, atr_at_entry=a,
            be_armed=False,
        )
        opp_streak = 0

    return pd.DataFrame(trades)


# ── Metrics ─────────────────────────────────────────────────────────────

def summarise(trades: pd.DataFrame, span_days: float, name: str) -> dict:
    if trades.empty:
        return {"name": name, "n": 0}
    eq_ret = trades["equity_ret_pct"].values
    eq_after = trades["equity_after"].values
    curve = np.concatenate([[INITIAL_CAPITAL], eq_after])
    wins = eq_ret[eq_ret > 0]
    losses = eq_ret[eq_ret < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf
    trades_per_year = len(trades) * (365.25 / span_days) if span_days > 0 else 0
    sharpe_pt = float(eq_ret.mean() / eq_ret.std()) if eq_ret.std() > 0 else 0.0
    sharpe_ann = sharpe_pt * np.sqrt(trades_per_year)
    by_reason = trades.groupby("exit_reason").agg(
        n=("net_pct", "size"),
        wr=("win", "mean"),
        avg_net_bps=("net_pct", lambda x: x.mean() * 10000),
        avg_bars=("bars_held", "mean"),
    ).to_dict("index")
    return {
        "name": name,
        "n": int(len(trades)),
        "wr_pct": float((trades["win"] == 1).mean() * 100),
        "avg_net_bps": float(trades["net_pct"].mean() * 10000),
        "roi_pct": float((eq_after[-1] / INITIAL_CAPITAL - 1.0) * 100),
        "mdd_pct": _max_drawdown(curve),
        "sharpe_ann": float(sharpe_ann),
        "profit_factor": pf if pf != np.inf else float("inf"),
        "avg_hold_h": float(trades["bars_held"].mean()),
        "by_reason": by_reason,
    }


def _fmt_pf(pf: float) -> str:
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def print_compare(results: list[dict]) -> None:
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    print()
    print("=" * 95)
    print(f"{'variant':<18} {'n':>4} {'WR%':>6} {'net bps':>9} "
          f"{'ROI %':>8} {'MDD %':>7} {'Sharpe':>7} {'PF':>5} {'hold h':>7}")
    print("-" * 95)
    for r in results:
        if r["n"] == 0:
            print(f"{r['name']:<18} {'-':>4}")
            continue
        line = (f"{r['name']:<18} {r['n']:>4} "
                f"{r['wr_pct']:>6.1f} {r['avg_net_bps']:>+9.1f} "
                f"{r['roi_pct']:>+8.1f} {r['mdd_pct']:>7.2f} "
                f"{r['sharpe_ann']:>7.2f} {_fmt_pf(r['profit_factor']):>5} "
                f"{r['avg_hold_h']:>7.1f}")
        if baseline is not None and r["name"] != "baseline":
            d_sh = r["sharpe_ann"] - baseline["sharpe_ann"]
            d_net = r["avg_net_bps"] - baseline["avg_net_bps"]
            line += f"   Δ Sharpe {d_sh:+.2f}, Δ net {d_net:+.1f}bps"
        print(line)
    print("=" * 95)
    print()
    # Per-reason breakdown for baseline
    if baseline is not None and baseline["by_reason"]:
        print("Baseline 出場理由拆解:")
        for reason, m in baseline["by_reason"].items():
            print(f"  {reason:<12} n={m['n']:>3}  WR={m['wr']*100:>5.1f}%  "
                  f"net={m['avg_net_bps']:>+7.1f}bps  avg {m['avg_bars']:>4.0f}h")
        print()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=None,
                    help="subset to run; default = all")
    ap.add_argument("--entry-mode", default="signal_close",
                    choices=["signal_close", "next_open"])
    args = ap.parse_args()

    print("Loading OOS data + decoding signals (may take 10-30s)...")
    df = load_data()
    df["atr"] = _atr_wilder(df, ATR_PERIOD)
    direction, tier, warmup_n = decode_signals(df)
    df = df.iloc[warmup_n:].copy()
    direction = direction[warmup_n:]
    tier = tier[warmup_n:]
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    print(f"OOS window: {df.index[0]} → {df.index[-1]}  ({span_days:.1f} days)")
    print()

    catalog = variants_catalog()
    if args.variants:
        unknown = [v for v in args.variants if v not in catalog]
        if unknown:
            raise SystemExit(f"unknown variants: {unknown}. "
                             f"Available: {list(catalog)}")
        run_set = args.variants
    else:
        run_set = list(catalog)

    results = []
    for name in run_set:
        policy = catalog[name]
        print(f"Running {name}...", flush=True)
        trades = simulate_with_policy(df, direction, tier, policy,
                                      entry_mode=args.entry_mode)
        s = summarise(trades, span_days, name)
        results.append(s)

    print_compare(results)


if __name__ == "__main__":
    main()
