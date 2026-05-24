"""
V7.1 stress test battery — survival, not profit.

Runs the WF-OOS backtest under three execution models + extreme stress
overlays, to answer "does the strategy stay alive across regimes" rather
than "how much does it earn."

Execution models
----------------
A — resting_stop (optimistic, current paper assumption):
    Pre-placed exchange stop. Triggers intrabar when bar_low ≤ stop (LONG).
    Fill at min(stop, bar_open). 0 extra slippage.
B — resting_stop + slip (realistic, what live trading will be):
    Same trigger + fill model, plus extra slippage added to trail_stop exits
    (default +30 bps round-trip — typical BTC perp stop in mildly stressed
    book).
C — poll_close (pessimistic / "no resting stop"):
    No exchange order. Check at bar close: close ≤ stop triggers exit at
    close. Intrabar spikes that recover by close are MISSED — protects you
    when crash recovers, crushes you when it doesn't.

Stress overlays
---------------
S1 — slippage gradient: vary +slippage on Model B, find break-even
S2 — flash crash injection: drop one -5/-10/-15/-20% gap-down bar at random
     OOS positions, re-run, measure single-event impact under each model
S3 — block bootstrap MDD: resample real trade equity returns in 8-trade
     blocks (preserves autocorrelation), report 99th-percentile MDD and
     probability of breaching the -15% kill switch

Usage:
    python research/stress_test_v7.py
"""
from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd

import research.v71_v7_sizing_1x as bt


KILL_SWITCH_DD_PCT = -15.0     # mirrors paper_trading.V7_KILL_SWITCH_MDD_PCT
DEFAULT_SLIP_BPS = 30           # realistic baseline for Model B


# ── Sim with execution-model switch ─────────────────────────────────────────

def simulate_exec(df: pd.DataFrame, direction, tier,
                  exec_model: str = "resting_stop",
                  extra_slip: float = 0.0) -> pd.DataFrame:
    """Copy of bt.simulate with an exec_model branch on the trail_stop check.

    exec_model:
        "resting_stop" — intrabar trigger via bar_low/bar_high, fill at
                         min(stop, bar_open) (LONG) / max(stop, bar_open)
        "poll_close"   — close-only check, fill at close
    """
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    atr = df["atr"].values
    ts = df.index
    n = len(df)

    equity = bt.INITIAL_CAPITAL
    pos = None
    trades = []

    for i in range(n):
        # ── manage open ──
        if pos is not None:
            bars_held = int((ts[i] - pos["entry_ts"]).total_seconds() / 3600)
            sd = pos["stop_dist"]
            prev_ext = pos["trail_extreme"]
            cur_stop = (prev_ext - sd) if pos["side"] == "LONG" else (prev_ext + sd)

            exit_price = exit_reason = None
            # 1) trailing stop
            if exec_model == "resting_stop":
                if pos["side"] == "LONG" and lo[i] <= cur_stop:
                    exit_price, exit_reason = min(cur_stop, o[i]), "trail_stop"
                elif pos["side"] == "SHORT" and h[i] >= cur_stop:
                    exit_price, exit_reason = max(cur_stop, o[i]), "trail_stop"
            elif exec_model == "poll_close":
                if pos["side"] == "LONG" and c[i] <= cur_stop:
                    exit_price, exit_reason = c[i], "trail_stop"
                elif pos["side"] == "SHORT" and c[i] >= cur_stop:
                    exit_price, exit_reason = c[i], "trail_stop"
            else:
                raise ValueError(f"unknown exec_model: {exec_model}")

            if exit_reason is None and bars_held >= bt.TIME_CAP_HOURS:
                exit_price, exit_reason = c[i], "time_cap"
            if exit_reason is None:
                opp = ((pos["side"] == "LONG" and direction[i] == "DOWN") or
                       (pos["side"] == "SHORT" and direction[i] == "UP"))
                if opp:
                    exit_price, exit_reason = c[i], "opp_signal"
            if exit_reason is None and i == n - 1:
                exit_price, exit_reason = c[i], "data_end"

            if exit_reason is None:
                pos["trail_extreme"] = (max(prev_ext, h[i])
                                         if pos["side"] == "LONG"
                                         else min(prev_ext, lo[i]))
                continue

            cost = bt.TAKER_COST + (extra_slip if exit_reason == "trail_stop" else 0.0)
            gross = (exit_price / pos["entry_price"] - 1.0
                     if pos["side"] == "LONG"
                     else -(exit_price / pos["entry_price"] - 1.0))
            net = gross - cost
            equity_ret = pos["size_frac"] * net
            equity = max(equity * (1.0 + equity_ret), 0.0)
            trades.append(dict(
                entry_ts=pos["entry_ts"], exit_ts=ts[i], side=pos["side"],
                tier=pos["tier"], entry_price=pos["entry_price"],
                exit_price=exit_price, exit_reason=exit_reason,
                bars_held=bars_held, gross_pct=gross, net_pct=net,
                size_frac=pos["size_frac"],
                equity_ret_pct=equity_ret * 100.0, equity_after=equity,
                win=int(gross > 0)))
            pos = None
            continue

        # ── flat: maybe open ──
        if not df["in_oos"].values[i]:
            continue
        if direction[i] not in ("UP", "DOWN"):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry_price = c[i]
        stop_dist = bt.TRAIL_MULT * a
        stop_pct = stop_dist / entry_price
        size_frac = min(bt.MAX_LEVERAGE,
                        bt.RISK_FRAC / stop_pct) if stop_pct > 0 else bt.MAX_LEVERAGE
        pos = dict(
            side="LONG" if direction[i] == "UP" else "SHORT",
            tier=tier[i], entry_ts=ts[i], entry_price=entry_price,
            stop_dist=stop_dist, trail_extreme=entry_price, size_frac=size_frac)

    return pd.DataFrame(trades)


# ── Headline summary, with kill-switch breach flag ──────────────────────────

def _max_dd(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak) * 100.0)


def hdr(label: str, trades: pd.DataFrame, span_days: float) -> dict:
    if trades.empty:
        return {"label": label, "n": 0, "killed": False}
    s = bt.summarize(trades, span_days)
    eq = np.concatenate([[bt.INITIAL_CAPITAL], trades["equity_after"].values])
    peak = np.maximum.accumulate(eq)
    dd_path = (eq / peak - 1.0) * 100.0
    killed = bool((dd_path <= KILL_SWITCH_DD_PCT).any())
    s["label"] = label
    s["killed"] = killed
    s["worst_single_pct"] = float(trades["equity_ret_pct"].min())
    return s


def fmt(r: dict) -> str:
    if r["n"] == 0:
        return f"  {r['label']:<32} (no trades)"
    kill_flag = "🚨KILLED" if r["killed"] else "—"
    return (f"  {r['label']:<32} n={r['n']:>3} WR={r['wr_pct']:>4.1f}% "
            f"ROI {r['roi_pct']:>+6.1f}% MDD {r['mdd_pct']:>4.1f}% "
            f"Sharpe {r['sharpe_calendar_ann']:>4.1f} "
            f"worst={r['worst_single_pct']:>+5.1f}% kill={kill_flag}")


# ── Test 1: three execution models, no injection ────────────────────────────

def test_1_three_models(df, direction, tier, span_days):
    print("=" * 78)
    print("  TEST 1 — Three execution models (clean OOS, no injection)")
    print("=" * 78)
    print("  >> 量「執行模型假設」本身值多少。實盤越靠近 B 越誠實。")
    print()
    configs = [
        ("A optimistic (resting_stop, 0 slip)", "resting_stop", 0.0),
        (f"B realistic (resting_stop +{DEFAULT_SLIP_BPS}bps)", "resting_stop",
         DEFAULT_SLIP_BPS / 1e4),
        ("C pessimistic (poll close, no exchg stop)", "poll_close", 0.0),
    ]
    for label, em, slip in configs:
        tr = simulate_exec(df, direction, tier, exec_model=em, extra_slip=slip)
        print(fmt(hdr(label, tr, span_days)))
    print()


# ── Test 2: slippage gradient on Model B ────────────────────────────────────

def test_2_slippage_gradient(df, direction, tier, span_days):
    print("=" * 78)
    print("  TEST 2 — Slippage break-even on Model B")
    print("=" * 78)
    print("  >> trail_stop 上的額外滑價多少 bps 開始,策略變 net-negative?")
    print()
    for slip_bps in [0, 25, 50, 75, 100, 150, 200, 300, 500]:
        tr = simulate_exec(df, direction, tier, exec_model="resting_stop",
                           extra_slip=slip_bps / 1e4)
        r = hdr(f"+{slip_bps:>3} bps trail slip", tr, span_days)
        print(fmt(r))
    print()


# ── Test 3: flash crash injection ───────────────────────────────────────────

def _inject_flash_crash(df: pd.DataFrame, t_idx: int,
                         magnitude: float) -> pd.DataFrame:
    """Replace bar at t_idx with a synthetic crash + partial recovery bar.
    magnitude < 0 (e.g. -0.10 = -10% intrabar low).
    """
    df2 = df.copy()
    prev_close = float(df2["close"].iloc[t_idx - 1])
    crash_low = prev_close * (1 + magnitude)
    df2.iloc[t_idx, df2.columns.get_loc("open")] = prev_close
    df2.iloc[t_idx, df2.columns.get_loc("high")] = prev_close * 1.0005
    df2.iloc[t_idx, df2.columns.get_loc("low")] = crash_low
    # close: partial recovery (half of the dip retraces)
    df2.iloc[t_idx, df2.columns.get_loc("close")] = prev_close * (1 + magnitude * 0.5)
    return df2


def test_3_flash_crash(df, direction, tier, span_days, n_trials=60, seed=42):
    print("=" * 78)
    print("  TEST 3 — Flash crash injection")
    print("=" * 78)
    print(f"  >> 在 OOS 隨機位置注入單根崩盤 bar (low 跌 X%, 收盤回拉一半),")
    print(f"     每個 (magnitude × exec_model) 跑 {n_trials} 次,看單一事件下的最差結果。")
    print()
    rng = np.random.default_rng(seed)
    in_oos_idx = np.where(df["in_oos"].values)[0]
    # only inject inside OOS, with at least 100 bars before/after
    valid = in_oos_idx[(in_oos_idx > 100) & (in_oos_idx < len(df) - 100)]

    configs = [
        ("A resting_stop      ", "resting_stop", 0.0),
        (f"B resting +{DEFAULT_SLIP_BPS}bps  ", "resting_stop",
         DEFAULT_SLIP_BPS / 1e4),
        ("C poll_close         ", "poll_close", 0.0),
    ]

    for mag in [-0.05, -0.10, -0.15, -0.20]:
        print(f"  -- magnitude {mag*100:+.0f}% --")
        for label, em, slip in configs:
            rois, mdds, killed_count, worst_trade = [], [], 0, 0.0
            for _ in range(n_trials):
                t = int(rng.choice(valid))
                df_inj = _inject_flash_crash(df, t, mag)
                # recompute ATR on injected bars (the crash bar inflates ATR
                # for subsequent bars — which is correct: vol DID expand)
                df_inj["atr"] = bt._atr_wilder(df_inj)
                tr = simulate_exec(df_inj, direction, tier,
                                   exec_model=em, extra_slip=slip)
                r = hdr(label, tr, span_days)
                if r["n"] == 0:
                    continue
                rois.append(r["roi_pct"])
                mdds.append(r["mdd_pct"])
                killed_count += int(r["killed"])
                worst_trade = min(worst_trade, r["worst_single_pct"])
            if not rois:
                continue
            rois = np.array(rois); mdds = np.array(mdds)
            print(f"  {label} ROI med {np.median(rois):+5.1f}% / "
                  f"p5 {np.percentile(rois,5):+5.1f}% | "
                  f"MDD med {np.median(mdds):4.1f}% / "
                  f"worst {mdds.min():4.1f}% | "
                  f"killed {killed_count}/{n_trials} | "
                  f"worst-trade {worst_trade:+.1f}%")
        print()


# ── Test 4: block bootstrap on Model B trades ───────────────────────────────

def test_4_block_bootstrap(df, direction, tier, span_days,
                            block=8, n_iter=10000, seed=7):
    print("=" * 78)
    print("  TEST 4 — Block-bootstrap MDD distribution (Model B base)")
    print("=" * 78)
    print(f"  >> 保留交易自相關 (block={block}),重排 {n_iter} 條可能的成交順序,")
    print(f"     看 kill switch ({KILL_SWITCH_DD_PCT}% DD) 多容易被自然觸發。")
    print()

    tr = simulate_exec(df, direction, tier, exec_model="resting_stop",
                       extra_slip=DEFAULT_SLIP_BPS / 1e4)
    if tr.empty:
        print("  no baseline trades")
        return

    eqr = tr["equity_ret_pct"].values / 100.0
    n = len(eqr)
    rng = np.random.default_rng(seed)

    def block_indices():
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend(range(start, min(start + block, n)))
        return np.array(idx[:n])

    rois, mdds, killed = [], [], 0
    for _ in range(n_iter):
        sample = eqr[block_indices()]
        curve = bt.INITIAL_CAPITAL * np.cumprod(1.0 + sample)
        rois.append((curve[-1] / bt.INITIAL_CAPITAL - 1.0) * 100.0)
        mdd = _max_dd(np.concatenate([[bt.INITIAL_CAPITAL], curve]))
        mdds.append(mdd)
        # max_dd returns positive; killed if peak-to-trough >= 15%
        if mdd >= -KILL_SWITCH_DD_PCT:
            killed += 1
    rois = np.array(rois); mdds = np.array(mdds)
    print(f"  baseline (Model B, +{DEFAULT_SLIP_BPS}bps): n={n} trades, ROI {(tr['equity_after'].iloc[-1]/bt.INITIAL_CAPITAL-1)*100:+.1f}%")
    print(f"  ROI    p1={np.percentile(rois,1):+5.1f}%  p5={np.percentile(rois,5):+5.1f}%  "
          f"p50={np.percentile(rois,50):+5.1f}%  p95={np.percentile(rois,95):+5.1f}%")
    print(f"  MDD    p50={np.percentile(mdds,50):4.1f}%  p95={np.percentile(mdds,95):4.1f}%  "
          f"p99={np.percentile(mdds,99):4.1f}%  worst={mdds.max():4.1f}%")
    print(f"  P(MDD breaches kill switch {KILL_SWITCH_DD_PCT}%) = "
          f"{killed/n_iter*100:.2f}%")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading WF-OOS data + decoding signals...")
    df = bt.load_data()
    span = (df.index.max() - df.index.min()).total_seconds() / 86400
    direction, tier, _ = bt.decode_signals(df)
    print(f"OOS span {span:.0f} days, {df['in_oos'].sum()} OOS bars\n")

    test_1_three_models(df, direction, tier, span)
    test_2_slippage_gradient(df, direction, tier, span)
    test_3_flash_crash(df, direction, tier, span, n_trials=60)
    test_4_block_bootstrap(df, direction, tier, span)

    print("=" * 78)
    print("  Stress test complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
