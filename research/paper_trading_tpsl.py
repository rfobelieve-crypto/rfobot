"""
TP/SL paper-trading backtest.

User question (2026-05-10): how does a fixed risk:reward 1:1.5 strategy
perform across all settled signals?

Rules:
    Entry @ tracked_signals.entry_price (= bar close at signal_time)
    TP    @ entry × (1 ± SL × RR × sign)   where sign = +1 (UP) or -1 (DOWN)
    SL    @ entry × (1 ∓ SL × sign)
    Exit  : whichever (TP, SL) is touched first by subsequent 1h bar's
            high/low.  If neither is touched within `timeout` bars,
            force-exit at timeout bar's close.

Conservative-first heuristic for ambiguous bars (both TP and SL touched in
the same 1h bar): assume SL hits first.  Real intraperiod path is unknown
without 1m data; using SL-first is the standard pessimistic assumption.

Sweeps a grid of SL distances; RR is fixed (1.5) per the user's spec.
For each grid point, reports n, WR, avg_net_bps, Sharpe, MDD, profit factor.

Reads:
    MySQL tracked_signals (filled=1)
    MySQL indicator_history (1h OHLC, dt index)

Cost: 13 bps round-trip (same as paper_trading.py: taker 10 + slip 2 + fund 1).
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.db import get_db_conn
from indicator.paper_trading import fetch_paper_signals, COST_BPS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COST = COST_BPS / 10000.0


def fetch_klines_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Bulk-fetch 1h OHLC from indicator_history covering [start, end]."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, `open`, high, low, `close`
                FROM indicator_history
                WHERE dt >= %s AND dt <= %s
                ORDER BY dt ASC
            """, (start.strftime("%Y-%m-%d %H:%M:%S"),
                  end.strftime("%Y-%m-%d %H:%M:%S")))
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df = df.set_index("dt").sort_index()
    return df


def _find_exit_for_signal(
    entry_price: float,
    direction: str,
    sl_dist: float,
    rr: float | None,
    bars: pd.DataFrame,
    timeout_bars: int,
) -> tuple[float, int, str]:
    """
    Walk forward through bars (in chronological order) and return
    (exit_price, bars_held, exit_reason) for one signal.

    sl_dist : SL distance as a fraction of entry (e.g. 0.005 = 0.5%)
    rr      : reward-to-risk ratio.  None or 0 = SL-only, no TP
              (timeout-bar's close is the exit if SL never hits).
    """
    if len(bars) == 0:
        return entry_price, 0, "no_bars"

    use_tp = rr is not None and rr > 0
    tp_dist = sl_dist * rr if use_tp else None

    if direction == "UP":
        sl_price = entry_price * (1.0 - sl_dist)
        tp_price = entry_price * (1.0 + tp_dist) if use_tp else None
    else:  # DOWN — short, profit when price falls
        sl_price = entry_price * (1.0 + sl_dist)
        tp_price = entry_price * (1.0 - tp_dist) if use_tp else None

    bars_to_check = bars.iloc[:timeout_bars]
    for i in range(len(bars_to_check)):
        row = bars_to_check.iloc[i]
        hi, lo = row["high"], row["low"]

        if direction == "UP":
            sl_hit = lo <= sl_price
            tp_hit = (hi >= tp_price) if use_tp else False
        else:
            sl_hit = hi >= sl_price
            tp_hit = (lo <= tp_price) if use_tp else False

        if tp_hit and sl_hit:
            # Both touched same bar — conservative: SL first
            return sl_price, i + 1, "sl_ambig"
        if sl_hit:
            return sl_price, i + 1, "sl"
        if tp_hit:
            return tp_price, i + 1, "tp"

    # Timeout — exit at last available bar's close
    last = bars_to_check.iloc[-1]
    return float(last["close"]), len(bars_to_check), "timeout"


def backtest_tpsl(
    signals: pd.DataFrame,
    klines: pd.DataFrame,
    sl_dist: float,
    rr: float = 1.5,
    timeout_hours: int = 24,
) -> pd.DataFrame:
    """Apply TP/SL rule to each signal, return per-trade results."""
    if signals.empty or klines.empty:
        return pd.DataFrame()

    out_rows = []
    klines_idx = klines.index
    klines_start = klines_idx.min()

    for _, sig in signals.iterrows():
        st = pd.Timestamp(sig["signal_time"])
        if st.tz is not None:
            st = st.tz_localize(None)

        # Bars strictly after signal_time (entry happens AT close of bar at st)
        future_idx = klines_idx[klines_idx > st]
        if len(future_idx) == 0:
            continue
        # CRITICAL: skip signals where klines coverage starts AFTER signal_time.
        # If klines_start > signal_time + 1.5h we'd be using unrelated future
        # bars (data gap silently producing wrong PnL).  E.g. indicator_history
        # only goes back to 2026-04-03 — older signals must be skipped.
        if klines_start > st + timedelta(hours=1, minutes=30):
            continue
        future = klines.loc[future_idx]

        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=float(sig["entry_price"]),
            direction=sig["direction"],
            sl_dist=sl_dist,
            rr=rr,
            bars=future,
            timeout_bars=timeout_hours,
        )

        sign = 1.0 if sig["direction"] == "UP" else -1.0
        gross_ret = (exit_price / float(sig["entry_price"]) - 1.0) * sign
        net_ret = gross_ret - COST

        out_rows.append({
            "signal_time": sig["signal_time"],
            "direction": sig["direction"],
            "strength": sig["strength"],
            "confidence": sig["confidence"],
            "regime": sig["regime"],
            "entry_price": sig["entry_price"],
            "exit_price": exit_price,
            "bars_held": bars_held,
            "exit_reason": reason,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "win": int(gross_ret > 0),
        })

    return pd.DataFrame(out_rows)


def summarize(results: pd.DataFrame, label: str) -> dict:
    if results.empty:
        return {"label": label, "n": 0}
    n = len(results)
    net = results["net_ret"].values
    cum = np.cumsum(net)
    running_max = np.maximum.accumulate(cum)
    mdd = (cum - running_max).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    by_reason = results["exit_reason"].value_counts().to_dict()
    return {
        "label": label,
        "n": n,
        "wr": float(results["win"].mean()),
        "avg_gross_bps": float(results["gross_ret"].mean()) * 10000,
        "avg_net_bps": float(net.mean()) * 10000,
        "median_net_bps": float(np.median(net)) * 10000,
        "sharpe": float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        "cum_net_pct": float(cum[-1]) * 100,
        "mdd_pct": float(mdd),
        "profit_factor": float(pf),
        "avg_bars_held": float(results["bars_held"].mean()),
        "exit_reasons": by_reason,
    }


def _print_sweep_table(
    title: str,
    rr: float | None,
    timeout_hours: int,
    sweep_results: list,
):
    rr_str = f"1:{rr}" if rr is not None and rr > 0 else "SL-only"
    print(f"\n{'=' * 100}")
    print(f"{title}  (RR={rr_str}, timeout={timeout_hours}h, cost={COST_BPS:.1f} bps)")
    print(f"{'=' * 100}")
    print(f"{'SL%':>6} {'TP%':>6}  {'n':>5} {'WR':>6} "
          f"{'avg_gross':>9} {'avg_net':>8} {'sharpe':>7} "
          f"{'cum%':>7} {'MDD%':>7} {'PF':>5} "
          f"{'bars':>5} {'TP/SL/TO':>16}")
    print("-" * 100)
    for sl, s, _ in sweep_results:
        if s.get("n", 0) == 0:
            print(f"  {sl*100:>5.2f}% {'-':>6}   no data")
            continue
        reasons = s["exit_reasons"]
        tp = reasons.get("tp", 0)
        sl_n = reasons.get("sl", 0) + reasons.get("sl_ambig", 0)
        to = reasons.get("timeout", 0)
        tp_pct = (
            f"{sl*rr*100:>5.2f}%" if rr is not None and rr > 0 else "  -  "
        )
        print(f"  {sl*100:>5.2f}% {tp_pct}  "
              f"{s['n']:>5d} {s['wr']*100:>5.1f}% "
              f"{s['avg_gross_bps']:>+8.1f} "
              f"{s['avg_net_bps']:>+7.1f} "
              f"{s['sharpe']:>+7.3f} "
              f"{s['cum_net_pct']:>+6.1f}% "
              f"{s['mdd_pct']:>+6.1f}% "
              f"{s['profit_factor']:>5.2f} "
              f"{s['avg_bars_held']:>5.1f} "
              f"{tp:>3}/{sl_n:>3}/{to:>3}")


def run_strategy_sweep(
    name: str,
    rr: float | None,
    timeout_hours: int,
    sl_grid: list[float],
    signals: pd.DataFrame,
    klines: pd.DataFrame,
) -> tuple[float, dict, pd.DataFrame]:
    """Run a strategy across SL grid; return (best_sl, best_summary, best_results)."""
    sweep_results = []
    for sl in sl_grid:
        results = backtest_tpsl(signals, klines, sl_dist=sl, rr=rr,
                                 timeout_hours=timeout_hours)
        s = summarize(results, f"SL={sl*100:.2f}%")
        sweep_results.append((sl, s, results))

    _print_sweep_table(name, rr, timeout_hours, sweep_results)

    best_sl, best_s, best_results = max(
        sweep_results, key=lambda x: x[1].get("avg_net_bps", -1e9))
    return best_sl, best_s, best_results


def _print_tier_dir_breakdown(name: str, sl: float, rr: float | None,
                              results: pd.DataFrame):
    if results.empty:
        return
    tp_str = f"{sl*rr*100:.2f}%" if rr is not None and rr > 0 else "no TP"
    print(f"\n  {name} | best SL={sl*100:.2f}% TP={tp_str} — tier × direction:")
    print(f"  {'slice':<22} {'n':>5} {'WR':>6} {'avg_net':>8} {'sharpe':>7} "
          f"{'PF':>5} {'TP/SL/TO':>15}")
    for tier in ("Strong", "Moderate"):
        for direction in ("UP", "DOWN"):
            sub = results[
                (results["strength"] == tier)
                & (results["direction"] == direction)
            ]
            if len(sub) < 5:
                continue
            ss = summarize(sub, f"{tier}/{direction}")
            r = ss["exit_reasons"]
            tp = r.get("tp", 0)
            sln = r.get("sl", 0) + r.get("sl_ambig", 0)
            to = r.get("timeout", 0)
            print(f"    {ss['label']:<20} {ss['n']:>5d} "
                  f"{ss['wr']*100:>5.1f}% "
                  f"{ss['avg_net_bps']:>+7.1f} "
                  f"{ss['sharpe']:>+7.3f} "
                  f"{ss['profit_factor']:>5.2f} "
                  f"{tp:>3}/{sln:>3}/{to:>3}")


def main():
    logger.info("Fetching signals + klines...")
    signals = fetch_paper_signals()
    if signals.empty:
        logger.error("No signals — abort")
        return
    logger.info("Signals: n=%d, range %s ~ %s",
                len(signals), signals["signal_time"].min(),
                signals["signal_time"].max())

    start = pd.Timestamp(signals["signal_time"].min())
    end = pd.Timestamp(signals["signal_time"].max()) + timedelta(hours=48)
    klines = fetch_klines_range(start, end)
    logger.info("Klines: n=%d, range %s ~ %s",
                len(klines), klines.index.min(), klines.index.max())

    # Strategy A: 1:0.5 RR (TP < SL — "give signal room, lock small profit")
    a_sl, a_s, a_res = run_strategy_sweep(
        "STRATEGY A: 1:0.5 RR (TP < SL, 24h timeout)",
        rr=0.5, timeout_hours=24,
        sl_grid=[0.005, 0.007, 0.010, 0.012, 0.015, 0.020],
        signals=signals, klines=klines,
    )

    # Strategy B: 1:1 symmetric
    b_sl, b_s, b_res = run_strategy_sweep(
        "STRATEGY B: 1:1 symmetric (24h timeout)",
        rr=1.0, timeout_hours=24,
        sl_grid=[0.003, 0.005, 0.007, 0.010, 0.012, 0.015],
        signals=signals, klines=klines,
    )

    # Strategy C/D: SL only + 4h hold (no TP, force-exit at horizon)
    cd_sl, cd_s, cd_res = run_strategy_sweep(
        "STRATEGY C: SL-only + 4h hold (no TP)",
        rr=None, timeout_hours=4,
        sl_grid=[0.005, 0.007, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030],
        signals=signals, klines=klines,
    )

    # Reference baseline: persisted from paper_trading.py (4h hold, no TP/SL)
    # we can compute it on-the-fly here for fairness on the same n
    baseline_results = backtest_tpsl(signals, klines, sl_dist=0.99, rr=None,
                                       timeout_hours=4)
    baseline_s = summarize(baseline_results, "baseline 4h hold no TP/SL")

    print(f"\n{'=' * 100}")
    print("FINAL COMPARISON  (best SL of each strategy, restricted to n in klines coverage)")
    print(f"{'=' * 100}")
    print(f"{'strategy':<42} {'n':>5} {'WR':>6} "
          f"{'avg_net':>8} {'sharpe':>7} {'cum%':>7} {'MDD%':>7} {'PF':>5}")
    print("-" * 100)
    for name, s, sl, rr in [
        ("Baseline (4h hold, no TP/SL)", baseline_s, None, None),
        (f"A: 1:0.5 RR  SL={a_sl*100:.2f}% TP={a_sl*0.5*100:.2f}%", a_s, a_sl, 0.5),
        (f"B: 1:1   RR  SL={b_sl*100:.2f}% TP={b_sl*100:.2f}%", b_s, b_sl, 1.0),
        (f"C: SL-only+4h SL={cd_sl*100:.2f}% (no TP)", cd_s, cd_sl, None),
    ]:
        if s.get("n", 0) == 0:
            continue
        print(f"  {name:<40} {s['n']:>5d} {s['wr']*100:>5.1f}% "
              f"{s['avg_net_bps']:>+7.1f} {s['sharpe']:>+7.3f} "
              f"{s['cum_net_pct']:>+6.1f}% {s['mdd_pct']:>+6.1f}% "
              f"{s['profit_factor']:>5.2f}")

    # Tier × direction breakdown for the winner of each strategy
    print(f"\n{'=' * 100}")
    print("PER-STRATEGY: tier × direction breakdown (best SL)")
    print(f"{'=' * 100}")
    _print_tier_dir_breakdown("A (1:0.5)", a_sl, 0.5, a_res)
    _print_tier_dir_breakdown("B (1:1)",   b_sl, 1.0, b_res)
    _print_tier_dir_breakdown("C (SL+4h)", cd_sl, None, cd_res)

    # Save winners
    out_dir = PROJECT_ROOT / "research" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    a_res.to_csv(out_dir / f"tpsl_A_1to0.5_sl{int(a_sl*10000)}bp.csv", index=False)
    b_res.to_csv(out_dir / f"tpsl_B_1to1_sl{int(b_sl*10000)}bp.csv", index=False)
    cd_res.to_csv(out_dir / f"tpsl_C_SLonly4h_sl{int(cd_sl*10000)}bp.csv", index=False)
    logger.info("Saved per-strategy best trades to research/results/")


if __name__ == "__main__":
    main()
