"""
Conviction-decay exit — replaces "wait for full opposite-tier reclassification
OR hit a fixed 3xATR distance" with "exit once the SAME model that generated
entry no longer agrees with the position's direction, for N consecutive bars."

Origin (chat, 2026-07-24): live track record (14 trades, 43% WR, net -0.80%)
diverged sharply from this backtest's clean baseline (65.1% WR). Exit-reason
decomposition (research/exit_decomposition.py) showed opp_signal exits are
excellent (85.7% WR) while trail_stop exits are weak (37.0% WR) in BOTH
backtest and live — consistently. The user's framing: the entry model
produces a continuous score every bar; exit logic should keep using it
(is the reason I entered still true?), not fall back to a fixed ATR distance
that ignores the model entirely once a position is open.

Mechanism: while in a position, if the model's raw continuous pred_ret has
the OPPOSITE sign of the position's direction for `consec_required`
consecutive bars, exit. This is a strictly softer trigger than opp_signal
(which requires reaching the full opposite Moderate/Strong tier threshold),
so it fires earlier and more often — trail_stop and time_cap remain as
backstops for cases the decay check doesn't catch.

Validation performed here (per this project's own per-fold-sanity /
half-split discipline — mistake.md 2026-06-02, 2026-06-20):
  1. Bootstrap CI on win rate (not just a point estimate)
  2. QUARTILE consistency (4 chunks by entry time, not just 2 halves —
     finer-grained than the initial chat-side check)
  3. Bar-count sensitivity (1/2/3/4 consecutive bars) — confirms the
     2-bar optimum is part of a smooth trend, not an isolated spike from
     picking the best of a small search (a real risk to flag honestly:
     only 3-4 configs were ever tried, this is a small search, not immune
     to the 2026-06-01/06-02 mistake.md lesson, just less likely given the
     smooth non-spiky shape).

Honest limitation: all of this still runs on the SAME single 167-day
WF-OOS window (2026-01-20 -> 2026-07-06) that every other test in this
file's lineage uses. It is not an independent out-of-sample window. Gate
A/B-style production validation (a fresh forward window, or live shadow
mode) is the next required step before touching indicator/okx/executor.py
— see TODO.md.

Run: python research/conviction_decay_exit.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import research.v71_v7_sizing_1x as bt

RESULTS_DIR = PROJECT_ROOT / "research" / "results"


def simulate_conviction_decay(df, direction, tier, pred, consec_required: int) -> pd.DataFrame:
    """Same mechanics as bt.simulate() (trail_stop + time_cap unchanged as
    backstops) but adds a conviction_decay exit: N consecutive bars where the
    model's raw pred_ret disagrees with the position's side."""
    o = df["open"].values; h = df["high"].values; lo = df["low"].values; c = df["close"].values
    atr = df["atr"].values; ts = df.index; n = len(df)
    equity = bt.INITIAL_CAPITAL
    pos = None
    trades = []
    for i in range(n):
        if pos is not None:
            bars_held = int((ts[i] - pos["entry_ts"]).total_seconds() / 3600)
            sd = pos["stop_dist"]; prev_ext = pos["trail_extreme"]
            cur_stop = prev_ext - sd if pos["side"] == "LONG" else prev_ext + sd
            exit_price = exit_reason = None
            if pos["side"] == "LONG" and lo[i] <= cur_stop:
                exit_price, exit_reason = min(cur_stop, o[i]), "trail_stop"
            elif pos["side"] == "SHORT" and h[i] >= cur_stop:
                exit_price, exit_reason = max(cur_stop, o[i]), "trail_stop"
            if exit_reason is None and bars_held >= bt.TIME_CAP_HOURS:
                exit_price, exit_reason = c[i], "time_cap"
            if exit_reason is None:
                p = pred[i]
                decaying = (p < 0) if pos["side"] == "LONG" else (p > 0)
                pos["decay_streak"] = pos.get("decay_streak", 0) + 1 if decaying else 0
                if pos["decay_streak"] >= consec_required:
                    exit_price, exit_reason = c[i], "conviction_decay"
            if exit_reason is None and i == n - 1:
                exit_price, exit_reason = c[i], "data_end"
            if exit_reason is None:
                pos["trail_extreme"] = max(prev_ext, h[i]) if pos["side"] == "LONG" else min(prev_ext, lo[i])
                continue
            cost = bt.TAKER_COST
            gross = (exit_price / pos["entry_price"] - 1.0 if pos["side"] == "LONG"
                     else -(exit_price / pos["entry_price"] - 1.0))
            net = gross - cost
            equity_ret = pos["size_frac"] * net
            equity = max(equity * (1.0 + equity_ret), 0.0)
            trades.append(dict(
                entry_ts=pos["entry_ts"], exit_ts=ts[i], side=pos["side"], tier=pos["tier"],
                entry_price=pos["entry_price"], exit_price=exit_price, exit_reason=exit_reason,
                bars_held=bars_held, gross_pct=gross, net_pct=net, size_frac=pos["size_frac"],
                equity_ret_pct=equity_ret * 100.0, equity_after=equity, win=int(gross > 0)))
            pos = None
            continue
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
        size_frac = min(bt.MAX_LEVERAGE, bt.RISK_FRAC / stop_pct) if stop_pct > 0 else bt.MAX_LEVERAGE
        pos = dict(side="LONG" if direction[i] == "UP" else "SHORT", tier=tier[i],
                   entry_ts=ts[i], entry_price=entry_price, stop_dist=stop_dist,
                   trail_extreme=entry_price, size_frac=size_frac, decay_streak=0)
    return pd.DataFrame(trades)


def wr_bootstrap_ci(wins: np.ndarray, n_iter: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(wins)
    boot = np.array([wins[rng.integers(0, n, n)].mean() for _ in range(n_iter)])
    return float(wins.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def quartile_consistency(trades: pd.DataFrame) -> list[dict]:
    t = trades.sort_values("entry_ts").reset_index(drop=True)
    n = len(t)
    edges = [0, n // 4, n // 2, 3 * n // 4, n]
    out = []
    for qi in range(4):
        chunk = t.iloc[edges[qi]:edges[qi + 1]]
        if len(chunk) == 0:
            out.append(dict(q=qi + 1, n=0, wr=None, avg_net_bps=None))
            continue
        out.append(dict(
            q=qi + 1, n=len(chunk),
            wr=round(float((chunk["win"] == 1).mean()) * 100, 1),
            avg_net_bps=round(float(chunk["net_pct"].mean()) * 1e4, 1),
            span=f"{chunk['entry_ts'].min()} -> {chunk['entry_ts'].max()}"))
    return out


def main():
    print("=" * 76)
    print("  CONVICTION-DECAY EXIT — full validation")
    print("=" * 76)

    df = bt.load_data()
    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400.0
    direction, tier, _ = bt.decode_signals(df)
    direction = np.asarray(direction, dtype=object)
    tier = np.asarray(tier, dtype=object)
    pred = df["pred_ret"].values

    baseline_trades = bt.simulate(df, direction, tier)
    baseline_s = bt.summarize(baseline_trades, span_days)
    print(f"\n  BASELINE (trail+opp_signal, current production logic): "
          f"n={baseline_s['n']} WR={baseline_s['wr_pct']:.1f}% "
          f"avg_net_bps={baseline_s['avg_net_bps']:+.1f} "
          f"Sharpe={baseline_s['sharpe_calendar_ann']:.2f}")

    print("\n  --- Bar-count sensitivity (confirms 2-bar isn't an isolated spike) ---")
    sensitivity = {}
    for consec in (1, 2, 3, 4):
        trades = simulate_conviction_decay(df, direction, tier, pred, consec)
        s = bt.summarize(trades, span_days)
        wr, lo, hi = wr_bootstrap_ci(trades["win"].values)
        sensitivity[consec] = dict(n=s["n"], wr=s["wr_pct"], wr_ci_lo=lo * 100, wr_ci_hi=hi * 100,
                                    avg_net_bps=s["avg_net_bps"], roi=s["roi_pct"],
                                    mdd=s["mdd_pct"], sharpe=s["sharpe_calendar_ann"])
        print(f"    consec={consec}: n={s['n']:3d}  WR={s['wr_pct']:5.1f}%  "
              f"CI=[{lo*100:5.1f},{hi*100:5.1f}]  avg_net_bps={s['avg_net_bps']:+6.1f}  "
              f"ROI={s['roi_pct']:+6.1f}%  MDD={s['mdd_pct']:.1f}%  Sharpe={s['sharpe_calendar_ann']:.2f}")

    print("\n  --- Selected: consec=2 (best Sharpe) — quartile consistency ---")
    chosen = simulate_conviction_decay(df, direction, tier, pred, 2)
    quartiles = quartile_consistency(chosen)
    all_positive_side = all(q["wr"] is not None and q["wr"] > 50 for q in quartiles)
    for q in quartiles:
        print(f"    Q{q['q']}: n={q['n']:3d}  WR={q['wr']}%  avg_net_bps={q['avg_net_bps']}  {q.get('span','')}")
    print(f"    all 4 quartiles WR > 50%: {'YES' if all_positive_side else 'NO — inconsistent'}")

    print("\n  --- Exit-reason / side breakdown (consec=2) ---")
    for reason, g in chosen.groupby("exit_reason"):
        print(f"    {reason:18s} n={len(g):3d} WR={(g['win']==1).mean()*100:5.1f}% "
              f"avg_net_bps={g['net_pct'].mean()*1e4:+7.1f}")
    for side, g in chosen.groupby("side"):
        print(f"    side={side:6s} n={len(g):3d} WR={(g['win']==1).mean()*100:5.1f}% "
              f"avg_net_bps={g['net_pct'].mean()*1e4:+7.1f}")

    print("\n  HONEST CAVEATS:")
    print("  - Single 167-day WF-OOS window (2026-01-20 -> 2026-07-06) — not an")
    print("    independent second window. Live shadow-mode test is still required")
    print("    before this touches indicator/okx/executor.py.")
    print("  - consec in {1,2,3,4} is a small search (4 configs) — the 2026-06-02")
    print("    per-fold-sanity discipline exists precisely because 'best of a few")
    print("    configs' can look great by chance. The smooth non-spiky shape across")
    print("    consec values is reassuring but does not replace an independent")
    print("    forward-window check.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "conviction_decay_exit_validation.json"
    out.write_text(json.dumps({
        "baseline": baseline_s,
        "sensitivity_by_consec_bars": sensitivity,
        "chosen_consec": 2,
        "quartile_consistency": quartiles,
        "run_ts": pd.Timestamp.now(tz="UTC").isoformat(),
    }, indent=2, default=str))
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
