"""
Step 1: UP/DOWN asymmetric floor — empirical sweep.

Evidence (memory_project_up_down_asymmetry.md, 2026-05-09 OOS slice):
    DOWN sign_acc beats UP sign_acc by ~6pp at every threshold.
    Need a higher |pred| floor for UP signals to selectively kill the weakest
    UP triggers (where sign_acc is worst) and bring per-trade EV up.

Method:
    Take walk-forward OOS predictions (3696 rows, v7 path-return target).
    For each (UP_floor, DN_floor) pair in a small grid, simulate trades:
        - pred >= UP_floor   → take long
        - pred <= -DN_floor  → take short
        - else                → skip (NEUTRAL)
    Run OHLC TP/SL backtest on the triggered trades (TP=0.4%, SL=0.4%
    symmetric, 4h hold, 13 bps round-trip cost).
    Report per-config:
        n_trades, n_long / n_short, WR (overall, UP-only, DOWN-only),
        avg_net_bps (overall, UP-only, DOWN-only), Sharpe, MDD, PF.

Goal:
    Find the (UP_floor, DN_floor) that brings UP_net_bps closest to (or
    above) DOWN_net_bps without losing too much trade volume.  The optimal
    pair becomes ABS_FLOOR_STRONG_UP, ABS_FLOOR_STRONG_DN (with DN kept
    at current 0.0008 as baseline).

Inputs:
    research/results/dual_model/direction_concept_drift_oos.parquet
    market_data/raw_data/binance_klines_1h.parquet
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

V7_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_concept_drift_oos.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

COST_BPS = 13.0
COST = COST_BPS / 10000.0
TP_DIST = 0.004  # 0.4% symmetric
SL_DIST = 0.004
TIMEOUT_HOURS = 4


def simulate_trades(oos: pd.DataFrame, klines: pd.DataFrame,
                    up_floor: float, dn_floor: float) -> pd.DataFrame:
    """Take long if pred >= up_floor, short if pred <= -dn_floor."""
    rows = []
    klines_idx = klines.index
    for _, r in oos.iterrows():
        pred = float(r["pred"])
        if pred >= up_floor:
            direction = "UP"
        elif pred <= -dn_floor:
            direction = "DOWN"
        else:
            continue  # NEUTRAL

        ts = pd.Timestamp(r["ts"])
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in klines_idx:
            continue
        entry_price = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) == 0:
            continue
        future = klines.loc[future_idx]

        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry_price, direction=direction,
            sl_dist=SL_DIST, rr=TP_DIST / SL_DIST,
            bars=future, timeout_bars=TIMEOUT_HOURS,
        )
        sign = 1.0 if direction == "UP" else -1.0
        gross = (exit_price / entry_price - 1.0) * sign
        net = gross - COST
        rows.append({
            "ts": ts, "pred": pred, "direction": direction,
            "gross_ret": gross, "net_ret": net,
            "win": int(gross > 0), "exit_reason": reason,
        })
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) == 0:
        return {"n": 0}
    n = len(df)
    net = df["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    return {
        "n": n, "wr": float(df["win"].mean()),
        "avg_net_bps": float(net.mean()) * 10000,
        "sharpe": float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        "cum_pct": float(cum[-1]) * 100,
        "mdd_pct": float(mdd),
        "profit_factor": float(pf),
    }


def metrics_by_dir(df: pd.DataFrame) -> dict:
    up = df[df["direction"] == "UP"]
    dn = df[df["direction"] == "DOWN"]
    return {"up": metrics(up), "down": metrics(dn), "all": metrics(df)}


def main():
    logger.info("Loading OOS predictions and klines...")
    oos = pd.read_parquet(V7_OOS)
    klines = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    logger.info("OOS: n=%d, klines: n=%d", len(oos), len(klines))

    # Baseline (current production: symmetric Strong floor 0.0008)
    print(f"\n{'='*100}")
    print(f"Baseline: symmetric Strong floor (UP=DOWN=0.0008) — what production does now")
    print(f"{'='*100}")
    baseline = simulate_trades(oos, klines, 0.0008, 0.0008)
    b = metrics_by_dir(baseline)
    for d in ["all", "up", "down"]:
        s = b[d]
        if s.get("n", 0) == 0:
            print(f"  {d.upper():<6} (n=0)")
            continue
        print(f"  {d.upper():<6} n={s['n']:>4}  WR={s['wr']*100:>5.1f}%  "
              f"net={s['avg_net_bps']:>+6.1f} bps  "
              f"sharpe={s['sharpe']:>+6.3f}  "
              f"cum={s['cum_pct']:>+6.1f}%")

    # Now sweep UP floor (keep DN at 0.0008)
    print(f"\n{'='*100}")
    print(f"Sweep UP floor, DN floor = 0.0008 (baseline) — find UP floor that fixes asymmetry")
    print(f"{'='*100}")
    print(f"  {'UP_floor':>9} {'DN_floor':>9}  {'n_UP':>5} {'n_DN':>5}  "
          f"{'WR_UP':>7} {'WR_DN':>7}  {'net_UP':>8} {'net_DN':>8}  "
          f"{'net_ALL':>9} {'cum_ALL':>9}")
    print("-" * 105)

    sweep = []
    for up_floor in [0.0008, 0.0010, 0.0012, 0.0014, 0.0016, 0.0018, 0.0020, 0.0025]:
        df = simulate_trades(oos, klines, up_floor, 0.0008)
        m = metrics_by_dir(df)
        sweep.append((up_floor, 0.0008, df, m))
        u, d_, a = m["up"], m["down"], m["all"]
        print(f"  {up_floor:>9.4f} {0.0008:>9.4f}  "
              f"{u.get('n',0):>5} {d_.get('n',0):>5}  "
              f"{u.get('wr',0)*100:>6.1f}% {d_.get('wr',0)*100:>6.1f}%  "
              f"{u.get('avg_net_bps',0):>+7.1f} {d_.get('avg_net_bps',0):>+7.1f}  "
              f"{a.get('avg_net_bps',0):>+8.1f} {a.get('cum_pct',0):>+8.1f}%")

    # Find best by overall net_bps
    best_up, best_dn, best_df, best_m = max(sweep, key=lambda x: x[3]["all"].get("avg_net_bps", -1e9))
    print(f"\n>> BEST overall by avg_net_bps:")
    print(f"   UP_floor={best_up}, DN_floor={best_dn}")
    print(f"   overall n={best_m['all']['n']}, WR={best_m['all']['wr']*100:.1f}%, "
          f"net={best_m['all']['avg_net_bps']:+.1f} bps, Sharpe={best_m['all']['sharpe']:+.3f}")
    print(f"   UP n={best_m['up']['n']}, net={best_m['up']['avg_net_bps']:+.1f} bps")
    print(f"   DOWN n={best_m['down']['n']}, net={best_m['down']['avg_net_bps']:+.1f} bps")

    # Also try sweep DN floor (kept UP=baseline 0.0008) — sanity check
    print(f"\n{'='*100}")
    print(f"Sanity: sweep DN floor with UP=0.0008 — to verify DN side already optimal at 0.0008")
    print(f"{'='*100}")
    print(f"  {'UP_floor':>9} {'DN_floor':>9}  {'n_UP':>5} {'n_DN':>5}  "
          f"{'WR_UP':>7} {'WR_DN':>7}  {'net_UP':>8} {'net_DN':>8}  "
          f"{'net_ALL':>9}")
    print("-" * 95)
    for dn_floor in [0.0005, 0.0008, 0.0010, 0.0012, 0.0015]:
        df = simulate_trades(oos, klines, 0.0008, dn_floor)
        m = metrics_by_dir(df)
        u, d_, a = m["up"], m["down"], m["all"]
        print(f"  {0.0008:>9.4f} {dn_floor:>9.4f}  "
              f"{u.get('n',0):>5} {d_.get('n',0):>5}  "
              f"{u.get('wr',0)*100:>6.1f}% {d_.get('wr',0)*100:>6.1f}%  "
              f"{u.get('avg_net_bps',0):>+7.1f} {d_.get('avg_net_bps',0):>+7.1f}  "
              f"{a.get('avg_net_bps',0):>+8.1f}")


if __name__ == "__main__":
    main()
