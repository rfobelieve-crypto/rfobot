"""
Compare v7 vs v8 direction model — by simulated OHLC TP/SL trade PnL.

Both models' OOS predictions over the same 3696-bar window are loaded.
For each bar:
    direction = sign(pred)         (skip if |pred| below floor)
    entry     = close[t]
    apply TP=0.5%, SL=0.3% with intra-bar high/low (paper_trading_tpsl style)
    timeout = 4 bars (model horizon)
    cost    = 13 bps round-trip

Reports per model: n, WR, avg_net_bps, Sharpe, MDD, profit_factor.
Sliced by: overall, |pred| quintile, fold-group time split.

Caveats:
    - SL/TP are symmetric per-direction (long: TP up 0.5%, SL down 0.3%;
      short: TP down 0.5%, SL up 0.3%) — both legs use the same |TP|/|SL|.
    - Same conservative ambig-bar rule as paper_trading_tpsl.py (SL first).
"""
from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

V7_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_concept_drift_oos.parquet"
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

TIMEOUT = 4
COST_BPS = 13.0
COST = COST_BPS / 10000.0

# Set in main() from CLI args
TP_DIST = 0.005
SL_DIST = 0.003


def load_klines() -> pd.DataFrame:
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]]
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    return k.sort_index()


def simulate_trades(oos: pd.DataFrame, klines: pd.DataFrame,
                    pred_floor: float = 0.0) -> pd.DataFrame:
    """For each OOS row with |pred| >= floor, simulate TP/SL trade."""
    rows = []
    klines_idx = klines.index

    for _, r in oos.iterrows():
        pred = float(r["pred"])
        if abs(pred) < pred_floor:
            continue
        ts = pd.Timestamp(r["ts"])
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

        # Need bars at ts (entry) and ts+1..+H (price action)
        if ts not in klines_idx:
            continue
        entry_price = float(klines.loc[ts, "close"])

        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) < 1:
            continue
        future = klines.loc[future_idx]

        direction = "UP" if pred > 0 else "DOWN"
        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry_price,
            direction=direction,
            sl_dist=SL_DIST,
            rr=TP_DIST / SL_DIST,  # = 1.667
            bars=future,
            timeout_bars=TIMEOUT,
        )
        sign = 1.0 if direction == "UP" else -1.0
        gross = (exit_price / entry_price - 1.0) * sign
        net = gross - COST
        rows.append({
            "ts": ts,
            "pred": pred,
            "abs_pred": abs(pred),
            "direction": direction,
            "gross_ret": gross,
            "net_ret": net,
            "exit_reason": reason,
            "bars_held": bars_held,
            "win": int(gross > 0),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0}
    n = len(df)
    net = df["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    reasons = df["exit_reason"].value_counts().to_dict()
    return {
        "label": label,
        "n": n,
        "wr": float(df["win"].mean()),
        "avg_gross_bps": float(df["gross_ret"].mean()) * 10000,
        "avg_net_bps": float(net.mean()) * 10000,
        "sharpe": float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        "cum_pct": float(cum[-1]) * 100,
        "mdd_pct": float(mdd),
        "profit_factor": float(pf),
        "exit_reasons": reasons,
    }


def print_row(s: dict):
    if s.get("n", 0) == 0:
        print(f"  {s['label']:<28} (no data)")
        return
    r = s["exit_reasons"]
    tp = r.get("tp", 0)
    sln = r.get("sl", 0) + r.get("sl_ambig", 0)
    to = r.get("timeout", 0)
    print(f"  {s['label']:<28} {s['n']:>5d} {s['wr']*100:>5.1f}% "
          f"{s['avg_net_bps']:>+7.1f} {s['sharpe']:>+7.3f} "
          f"{s['cum_pct']:>+6.1f}% {s['mdd_pct']:>+6.1f}% "
          f"{s['profit_factor']:>5.2f} {tp:>4}/{sln:>4}/{to:>3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=float, default=0.005,
                        help="TP distance for OHLC simulation")
    parser.add_argument("--sl", type=float, default=0.003,
                        help="SL distance for OHLC simulation")
    parser.add_argument("--v8-suffix", type=str, default="TP50_SL30",
                        help="v8 OOS file suffix matching its label barriers")
    args = parser.parse_args()

    global TP_DIST, SL_DIST
    TP_DIST = args.tp
    SL_DIST = args.sl
    v8_path = RESULTS_DIR / f"direction_v8_{args.v8_suffix}_oos.parquet"

    klines = load_klines()
    logger.info("Klines: n=%d, range %s ~ %s",
                len(klines), klines.index.min(), klines.index.max())
    logger.info("OHLC simulation: TP=%.2f%% SL=%.2f%% v8=%s",
                TP_DIST*100, SL_DIST*100, v8_path.name)

    v7 = pd.read_parquet(V7_OOS)
    v8 = pd.read_parquet(v8_path)
    logger.info("v7 OOS: n=%d  v8 OOS: n=%d", len(v7), len(v8))

    # Align on ts
    v7 = v7.rename(columns={"y": "y_v7"}) if "y" in v7.columns else v7
    v7_pred = v7[["ts", "pred"]].copy()
    v8_pred = v8[["ts", "pred"]].copy()

    # No floor — every signal triggers (most permissive view)
    print(f"\n{'='*100}")
    print(f"OHLC-AWARE TRADE PnL  (TP={TP_DIST*100:.1f}% SL={SL_DIST*100:.1f}% "
          f"timeout={TIMEOUT}h cost={COST_BPS:.0f}bps)")
    print(f"{'='*100}")
    print(f"  {'slice':<28} {'n':>5} {'WR':>6} {'net_bps':>8} {'sharpe':>7} "
          f"{'cum%':>7} {'MDD%':>7} {'PF':>5} {'TP/SL/TO':>14}")
    print("-" * 100)

    v7_trades = simulate_trades(v7_pred, klines)
    v8_trades = simulate_trades(v8_pred, klines)
    print(">> all signals (no |pred| floor):")
    print_row(summarize(v7_trades, "v7 (TWAP target)"))
    print_row(summarize(v8_trades, "v8 (path-clip target)"))

    # |pred| quintile slice
    print(f"\n>> |pred| quintile slice (does each model's high-conviction subset trade better?):")
    for label, df in [("v7", v7_trades), ("v8", v8_trades)]:
        if df.empty:
            continue
        df = df.copy()
        try:
            df["q"] = pd.qcut(df["abs_pred"], q=5,
                                labels=["Q1_lo", "Q2", "Q3", "Q4", "Q5_hi"],
                                duplicates="drop")
        except ValueError:
            continue
        for q, sub in df.groupby("q", observed=True):
            print_row(summarize(sub, f"{label} | {q}"))

    # Time split — first half vs second half
    print(f"\n>> Time split (front-half vs back-half — drift / regime check):")
    for label, df in [("v7", v7_trades), ("v8", v8_trades)]:
        if df.empty:
            continue
        df = df.sort_values("ts").reset_index(drop=True)
        mid = len(df) // 2
        print_row(summarize(df.iloc[:mid], f"{label} | front half"))
        print_row(summarize(df.iloc[mid:], f"{label} | back  half"))


if __name__ == "__main__":
    main()
