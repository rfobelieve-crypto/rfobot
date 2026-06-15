"""
Task 7: LDC ensemble evaluation.

Three questions to answer (in order):
    1. Standalone: are LDC entries net-positive after OHLC TP/SL + cost?
       If clearly negative → LDC has no alpha, no integration value.

    2. Correlation with v7: does LDC fire at different times than v7?
       High overlap = redundant info; low overlap = potential ensemble.

    3. Ensemble: does v7 + LDC confirm-or-veto layer outperform pure v7?
       This is the actual production question.

Inputs:
    research/results/ldc_entries.parquet                  (77 LDC entries)
    research/results/dual_model/direction_concept_drift_oos.parquet (v7 OOS, 3696 rows)
    market_data/raw_data/binance_klines_1h.parquet        (OHLC for TP/SL sim)
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

LDC_ENTRIES = PROJECT_ROOT / "research" / "results" / "ldc_entries.parquet"
V7_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_concept_drift_oos.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

COST_BPS = 13.0
COST = COST_BPS / 10000.0
TIMEOUT_HOURS = 24


# ─── 1. LDC standalone OHLC backtest ────────────────────────────────────────

def backtest_ldc_standalone(klines: pd.DataFrame, entries: pd.DataFrame,
                              tp: float, sl: float) -> pd.DataFrame:
    rows = []
    klines_idx = klines.index
    for ts, r in entries.iterrows():
        if ts not in klines_idx:
            continue
        entry_price = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) == 0:
            continue
        future = klines.loc[future_idx]
        direction = "UP" if r["direction"] == "long" else "DOWN"
        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry_price,
            direction=direction,
            sl_dist=sl,
            rr=tp / sl,
            bars=future,
            timeout_bars=TIMEOUT_HOURS,
        )
        sign = 1.0 if direction == "UP" else -1.0
        gross = (exit_price / entry_price - 1.0) * sign
        net = gross - COST
        rows.append({
            "ts": ts, "direction": direction,
            "gross_ret": gross, "net_ret": net,
            "win": int(gross > 0), "exit_reason": reason,
            "bars_held": bars_held,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str) -> dict:
    if df.empty or len(df) == 0:
        return {"name": name, "n": 0}
    n = len(df)
    net = df["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    return {
        "name": name, "n": n,
        "wr": float(df["win"].mean()),
        "avg_gross_bps": float(df["gross_ret"].mean()) * 10000,
        "avg_net_bps": float(net.mean()) * 10000,
        "sharpe": float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        "cum_pct": float(cum[-1]) * 100,
        "mdd_pct": float(mdd),
        "profit_factor": float(pf),
    }


def print_row(s: dict):
    if s.get("n", 0) == 0:
        print(f"  {s['name']:<35} (no data)")
        return
    print(f"  {s['name']:<35} {s['n']:>5d} {s['wr']*100:>5.1f}% "
          f"{s['avg_net_bps']:>+7.1f} {s['sharpe']:>+7.3f} "
          f"{s['cum_pct']:>+6.1f}% {s['mdd_pct']:>+6.1f}% "
          f"{s['profit_factor']:>5.2f}")


# ─── 2. v7 vs LDC timing overlap ─────────────────────────────────────────────

def compute_overlap(v7_oos: pd.DataFrame, ldc_entries: pd.DataFrame,
                    window_hours: int = 4) -> dict:
    """For each LDC entry, find v7 OOS rows within ±window_hours.
    Build a confusion matrix of (v7 direction, LDC direction)."""
    v7 = v7_oos.copy()
    v7["ts"] = pd.to_datetime(v7["ts"])
    if v7["ts"].dt.tz is not None:
        v7["ts"] = v7["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    v7["v7_dir"] = np.sign(v7["pred"]).astype(int)

    ldc = ldc_entries.copy()
    ldc.index = pd.to_datetime(ldc.index)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)
    ldc["ldc_dir"] = np.where(ldc["direction"] == "long", 1, -1)

    matched = []
    for ldc_ts, r in ldc.iterrows():
        w = v7[(v7["ts"] >= ldc_ts - pd.Timedelta(hours=window_hours))
               & (v7["ts"] <= ldc_ts + pd.Timedelta(hours=window_hours))]
        if len(w) == 0:
            continue
        # Closest v7 row
        closest = w.iloc[(w["ts"] - ldc_ts).abs().argsort()[:1]]
        matched.append({
            "ldc_ts": ldc_ts, "ldc_dir": int(r["ldc_dir"]),
            "v7_ts": closest["ts"].iloc[0], "v7_dir": int(closest["v7_dir"].iloc[0]),
            "v7_pred": float(closest["pred"].iloc[0]),
        })
    m = pd.DataFrame(matched)
    print(f"\n=== v7 vs LDC overlap (±{window_hours}h matching) ===")
    print(f"  LDC entries total: {len(ldc)}")
    print(f"  Matched to v7 OOS: {len(m)}")
    if len(m) == 0:
        return {"matched": 0}

    # Confusion matrix
    agree = (m["v7_dir"] == m["ldc_dir"]).sum()
    disagree = (m["v7_dir"] != m["ldc_dir"]).sum()
    print(f"  Agreement (same dir): {agree} ({agree/len(m)*100:.1f}%)")
    print(f"  Disagreement:         {disagree} ({disagree/len(m)*100:.1f}%)")
    print(f"  → If random, 50/50 expected. {agree/len(m)*100:.1f}% agreement says:")
    if agree / len(m) > 0.65:
        print("    sources are CORRELATED — limited ensemble value")
    elif agree / len(m) < 0.35:
        print("    sources are ANTI-correlated — potentially useful as VETO layer")
    else:
        print("    sources are INDEPENDENT — best case for ensemble alpha")
    return {"matched": len(m), "agree": int(agree), "disagree": int(disagree), "df": m}


# ─── 3. Confirm-or-veto ensemble backtest ────────────────────────────────────

def backtest_ensemble(klines: pd.DataFrame, v7_oos: pd.DataFrame,
                       ldc_entries: pd.DataFrame, sl: float, tp: float,
                       conf_thresh: float, lookback_hours: int = 4) -> pd.DataFrame:
    """
    For each v7 OOS row with |pred| >= conf_thresh:
        direction = sign(v7_pred)
        Look at LDC entries in [v7_ts - lookback_hours, v7_ts]
        If most recent LDC entry direction == v7 direction → take trade (CONFIRMED)
        If most recent LDC entry direction != v7 direction → SKIP (vetoed by LDC)
        If no LDC entry in window → take trade (pure v7, no LDC opinion)
    """
    v7 = v7_oos.copy()
    v7["ts"] = pd.to_datetime(v7["ts"])
    if v7["ts"].dt.tz is not None:
        v7["ts"] = v7["ts"].dt.tz_convert("UTC").dt.tz_localize(None)

    ldc = ldc_entries.copy()
    ldc.index = pd.to_datetime(ldc.index)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)
    ldc_long_ts = ldc[ldc["direction"] == "long"].index
    ldc_short_ts = ldc[ldc["direction"] == "short"].index

    klines_idx = klines.index
    rows_pure = []
    rows_ensemble = []

    for _, r in v7.iterrows():
        pred = float(r["pred"])
        if abs(pred) < conf_thresh:
            continue
        ts = pd.Timestamp(r["ts"])
        if ts not in klines_idx:
            continue

        entry_price = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) == 0:
            continue
        future = klines.loc[future_idx]
        v7_dir_str = "UP" if pred > 0 else "DOWN"
        v7_dir_int = 1 if pred > 0 else -1

        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry_price, direction=v7_dir_str,
            sl_dist=sl, rr=tp / sl, bars=future, timeout_bars=TIMEOUT_HOURS,
        )
        sign = 1.0 if v7_dir_str == "UP" else -1.0
        gross = (exit_price / entry_price - 1.0) * sign
        net = gross - COST

        # Most recent LDC entry within lookback
        recent_long = ldc_long_ts[(ldc_long_ts >= ts - pd.Timedelta(hours=lookback_hours))
                                    & (ldc_long_ts <= ts)]
        recent_short = ldc_short_ts[(ldc_short_ts >= ts - pd.Timedelta(hours=lookback_hours))
                                      & (ldc_short_ts <= ts)]
        most_recent_ts = None
        ldc_dir = 0
        if len(recent_long) > 0:
            most_recent_ts = recent_long.max()
            ldc_dir = 1
        if len(recent_short) > 0 and (most_recent_ts is None
                                        or recent_short.max() > most_recent_ts):
            most_recent_ts = recent_short.max()
            ldc_dir = -1

        # Pure v7 baseline: always take trade
        rows_pure.append({"ts": ts, "v7_dir": v7_dir_int, "ldc_dir": ldc_dir,
                          "gross_ret": gross, "net_ret": net, "win": int(gross > 0)})

        # Ensemble: skip if LDC vetoes
        if ldc_dir != 0 and ldc_dir != v7_dir_int:
            continue  # VETOED
        rows_ensemble.append({"ts": ts, "v7_dir": v7_dir_int, "ldc_dir": ldc_dir,
                              "gross_ret": gross, "net_ret": net,
                              "win": int(gross > 0),
                              "ldc_confirms": int(ldc_dir == v7_dir_int)})

    return pd.DataFrame(rows_pure), pd.DataFrame(rows_ensemble)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading data...")
    klines = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    ldc = pd.read_parquet(LDC_ENTRIES)
    v7 = pd.read_parquet(V7_OOS)
    logger.info("Klines: n=%d, LDC entries: n=%d, v7 OOS: n=%d",
                len(klines), len(ldc), len(v7))

    # ─── Q1: LDC standalone ─────
    print(f"\n{'='*95}")
    print(f"Q1: LDC standalone OHLC backtest (cost={COST_BPS:.0f} bps)")
    print(f"{'='*95}")
    print(f"  {'name':<35} {'n':>5} {'WR':>6} {'net_bps':>8} {'sharpe':>7} "
          f"{'cum%':>7} {'MDD%':>7} {'PF':>5}")
    print("-" * 95)
    for label, (tp, sl) in [
        ("LDC | TP=0.5% SL=0.3%", (0.005, 0.003)),
        ("LDC | TP=0.4% SL=0.4%", (0.004, 0.004)),
        ("LDC | TP=1.0% SL=0.5%", (0.010, 0.005)),
        ("LDC | TP=0.3% SL=0.3%", (0.003, 0.003)),
    ]:
        bt = backtest_ldc_standalone(klines, ldc, tp, sl)
        print_row(summarize(bt, label))

    # ─── Q2: v7 vs LDC overlap ─────
    compute_overlap(v7, ldc, window_hours=4)

    # ─── Q3: ensemble ─────
    print(f"\n{'='*95}")
    print(f"Q3: Ensemble — pure v7 vs v7 + LDC confirm-or-veto")
    print(f"{'='*95}")
    print(f"  Strategy: if LDC entry in past 4h has opposite dir to v7 → SKIP trade")
    print(f"  Otherwise: take v7 trade as usual")
    print(f"  TP/SL: 0.4%/0.4% sym (best from earlier paper_trading_tpsl)")
    print()
    print(f"  {'config':<55} {'n':>5} {'WR':>6} {'net_bps':>8} {'sharpe':>7} "
          f"{'cum%':>7} {'MDD%':>7}")
    print("-" * 95)

    # Try different conf thresholds
    for conf_thr in [0.0005, 0.0008, 0.0010, 0.0012, 0.0015, 0.0020]:
        pure, ens = backtest_ensemble(klines, v7, ldc,
                                        sl=0.004, tp=0.004,
                                        conf_thresh=conf_thr,
                                        lookback_hours=4)
        s_pure = summarize(pure, f"v7 pure |pred|>={conf_thr*10000:.0f}bps")
        s_ens = summarize(ens, f"v7+LDC veto |pred|>={conf_thr*10000:.0f}bps")
        print_row(s_pure)
        print_row(s_ens)
        # Show how often LDC vetoed
        n_pure = len(pure)
        n_ens = len(ens)
        if n_pure > 0:
            print(f"    └ LDC vetoed {n_pure - n_ens} of {n_pure} ({(n_pure-n_ens)/n_pure*100:.1f}%); "
                  f"of accepted, {ens['ldc_confirms'].sum() if not ens.empty else 0} confirmed by LDC")
        print()


if __name__ == "__main__":
    main()
