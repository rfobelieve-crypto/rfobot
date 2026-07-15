"""Cancellation-SHOCK lead test — second pre-registered variant family.

Hypothesis (user, 2026-07-15): the information in cancellation data is in
its CHANGE MAGNITUDE vs its own recent baseline, not in the level — bursts
are "someone doing something unusual right now"; levels are regime/session
colour.

PRE-REGISTERED 2026-07-15. Honesty caveat: registered after ~6 days of
depth_deltas had been eyeballed on the monitor chart (the original
cancel_lead_ic.py was registered 2026-07-10 before any data). Definitions
below are frozen NOW and must not be tuned; the powered verdict uses mostly
yet-uncollected data (n>=40,000, ~2026-08-10, same checkpoint as family 1).

FROZEN DEFINITIONS (trailing-only, no look-ahead):
  baseline(t)        = median(total_cancel over trailing 60 min, min 30)
  intensity_shock(t) = total_cancel(t) / baseline(t)          [unsigned]
  skew_raw(t)        = (ask_cancel - bid_cancel)/(ask_cancel + bid_cancel)
  skew_shock(t)      = skew_raw(t) - mean(skew_raw over trailing 60 min)

  TEST A (direction): Spearman IC of skew_shock vs signed forward mid
         return, h in {5,15,30,60} min, non-overlapping stride.
  TEST B (volatility): Spearman IC of intensity_shock vs |forward mid
         return|, same horizons/stride.

  PASS  same bar as family 1 (CI clear of 0, |IC|>=0.02, halves agree)
        BUT family-wise: with 2 registered families, a single marginal
        pass in one family = weak evidence; strong claim needs the passing
        cell to survive at 99% CI or replicate in the other family/next
        30 days. Runs before n>=40,000 print SMOKE — plumbing, not evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn

RNG = np.random.default_rng(7)
HORIZONS_MIN = (5, 15, 30, 60)
POWERED_N = 40_000
BASE_WIN = 60
BASE_MINP = 30


def load_joined() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minute_start_ms, bid_cancel_qty, ask_cancel_qty "
                "FROM depth_deltas_1m WHERE canonical_symbol='BTC-USD' "
                "AND exchange='binance' "  # spot series only — the registered one
                "ORDER BY minute_start_ms")
            dd = pd.DataFrame(cur.fetchall())
            cur.execute(
                "SELECT ts_ms, mid_price FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol='BTC-USD' AND ts_ms >= %s "
                "ORDER BY ts_ms",
                (int(dd["minute_start_ms"].min()) if len(dd) else 0,))
            ob = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    if dd.empty or ob.empty:
        return pd.DataFrame()
    dd["minute"] = dd["minute_start_ms"] // 60_000
    ob["minute"] = ob["ts_ms"] // 60_000
    mid = ob.groupby("minute")["mid_price"].last().astype(float)

    bid = dd["bid_cancel_qty"].astype(float)
    ask = dd["ask_cancel_qty"].astype(float)
    tot = bid + ask
    baseline = tot.rolling(BASE_WIN, min_periods=BASE_MINP).median()
    dd["intensity_shock"] = np.where(baseline > 0, tot / baseline, np.nan)
    skew_raw = pd.Series(np.where(tot > 0, (ask - bid) / tot, np.nan))
    dd["skew_shock"] = skew_raw - skew_raw.rolling(BASE_WIN, min_periods=BASE_MINP).mean()

    df = (dd.set_index("minute")[["intensity_shock", "skew_shock"]]
          .join(mid.rename("mid"), how="inner"))
    return df.dropna(subset=["mid"])


def boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000):
    base = spearmanr(x, y)[0]
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        bs.append(spearmanr(x[i], y[i])[0])
    return base, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def run_block(df: pd.DataFrame, sig_col: str, target_abs: bool, label: str):
    print(f"\n── {label} ──")
    print(f"{'h(min)':>7} | {'IC':>7} {'CI':>18} {'n_nonovl':>9} | halves sign agree")
    for h in HORIZONS_MIN:
        fwd = df["mid"].shift(-h) / df["mid"] - 1.0
        if target_abs:
            fwd = fwd.abs()
        sub = pd.DataFrame({"x": df[sig_col], "y": fwd}).dropna()
        sub = sub.iloc[::h]
        if len(sub) < 30:
            print(f"{h:>7} | insufficient non-overlapping samples ({len(sub)})")
            continue
        x, y = sub["x"].to_numpy(), sub["y"].to_numpy()
        ic, lo, hi = boot_ci(x, y)
        half = len(sub) // 2
        s1 = spearmanr(x[:half], y[:half])[0]
        s2 = spearmanr(x[half:], y[half:])[0]
        agree = np.sign(s1) == np.sign(s2)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"{h:>7} | {ic:>+7.3f} [{lo:>+7.3f},{hi:>+7.3f}]{star} {len(sub):>8} | "
              f"{s1:+.3f}/{s2:+.3f} {'✓' if agree else '✗'}")


def main() -> int:
    df = load_joined()
    n = len(df)
    if n < 200:
        print(f"only {n} joined minutes — collector too young")
        return 0
    tag = ("POWERED CHECKPOINT" if n >= POWERED_N
           else f"SMOKE (n={n} < {POWERED_N} — plumbing check, NOT evidence)")
    span_h = (df.index.max() - df.index.min()) / 60
    print(f"{tag}\njoined minutes n={n}, span ≈ {span_h:.1f}h")

    run_block(df, "skew_shock", target_abs=False,
              label="TEST A: skew_shock → signed forward return (direction)")
    run_block(df, "intensity_shock", target_abs=True,
              label="TEST B: intensity_shock → |forward return| (volatility)")

    print(f"\nPASS gate (only at n>={POWERED_N}): CI clear of 0 AND |IC|>=0.02 AND "
          f"halves agree; family-wise caveat in module docstring applies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
