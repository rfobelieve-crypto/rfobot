"""Cancellation-lead test — does per-side cancel asymmetry LEAD price at
sub-hour horizons? The first (and cheapest) life-or-death checkpoint for
the differentiated cancellation-data bet.

PRE-REGISTERED (2026-07-10, before any depth_deltas results seen):
  signal   cancel_skew(t) = (ask_cancel - bid_cancel) / (ask_cancel + bid_cancel)
           positive = asks being pulled harder -> upward vacuum hypothesis
  target   forward mid return over h ∈ {5, 15, 30, 60} minutes
           (mids from orderbook_snapshots_1m, matched by minute)
  metric   Spearman IC on NON-OVERLAPPING samples (stride = h, avoids the
           overlapping-return autocorrelation that fakes significance),
           bootstrap 95% CI on those samples.
  PASS     at checkpoint n_minutes >= 40,000 (~4 weeks of collection,
           ~2026-08-10): some horizon has CI clear of 0 AND |IC| >= 0.02
           AND first/second half agree in sign. Otherwise the bar-level
           lead claim is dead and only the squeeze-event-conditioned test
           (3-6mo) remains.
  Runs before the checkpoint print SMOKE — plumbing check, not evidence.

No thresholds tuned on data; definitions frozen at registration.
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


def load_joined() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minute_start_ms, bid_cancel_qty, ask_cancel_qty, "
                "       bid_add_qty, ask_add_qty "
                "FROM depth_deltas_1m WHERE canonical_symbol='BTC-USD' "
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
    tot = dd["bid_cancel_qty"].astype(float) + dd["ask_cancel_qty"].astype(float)
    dd["cancel_skew"] = np.where(
        tot > 0,
        (dd["ask_cancel_qty"].astype(float) - dd["bid_cancel_qty"].astype(float)) / tot,
        np.nan)
    df = dd.set_index("minute")[["cancel_skew"]].join(mid.rename("mid"), how="inner")
    return df.dropna()


def boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000):
    base = spearmanr(x, y)[0]
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        bs.append(spearmanr(x[i], y[i])[0])
    return base, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def main() -> int:
    df = load_joined()
    n = len(df)
    if n < 100:
        print(f"only {n} joined minutes — collector too young, nothing to test")
        return 0
    tag = ("POWERED CHECKPOINT" if n >= POWERED_N
           else f"SMOKE (n={n} < {POWERED_N} — plumbing check, NOT evidence)")
    span_h = (df.index.max() - df.index.min()) / 60
    print(f"{tag}\njoined minutes n={n}, span ≈ {span_h:.1f}h\n")
    print(f"{'h(min)':>7} | {'IC':>7} {'CI':>18} {'n_nonovl':>9} | halves sign agree")
    for h in HORIZONS_MIN:
        fwd = df["mid"].shift(-h) / df["mid"] - 1.0
        sub = pd.DataFrame({"x": df["cancel_skew"], "y": fwd}).dropna()
        sub = sub.iloc[::h]                     # non-overlapping stride
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
    print(f"\nPASS gate (only at n>={POWERED_N}): CI clear of 0 AND |IC|>=0.02 "
          f"AND halves agree, any horizon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
