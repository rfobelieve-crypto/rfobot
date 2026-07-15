"""Cancel-skew FLIP events — third registered variant (event-study form).

Hypothesis (user, 2026-07-15): what matters is when the abandoned side
SWITCHES — red→green (bid-pulling stops, ask-pulling starts = ceiling being
dismantled = bullish) and green→red (reverse = bearish).

PRE-REGISTERED 2026-07-15 (same honesty caveat as cancel_shock_ic.py:
~6 days of data already eyeballed; definitions frozen now, no tuning).
Family-wise: this is the THIRD registered family — a single marginal pass
across three families is weak; see cancel_shock_ic.py docstring.

FROZEN DEFINITIONS (identical transforms to the monitor/review charts):
  skew_s(t)   = 15m rolling mean of (skew_raw − full-window mean)
  deep GREEN  = skew_s >= +0.30 sustained >= 4 min
  deep RED    = skew_s <= −0.30 sustained >= 4 min
  FLIP        = a deep episode of one colour starts within <= 60 min of the
                END of a deep episode of the opposite colour.
                Direction: red→green = bullish call, green→red = bearish.
  Event time  = first minute of the NEW episode.
  Outcome     = signed forward mid return at {15, 30, 60} min from event
                (sign-adjusted: bullish flip keeps sign, bearish flips it,
                so >0 = hypothesis-consistent).
  VERDICT     only at n_flips >= 30 per direction (matches the compound-
  trigger discipline from mistake.md 2026-06-20); before that: descriptive.

Also prints raw zero-crossing count to show why unqualified flips are noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn

SMOOTH = 15
DEEP = 0.30
MIN_LEN = 4
GAP_MAX = 60
HORIZONS = (15, 30, 60)


def load() -> pd.DataFrame:
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
                "WHERE canonical_symbol='BTC-USD' AND ts_ms >= %s ORDER BY ts_ms",
                (int(dd["minute_start_ms"].min()) if len(dd) else 0,))
            ob = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    dd["minute"] = dd["minute_start_ms"] // 60_000
    ob["minute"] = ob["ts_ms"] // 60_000
    mid = ob.groupby("minute")["mid_price"].last().astype(float)
    bid = dd["bid_cancel_qty"].astype(float)
    ask = dd["ask_cancel_qty"].astype(float)
    tot = bid + ask
    skew = pd.Series(np.where(tot > 0, (ask - bid) / tot, np.nan))
    skew = skew - skew.mean()
    dd["skew_s"] = skew.rolling(SMOOTH, min_periods=max(3, SMOOTH // 3)).mean()
    df = dd.set_index("minute")[["skew_s"]].join(mid.rename("mid"), how="inner")
    return df.dropna()


def episodes(z: pd.Series, sign: int) -> list[tuple[int, int]]:
    """Contiguous runs (start_minute, end_minute) where sign*z >= DEEP, len>=MIN_LEN."""
    mask = (sign * z) >= DEEP
    out, start = [], None
    prev = None
    for m, flag in mask.items():
        if flag and start is None:
            start = m
        elif not flag and start is not None:
            if prev - start + 1 >= MIN_LEN:
                out.append((start, prev))
            start = None
        prev = m
    if start is not None and prev - start + 1 >= MIN_LEN:
        out.append((start, prev))
    return out


def main() -> int:
    df = load()
    z = df["skew_s"]
    n = len(df)
    span_h = (df.index.max() - df.index.min()) / 60
    zero_cross = int((np.sign(z).diff().abs() > 0).sum())
    print(f"n={n} minutes, span ≈ {span_h:.1f}h")
    print(f"raw zero-crossings of smoothed skew: {zero_cross} "
          f"(≈ {zero_cross / (span_h / 24):.0f}/day — unqualified flips are noise)\n")

    greens = episodes(z, +1)
    reds = episodes(z, -1)
    print(f"deep GREEN episodes (>=+{DEEP}, >={MIN_LEN}min): {len(greens)}")
    print(f"deep RED episodes   (<=-{DEEP}, >={MIN_LEN}min): {len(reds)}")

    tagged = ([(s, e, +1) for s, e in greens] + [(s, e, -1) for s, e in reds])
    tagged.sort()
    flips = []  # (event_minute, direction) direction=+1 bullish red→green
    for (s1, e1, c1), (s2, e2, c2) in zip(tagged, tagged[1:]):
        if c1 != c2 and 0 < s2 - e1 <= GAP_MAX:
            flips.append((s2, c2))

    bull = [m for m, d in flips if d > 0]
    bear = [m for m, d in flips if d < 0]
    print(f"\nFLIP events (opposite deep episode within {GAP_MAX}min): "
          f"red→green {len(bull)}, green→red {len(bear)}")
    verdict_ready = min(len(bull), len(bear)) >= 30
    print("STATUS:", "VERDICT-READY" if verdict_ready
          else f"DESCRIPTIVE ONLY (need >=30 per direction)")

    mid = df["mid"]
    for h in HORIZONS:
        fwd = (mid.shift(-h) / mid - 1.0) * 1e4  # bps
        rows = []
        for m, d in flips:
            if m in fwd.index and not np.isnan(fwd.loc[m]):
                rows.append(d * fwd.loc[m])  # sign-adjusted
        if rows:
            arr = np.array(rows)
            print(f"h={h:>2}m: sign-adj fwd ret mean {arr.mean():+7.1f} bps, "
                  f"median {np.median(arr):+7.1f}, hit-rate {(arr > 0).mean():.0%}, "
                  f"n={len(arr)}")
    print("\n(sign-adjusted: >0 = flip direction was right. Descriptive until "
          "n>=30/direction; definitions frozen, do not tune.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
