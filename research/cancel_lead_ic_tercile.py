"""F1b — cancellation lead-IC conditioned on trading-volume tercile.

PRE-REGISTERED 2026-07-17 (TODO.md §4.5, written before this script or any
tercile-split result existed). Core hypothesis: cancel-flow's information
value is inversely related to trading activity — two channels compete for
"who's informed right now": when taker flow is loud, price is driven by
aggressors and cancels are mostly protective noise/redundant with what taker
flow already says; when taker flow is quiet, the order-book (cancel/add) is
the only thing still moving, so market-maker repositioning leaks direction
first. Prediction: cancel_lead_ic significant in LOW-volume minutes,
~zero in HIGH-volume minutes.

This is the standalone-signal case the user pointed at 2026-07-20: unlike
the shock-gated v1 playbooks (absorption/true_break/vacuum, all requiring
shock>=3.0 or |skew15|/|net15|>=0.30), F1b's whole premise is that the
actionable regime is exactly the QUIET one those gates are built to ignore
(cf. this morning's 07:44-07:57 case: net15 peaked at +0.178, well under
the 0.30 vacuum threshold, shock never cleared the gate — zero events
logged despite a textbook pre-breakout leak). If F1b passes, a genuinely
gate-free "quiet-period lean" state becomes admissible for a def v2 — not
before.

Definitions reused verbatim from already-frozen specs (no new tuning):
  cancel_skew(t)   = (ask_cancel - bid_cancel)/(ask_cancel+bid_cancel)
                     [cancel_lead_ic.py, frozen 2026-07-10]
  vshock(t)        = volume_usd(t) / trailing-60m-median(volume_usd)
                     [cancel_playbook_watcher.compute_features, frozen
                     v1-2026-07-16] — reused as the volume-activity measure
                     so this test shares one volume definition with the
                     rest of the system rather than inventing a second one.
  tercile           rank-based cut of vshock over the analysis sample
                     (low/mid/high, ~equal count each)
  target            forward mid return, h in {5,15,30,60}min, non-overlapping
                     stride (same anti-autocorrelation discipline as F1/F2)
  metric            Spearman IC + bootstrap 95% CI, first/second-half sign
                     agreement, PER TERCILE PER HORIZON

PASS (per cell): CI clear of 0 AND |IC|>=0.02 AND halves agree — same bar
as F1/F2. Power caveat: splitting into terciles divides the sample ~3x, so
the POWERED_N checkpoint (40,000 joined minutes) only gives ~13,300 per
tercile — this test needs MORE calendar time than F1's raw checkpoint to
be properly powered, not less. Runs before that only as SMOKE (plumbing,
not evidence) — printed and labelled honestly, never treated as a verdict.

No thresholds tuned on data; definitions frozen at registration (2026-07-17)
before any tercile-conditioned number was seen.
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
POWERED_N = 40_000          # same checkpoint as F1; ~/3 per tercile in practice
BASE_WIN = 60
BASE_MINP = 30
TERCILE_LABELS = ("low-vol", "mid-vol", "high-vol")


def load_joined() -> pd.DataFrame:
    """cancel_skew + vshock + mid, minute-indexed, inner-joined."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minute_start_ms, bid_cancel_qty, ask_cancel_qty "
                "FROM depth_deltas_1m WHERE canonical_symbol='BTC-USD' "
                "AND exchange='binance' ORDER BY minute_start_ms")
            dd = pd.DataFrame(cur.fetchall())
            cur.execute(
                "SELECT window_start, volume_usd FROM flow_bars_1m "
                "WHERE canonical_symbol='BTC-USD' AND exchange_scope='all' "
                "ORDER BY window_start")
            fb = pd.DataFrame(cur.fetchall())
            cur.execute(
                "SELECT ts_ms, mid_price FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol='BTC-USD' ORDER BY ts_ms")
            ob = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    if dd.empty or fb.empty or ob.empty:
        return pd.DataFrame()

    dd["minute"] = dd["minute_start_ms"] // 60_000
    fb["minute"] = fb["window_start"] // 60_000
    ob["minute"] = ob["ts_ms"] // 60_000

    bid = dd["bid_cancel_qty"].astype(float)
    ask = dd["ask_cancel_qty"].astype(float)
    tot = bid + ask
    dd["cancel_skew"] = np.where(tot > 0, (ask - bid) / tot, np.nan)

    vol = fb.set_index("minute")["volume_usd"].astype(float).sort_index()
    vbase = vol.rolling(BASE_WIN, min_periods=BASE_MINP).median()
    vshock = (vol / vbase.replace(0, np.nan)).rename("vshock")

    mid = ob.groupby("minute")["mid_price"].last().astype(float).rename("mid")

    df = (dd.set_index("minute")[["cancel_skew"]]
          .join(vshock, how="inner")
          .join(mid, how="inner"))
    return df.dropna()


def boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000):
    base = spearmanr(x, y)[0]
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        bs.append(spearmanr(x[i], y[i])[0])
    return base, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def run_tercile(df: pd.DataFrame, label: str) -> None:
    print(f"\n── {label} (n={len(df)}) ──")
    print(f"{'h(min)':>7} | {'IC':>7} {'CI':>18} {'n_nonovl':>9} | halves sign agree")
    for h in HORIZONS_MIN:
        fwd = df["mid"].shift(-h) / df["mid"] - 1.0
        sub = pd.DataFrame({"x": df["cancel_skew"], "y": fwd}).dropna()
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
    if n < 300:
        print(f"only {n} joined minutes — collector too young for a 3-way split")
        return 0
    tag = ("POWERED CHECKPOINT" if n >= POWERED_N
           else f"SMOKE (n={n} < {POWERED_N} — plumbing check, NOT evidence; "
                f"per-tercile power is ~1/3 of this, needs more calendar time "
                f"than F1's own checkpoint)")
    span_h = (df.index.max() - df.index.min()) / 60
    print(f"{tag}\njoined minutes n={n}, span ~ {span_h:.1f}h\n")

    cuts = df["vshock"].quantile([1 / 3, 2 / 3]).to_numpy()
    tercile = np.digitize(df["vshock"], cuts)   # 0=low,1=mid,2=high
    for i, label in enumerate(TERCILE_LABELS):
        run_tercile(df[tercile == i], label)

    print(f"\nPASS gate (only at powered n, per cell): CI clear of 0 AND "
          f"|IC|>=0.02 AND halves agree. Prediction being tested: low-vol "
          f"cell carries the signal, high-vol cell ~zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
