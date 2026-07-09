"""Squeeze events × liquidity asymmetry — pre-registered hypothesis test.

Hypothesis (path of least resistance / liquidity vacuum): price breaks
toward the side where resting liquidity is thinner or being withdrawn.
Cancellations lead fills — makers pull the side about to break first.

PRE-REGISTERED HYPOTHESES (written 2026-07-07, BEFORE looking at data):
  H1  Events whose breakout direction matches the THIN side at squeeze
      end have a lower sl_first rate than mismatched events.
  H2  When liquidity is persistently withdrawn from one side during the
      squeeze window, breakout goes toward that side > 55% of the time
      (bootstrap CI lower bound > 50%).
  H3  Triple-confluence subset (thin side + withdrawal side + taker-flow
      side all agree with breakout direction): mean r_scaleout CI lower
      bound > 0.

Data sources (all EXISTING, nothing downloaded):
  · events.parquet             — research/squeeze_events_cli.py output
  · orderbook_snapshots_1m     — MySQL (market_data L20 collector)
  · flow_bars_1m               — MySQL (taker flow / CVD)

Flags are CATEGORICAL (terciles / sign buckets), never tuned thresholds
— mistake.md 2026-06-20 discipline. Cells with n < 100 are reported but
flagged UNRELIABLE. Stability check: first half vs second half must
agree in sign.

Usage:
    python research/squeeze_flow_join.py                       # default paths
    python research/squeeze_flow_join.py --events research/results/squeeze_events.parquet
    python research/squeeze_flow_join.py --pre-window 180 --post-window 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Report strings use ⚠️ / box-drawing; force UTF-8 so a cp950 Windows console
# (the local dev machine) doesn't UnicodeEncodeError mid-report.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn

RNG = np.random.default_rng(7)
N_BOOT = 2000


def _fetch_window(table: str, cols: str, t0_ms: int, t1_ms: int) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT ts_ms, {cols} FROM {table} "
                f"WHERE canonical_symbol='BTC-USD' AND ts_ms BETWEEN %s AND %s "
                f"ORDER BY ts_ms",
                (t0_ms, t1_ms))
            return pd.DataFrame(cur.fetchall())
    finally:
        conn.close()


def fetch_book(t0_ms: int, t1_ms: int) -> pd.DataFrame:
    return _fetch_window(
        "orderbook_snapshots_1m",
        "bid_depth_usd_l20, ask_depth_usd_l20, imbalance_l20",
        t0_ms, t1_ms)


def fetch_flow(t0_ms: int, t1_ms: int) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT window_start AS ts_ms, delta_usd "
                "FROM flow_bars_1m "
                "WHERE canonical_symbol='BTC-USD' "
                "  AND window_start BETWEEN %s AND %s "
                "ORDER BY window_start",
                (t0_ms, t1_ms))
            return pd.DataFrame(cur.fetchall())
    finally:
        conn.close()


def compute_flags(ev: pd.Series, pre_min: int, post_min: int) -> dict | None:
    """Three pre-registered categorical flags for one event.

    thin_side       +1 = asks thinner (up is path of least resistance),
                    -1 = bids thinner, 0 = balanced (middle tercile-ish)
    withdraw_side   +1 = ask depth shrank during squeeze, -1 = bid shrank,
                    0 = mixed  (liquidity-migration direction)
    taker_side      sign of cumulative delta over the squeeze tail
    """
    bo_ms = int(pd.Timestamp(ev["breakout_ts"]).value // 1_000_000)
    t0 = bo_ms - pre_min * 60_000
    book = fetch_book(t0, bo_ms)
    flow = fetch_flow(t0, bo_ms + post_min * 60_000)
    if len(book) < 10:
        return None

    bid = book["bid_depth_usd_l20"].astype(float)
    ask = book["ask_depth_usd_l20"].astype(float)

    # Flag 1 — static thinness at squeeze end (last 15 min average)
    tail = book.tail(15)
    b_end, a_end = tail["bid_depth_usd_l20"].mean(), tail["ask_depth_usd_l20"].mean()
    ratio = a_end / b_end if b_end > 0 else np.nan
    thin_side = 1 if ratio < 0.8 else (-1 if ratio > 1.25 else 0)

    # Flag 2 — withdrawal direction over the window (first vs last third)
    k = max(len(book) // 3, 5)
    d_bid = bid.tail(k).mean() - bid.head(k).mean()
    d_ask = ask.tail(k).mean() - ask.head(k).mean()
    if d_ask < 0 <= d_bid:
        withdraw_side = 1          # asks pulled → up is vacuum
    elif d_bid < 0 <= d_ask:
        withdraw_side = -1
    else:
        withdraw_side = 0          # both grew / both shrank → mixed

    # Flag 3 — taker flow over squeeze tail (same pre-window)
    taker_side = 0
    if len(flow):
        cum = float(flow.loc[flow["ts_ms"] <= bo_ms, "delta_usd"].astype(float).sum())
        taker_side = int(np.sign(cum))

    d = int(ev["direction"])
    return dict(
        thin_match=(thin_side == d), thin_side=thin_side,
        withdraw_match=(withdraw_side == d), withdraw_side=withdraw_side,
        taker_match=(taker_side == d),
        confluence=(thin_side == d and withdraw_side == d and taker_side == d),
    )


def boot_ci(x: np.ndarray, stat=np.mean) -> tuple[float, float, float]:
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    boots = [stat(RNG.choice(x, len(x), replace=True)) for _ in range(N_BOOT)]
    return (float(stat(x)), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def report_cell(name: str, sub: pd.DataFrame, base_sl: float) -> None:
    n = len(sub)
    tag = "" if n >= 100 else "  ⚠️ n<100 UNRELIABLE"
    if n == 0:
        print(f"  {name:<28} n=0{tag}")
        return
    sl = sub["sl_first"].astype(float).to_numpy()
    rs = sub["r_scaleout"].astype(float).to_numpy()
    m, lo, hi = boot_ci(sl)
    rm, rlo, rhi = boot_ci(rs)
    print(f"  {name:<28} n={n:<5} sl_first={m*100:5.1f}% "
          f"CI[{lo*100:.1f},{hi*100:.1f}] (base {base_sl*100:.1f}%)  "
          f"r_scaleout={rm:+.3f} CI[{rlo:+.3f},{rhi:+.3f}]{tag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(
        PROJECT_ROOT / "research" / "results" / "squeeze_events.parquet"))
    ap.add_argument("--pre-window", type=int, default=180,
                    help="minutes before breakout for book/flow window")
    ap.add_argument("--post-window", type=int, default=15)
    args = ap.parse_args()

    ev = pd.read_parquet(args.events)
    ev = ev[ev["resolved"] == True].reset_index(drop=True)  # noqa: E712
    print(f"{len(ev)} resolved events loaded")
    if len(ev) < 30:
        print("⚠️ fewer than 30 resolved events — run a longer date range first")

    rows = []
    skipped = 0
    for _, e in ev.iterrows():
        f = compute_flags(e, args.pre_window, args.post_window)
        if f is None:
            skipped += 1
            continue
        rows.append({**e.to_dict(), **f})
    df = pd.DataFrame(rows)
    print(f"{len(df)} events with book coverage, {skipped} skipped "
          f"(no snapshots in window — collector younger than event)")
    if df.empty:
        return 1

    base_sl = float(df["sl_first"].astype(float).mean())

    def _section(title: str) -> None:
        print("\n" + "=" * 88)
        print(title)
        print("=" * 88)

    _section(f"H1 — thin-side match vs mismatch  (baseline sl_first {base_sl*100:.1f}%)")
    report_cell("thin_match=True", df[df["thin_match"]], base_sl)
    report_cell("thin_match=False (thin≠0)",
                df[(~df["thin_match"]) & (df["thin_side"] != 0)], base_sl)
    report_cell("balanced (thin=0)", df[df["thin_side"] == 0], base_sl)

    _section("H2 — does withdrawal direction predict breakout direction?")
    w = df[df["withdraw_side"] != 0]
    if len(w):
        agree = (w["withdraw_side"] == w["direction"]).astype(float).to_numpy()
        m, lo, hi = boot_ci(agree)
        verdict = "PASS" if lo > 0.5 else "no"
        print(f"  agreement {m*100:.1f}%  CI[{lo*100:.1f},{hi*100:.1f}]  "
              f"n={len(w)}  (pre-registered: CI-lo > 50%) → {verdict}")
    else:
        print("  no events with a clear withdrawal side")

    _section("H3 — triple confluence subset")
    report_cell("confluence=True", df[df["confluence"]], base_sl)
    report_cell("confluence=False", df[~df["confluence"]], base_sl)

    _section("Stability — first half vs second half (sign must agree)")
    df = df.sort_values("breakout_ts").reset_index(drop=True)
    half = len(df) // 2
    for name, sub in (("H1 thin_match", df[df["thin_match"]]),
                      ("H3 confluence", df[df["confluence"]])):
        a = sub[sub.index < half]
        b = sub[sub.index >= half]
        for label, s in (("  first half", a), ("  second half", b)):
            report_cell(f"{name}{label}", s, base_sl)

    out = PROJECT_ROOT / "research" / "results" / "squeeze_flow_flags.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote per-event flag table → {out}")
    print("\nDiscipline reminders: cells with n<100 are NOT conclusions; "
          "H-pass requires both halves agreeing in sign; no new thresholds "
          "may be tuned on this data after looking at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
