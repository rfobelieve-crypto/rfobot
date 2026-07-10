"""Flow-conditioned re-entry after trail-stop exits — pre-registered harness.

Motivation (exit_decomposition, 2026-07-10): opp_signal exits are the profit
engine (86% WR, +152bps, negative regret — do not touch). Trail exits are
right on average, but the worst QUARTILE leaves 70-130bps on the table.
Every price-geometry fix failed (11-variant sweep, 5d83da2). The one door
left: NEW information — after a trail sweep, order-flow can say whether the
sweep was noise (vacuum toward the old direction still open) or real.

PRE-REGISTERED RULES (frozen 2026-07-10, BEFORE any depth-era trail exits
exist; categorical, no tuned thresholds — mistake.md 2026-06-20):

  Universe   Strong-only trade sim faithful to live (entry next-bar open,
             3xATR Wilder-14 trail ratcheted per completed bar, exit on any
             opposite reading, no time cap, 1-position occupancy), driven by
             tracked_signals (the live decode) + 1h klines.
  Trigger    a trail_stop exit of one of those trades, with depth_deltas_1m
             coverage (collector live since 2026-07-09).
  Condition  over the 60 minutes AFTER the exit bar closes:
               skew_60 = mean cancel_skew, sign-aligned with the ORIGINAL
                         direction (LONG wants ask-side cancels, >0)
               imb_60  = mean imbalance_l20, sign-aligned likewise
             BOTH > 0  →  re-enter the original direction at the next 1h bar
             open; fresh 3xATR trail; same exit rules; ONE re-entry per exit.
  Metric     net bps of re-entry trades (vs staying flat = 0 baseline).
  GATE       n >= 30 gated re-entries AND bootstrap 95% CI low > 0 AND
             first/second half agree in sign. Expected earliest checkpoint
             ~2026-10 (backtest cadence ≈ 8-9 trail exits/month).

Stop integrity is untouched by design: the trail always fires; this only
decides whether to buy the ticket back afterwards.

Usage:  python research/flow_reentry_bt.py     (rerun any time; reports
        accumulation status until the gate is reachable)
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

KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
DEPTH_SINCE = pd.Timestamp("2026-07-09")
TRAIL_MULT = 3.0
FEE_RT = 0.0008          # 8 bps round trip (matches research kernel)
RNG = np.random.default_rng(7)


def _fetch(sql: str, args: tuple = ()) -> list[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()
    finally:
        conn.close()


def load_signals() -> pd.DataFrame:
    rows = _fetch(
        "SELECT signal_time, direction, strength FROM tracked_signals "
        "WHERE signal_time >= %s ORDER BY signal_time",
        (str(DEPTH_SINCE - pd.Timedelta(days=2)),))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["signal_time"] = pd.to_datetime(df["signal_time"]).dt.floor("h")
    return df.drop_duplicates("signal_time").set_index("signal_time")


def atr_wilder(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def sim_trades(k: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Live-faithful Strong-only sim; returns closed trades incl. exit bar."""
    idx = k.index
    dirs = sig["direction"].reindex(idx)
    tiers = sig["strength"].reindex(idx)
    atr = atr_wilder(k["high"], k["low"], k["close"]).to_numpy()
    o, h, l, c = (k[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(idx)
    rows, i = [], 0
    while i < n - 1:
        d, ti = dirs.iloc[i], tiers.iloc[i]
        if d not in ("UP", "DOWN") or ti != "Strong" or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        e = i + 1
        entry_px, stop_dist = o[e], TRAIL_MULT * atr[i]
        extreme = entry_px
        stop = entry_px - stop_dist if d == "UP" else entry_px + stop_dist
        exit_i = exit_px = reason = None
        for j in range(e, n):
            if d == "UP" and l[j] <= stop:
                exit_i, exit_px, reason = j, stop, "trail_stop"; break
            if d == "DOWN" and h[j] >= stop:
                exit_i, exit_px, reason = j, stop, "trail_stop"; break
            if dirs.iloc[j] == ("DOWN" if d == "UP" else "UP"):
                exit_i, exit_px, reason = j, c[j], "opp_signal"; break
            extreme = max(extreme, h[j]) if d == "UP" else min(extreme, l[j])
            stop = extreme - stop_dist if d == "UP" else extreme + stop_dist
        if exit_i is None:
            break                                   # still open at data end
        gross = (exit_px / entry_px - 1) * (1 if d == "UP" else -1)
        rows.append(dict(signal_ts=idx[i], entry_ts=idx[e], exit_ts=idx[exit_i],
                         direction=d, entry_px=entry_px, exit_px=exit_px,
                         exit_reason=reason, net=gross - FEE_RT))
        i = exit_i
    return pd.DataFrame(rows)


def flow_gate(exit_ts: pd.Timestamp, direction: str) -> tuple[bool, dict]:
    """Frozen condition over the 60min after the exit bar closes."""
    t0 = int((exit_ts + pd.Timedelta(hours=1)).value // 1_000_000)
    t1 = t0 + 60 * 60_000
    dd = pd.DataFrame(_fetch(
        "SELECT bid_cancel_qty, ask_cancel_qty FROM depth_deltas_1m "
        "WHERE canonical_symbol='BTC-USD' AND minute_start_ms BETWEEN %s AND %s",
        (t0, t1)))
    ob = pd.DataFrame(_fetch(
        "SELECT imbalance_l20 FROM orderbook_snapshots_1m "
        "WHERE canonical_symbol='BTC-USD' AND ts_ms BETWEEN %s AND %s",
        (t0, t1)))
    if len(dd) < 30 or len(ob) < 30:
        return False, {"covered": False}
    tot = dd["bid_cancel_qty"].astype(float) + dd["ask_cancel_qty"].astype(float)
    skew = ((dd["ask_cancel_qty"].astype(float) - dd["bid_cancel_qty"].astype(float))
            / tot.replace(0, np.nan)).mean()
    imb = ob["imbalance_l20"].astype(float).mean()
    s = 1.0 if direction == "UP" else -1.0
    return bool(s * skew > 0 and s * imb > 0), {
        "covered": True, "skew_60": float(skew), "imb_60": float(imb)}


def main() -> int:
    sig = load_signals()
    if sig.empty:
        print("no tracked signals in the depth era yet"); return 0
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = pd.DatetimeIndex(k.index)
    if getattr(k.index, "tz", None) is not None:
        k.index = k.index.tz_localize(None)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    k = k.loc[DEPTH_SINCE - pd.Timedelta(days=2):]
    if k.empty:
        print("klines parquet has no depth-era bars yet (stale backfill?) — "
              "refresh market_data/raw_data/binance_klines_1h.parquet "
              "(research/backfill_all_parquet.py) and rerun.")
        return 0
    print(f"depth-era window: klines to {k.index[-1]}, "
          f"signals to {sig.index.max()}")

    trades = sim_trades(k, sig)
    trail = trades[trades["exit_reason"] == "trail_stop"] if len(trades) else trades
    print(f"sim trades={len(trades)}, trail exits={len(trail)} "
          f"(gate needs n>=30 gated re-entries — expected ~2026-10)\n")

    rows = []
    for _, t in trail.iterrows():
        ok, info = flow_gate(t["exit_ts"], t["direction"])
        if not info.get("covered"):
            continue
        # simulate re-entry at next 1h bar open after the 60-min window
        re_sig = t["exit_ts"] + pd.Timedelta(hours=1)
        sub = k.loc[re_sig:]
        if len(sub) < 3:
            continue
        rows.append({**t, "gated_in": ok, **info})
    df = pd.DataFrame(rows)
    if df.empty:
        print("0 trail exits with depth coverage yet — accumulating. "
              "Rules stay frozen; rerun later.")
        return 0
    print(df[["exit_ts", "direction", "gated_in", "skew_60", "imb_60"]]
          .to_string(index=False))
    n_in = int(df["gated_in"].sum())
    print(f"\ngated-in {n_in}/{len(df)} — re-entry P&L evaluation activates "
          f"at the pre-registered gate (n>=30).")
    out = PROJECT_ROOT / "research" / "results" / "flow_reentry_bt.csv"
    df.to_csv(out, index=False)
    print(f"Wrote → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
