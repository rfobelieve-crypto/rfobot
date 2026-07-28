"""Shadow execution window — would order-flow-timed entries beat immediate
market entries on live V7 signals?

Motivation: live cohort shows gross +0.97% eaten to net +0.05% (~7.7 bps/trade
cost). This harness replays each live entry against the following 60 minutes
of recorded 1m microstructure and computes what two order-flow-aware entry
rules WOULD have got. Pure retrospective analysis: reads DB, writes one
parquet + stdout. Touches nothing live.

PRE-REGISTERED RULES (written 2026-07-10 BEFORE looking at any results;
categorical, no tuned thresholds — mistake.md 2026-06-20 discipline):

  Window: 60 minutes from actual entry_time. Baseline = actual entry price,
  taker fee.

  R1 "flow-timed taker": enter (taker) at the first minute whose flow agrees
     with trade direction — cancel_skew sign match when depth_deltas_1m
     covers that minute (skew = (ask_cancel-bid_cancel)/(ask_cancel+
     bid_cancel); LONG wants >0, SHORT wants <0), else imbalance_l20 sign
     match (LONG >0 / SHORT <0). Price = ask_l1 for LONG / bid_l1 for SHORT.
     No agreeing minute → taker at window end (never skip the signal).

  R2 "maker at signal price": passive limit at the ACTUAL entry price.
     Fill proxy: some later snapshot's mid crosses the limit (LONG: mid <=
     px; SHORT: mid >= px). Filled → same price but maker fee. Not filled →
     taker at window end (price may be worse — adverse selection cost is
     precisely what we're measuring).

  Fees: taker 5 bps, maker 2 bps (OKX BTC-USDT-SWAP retail tier).
  Metric per trade: edge_bps = direction-signed price improvement vs actual
  + fee saving. Verdict metric: mean edge_bps with bootstrap 95% CI; switch
  the live executor ONLY if CI low > 0 on n >= 30 (accumulating harness —
  early runs are directional, not conclusions).

  Known proxy limits: 1m snapshots miss intraminute extremes (fill proxy is
  conservative-ish but imperfect); depth_deltas only exists from 2026-07-09
  so earlier trades use the imbalance-only R1 variant (flagged per row).

CORRECTION (2026-07-28): `entry_time` / `signal_time` are bar-OPEN labels —
the signal actually fires (and the live order fills) at label+1h+~2.5min
(= `created_at`; verified: all v7_okx_positions rows have created_at =
entry_time + 1h02m28s, and the recorded fill price matches the 1m price at
created_at, NOT at the label). The original implementation anchored the
window at the LABEL, replaying the signal's own formation bar — pure
lookahead. Live-R1's +16.7 bias (any pre-close minute beats the close of a
momentum bar) and signals-R2's -76bps "misses" (bars closing on their
extreme) were artifacts of that. Every parquet produced before this date is
void. The window now anchors at created_at (guarded: if created_at is
missing or not within label+[55,75]min, fall back to label+65min), the
signals baseline uses tracked_signals.entry_price (the fire-time price),
and each row carries `anchor_gap_bps` (baseline vs first post-anchor mid —
spread-scale means the anchor is right, bar-scale means it is wrong).
Pre-registered R1/R2 rules and the switch gate are UNCHANGED; evidence
accumulation restarts 2026-07-28.

Usage:  python research/shadow_exec_window.py [--window-min 60]
"""
from __future__ import annotations

import argparse
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

TAKER_BPS = 5.0
MAKER_BPS = 2.0
RNG = np.random.default_rng(7)


def _fetch(sql: str, args: tuple = ()) -> list[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()
    finally:
        conn.close()


def _anchor(label, created) -> pd.Timestamp:
    """Fire/fill wall time for a bar-open label. created_at when sane,
    else label+65min (bar close + typical cycle latency)."""
    lab = pd.Timestamp(label)
    if created is not None:
        gap_min = (pd.Timestamp(created) - lab).total_seconds() / 60.0
        if 55.0 <= gap_min <= 75.0:
            return pd.Timestamp(created)
    return lab + pd.Timedelta(minutes=65)


def load_entries() -> list[dict]:
    rows = _fetch(
        "SELECT id, entry_time, created_at, direction, entry_price "
        "FROM v7_okx_positions "
        "WHERE entry_time >= '2026-06-07' "
        "  AND (model_version IS NULL OR model_version NOT LIKE 'manual_test%%') "
        "ORDER BY entry_time")
    for r in rows:
        r["anchor_time"] = _anchor(r["entry_time"], r.get("created_at"))
    return rows


def load_window(t0_ms: int, t1_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    book = pd.DataFrame(_fetch(
        "SELECT ts_ms, mid_price, bid_l1_price, ask_l1_price, imbalance_l20 "
        "FROM orderbook_snapshots_1m "
        "WHERE canonical_symbol='BTC-USD' AND ts_ms BETWEEN %s AND %s "
        "ORDER BY ts_ms", (t0_ms, t1_ms)))
    dd = pd.DataFrame(_fetch(
        "SELECT minute_start_ms, bid_cancel_qty, ask_cancel_qty "
        "FROM depth_deltas_1m "
        "WHERE canonical_symbol='BTC-USD' AND minute_start_ms BETWEEN %s AND %s "
        "ORDER BY minute_start_ms", (t0_ms, t1_ms)))
    return book, dd


def shadow_one(ev: dict, window_min: int) -> dict | None:
    d = 1 if ev["direction"] == "LONG" else -1
    # Anchor at FIRE time (created_at), never the bar-open label — see the
    # 2026-07-28 CORRECTION in the module docstring.
    t0 = int(pd.Timestamp(ev["anchor_time"]).value // 1_000_000)
    t1 = t0 + window_min * 60_000
    book, dd = load_window(t0, t1)
    if len(book) < 5:
        return None
    for c in ("mid_price", "bid_l1_price", "ask_l1_price", "imbalance_l20"):
        book[c] = book[c].astype(float)
    # Hypothetical (signal-level) entries carry no fill price — the baseline
    # is the taker price at the first snapshot after the signal.
    px0 = float(ev.get("entry_price") or 0)
    if px0 <= 0:
        first = book.iloc[0]
        px0 = float(first["ask_l1_price"] if d == 1 else first["bid_l1_price"])
    # Anchor sanity: baseline should sit spread-distance from the first
    # post-anchor mid. Bar-scale gaps mean the anchor is wrong.
    first_mid = float(book.iloc[0]["mid_price"])
    anchor_gap_bps = abs(px0 - first_mid) / first_mid * 1e4

    skew_by_min: dict[int, float] = {}
    if len(dd):
        for _, r in dd.iterrows():
            tot = float(r["bid_cancel_qty"]) + float(r["ask_cancel_qty"])
            if tot > 0:
                skew_by_min[int(r["minute_start_ms"])] = (
                    (float(r["ask_cancel_qty"]) - float(r["bid_cancel_qty"])) / tot)

    def taker_px(row) -> float:
        return float(row["ask_l1_price"] if d == 1 else row["bid_l1_price"])

    # R1 — first flow-agreeing minute
    r1_px, r1_used_skew = None, False
    for _, row in book.iterrows():
        mkey = int(row["ts_ms"]) // 60_000 * 60_000
        if mkey in skew_by_min:
            agree = (d * skew_by_min[mkey]) > 0
            used = True
        else:
            agree = (d * float(row["imbalance_l20"])) > 0
            used = False
        if agree:
            r1_px, r1_used_skew = taker_px(row), used
            break
    r1_fallback = r1_px is None
    if r1_fallback:
        r1_px = taker_px(book.iloc[-1])

    # R2 — maker limit at actual entry price
    later = book[book["ts_ms"] > int(book.iloc[0]["ts_ms"])]
    crossed = ((later["mid_price"] <= px0) if d == 1
               else (later["mid_price"] >= px0)).any()
    if crossed:
        r2_px, r2_fee, r2_filled = px0, MAKER_BPS, True
    else:
        r2_px, r2_fee, r2_filled = taker_px(book.iloc[-1]), TAKER_BPS, False

    def edge(shadow_px: float, shadow_fee: float) -> float:
        price_bps = d * (px0 - shadow_px) / px0 * 1e4   # LONG cheaper = +
        return price_bps + (TAKER_BPS - shadow_fee)

    return {
        "id": ev["id"], "entry_time": ev["entry_time"],
        "anchor_time": ev["anchor_time"], "anchor_gap_bps": anchor_gap_bps,
        "direction": ev["direction"], "entry_price": px0,
        "book_minutes": len(book), "skew_minutes": len(skew_by_min),
        "r1_px": r1_px, "r1_edge_bps": edge(r1_px, TAKER_BPS),
        "r1_used_skew": r1_used_skew, "r1_fallback": r1_fallback,
        "r2_px": r2_px, "r2_edge_bps": edge(r2_px, r2_fee),
        "r2_filled": r2_filled,
    }


def boot_ci(x: np.ndarray) -> tuple[float, float, float]:
    if len(x) == 0:
        return (np.nan,) * 3
    bs = [np.mean(RNG.choice(x, len(x), replace=True)) for _ in range(2000)]
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_signal_entries() -> list[dict]:
    """Hypothetical entries from ALL tracked Strong signals with book
    coverage (L20 collector live since 2026-05-11). SUPPORTING evidence to
    accelerate confidence — the pre-registered switch gate stays on live
    entries (n >= 30, CI-low > 0)."""
    rows = _fetch(
        "SELECT id, signal_time, created_at, direction, entry_price "
        "FROM tracked_signals "
        "WHERE strength='Strong' AND direction IN ('UP','DOWN') "
        "  AND signal_time >= '2026-05-11' ORDER BY signal_time")
    return [{"id": f"s{r['id']}", "entry_time": r["signal_time"],
             "anchor_time": _anchor(r["signal_time"], r.get("created_at")),
             "direction": "LONG" if r["direction"] == "UP" else "SHORT",
             # fire-time price recorded by the indicator (verified against
             # live fills to the tick) — the honest baseline
             "entry_price": r["entry_price"]} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=int, default=60)
    ap.add_argument("--signals", action="store_true",
                    help="signal-level backfill (hypothetical entries at every "
                         "tracked Strong signal) — supporting evidence only")
    args = ap.parse_args()

    if args.signals:
        entries = load_signal_entries()
        print(f"{len(entries)} hypothetical entries from tracked Strong signals "
              f"since 2026-05-11 (SUPPORTING evidence — switch gate stays on "
              f"live entries)")
    else:
        entries = load_entries()
        print(f"{len(entries)} live entries since 2026-06-07 (manual_test excluded)")
    rows, skipped = [], 0
    for ev in entries:
        r = shadow_one(ev, args.window_min)
        if r is None:
            skipped += 1
            continue
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"{len(df)} with book coverage, {skipped} skipped")
    if df.empty:
        return 1
    print(f"anchor sanity: |baseline - first mid| mean "
          f"{df['anchor_gap_bps'].mean():.1f} bps, max "
          f"{df['anchor_gap_bps'].max():.1f} bps "
          f"(spread-scale = anchored right; bar-scale = anchored wrong)\n")

    pd.set_option("display.width", 160)
    cols = ["id", "direction", "entry_price", "r1_edge_bps", "r1_fallback",
            "r1_used_skew", "r2_edge_bps", "r2_filled"]
    print(df[cols].to_string(index=False,
                             float_format=lambda v: f"{v:.1f}"))

    for name in ("r1", "r2"):
        m, lo, hi = boot_ci(df[f"{name}_edge_bps"].to_numpy(float))
        extra = (f"fallback {int(df['r1_fallback'].sum())}/{len(df)}, "
                 f"skew-informed {int(df['r1_used_skew'].sum())}" if name == "r1"
                 else f"filled {int(df['r2_filled'].sum())}/{len(df)}")
        print(f"\n{name.upper()} mean edge {m:+.1f} bps  CI[{lo:+.1f},{hi:+.1f}]  ({extra})")

    out = PROJECT_ROOT / "research" / "results" / (
        "shadow_exec_window_signals.parquet" if args.signals
        else "shadow_exec_window.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nWrote → {out}")
    print(f"\nDiscipline: switch live execution ONLY if CI-low > 0 at n >= 30. "
          f"Current n={len(df)} — directional evidence, not a conclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
