# -*- coding: utf-8 -*-
"""Stage 3 — sweep events.

    data/events/{COIN}.parquet
        event_id, level_id, side, t_sweep, sweep_lvl,
        cross_type, cross_depth_ticks, cross_depth_atr,
        hour_ts, dup_rank, dup_group

Rule (frozen, from the plan)
    scan 1-MINUTE bars strictly after the level's confirmed_at
        sellside : first bar with low  < price - k*tick
        buyside  : first bar with high > price + k*tick
    k = 2 ticks, pre-registered
    t_sweep = the CLOSE of that minute -- the earliest moment the breach is
              established by a completed bar.  (Deviation from the plan's
              "the ts of that bar", which is its OPEN: at the open the breach
              has not happened yet, so stamping it there would be look-ahead
              by up to 59 seconds.  Stated, not silently done.)

Dedup (frozen)
    same coin, same side, |sweep_lvl difference| < 3 ticks AND
    |t_sweep difference| < cooldown  ->  keep the earliest
    cooldown main = 300s; sensitivity 60 / 900.

Regression gate
    Against the FROZEN engine's own historical event table
    (sweep_core.detect_sweeps on the 1h cache) -- same level definition, so
    the comparison is meaningful.  Match = same side, |level diff| <= 5 ticks,
    and this pipeline's t_sweep falls inside the frozen engine's piercing
    HOUR.  The plan asks for |t_sweep diff| < 5 min, but the frozen engine
    only resolves the event to an hourly bar; a 5-minute tolerance cannot be
    evaluated against it, so "inside the same hour" is the tightest honest
    equivalent.

    The forward shadow ledger is NOT the primary reference: it draws four
    pool types (session 2088 / pdh_pdl 1215 / swing 1076 / pwh_pwl 224) of
    which only `swing` is a pivot, and only 1,411 of its rows are core9.  It
    is reported as a secondary check on the `swing` subset only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sweep_failure"))
import sweep_core as sc  # noqa: E402

BARS = HERE / "data" / "bars"
LEVELS = HERE / "data" / "levels"
OUT = HERE / "data" / "events"
QUALITY = HERE / "data" / "quality"
CACHE_1H = HERE.parent / "sweep_failure" / ".cache"
MIN_MS = 60_000
HOUR_MS = 3_600_000
K_TICKS = 2
DEDUP_TICKS = 3
COOLDOWNS = [60, 300, 900]
COOLDOWN_MAIN = 300
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def build(sym):
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "high", "low", "close", "atr_h14", "tick_size"])
    lv = pd.read_parquet(LEVELS / f"{sym}.parquet")
    tick = float(b["tick_size"].iloc[0])
    ts = b["ts"].to_numpy(np.int64)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    atr = b.set_index("ts")["atr_h14"]

    rows = []
    for r in lv.itertuples():
        i0 = int(np.searchsorted(ts, r.confirmed_at, side="left"))
        thr = r.price + K_TICKS * tick if r.side == "buyside" else r.price - K_TICKS * tick
        j = np.flatnonzero(hi[i0:] > thr) if r.side == "buyside" \
            else np.flatnonzero(lo[i0:] < thr)
        if len(j) == 0:
            continue
        k = i0 + int(j[0])
        t_sweep = int(ts[k] + MIN_MS)
        depth = (hi[k] - r.price) if r.side == "buyside" else (r.price - lo[k])
        a = atr.get(t_sweep, np.nan)
        rows.append(dict(level_id=r.level_id, side=r.side, t_sweep=t_sweep,
                         sweep_lvl=float(r.price),
                         cross_type=("high_break" if r.side == "buyside" else "low_break"),
                         cross_depth_ticks=float(depth / tick),
                         cross_depth_atr=float(depth / a) if a and np.isfinite(a) else np.nan,
                         hour_ts=int(t_sweep - MIN_MS) // HOUR_MS * HOUR_MS,
                         minute_in_hour=int(((t_sweep - MIN_MS) % HOUR_MS) // MIN_MS)))
    ev = pd.DataFrame(rows).sort_values(["t_sweep", "side", "sweep_lvl"]).reset_index(drop=True)
    ev.insert(0, "event_id", [f"{sym}-E{i:06d}" for i in range(len(ev))])
    return ev, tick


def dedup(ev, tick, cooldown):
    """Keep the earliest of any group within 3 ticks and `cooldown` seconds."""
    keep, dropped = [], 0
    for side, g in ev.groupby("side", sort=False):
        g = g.sort_values("t_sweep")
        anchors = []           # (price, t_sweep) kept so far, recent ones only
        for r in g.itertuples():
            anchors = [x for x in anchors if r.t_sweep - x[1] <= cooldown * 1000]
            if any(abs(r.sweep_lvl - p) < DEDUP_TICKS * tick for p, _ in anchors):
                dropped += 1
                continue
            keep.append(r.Index)
            anchors.append((r.sweep_lvl, r.t_sweep))
    return ev.loc[sorted(keep)].reset_index(drop=True), dropped


def run_asserts(ev, lv, b, tick):
    f = []
    m = ev.merge(lv[["level_id", "confirmed_at"]], on="level_id", how="left")
    if not (m["t_sweep"] > m["confirmed_at"]).all():
        f.append("t_sweep <= confirmed_at")
    bb = b.set_index("ts")
    bad = 0
    for r in ev.sample(min(2000, len(ev)), random_state=0).itertuples():
        bar = bb.loc[r.t_sweep - MIN_MS]
        if r.side == "sellside" and not (bar["low"] < r.sweep_lvl - K_TICKS * tick):
            bad += 1
        if r.side == "buyside" and not (bar["high"] > r.sweep_lvl + K_TICKS * tick):
            bad += 1
    if bad:
        f.append(f"{bad} sampled events do not actually breach by k ticks")
    for side, g in ev.groupby("side"):
        g = g.sort_values("t_sweep")
        p, t = g["sweep_lvl"].to_numpy(), g["t_sweep"].to_numpy()
        for i in range(1, len(g)):
            back = np.flatnonzero(t[i] - t[:i] < COOLDOWN_MAIN * 1000)
            if len(back) and (np.abs(p[i] - p[back]) < DEDUP_TICKS * tick).any():
                f.append("dedup violated after dedup step")
                break
    return f


def regression_vs_frozen(sym, ev, tick):
    """Two-way hit rate against sweep_core.detect_sweeps on the 1h cache."""
    b1 = sc.load_csv(str(CACHE_1H / f"{sym}USDT_1h.csv"))
    ref = []
    for e in sc.detect_sweeps(b1):
        ref.append((("buyside" if e["kind"] == "buy" else "sellside"),
                    float(e["level"]), int(b1[e["j"]][0]) * 1000))
    ours = [(r.side, float(r.sweep_lvl), int(r.t_sweep)) for r in ev.itertuples()]
    tol = 5 * tick

    by_side_ref = {}
    for s, p, h in ref:
        by_side_ref.setdefault(s, []).append((p, h))
    hit_ours = 0
    for s, p, t in ours:
        cands = by_side_ref.get(s, [])
        if any(abs(p - rp) <= tol and rh <= t - MIN_MS < rh + HOUR_MS for rp, rh in cands):
            hit_ours += 1
    by_side_ours = {}
    for s, p, t in ours:
        by_side_ours.setdefault(s, []).append((p, t))
    hit_ref = 0
    for s, p, h in ref:
        cands = by_side_ours.get(s, [])
        if any(abs(p - op) <= tol and h <= ot - MIN_MS < h + HOUR_MS for op, ot in cands):
            hit_ref += 1
    return dict(n_ours=len(ours), n_ref=len(ref),
                ours_in_ref=hit_ours / len(ours) if ours else 0.0,
                ref_in_ours=hit_ref / len(ref) if ref else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    QUALITY.mkdir(parents=True, exist_ok=True)
    summary, allok = {}, True
    for s in [x.strip().upper() for x in a.syms.split(",") if x.strip()]:
        if not (LEVELS / f"{s}.parquet").exists():
            print(f"{s:5s} no levels, skipped")
            continue
        raw, tick = build(s)
        sens = {}
        for cd in COOLDOWNS:
            _, dr = dedup(raw, tick, cd)
            sens[cd] = int(dr)
        ev, dropped = dedup(raw, tick, COOLDOWN_MAIN)
        b = pd.read_parquet(BARS / f"{s}.parquet", columns=["ts", "high", "low"])
        lv = pd.read_parquet(LEVELS / f"{s}.parquet")
        fails = run_asserts(ev, lv, b, tick)
        reg = regression_vs_frozen(s, ev, tick)
        ev.to_parquet(OUT / f"{s}.parquet", index=False)

        # the duplication the registered dedup does NOT catch: different
        # prices, same hourly bar
        per_hour = ev.groupby(["side", "hour_ts"]).size()
        dup_rate = 1 - len(per_hour) / len(ev)
        ok = (not fails) and reg["ours_in_ref"] > 0.85 and reg["ref_in_ours"] > 0.85
        allok &= ok
        summary[s] = dict(raw=int(len(raw)), kept=int(len(ev)),
                          dropped_by_cooldown=sens, regression=reg,
                          same_hour_dup_rate=float(dup_rate),
                          max_events_one_hour=int(per_hour.max()),
                          asserts=fails)
        print(f"{s:5s} raw={len(raw):5,} kept={len(ev):5,} "
              f"dedup(60/300/900)={sens[60]}/{sens[300]}/{sens[900]}  "
              f"regr ours->ref={reg['ours_in_ref']*100:.2f}% ref->ours={reg['ref_in_ours']*100:.2f}%  "
              f"same-hour dup={dup_rate*100:.1f}% (max {per_hour.max()})  "
              f"{'PASS' if ok else 'FAIL ' + str(fails)}")
    (QUALITY / "stage3_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    print("\nStage 3 gate (>85% both ways):", "ALL PASS" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
