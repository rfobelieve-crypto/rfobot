# -*- coding: utf-8 -*-
"""Stage 1 validation — measure the two allocation rules against TICK truth.

The plan asks for the distribution of |POC_uniform - POC_close| / ATR, and
reads "median > 0.5 ATR" as "POC precision cannot carry H1".  That number says
how much the two rules DISAGREE.  It does not say which is right — and neither
does comparing a 5-minute reconstruction against a 1-minute one built by the
same rule.  That comparison is circular; retiring it is why this file exists.

With real prints there is no allocation rule at all: every print carries its
own price, so the volume-at-price histogram is the thing itself.

    POC_tick     ground truth
    POC_uniform  the registered main method, from 1-minute bars
    POC_close    the registered sensitivity, from the same bars

Reported: each rule's error against truth, and the disagreement the plan asked
for, under BOTH candidate bin sizes (atr_1h and atr_h14 — see bars.py).

Instrument guard
    The prints and the bars must be the same market.  The perp aggTrades
    already in this repo are NOT (verified 2026-09-06: a minute's print
    quantity matched fapi 1m volume at ratio 1.000, while spot was 10.3x off),
    so this reads SPOT aggTrades pulled by fetch_ticks.py.  The check runs on
    every sample and the script refuses to report if it fails.

Timestamp units
    Binance's spot vision aggTrades are MICROseconds; the perp files in this
    repo are milliseconds.  Same provider, different endpoints, different unit
    (mistake.md 2026-04-12) — detected, never assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from profile import Bars, build_profile  # noqa: E402

TICKS = HERE / "data" / "ticks"
BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "quality"
MIN_MS = 60_000
COLS = ["agg_id", "price", "quantity", "first_id", "last_id",
        "transact_time", "is_buyer_maker", "is_best_match"]


def load_day(sym, day):
    """Returns (price, quantity, ts_ms, unit) or None.  Unit detected, not assumed."""
    p = TICKS / f"{sym}_{day}.zip"
    if not p.exists():
        return None
    with zipfile.ZipFile(p) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            head = f.readline().decode(errors="ignore")
        has_header = not head.split(",")[0].strip().strip('"').isdigit()
        with z.open(name) as f:
            if has_header:
                d = pd.read_csv(f)
                d.columns = [c.strip() for c in d.columns]
                d = d[["price", "quantity", "transact_time"]]
            else:
                d = pd.read_csv(f, header=None, names=COLS, usecols=[1, 2, 5])
                d = d[["price", "quantity", "transact_time"]]
    t = d["transact_time"].to_numpy(np.int64)
    mx = int(t.max())
    if mx > 1e15:
        unit, ms = "us", t // 1000
    elif mx > 1e12:
        unit, ms = "ms", t
    else:
        unit, ms = "s", t * 1000
    order = np.argsort(ms, kind="stable")
    return (d["price"].to_numpy(np.float64)[order],
            d["quantity"].to_numpy(np.float64)[order], ms[order], unit)


def tick_poc(px, qty, bin_size):
    ids = np.floor(px / bin_size).astype(np.int64)
    uniq, inv = np.unique(ids, return_inverse=True)
    w = np.bincount(inv, weights=qty, minlength=len(uniq))
    tied = uniq[w >= w.max() - 1e-12]
    return float(np.median((tied + 0.5) * bin_size))   # same tie rule as Stage 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="BTC")
    ap.add_argument("--per-day", type=int, default=10)
    ap.add_argument("--lookback-h", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()

    bdf = pd.read_parquet(BARS / f"{a.sym}.parquet")
    bars = Bars(bdf.ts, bdf.open, bdf.high, bdf.low, bdf.close, bdf.volume)
    tick = float(bdf.tick_size.iloc[0])
    atr1 = bdf.set_index("ts")["atr_1h"]
    atrh = bdf.set_index("ts")["atr_h14"]
    vol = bdf.set_index("ts")["volume"]

    have = sorted(p.stem.split("_", 1)[1] for p in TICKS.glob(f"{a.sym}_*.zip"))
    ref_days = [d for d in have
                if (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") in set(have)]
    if not ref_days:
        sys.exit("no usable tick days -- run fetch_ticks.py")
    rng = np.random.default_rng(a.seed)
    print(f"tick days on disk={len(have)} usable as reference={len(ref_days)}", flush=True)

    rows, xcheck, units = [], [], set()
    for day in ref_days:
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        A, B = load_day(a.sym, prev), load_day(a.sym, day)
        if A is None or B is None:
            continue
        units.add(A[3])
        units.add(B[3])
        t_px = np.concatenate([A[0], B[0]])
        t_q = np.concatenate([A[1], B[1]])
        t_ts = np.concatenate([A[2], B[2]])
        day0 = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
        for m in sorted(rng.choice(np.arange(0, 1440), size=a.per_day, replace=False)):
            t_ref = day0 + int(m) * MIN_MS
            if t_ref not in atr1.index:
                continue
            mstart = t_ref - MIN_MS
            sel = (t_ts >= mstart) & (t_ts < t_ref)
            if sel.any() and mstart in vol.index and vol[mstart] > 0:
                xcheck.append(abs(t_q[sel].sum() - vol[mstart]) / vol[mstart])
            w = (t_ts >= t_ref - int(a.lookback_h * 3600_000)) & (t_ts < t_ref)
            if w.sum() < 1000:
                continue
            for tag, atr in (("atr_1h", atr1.get(t_ref)), ("atr_h14", atrh.get(t_ref))):
                if atr is None or not np.isfinite(atr) or atr <= 0:
                    continue
                bs = max(tick, atr / 20.0)
                truth = tick_poc(t_px[w], t_q[w], bs)
                pu = build_profile(bars, t_ref, ("time", a.lookback_h), "uniform", bs)
                pc = build_profile(bars, t_ref, ("time", a.lookback_h), "close", bs)
                if pu is None or pc is None:
                    continue
                rows.append(dict(t_ref=t_ref, day=day, atr_kind=tag, atr=float(atr),
                                 bin_size=bs, n_prints=int(w.sum()),
                                 poc_tick=truth, poc_uniform=pu.poc, poc_close=pc.poc,
                                 err_uniform=abs(pu.poc - truth) / atr,
                                 err_close=abs(pc.poc - truth) / atr,
                                 disagree=abs(pu.poc - pc.poc) / atr))
        print(f"  {day}: samples={len(rows)}", flush=True)

    if not rows:
        sys.exit("no samples produced")
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"stage1_tick_truth_{a.sym}.csv", index=False)

    xa = np.array(xcheck)
    same = bool(xa.max() < 1e-6)
    print("\ninstrument guard (print quantity in a minute vs that bar's volume)")
    print(f"  tick timestamp units seen: {sorted(units)}")
    print(f"  n={len(xa)}  median={np.median(xa):.3e}  max={xa.max():.3e}"
          f"  -> {'SAME INSTRUMENT' if same else 'MISMATCH'}")
    if not same:
        print("\n  REFUSING to report allocation errors: the prints and the bars"
              "\n  are not the same market, so 'truth' would be truth for"
              "\n  something else.")
        sys.exit(2)

    summary = {"instrument_guard": {"n": int(len(xa)), "median": float(np.median(xa)),
                                    "max": float(xa.max()), "units": sorted(units)},
               "n_days": int(d.day.nunique()), "lookback_h": a.lookback_h}
    for tag, g in d.groupby("atr_kind"):
        def q(x):
            return dict(median=float(np.median(x)), mean=float(np.mean(x)),
                        q75=float(np.percentile(x, 75)), q90=float(np.percentile(x, 90)),
                        max=float(np.max(x)))
        summary[tag] = dict(n=int(len(g)), bin_size_median=float(g.bin_size.median()),
                            atr_median=float(g.atr.median()),
                            err_uniform=q(g.err_uniform), err_close=q(g.err_close),
                            disagree=q(g.disagree),
                            uniform_beats_close=float((g.err_uniform < g.err_close).mean()),
                            exact_bin_uniform=float((g.err_uniform < 1e-9).mean()),
                            exact_bin_close=float((g.err_close < 1e-9).mean()))
        s = summary[tag]
        print(f"\n=== bin from {tag}  (median ATR={s['atr_median']:.4g}, "
              f"median bin={s['bin_size_median']:.4g}, n={s['n']}) ===")
        for k in ("err_uniform", "err_close", "disagree"):
            v = s[k]
            print(f"  {k:12s} median={v['median']:.4f} mean={v['mean']:.4f} "
                  f"q75={v['q75']:.4f} q90={v['q90']:.4f} max={v['max']:.3f}")
        print(f"  uniform closer to truth than close-bin: "
              f"{s['uniform_beats_close']*100:.1f}% of samples")
        print(f"  landed in the true POC bin: uniform {s['exact_bin_uniform']*100:.1f}%, "
              f"close {s['exact_bin_close']*100:.1f}%")
        med = s["disagree"]["median"]
        print(f"  PLAN CRITERION  median disagreement = {med:.4f} ATR vs 0.5 -> "
              f"{'OK' if med <= 0.5 else 'POC PRECISION INSUFFICIENT'}")

    (OUT / f"stage1_tick_truth_{a.sym}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwritten ->", OUT / f"stage1_tick_truth_{a.sym}.json")


if __name__ == "__main__":
    main()
