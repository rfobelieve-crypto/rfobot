# -*- coding: utf-8 -*-
"""Stage 5 — labels.  Reads bars and events.  Never reads Stage 4.

    data/labels/{COIN}.parquet
        event_id, r_900 .. r_14400, r_norm_900 .. r_norm_14400,
        mfe_4h, mae_4h, t_extreme_4h, base_px

    r_tau      = side_sign_cont * (close(t_sweep + tau) - base) / base
    r_norm_tau = side_sign_cont * (close(t_sweep + tau) - base) / atr_h14
    side_sign_cont : sellside -1 (continuing DOWN is positive), buyside +1
    base           = close of the bar that ends at t_sweep, i.e. the piercing
                     minute's close -- known at t_sweep, and symmetric with the
                     endpoints, which are also bar closes.

Deviation from the plan, stated
    The plan writes `bars_post = bars[bars.ts > t_sweep]`, strictly greater.
    `ts` here is a bar's OPEN time and the index is left-closed, so the bar
    with ts == t_sweep spans [t_sweep, t_sweep+60s) -- entirely in the future.
    Excluding it would silently discard the first minute of the response, the
    most informative one, from MFE/MAE.  This uses `ts >= t_sweep`, which is
    what "strictly after the moment t_sweep" means under this indexing.  The
    tau endpoints are unaffected either way.

Stage 4 and Stage 5 are separate scripts over separate inputs; joining them is
Stage 6's job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BARS = HERE / "data" / "bars"
EVENTS = HERE / "data" / "events"
OUT = HERE / "data" / "labels"
QUALITY = HERE / "data" / "quality"
MIN_MS = 60_000
TAUS = [900, 1800, 3600, 7200, 14400]
H4 = 14400_000 // 1000 * 1000
HORIZON_MS = 4 * 3_600_000
EXTREME_ABS = 10.0          # |r_norm| above this is an instrument suspicion
EXTREME_MAX_FRAC = 0.001    # ... and more than 0.1% of them means STOP
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def build(sym):
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "high", "low", "close", "atr_h14"])
    ev = pd.read_parquet(EVENTS / f"{sym}.parquet")
    ts = b["ts"].to_numpy(np.int64)
    hi = b["high"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    cl = b["close"].to_numpy(float)
    atr = b.set_index("ts")["atr_h14"]
    pos = {t: i for i, t in enumerate(ts)}

    rows = []
    for r in ev.itertuples():
        t = int(r.t_sweep)
        a = atr.get(t, np.nan)
        i_p = pos.get(t - MIN_MS)                 # the piercing bar
        if i_p is None or not np.isfinite(a) or a <= 0:
            continue
        base = cl[i_p]
        if not np.isfinite(base) or base <= 0:
            continue
        cont = -1.0 if r.side == "sellside" else 1.0
        rec = dict(event_id=r.event_id, base_px=float(base))
        for tau in TAUS:
            j = pos.get(t + tau * 1000 - MIN_MS)
            if j is None or not np.isfinite(cl[j]):
                rec[f"r_{tau}"] = np.nan
                rec[f"r_norm_{tau}"] = np.nan
                continue
            d = cont * (cl[j] - base)
            rec[f"r_{tau}"] = float(d / base)
            rec[f"r_norm_{tau}"] = float(d / a)

        # response window: bars opening at or after t_sweep, through +4h
        k0 = int(np.searchsorted(ts, t, side="left"))
        k1 = int(np.searchsorted(ts, t + HORIZON_MS, side="right")) - 1
        assert k0 >= i_p + 1, "label window overlaps the piercing bar"
        assert ts[k0] >= t, "label window starts before t_sweep"
        if k1 >= k0:
            H = hi[k0:k1 + 1]
            L = lo[k0:k1 + 1]
            if r.side == "sellside":
                m = int(np.nanargmin(L))
                fav, adv = L[m], np.nanmax(H)
            else:
                m = int(np.nanargmax(H))
                fav, adv = H[m], np.nanmin(L)
            rec["mfe_4h"] = float(cont * (fav - base) / a)
            rec["mae_4h"] = float(cont * (adv - base) / a)
            rec["t_extreme_4h"] = int(ts[k0 + m] + MIN_MS - t)
        else:
            rec["mfe_4h"] = rec["mae_4h"] = np.nan
            rec["t_extreme_4h"] = -1
        rows.append(rec)
    return pd.DataFrame(rows)


def run_asserts(lb):
    f, detail = [], {}
    for tau in TAUS:
        x = lb[f"r_norm_{tau}"].dropna()
        frac = float((x.abs() > EXTREME_ABS).mean()) if len(x) else 0.0
        detail[f"r_norm_{tau}"] = dict(n=int(len(x)),
                                       frac_gt_10=frac,
                                       max_abs=float(x.abs().max()) if len(x) else 0.0,
                                       std=float(x.std()) if len(x) else 0.0)
        if frac > EXTREME_MAX_FRAC:
            f.append(f"|r_norm_{tau}| > {EXTREME_ABS} in {frac*100:.3f}% "
                     f"(> {EXTREME_MAX_FRAC*100:.1f}%) -- suspect ATR or price data")
    if (lb["base_px"] <= 0).any():
        f.append("non-positive base price")
    return f, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    QUALITY.mkdir(parents=True, exist_ok=True)
    summary, allok = {}, True
    for s in [x.strip().upper() for x in a.syms.split(",") if x.strip()]:
        if not (EVENTS / f"{s}.parquet").exists():
            print(f"{s:5s} no events, skipped")
            continue
        lb = build(s)
        lb.to_parquet(OUT / f"{s}.parquet", index=False)
        fails, detail = run_asserts(lb)
        allok &= not fails
        summary[s] = dict(n=int(len(lb)), asserts=fails, extremes=detail)
        d = detail["r_norm_3600"]
        print(f"{s:5s} n={len(lb):5,}  r_norm_1h std={d['std']:.3f} "
              f"max|.|={d['max_abs']:.2f} frac>10={d['frac_gt_10']*100:.3f}%  "
              f"{'ok' if not fails else 'FAIL ' + str(fails)}")
    (QUALITY / "stage5_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    print("\nStage 5 gate:", "ALL PASS" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
