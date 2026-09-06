# -*- coding: utf-8 -*-
"""TODO 1.00 report 01 — how much POC does the 5m reconstruction lose?

The study rebuilds the volume profile from 5m bars by spreading each bar's
volume uniformly across the bins its high-low touches.  That is an
approximation of the true traded distribution.  This script measures the
approximation error directly, instead of disclosing it as a sentence:

    for every core-event that falls inside the 1m calibration window
    (BTC/ETH, 365d), build the SAME L2 profile from 1m bars (ground truth)
    and from 5m bars, and report |POC_1m - POC_5m| / ATR.

Amendment C anchors the degeneracy guard on the MEDIAN of this displacement:
    q75(POC_dist) - q25(POC_dist) must exceed 3x it.

Also reports the alternative allocation (all volume into the close bin) on the
same events, which is spec 1.2's registered sensitivity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc            # noqa: E402
import poc_profile as pp           # noqa: E402

SYMS = ["BTC", "ETH"]
RES = HERE.parent / "results" / "poc_profile"




class M1Frame(pp.M5Frame):
    """Same interface, 60s bars."""

    def __init__(self, path):
        rows = sc.load_csv(str(path))
        rows.sort(key=lambda r: r[0])
        self.ot = [r[0] for r in rows]
        self.ct = [r[0] + 60 for r in rows]
        self.o = [r[1] for r in rows]
        self.h = [r[2] for r in rows]
        self.l = [r[3] for r in rows]
        self.c = [r[4] for r in rows]
        self.v = [r[5] for r in rows]
        self._ot_ix = {t: i for i, t in enumerate(self.ot)}
        self._ct_ix = {t: i for i, t in enumerate(self.ct)}


def poc_of(fr, t_sweep, bin_size, lookback_s, alt=False):
    hi = fr.idx_at_or_before_close(t_sweep - 1)
    pr = pp.build_profile(fr, hi, t_sweep - lookback_s, bin_size,
                          alt_close_only=alt)
    if pr is None:
        return None
    bins = pr[0]
    b = max(bins, key=lambda k: bins[k])
    return (b + 0.5) * bin_size


def main():
    ticks = pp.tick_sizes(SYMS)
    out = {}
    for sym in SYMS:
        b1 = sc.load_csv(str(pp.CACHE / f"{sym}USDT_1h.csv"))
        atr = sc.atr14(b1)
        f5 = pp.M5Frame(pp.CACHE / "m5" / (sym + "_5m.csv"))
        p1 = pp.CACHE / "m1v" / (sym + "_1m.csv")
        if not p1.exists():
            print(sym, "no 1m calibration data, skipped")
            continue
        f1 = M1Frame(p1)
        t_lo = f1.ct[0] + 86400          # need a full L2 window inside coverage
        tick = ticks.get(sym, 0.0)

        d_uni, d_alt, n = [], [], 0
        for e in sc.detect_sweeps(b1):
            j, lvl, kind = e["j"], e["level"], e["kind"]
            A = atr[j]
            if A is None or A <= 0:
                continue
            hour_open = b1[j][0]
            i0 = f5.by_open(hour_open)
            if i0 is None:
                continue
            pierce = None
            for k in range(i0, min(i0 + 12, len(f5))):
                if f5.ot[k] >= hour_open + 3600:
                    break
                if (kind == "buy" and f5.h[k] > lvl) or \
                   (kind == "sell" and f5.l[k] < lvl):
                    pierce = k
                    break
            if pierce is None:
                continue
            t_sweep = f5.ct[pierce]
            if t_sweep < t_lo or t_sweep > f1.ct[-1]:
                continue
            bs = max(tick, A / 20.0)
            a5 = poc_of(f5, t_sweep, bs, 86400)
            a1 = poc_of(f1, t_sweep, bs, 86400)
            aa = poc_of(f5, t_sweep, bs, 86400, alt=True)
            if a5 is None or a1 is None:
                continue
            n += 1
            d_uni.append(abs(a1 - a5) / A)
            if aa is not None:
                d_alt.append(abs(a1 - aa) / A)
        du, da = np.array(d_uni), np.array(d_alt)
        out[sym] = dict(
            n=n,
            uniform_median=float(np.median(du)), uniform_mean=float(du.mean()),
            uniform_q90=float(np.percentile(du, 90)),
            uniform_frac_same_bin=float((du < 1e-9).mean()),
            altclose_median=float(np.median(da)),
            altclose_q90=float(np.percentile(da, 90)),
        )
        print(f"{sym}: n={n}  |POC_1m - POC_5m|/ATR  median={out[sym]['uniform_median']:.4f}"
              f"  mean={out[sym]['uniform_mean']:.4f}  q90={out[sym]['uniform_q90']:.4f}"
              f"  exact-match={out[sym]['uniform_frac_same_bin']:.3f}")
        print(f"      alt(close-bin) vs 1m: median={out[sym]['altclose_median']:.4f}"
              f"  q90={out[sym]['altclose_q90']:.4f}")

    if out:
        med = float(np.mean([v["uniform_median"] for v in out.values()]))
        out["_anchor"] = dict(
            median_displacement_atr=med,
            guard_threshold_3x=3 * med,
            note="amendment C: q75-q25 of poc_dist must exceed guard_threshold_3x")
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "01_profile_method.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nanchor: median displacement {med:.4f} ATR -> guard 3x = {3*med:.4f} ATR")
        print("written ->", RES / "01_profile_method.json")


if __name__ == "__main__":
    main()
