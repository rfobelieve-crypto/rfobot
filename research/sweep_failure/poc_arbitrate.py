# -*- coding: utf-8 -*-
"""TODO 1.00 report 04b — which allocation does the 1m data side with?

Why this exists.  The registered 1.2 sensitivity (all volume into the close
bin) was run and it MOVED the mechanism conclusion:

    sellside L2, beta1 on poc_dist WITH vwap_dist in the regression
        uniform allocation   +0.050  (t +1.43)   -> POC absorbed by VWAP
        close-bin allocation -0.065  (t -3.06)   -> POC survives, VWAP weakens

H1's verdict (REJECT versus the hypothesised direction) is the same either
way, but the EXPLANATION is not.  Amendment D (committed before any H1 result
was seen) pre-committed the tie-break: the 1m data arbitrates, because it is
ground truth for both allocations.

So: same events, same window, same bins, four POC estimators
    5m-uniform / 5m-close / 1m-uniform / 1m-close
and the same regression on each.  BTC+ETH only (the 1m calibration window),
so n is ~1.1k, not 13k -- underpowered for a verdict, decisive for the
question "is the flip a 5m artifact?".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc            # noqa: E402
import poc_profile as pp           # noqa: E402
from poc_calib import M1Frame      # noqa: E402

SYMS = ["BTC", "ETH"]
RES = HERE.parent / "results" / "poc_profile"
LOOKBACK = 86400          # L2, the main lookback
TAU = 3600                # main horizon


def profile_at(fr, t_sweep, bin_size, lvl, side_sign, alt):
    hi = fr.idx_at_or_before_close(t_sweep - 1)
    pr = pp.build_profile(fr, hi, t_sweep - LOOKBACK, bin_size, alt_close_only=alt)
    if pr is None:
        return None, None
    bins, tot, vwap, _ = pr
    b = max(bins, key=lambda k: bins[k])
    poc = (b + 0.5) * bin_size
    return side_sign * (poc - lvl), side_sign * (vwap - lvl)


def build():
    ticks = pp.tick_sizes(SYMS)
    rows = []
    for sym in SYMS:
        b1 = sc.load_csv(str(pp.CACHE / f"{sym}USDT_1h.csv"))
        atr = sc.atr14(b1)
        c1 = [x[sc.C] for x in b1]
        f5 = pp.M5Frame(pp.CACHE / "m5" / (sym + "_5m.csv"))
        f1 = M1Frame(pp.CACHE / "m1v" / (sym + "_1m.csv"))
        t_lo = f1.ct[0] + LOOKBACK
        tick = ticks.get(sym, 0.0)
        import time as _t
        for e in sc.detect_sweeps(b1):
            j, lvl, kind = e["j"], e["level"], e["kind"]
            A = atr[j]
            if A is None or A <= 0 or j < 25:
                continue
            i0 = f5.by_open(b1[j][0])
            if i0 is None:
                continue
            pierce = None
            for k in range(i0, min(i0 + 12, len(f5))):
                if f5.ot[k] >= b1[j][0] + 3600:
                    break
                if (kind == "buy" and f5.h[k] > lvl) or \
                   (kind == "sell" and f5.l[k] < lvl):
                    pierce = k
                    break
            if pierce is None:
                continue
            ts = f5.ct[pierce]
            if ts < t_lo or ts + TAU > f1.ct[-1]:
                continue
            kk = f5.by_close(ts + TAU)
            if kk is None:
                continue
            base = f5.c[pierce]
            side_sign = 1.0 if kind == "sell" else -1.0
            cont = -1.0 if kind == "sell" else 1.0
            bs = max(tick, A / 20.0)
            r = dict(sym=sym, side="sellside" if kind == "sell" else "buyside",
                     t_sweep=ts, day=_t.strftime("%Y-%m-%d", _t.gmtime(ts)),
                     utc_hour=_t.gmtime(ts).tm_hour, atr=A,
                     r=cont * (f5.c[kk] - base) / A,
                     pierce_atr=((f5.h[pierce] - lvl) if kind == "buy"
                                 else (lvl - f5.l[pierce])) / A)
            ok = True
            for tag, fr, alt in (("m5u", f5, False), ("m5c", f5, True),
                                 ("m1u", f1, False), ("m1c", f1, True)):
                p, v = profile_at(fr, ts, bs, lvl, side_sign, alt)
                if p is None:
                    ok = False
                    break
                r["poc_" + tag] = p / A
                r["vwap_" + tag] = v / A
            if not ok:
                continue
            r["trend_24h"] = (c1[j - 1] - c1[j - 24]) / A
            path = sum(abs(c1[k] - c1[k - 1]) for k in range(j - 23, j))
            r["er_24h"] = abs(c1[j - 1] - c1[j - 24]) / path if path > 0 else 0.0
            rets = [(c1[k] - c1[k - 1]) / c1[k - 1] for k in range(j - 23, j)]
            mu = sum(rets) / len(rets)
            r["rv_24h"] = float(np.std(rets, ddof=1))
            lo12 = max(1, pierce - 12)
            r5 = [(f5.c[k] - f5.c[k - 1]) / f5.c[k - 1] for k in range(lo12, pierce)]
            r["rv_1h"] = float(np.std(r5, ddof=1) * np.sqrt(12)) if len(r5) > 2 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def reg(d, tag):
    import statsmodels.api as sm
    dd = d.dropna(subset=["poc_" + tag, "vwap_" + tag, "r", "trend_24h",
                          "er_24h", "rv_1h", "rv_24h", "pierce_atr"]).sort_values("t_sweep")
    X = pd.DataFrame({
        "poc": dd["poc_" + tag].to_numpy(float),
        "vwap": dd["vwap_" + tag].to_numpy(float),
        "trend_24h": dd["trend_24h"].to_numpy(float),
        "er_24h": dd["er_24h"].to_numpy(float),
        "rv_1h": dd["rv_1h"].to_numpy(float),
        "rv_24h": dd["rv_24h"].to_numpy(float),
        "pierce_atr": dd["pierce_atr"].to_numpy(float)}, index=dd.index)
    sess = pd.cut(dd["utc_hour"], [-1, 6, 12, 15, 23],
                  labels=["asia", "eu", "us_open", "us_rest"])
    X = pd.concat([X, pd.get_dummies(sess, prefix="s", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    y = dd["r"].to_numpy(float)
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    mn = sm.OLS(y, X.drop(columns=["vwap"])).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    return dict(alloc=tag, n=len(dd),
                beta_poc=float(m.params["poc"]), t_poc=float(m.tvalues["poc"]),
                beta_vwap=float(m.params["vwap"]), t_vwap=float(m.tvalues["vwap"]),
                beta_poc_no_vwap=float(mn.params["poc"]),
                t_poc_no_vwap=float(mn.tvalues["poc"]))


def main():
    d = build()
    RES.mkdir(parents=True, exist_ok=True)
    d.to_csv(RES / "04b_arbitrate_events.csv", index=False)
    print("events with all four POC estimators:", len(d),
          d.side.value_counts().to_dict())
    out = {}
    for side in ["sellside", "buyside"]:
        s = d[d.side == side]
        rows = [reg(s, t) for t in ("m5u", "m5c", "m1u", "m1c")]
        out[side] = rows
        print(f"\n{side}  (L2, tau=1h, same events, only the allocation changes)")
        print(pd.DataFrame(rows).to_string(index=False,
                                           float_format=lambda x: f"{x:.4f}"))
    corr = {}
    for side in ["sellside", "buyside"]:
        s = d[d.side == side]
        corr[side] = {a: float(s["poc_" + a].corr(s["poc_m1u"]))
                      for a in ("m5u", "m5c", "m1c")}
    print("\ncorr of each estimator with the 1m-uniform reference:")
    print(pd.DataFrame(corr).to_string(float_format=lambda x: f"{x:.4f}"))
    (RES / "04b_arbitrate.json").write_text(
        json.dumps(dict(reg=out, corr_with_m1u=corr, n=len(d)), indent=2),
        encoding="utf-8")
    print("\nwritten ->", RES / "04b_arbitrate.json")


if __name__ == "__main__":
    main()
