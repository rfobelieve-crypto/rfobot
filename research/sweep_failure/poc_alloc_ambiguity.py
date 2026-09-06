# -*- coding: utf-8 -*-
"""TODO 1.00 report 01b — allocation ambiguity: uniform vs close-bin, SAME bars.

Asked for by the user 2026-09-06 (mid-run):

    "1 分鐘 bar 均勻分配和全部分到 close 這兩種方法算出的 POC 會不同。
     先算兩者的 POC 差異分布——如果差異中位數超過 0.5 ATR，那你的 POC
     精度根本撐不起 H1，這時候的正確結論是「需要逐筆資料」，不是硬跑。"

This is NOT what 01_profile_method.json measured.  That one compared
5m-reconstruction against the 1m ground truth, i.e. RESOLUTION LOSS.
This one holds the bars fixed and varies only the allocation rule, i.e. the
irreducible AMBIGUITY of not having tick data.  Two different quantities.

Reported, per event, in ATR units:
    A  |POC(1m uniform)  - POC(1m close-bin)|   <- the user's criterion
    B  |POC(5m uniform)  - POC(5m close-bin)|   <- same, at the study's bars
    C  |POC(5m uniform)  - POC(1m uniform)|     <- resolution loss (already known)

Plus the reliability ratio that decides whether the error matters for H1:
    var(POC_dist across events) vs var(measurement error)
    attenuation of beta1 ~= var_true / (var_true + var_err)
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
from poc_calib import M1Frame, poc_of   # noqa: E402

SYMS = ["BTC", "ETH"]
RES = HERE.parent / "results" / "poc_profile"
CRIT = 0.5          # user's stop threshold, ATR units, median of A


def main():
    ticks = pp.tick_sizes(SYMS)
    out = {}
    for sym in SYMS:
        b1 = sc.load_csv(str(pp.CACHE / f"{sym}USDT_1h.csv"))
        atr = sc.atr14(b1)
        f5 = pp.M5Frame(pp.CACHE / "m5" / (sym + "_5m.csv"))
        f1 = M1Frame(pp.CACHE / "m1v" / (sym + "_1m.csv"))
        t_lo = f1.ct[0] + 86400
        tick = ticks.get(sym, 0.0)

        A, Bd, Cd, span = [], [], [], []
        for e in sc.detect_sweeps(b1):
            j, lvl, kind = e["j"], e["level"], e["kind"]
            a = atr[j]
            if a is None or a <= 0:
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
            if ts < t_lo or ts > f1.ct[-1]:
                continue
            bs = max(tick, a / 20.0)
            p1u = poc_of(f1, ts, bs, 86400)
            p1c = poc_of(f1, ts, bs, 86400, alt=True)
            p5u = poc_of(f5, ts, bs, 86400)
            p5c = poc_of(f5, ts, bs, 86400, alt=True)
            if None in (p1u, p1c, p5u, p5c):
                continue
            A.append(abs(p1u - p1c) / a)
            Bd.append(abs(p5u - p5c) / a)
            Cd.append(abs(p5u - p1u) / a)
            span.append(abs(p1u - lvl) / a)      # |POC_dist| for the true POC

        A, Bd, Cd, span = map(np.asarray, (A, Bd, Cd, span))
        def d(x):
            return dict(median=float(np.median(x)), mean=float(x.mean()),
                        q75=float(np.percentile(x, 75)),
                        q90=float(np.percentile(x, 90)),
                        max=float(x.max()),
                        frac_same_bin=float((x < 1e-9).mean()))
        out[sym] = dict(n=int(len(A)),
                        A_1m_uniform_vs_1m_close=d(A),
                        B_5m_uniform_vs_5m_close=d(Bd),
                        C_5m_vs_1m_resolution=d(Cd),
                        poc_dist_sd=float(span.std()))
        print(f"\n{sym}  n={len(A)}   (ATR units)")
        for k in ("A_1m_uniform_vs_1m_close", "B_5m_uniform_vs_5m_close",
                  "C_5m_vs_1m_resolution"):
            v = out[sym][k]
            print(f"  {k:28s} median={v['median']:.4f} mean={v['mean']:.4f}"
                  f" q75={v['q75']:.4f} q90={v['q90']:.4f} max={v['max']:.3f}"
                  f" same-bin={v['frac_same_bin']:.3f}")
        # attenuation: error variance vs the across-event variance of POC_dist
        for name, err in (("A(1m alloc)", A), ("B(5m alloc)", Bd),
                          ("C(5m res)", Cd)):
            # |diff| of two estimators -> sd of one estimator's error ~ mean|d|/ (2/sqrt(pi)) /sqrt(2)
            sd_err = float(np.mean(err)) * np.sqrt(np.pi / 2) / np.sqrt(2)
            rel = span.var() / (span.var() + sd_err ** 2)
            print(f"    {name:12s} sd_err~{sd_err:.4f}  sd(POC_dist)={span.std():.3f}"
                  f"  reliability={rel:.4f}  -> beta1 attenuated to {rel*100:.1f}%")
            out[sym].setdefault("attenuation", {})[name] = dict(
                sd_err=sd_err, reliability=float(rel))

    med_A = float(np.mean([out[s]["A_1m_uniform_vs_1m_close"]["median"] for s in out]))
    out["_verdict"] = dict(
        criterion="median |POC(1m uniform) - POC(1m close-bin)| in ATR",
        threshold=CRIT, measured=med_A,
        passes=bool(med_A <= CRIT),
        note=("<= threshold: bar-level allocation ambiguity does not dominate; "
              "tick data not required for H1's resolution."
              if med_A <= CRIT else
              "> threshold: POC precision cannot support H1; correct conclusion "
              "is INCONCLUSIVE-DATA, need tick data."))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "01b_alloc_ambiguity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n== user criterion: median A = {med_A:.4f} ATR vs threshold {CRIT} "
          f"-> {'PASS (no tick data needed)' if med_A <= CRIT else 'FAIL (need tick data)'}")
    print("written ->", RES / "01b_alloc_ambiguity.json")


if __name__ == "__main__":
    main()
