# -*- coding: utf-8 -*-
"""Stage 6 — H1 judgment.  Criteria are PREREG_STAGE6.md, restated here only
as executable code, never loosened.

PRIMARY (per side, L2, tau=1h)
    r_norm_1h ~ poc_dist_L2 + vwap_dist_L2 + trend_24h + er_24h
                + rv_1h + rv_24h + pierce_atr + session FE + coin FE
    SE clustered by UTC day, pooled across coins.

    PASS          |t| > 2.5 AND beta1 > 0 (the sign H1 predicts)
    REJECT        |t| < 1.5, OR |t| > 2.5 with the WRONG sign
    INCONCLUSIVE  1.5 <= |t| <= 2.5

    The three-way split is deliberate: a two-way one records "could not
    measure" as "measured and lost" (mistake.md 2026-09-04).

    Measured resolution limit, registered before the run: MDE(CI excludes 0)
    = 0.238 (sellside) / 0.167 (buyside) ATR across the IQR, versus an
    economic anchor of 0.089.  Below ~0.2 ATR this design returns
    INCONCLUSIVE, NOT "no effect".

Everything else in this file is robustness: reported in full, never gating.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from power import load, delta_pooled, se_of  # noqa: E402

OUT = HERE / "data" / "results"
LB_MAIN, TAU_MAIN = "L2", 3600
LOOKBACKS = ["L1", "L2", "L3"]
TAUS = [900, 1800, 3600, 7200, 14400]
SIDES = ["sellside", "buyside"]
ECON_ANCHOR = 0.089
RNG = np.random.default_rng(20260906)


def regression(d, lb, tau, per_coin=False):
    import statsmodels.api as sm
    poc, vwap, y = f"poc_dist_{lb}", f"vwap_dist_{lb}", f"r_norm_{tau}"
    d = d.dropna(subset=[poc, vwap, y, "trend_24h", "er_24h", "rv_1h",
                         "rv_24h", "pierce_atr"])
    if len(d) < 100:
        return None
    X = pd.DataFrame({"poc": d[poc].to_numpy(float),
                      "vwap": d[vwap].to_numpy(float),
                      "trend_24h": d["trend_24h"].to_numpy(float),
                      "er_24h": d["er_24h"].to_numpy(float),
                      "rv_1h": d["rv_1h"].to_numpy(float),
                      "rv_24h": d["rv_24h"].to_numpy(float),
                      "pierce_atr": d["pierce_atr"].to_numpy(float)}, index=d.index)
    X = pd.concat([X, pd.get_dummies(d["session"], prefix="s",
                                     drop_first=True).astype(float)], axis=1)
    if not per_coin and d["sym"].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(d["sym"], prefix="c",
                                         drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    f = sm.OLS(d[y].to_numpy(float), X).fit(cov_type="cluster",
                                            cov_kwds={"groups": d["day"].values})
    fn = sm.OLS(d[y].to_numpy(float), X.drop(columns=["vwap"])).fit(
        cov_type="cluster", cov_kwds={"groups": d["day"].values})
    iqr = float(d[poc].quantile(.75) - d[poc].quantile(.25))
    return dict(n=int(len(d)), n_days=int(d["day"].nunique()),
                beta=float(f.params["poc"]), t=float(f.tvalues["poc"]),
                p=float(f.pvalues["poc"]), se=float(f.bse["poc"]),
                beta_vwap=float(f.params["vwap"]), t_vwap=float(f.tvalues["vwap"]),
                beta_no_vwap=float(fn.params["poc"]), t_no_vwap=float(fn.tvalues["poc"]),
                iqr=iqr, effect_across_iqr=float(f.params["poc"] * iqr))


def verdict_of(t, beta):
    if abs(t) < 1.5:
        return "REJECT"
    if abs(t) > 2.5:
        return "PASS" if beta > 0 else "REJECT (significant, WRONG sign)"
    return "INCONCLUSIVE"


def delta_variants(d, lb, tau):
    """Three point-estimate versions, all reported (PREREG section 5)."""
    poc, y = f"poc_dist_{lb}", f"r_norm_{tau}"
    s = d.dropna(subset=[poc, y, "trend_24h"])
    out = {}
    out["mean_arms"] = float(delta_pooled(s[poc].to_numpy(float),
                                          s["trend_24h"].to_numpy(float),
                                          s[y].to_numpy(float)))
    def dmed(p, t, yy):
        t1, t2 = np.quantile(t, [1/3, 2/3]); num, den = 0.0, 0
        for lo, hi in ((-np.inf, t1), (t1, t2), (t2, np.inf)):
            m = (t >= lo) & (t < hi)
            if m.sum() < 20:
                continue
            q25, q75 = np.quantile(p[m], [.25, .75])
            a, b = p[m] >= q75, p[m] <= q25
            if a.sum() < 5 or b.sum() < 5:
                continue
            num += (np.median(yy[m][a]) - np.median(yy[m][b])) * m.sum(); den += m.sum()
        return num / den if den else np.nan
    out["median_arms"] = float(dmed(s[poc].to_numpy(float),
                                    s["trend_24h"].to_numpy(float),
                                    s[y].to_numpy(float)))
    g = s.groupby("day").agg(**{poc: (poc, "median"), "trend_24h": ("trend_24h", "median"),
                                y: (y, "median")}).reset_index()
    out["day_aggregated"] = float(delta_pooled(g[poc].to_numpy(float),
                                               g["trend_24h"].to_numpy(float),
                                               g[y].to_numpy(float)))
    return out


def quintiles(d, lb, tau):
    poc, y = f"poc_dist_{lb}", f"r_norm_{tau}"
    s = d.dropna(subset=[poc, y]).copy()
    s["q"] = pd.qcut(s[poc], 5, labels=["Q1_near", "Q2", "Q3", "Q4", "Q5_far"])
    g = s.groupby("q")[y].agg(["size", "mean", "median"])
    g["cont_rate"] = s.groupby("q")[y].apply(lambda x: float((x > 0).mean()))
    return g.reset_index().to_dict("records")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"pooled events={len(df):,}  UTC days={df.day.nunique():,}  "
          f"coins={df.sym.nunique()}\n")

    res = {"criteria": "PREREG_STAGE6.md", "lookback_main": LB_MAIN,
           "tau_main": TAU_MAIN, "econ_anchor": ECON_ANCHOR}

    # ---------------- PRIMARY ----------------
    print("== PRIMARY: regression, L2, tau=1h, day-clustered ==\n")
    print(f"{'side':10s} {'n':>6s} {'days':>5s} {'beta1':>9s} {'t':>7s} "
          f"{'x IQR':>8s} | {'beta(no vwap)':>13s} {'t':>7s} | "
          f"{'beta_vwap':>9s} {'t':>7s}")
    prim = {}
    for side in SIDES:
        r = regression(df[df.side == side], LB_MAIN, TAU_MAIN)
        v = verdict_of(r["t"], r["beta"])
        r["verdict"] = v
        prim[side] = r
        print(f"{side:10s} {r['n']:6,d} {r['n_days']:5,d} {r['beta']:+9.4f} "
              f"{r['t']:+7.2f} {r['effect_across_iqr']:+8.4f} | "
              f"{r['beta_no_vwap']:+13.4f} {r['t_no_vwap']:+7.2f} | "
              f"{r['beta_vwap']:+9.4f} {r['t_vwap']:+7.2f}")
    print()
    for side in SIDES:
        print(f"  {side:10s} -> {prim[side]['verdict']}")
    res["primary"] = prim

    # BH over the two primary tests
    ps = np.array([prim[s]["p"] for s in SIDES])
    order = np.argsort(ps)
    bh = ps[order] <= 0.05 * (np.arange(1, 3) / 2)
    res["bh"] = {SIDES[order[i]]: bool(bh[i]) for i in range(2)}
    print(f"\n  BH(2 primary tests, q=0.05): "
          + ", ".join(f"{k}={'pass' if v else 'fail'}" for k, v in res["bh"].items()))

    # ---------------- ROBUSTNESS ----------------
    print("\n== robustness: beta1 (t) across lookbacks x tau ==\n")
    grid = []
    hdr = f"{'side':10s} {'lb':4s}" + "".join(f"{t//60:>10d}m" for t in TAUS)
    print(hdr)
    for side in SIDES:
        for lb in LOOKBACKS:
            row = []
            for tau in TAUS:
                r = regression(df[df.side == side], lb, tau)
                row.append(r)
                grid.append(dict(side=side, lookback=lb, tau=tau,
                                 beta=r["beta"], t=r["t"], n=r["n"]))
            print(f"{side:10s} {lb:4s}" +
                  "".join(f"{x['beta']:+7.3f}({x['t']:+.1f})" for x in row))
    res["grid"] = grid

    print("\n== robustness: three delta versions (L2, tau=1h) ==\n")
    dv = {}
    for side in SIDES:
        dv[side] = delta_variants(df[df.side == side], LB_MAIN, TAU_MAIN)
        print(f"  {side:10s} " + "  ".join(f"{k}={v:+.4f}" for k, v in dv[side].items()))
    res["delta_variants"] = dv

    print("\n== robustness: quintile monotonicity (full grid, L2, tau=1h) ==\n")
    qs = {}
    for side in SIDES:
        qs[side] = quintiles(df[df.side == side], LB_MAIN, TAU_MAIN)
        print(f"  {side}")
        for r in qs[side]:
            print(f"    {r['q']:8s} n={r['size']:5d} mean={r['mean']:+7.4f} "
                  f"median={r['median']:+7.4f} cont={r['cont_rate']*100:5.1f}%")
    res["quintiles"] = qs

    print("\n== robustness: per coin (L2, tau=1h) ==\n")
    pc = []
    for side in SIDES:
        cells = []
        for sym, g in df[df.side == side].groupby("sym"):
            r = regression(g, LB_MAIN, TAU_MAIN, per_coin=True)
            if r:
                cells.append((sym, r["beta"], r["t"], r["n"]))
                pc.append(dict(side=side, sym=sym, beta=r["beta"], t=r["t"], n=r["n"]))
        pos = sum(1 for _, b, _, _ in cells if b > 0)
        print(f"  {side:10s} beta>0 in {pos}/{len(cells)} coins:  "
              + "  ".join(f"{s}={b:+.3f}({t:+.1f})" for s, b, t, _ in cells))
    res["per_coin"] = pc

    (OUT / "stage6_h1.json").write_text(json.dumps(res, indent=2, default=float),
                                        encoding="utf-8")
    print("\nwritten ->", OUT / "stage6_h1.json")


if __name__ == "__main__":
    main()
