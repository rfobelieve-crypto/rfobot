# -*- coding: utf-8 -*-
"""TODO 1.00 — H1 judgment (and the base-rate / calibration reports it needs).

Pre-registered: 341a523 (body), 7e1bb79 (amendments A/B), 04dff2a (amendment C).
The criteria live in TODO 1.00 and are restated here ONLY as executable code,
never loosened.  Every number this file prints that is not a criterion is a
sensitivity and does not enter PASS/REJECT.

H1 PASS (per side, per TODO 1.00 as amended):
    (a) pooled Delta 95% day-clustered CI lower bound > 0
        AND point estimate >= 0.089 ATR              <- economic anchor
    (b) beta1 on poc_dist, WITH vwap_dist in the regression, Newey-West t > 2.5
    (c) sign agreement across tau in {15m,30m,1h,2h,4h} >= 4/5
    (d) at least 2 of the 3 lookbacks satisfy (a)+(b)+(c) simultaneously
REJECT      Delta CI covers 0, or beta1 t < 1.5
INCONCLUSIVE  n < 150, or q75-q25 of poc_dist <= 3x the 5m POC reconstruction
              error measured on the 1m calibration set (amendment C)

Main judgment uses L2 x tau=1h only.  L1/L3 and the other taus are the
robustness legs named in (c)/(d).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "poc_profile"
TAUS = [900, 1800, 3600, 7200, 14400]
TAU_MAIN = 3600
LOOKBACKS = ["L1", "L2", "L3"]
LB_MAIN = "L2"
SIDES = ["sellside", "buyside"]
ECON_ANCHOR = 0.089        # frozen rule's whole per-trade edge, in ATR units
B = 2000
RNG = np.random.default_rng(20260906)


# --------------------------------------------------------------- bootstrap
def _delta_pooled(poc, trend, y):
    """Tercile-by-trend, within-bucket q75-vs-q25 difference, n-weighted."""
    if len(y) < 60:
        return np.nan
    t1, t2 = np.quantile(trend, [1 / 3, 2 / 3])
    num = 0.0
    den = 0
    for lo, hi in ((-np.inf, t1), (t1, t2), (t2, np.inf)):
        m = (trend >= lo) & (trend < hi)
        if m.sum() < 20:
            continue
        p, yy = poc[m], y[m]
        q25, q75 = np.quantile(p, [0.25, 0.75])
        hi_m, lo_m = p >= q75, p <= q25
        if hi_m.sum() < 5 or lo_m.sum() < 5:
            continue
        num += (yy[hi_m].mean() - yy[lo_m].mean()) * m.sum()
        den += m.sum()
    return num / den if den else np.nan


def _delta_by_bucket(poc, trend, y):
    t1, t2 = np.quantile(trend, [1 / 3, 2 / 3])
    out = []
    for name, (lo, hi) in zip(("T1_down", "T2_flat", "T3_up"),
                              ((-np.inf, t1), (t1, t2), (t2, np.inf))):
        m = (trend >= lo) & (trend < hi)
        p, yy = poc[m], y[m]
        q25, q75 = np.quantile(p, [0.25, 0.75])
        hi_m, lo_m = p >= q75, p <= q25
        out.append(dict(bucket=name, n=int(m.sum()),
                        n_hi=int(hi_m.sum()), n_lo=int(lo_m.sum()),
                        mean_hi=float(yy[hi_m].mean()),
                        mean_lo=float(yy[lo_m].mean()),
                        delta=float(yy[hi_m].mean() - yy[lo_m].mean())))
    return out


def day_boot(df, poc_col, y_col, stat=_delta_pooled, b=B):
    """Day-clustered bootstrap of `stat`.  Resamples whole UTC days."""
    days = df["day"].values
    uniq, inv = np.unique(days, return_inverse=True)
    idx_by_day = [np.where(inv == k)[0] for k in range(len(uniq))]
    poc = df[poc_col].to_numpy(float)
    tr = df["trend_24h"].to_numpy(float)
    y = df[y_col].to_numpy(float)
    point = stat(poc, tr, y)
    reps = np.empty(b)
    for i in range(b):
        pick = RNG.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([idx_by_day[k] for k in pick])
        reps[i] = stat(poc[ix], tr[ix], y[ix])
    reps = reps[~np.isnan(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, lo, hi, len(reps)


# -------------------------------------------------------------- regression
def regression(df, lb, tau, extra_interaction=False):
    import statsmodels.api as sm
    d = df.dropna(subset=[f"poc_dist_{lb}", f"vwap_dist_{lb}", f"r_{tau}",
                          "trend_24h", "er_24h", "rv_1h", "rv_24h",
                          "pierce_atr"]).sort_values("t_sweep")
    y = d[f"r_{tau}"].to_numpy(float)
    cols = {
        "poc_dist": d[f"poc_dist_{lb}"].to_numpy(float),
        "vwap_dist": d[f"vwap_dist_{lb}"].to_numpy(float),
        "trend_24h": d["trend_24h"].to_numpy(float),
        "er_24h": d["er_24h"].to_numpy(float),
        "rv_1h": d["rv_1h"].to_numpy(float),
        "rv_24h": d["rv_24h"].to_numpy(float),
        "pierce_atr": d["pierce_atr"].to_numpy(float),
    }
    if extra_interaction:
        cols["poc_depth"] = d[f"poc_depth_{lb}"].to_numpy(float)
        cols["poc_x_depth"] = cols["poc_dist"] * cols["poc_depth"]
    X = pd.DataFrame(cols, index=d.index)
    sess = pd.cut(d["utc_hour"], [-1, 6, 12, 15, 23],
                  labels=["asia", "eu", "us_open", "us_rest"])
    X = pd.concat([X, pd.get_dummies(sess, prefix="s", drop_first=True).astype(float)],
                  axis=1)
    X = sm.add_constant(X)
    nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    cl = sm.OLS(y, X).fit(cov_type="cluster",
                          cov_kwds={"groups": d["day"].values})
    out = dict(n=int(len(d)),
               beta1=float(nw.params["poc_dist"]),
               t_nw=float(nw.tvalues["poc_dist"]),
               p_nw=float(nw.pvalues["poc_dist"]),
               t_cluster=float(cl.tvalues["poc_dist"]),
               beta_vwap=float(nw.params["vwap_dist"]),
               t_vwap=float(nw.tvalues["vwap_dist"]),
               r2=float(nw.rsquared))
    # what beta1 looks like with the confound REMOVED (spec 6.3's point)
    Xn = X.drop(columns=["vwap_dist"])
    nn = sm.OLS(y, Xn).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    out["beta1_no_vwap"] = float(nn.params["poc_dist"])
    out["t_no_vwap"] = float(nn.tvalues["poc_dist"])
    if extra_interaction:
        out["beta_depth"] = float(nw.params["poc_depth"])
        out["t_depth"] = float(nw.tvalues["poc_depth"])
        out["beta_inter"] = float(nw.params["poc_x_depth"])
        out["t_inter"] = float(nw.tvalues["poc_x_depth"])
    return out


# ------------------------------------------------------------- BH procedure
def bh(pvals, q=0.05):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    crit = thresh[k - 1] if k else 0.0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out, crit


# ------------------------------------------------------------------ reports
def load(path=None):
    p = Path(path) if path else RES / "events.csv"
    d = pd.read_csv(p)
    d["day"] = d["day"].astype(str)
    return d


def report_base(d):
    lines = ["# 02 base rate (TODO 1.00)", ""]
    lines.append("POC always sits on the reverse side of the swept level; that")
    lines.append("is an arithmetic identity (level = recent extreme, POC =")
    lines.append("interior of the range), not a finding.  See amendment C.")
    lines.append("")
    rows = []
    for side in SIDES:
        s = d[d.side == side]
        for lb in LOOKBACKS:
            q = s[f"poc_dist_{lb}"].quantile([.05, .25, .5, .75, .95])
            rows.append(dict(side=side, lookback=lb, n=len(s),
                             frac_positive=float((s[f"poc_dist_{lb}"] > 0).mean()),
                             p05=q.iloc[0], q25=q.iloc[1], med=q.iloc[2],
                             q75=q.iloc[3], p95=q.iloc[4],
                             spread=q.iloc[3] - q.iloc[1]))
    base = pd.DataFrame(rows)
    cont = []
    for side in SIDES:
        s = d[d.side == side]
        t1, t2 = s.trend_24h.quantile([1 / 3, 2 / 3])
        for name, m in (("T1_down", s.trend_24h < t1),
                        ("T2_flat", (s.trend_24h >= t1) & (s.trend_24h < t2)),
                        ("T3_up", s.trend_24h >= t2)):
            y = s.loc[m, f"r_{TAU_MAIN}"]
            cont.append(dict(side=side, bucket=name, n=int(m.sum()),
                             cont_rate=float((y > 0).mean()),
                             mean_r=float(y.mean())))
    return base, pd.DataFrame(cont), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "h1"])
    ap.add_argument("--events", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    d = load(a.events or None)
    tag = ("_" + a.tag) if a.tag else ""

    base, cont, notes = report_base(d)
    base.to_csv(RES / f"02_base_rate{tag}.csv", index=False)
    cont.to_csv(RES / f"02_continuation_by_bucket{tag}.csv", index=False)
    print("== 02 base rate ==")
    print(base.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(cont.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if a.cmd == "base":
        return

    # ---------------- 03 Delta (main + lookbacks + windows) ----------------
    rows, pvals, pkeys = [], [], []
    for side in SIDES:
        s = d[d.side == side]
        for lb in LOOKBACKS:
            for tau in TAUS:
                sub = s.dropna(subset=[f"poc_dist_{lb}", f"r_{tau}", "trend_24h"])
                pt, lo, hi, nrep = day_boot(sub, f"poc_dist_{lb}", f"r_{tau}")
                rows.append(dict(side=side, lookback=lb, tau=tau, n=len(sub),
                                 delta=pt, ci_lo=lo, ci_hi=hi,
                                 main=(lb == LB_MAIN and tau == TAU_MAIN)))
    dl = pd.DataFrame(rows)
    dl.to_csv(RES / f"03_h1_delta{tag}.csv", index=False)
    print("\n== 03 Delta (pooled over trend terciles, day-clustered CI) ==")
    print(dl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # per-bucket detail for the main cell
    det = []
    for side in SIDES:
        s = d[d.side == side].dropna(
            subset=[f"poc_dist_{LB_MAIN}", f"r_{TAU_MAIN}", "trend_24h"])
        for r in _delta_by_bucket(s[f"poc_dist_{LB_MAIN}"].to_numpy(float),
                                  s["trend_24h"].to_numpy(float),
                                  s[f"r_{TAU_MAIN}"].to_numpy(float)):
            r["side"] = side
            det.append(r)
    pd.DataFrame(det).to_csv(RES / f"03_h1_delta_by_bucket{tag}.csv", index=False)
    print("\n== 03b per-tercile detail (L2, tau=1h) ==")
    print(pd.DataFrame(det).to_string(index=False,
                                      float_format=lambda x: f"{x:.4f}"))

    # ---------------- 04 regression with the VWAP confound ----------------
    reg = []
    for side in SIDES:
        s = d[d.side == side]
        for lb in LOOKBACKS:
            r = regression(s, lb, TAU_MAIN)
            r.update(side=side, lookback=lb)
            reg.append(r)
            pvals.append(r["p_nw"])
            pkeys.append(f"beta1_{side}_{lb}")
    rg = pd.DataFrame(reg)
    rg.to_csv(RES / f"04_h1_regression{tag}.csv", index=False)
    print("\n== 04 regression (tau=1h): beta1 with vs without vwap_dist ==")
    print(rg[["side", "lookback", "n", "beta1", "t_nw", "t_cluster",
              "beta1_no_vwap", "t_no_vwap", "beta_vwap", "t_vwap", "r2"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- 06 interaction (exploratory, not a criterion) -------
    inter = []
    for side in SIDES:
        r = regression(d[d.side == side], LB_MAIN, TAU_MAIN, extra_interaction=True)
        r.update(side=side, lookback=LB_MAIN)
        inter.append(r)
    pd.DataFrame(inter).to_csv(RES / f"06_h1_interaction{tag}.csv", index=False)

    # ---------------- verdict ----------------
    bh_pass, crit = bh(pvals)
    verdict = {}
    for side in SIDES:
        legs = {}
        for lb in LOOKBACKS:
            m = dl[(dl.side == side) & (dl.lookback == lb) & (dl.tau == TAU_MAIN)].iloc[0]
            r = rg[(rg.side == side) & (rg.lookback == lb)].iloc[0]
            signs = dl[(dl.side == side) & (dl.lookback == lb)].delta
            agree = int(max((signs > 0).sum(), (signs < 0).sum()))
            legs[lb] = dict(
                delta=float(m.delta), ci_lo=float(m.ci_lo), ci_hi=float(m.ci_hi),
                a_ci=bool(m.ci_lo > 0), a_econ=bool(m.delta >= ECON_ANCHOR),
                b_t=float(r.t_nw), b_pass=bool(r.t_nw > 2.5),
                b_reject=bool(abs(r.t_nw) < 1.5),
                c_agree=agree, c_pass=bool(agree >= 4),
                leg_pass=bool(m.ci_lo > 0 and m.delta >= ECON_ANCHOR
                              and r.t_nw > 2.5 and agree >= 4))
        n_pass = sum(v["leg_pass"] for v in legs.values())
        main = legs[LB_MAIN]
        if main["ci_lo"] <= 0 <= main["ci_hi"] or main["b_reject"]:
            v = "REJECT"
        elif n_pass >= 2:
            v = "PASS"
        else:
            v = "REJECT"
        verdict[side] = dict(verdict=v, n_legs_pass=n_pass, legs=legs,
                             n=int((d.side == side).sum()))
    verdict["_bh"] = dict(n_tests_here=len(pvals), crit=float(crit),
                          keys=pkeys, passed=[bool(x) for x in bh_pass])
    verdict["_criteria"] = dict(econ_anchor=ECON_ANCHOR, tau_main=TAU_MAIN,
                                lookback_main=LB_MAIN, B=B,
                                ci="day-clustered bootstrap")
    (RES / f"11_verdict{tag}.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    print("\n== 11 verdict ==")
    for side in SIDES:
        v = verdict[side]
        print(f"  {side:9s} n={v['n']:5d}  {v['verdict']}  legs_pass={v['n_legs_pass']}/3")
        for lb, L in v["legs"].items():
            print(f"      {lb}: delta={L['delta']:+.4f} CI[{L['ci_lo']:+.4f},"
                  f"{L['ci_hi']:+.4f}] econ={L['a_econ']} t={L['b_t']:+.2f}"
                  f" signs={L['c_agree']}/5 -> {L['leg_pass']}")
    print(f"\n  written -> {RES}")


if __name__ == "__main__":
    main()
