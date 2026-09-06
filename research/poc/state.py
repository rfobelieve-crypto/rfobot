# -*- coding: utf-8 -*-
"""Stage 4 — state variables, one row per event, all knowable at t_sweep.

    data/state/{COIN}.parquet
        event_id,
        poc_L1..L3, vwap_L1..L3, poc_dist_L*, vwap_dist_L*,
        poc_conc_L*, poc_depth_L*, next_hvn_L*, has_next_hvn_L*, nbars_L*,
        trend_24h, er_24h, rv_1h, rv_24h, session, atr_h14, bin_size

Frozen choices carried in from earlier stages
    bin_size = max(tick_size, atr_h14 / 20), frozen at t_sweep.
        atr_h14, not atr_1h — decided at the Stage 1 gate against TICK truth:
        median reconstruction error $27.4 vs $267, landed in the true POC bin
        41.5% vs 12.0%.  Nothing about any label entered that choice.
    lookbacks  L1 ('volume', 0.5 x 30-day average daily volume)
               L2 ('time', 24)   <- main
               L3 ('time', 72)
    side_sign  sellside +1, buyside -1, so poc_dist > 0 always means "the
               traded mass sits on the far side of the swept level".

The one thing this stage must not get wrong
    `bars_pre = bars[bars.ts < t_sweep]` — strictly less than.  The profile
    engine re-asserts it internally (a bar must have CLOSED before t_sweep),
    so a `<=` slip here fails loudly rather than quietly.

Stage 4 and Stage 5 are built by separate scripts reading separate inputs and
writing separate files.  Nothing here reads a label.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from profile import Bars, build_profile, depth_between  # noqa: E402

BARS = HERE / "data" / "bars"
EVENTS = HERE / "data" / "events"
OUT = HERE / "data" / "state"
QUALITY = HERE / "data" / "quality"
MIN_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
LOOKBACKS = {"L1": ("volume", 0.5), "L2": ("time", 24), "L3": ("time", 72)}
LB_MAIN = "L2"
L1_MIN_BARS = 60          # registered floor, see build()
ERR_IQR_FRAC = 0.10       # tick-truth error must stay under this share of the IQR
DISAGREE_MAX = 0.50       # the plan's registered allocation-disagreement bar
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def session_of(hour):
    if hour < 7:
        return "asia"
    if hour < 13:
        return "eu"
    if hour < 16:
        return "us_open"
    return "us_rest"


def build(sym):
    b = pd.read_parquet(BARS / f"{sym}.parquet")
    ev = pd.read_parquet(EVENTS / f"{sym}.parquet")
    bars = Bars(b.ts, b.open, b.high, b.low, b.close, b.volume)
    tick = float(b.tick_size.iloc[0])
    ts = bars.ts
    close = bars.close
    vol = np.nan_to_num(bars.volume, nan=0.0)
    cumvol = np.concatenate([[0.0], np.cumsum(vol)])
    atrh = b.set_index("ts")["atr_h14"]

    def close_at(t):
        """Close of the bar that ENDS at t (i.e. bar with ts == t - 1min)."""
        i = int(np.searchsorted(ts, t - MIN_MS))
        if i >= len(ts) or ts[i] != t - MIN_MS:
            return np.nan
        return close[i]

    rows = []
    for r in ev.itertuples():
        t = int(r.t_sweep)
        a = atrh.get(t, np.nan)
        if not np.isfinite(a) or a <= 0:
            continue
        bs = max(tick, a / 20.0)
        side_sign = 1.0 if r.side == "sellside" else -1.0

        # 30-day average daily volume, using only bars closed before t_sweep
        hi_i = int(np.searchsorted(ts, t - MIN_MS, side="right"))
        lo_i = int(np.searchsorted(ts, t - 30 * DAY_MS, side="left"))
        adv = (cumvol[hi_i] - cumvol[lo_i]) / 30.0 if hi_i > lo_i else np.nan

        rec = dict(event_id=r.event_id, side=r.side, t_sweep=t,
                   sweep_lvl=float(r.sweep_lvl), atr_h14=float(a), bin_size=bs,
                   hour_ts=int(r.hour_ts),
                   day=pd.Timestamp(t, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                   utc_hour=int(pd.Timestamp(t, unit="ms", tz="UTC").hour),
                   pierce_atr=float(r.cross_depth_atr))
        rec["session"] = session_of(rec["utc_hour"])

        ok = False
        for name, lb in LOOKBACKS.items():
            p = build_profile(bars, t, lb, "uniform", bs,
                              avg_daily_volume=adv if lb[0] == "volume" else None)
            if p is None:
                continue
            nh = p.next_hvn(side_sign)
            rec[f"poc_{name}"] = p.poc
            rec[f"vwap_{name}"] = p.vwap
            rec[f"poc_dist_{name}"] = side_sign * (p.poc - r.sweep_lvl) / a
            rec[f"vwap_dist_{name}"] = side_sign * (p.vwap - r.sweep_lvl) / a
            rec[f"poc_conc_{name}"] = float(
                p.volumes.max() / p.total_volume)
            rec[f"poc_depth_{name}"] = depth_between(p, p.poc, r.sweep_lvl)
            rec[f"next_hvn_{name}"] = np.nan if nh is None else nh
            rec[f"has_next_hvn_{name}"] = nh is not None
            rec[f"nbars_{name}"] = p.n_bars
            rec[f"first_ms_{name}"] = p.first_ms
            rec[f"last_ms_{name}"] = p.last_ms
            ok = True
        if not ok:
            continue
        # L1 floor (registered 2026-09-06, before any label was looked at):
        # the volume window collapses on the highest-volatility days -- 14 of
        # BTC's events had <60 bars, the shortest 7, and the most extreme were
        # all 2025-10-10.  Those 14 sat 6.05 ATR away from L2 versus 0.100 for
        # the sample.  A profile built from a few minutes of panic is not a
        # profile.  L1 is voided for such events; L2/L3 are untouched.
        if rec.get("nbars_L1", 0) < L1_MIN_BARS:
            for k in list(rec):
                if k.endswith("_L1"):
                    rec[k] = np.nan
            rec["l1_voided"] = True
        else:
            rec["l1_voided"] = False

        c0 = close_at(t)
        c24 = close_at(t - DAY_MS)
        rec["trend_24h"] = (c0 - c24) / a if np.isfinite(c0) and np.isfinite(c24) else np.nan
        hcl = np.array([close_at(t - k * HOUR_MS) for k in range(24, -1, -1)])
        if np.isfinite(hcl).all():
            step = np.abs(np.diff(hcl)).sum()
            rec["er_24h"] = abs(hcl[-1] - hcl[0]) / step if step > 0 else 0.0
            hr = np.diff(hcl) / hcl[:-1]
            rec["rv_24h"] = float(np.std(hr, ddof=1))
        else:
            rec["er_24h"] = rec["rv_24h"] = np.nan
        i = int(np.searchsorted(ts, t - MIN_MS, side="right"))
        seg = close[max(0, i - 61):i]
        seg = seg[np.isfinite(seg)]
        rec["rv_1h"] = float(np.std(np.diff(seg) / seg[:-1], ddof=1) * np.sqrt(60)) \
            if len(seg) > 3 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def run_asserts(st, sym):
    f = []
    for name in LOOKBACKS:
        c = f"poc_dist_{name}"
        if c not in st:
            f.append(f"{c} missing")
            continue
        d = st[f"poc_depth_{name}"].dropna()
        if not ((d >= -1e-9) & (d <= 1 + 1e-9)).all():
            f.append(f"poc_depth_{name} outside [0,1]")
    # sign convention: reconstruct poc_dist from raw prices and compare
    s = st[st.side == "sellside"]
    chk = ((s[f"poc_{LB_MAIN}"] - s["sweep_lvl"]) / s["atr_h14"] - s[f"poc_dist_{LB_MAIN}"]).abs()
    if chk.max() > 1e-9:
        f.append("sellside poc_dist sign/scale wrong")
    bq = st[st.side == "buyside"]
    chk2 = (-(bq[f"poc_{LB_MAIN}"] - bq["sweep_lvl"]) / bq["atr_h14"]
            - bq[f"poc_dist_{LB_MAIN}"]).abs()
    if chk2.max() > 1e-9:
        f.append("buyside poc_dist sign/scale wrong")
    # --- replaces the plan's "three lookbacks must correlate > 0.5" assert ---
    # That one was disproved as a definition-error detector on 2026-09-06: the
    # zero-error control (vwap_dist, no binning, no argmax) fails it in exactly
    # the same pattern (0.609 / 0.566 / 0.257), so it fires on a quantity that
    # cannot BE wrong.  Different windows measure different locations; that is
    # what they are for.  The two checks below can actually detect a broken
    # definition, and neither fires on an error-free quantity.

    # A1: every lookback is a suffix of the same series, so they must all END
    # on the same bar.  An off-by-one in any one window's right edge shows here.
    ends = [c for c in st.columns if c.startswith("last_ms_")]
    if len(ends) > 1:
        sub = st[ends].dropna()
        if len(sub) and not (sub.nunique(axis=1) == 1).all():
            f.append("lookback windows do not end on the same bar")
    # and the right edge must be the last bar closed before t_sweep
    for c in ends:
        m = st[[c, "t_sweep"]].dropna()
        if len(m) and not (m[c] + 60_000 <= m["t_sweep"]).all():
            f.append(f"{c} is not strictly before t_sweep")

    # A2: the correlation table is REPORTED, never gated -- see above.
    return f


def distribution_report(sym, st):
    """The number the plan wants BEFORE any label is looked at."""
    out = {}
    for side, g in st.groupby("side"):
        q = g[f"poc_dist_{LB_MAIN}"].quantile([.05, .25, .5, .75, .95])
        out[side] = dict(n=int(len(g)),
                         p05=float(q.iloc[0]), q25=float(q.iloc[1]),
                         med=float(q.iloc[2]), q75=float(q.iloc[3]),
                         p95=float(q.iloc[4]),
                         iqr=float(q.iloc[3] - q.iloc[1]),
                         frac_positive=float((g[f"poc_dist_{LB_MAIN}"] > 0).mean()),
                         outside_iqr=float(((g[f"poc_dist_{LB_MAIN}"] < q.iloc[1])
                                            | (g[f"poc_dist_{LB_MAIN}"] > q.iloc[3])).mean()))
    return out


def correlation_table(st):
    """Reported, not gated.  See run_asserts for why."""
    from scipy.stats import spearmanr
    out = {}
    for x, y in (("L1", "L2"), ("L2", "L3"), ("L1", "L3")):
        for pref in ("poc_dist", "vwap_dist"):
            m = st[[f"{pref}_{x}", f"{pref}_{y}"]].dropna()
            out[f"{pref}_{x}_{y}"] = (float(spearmanr(m.iloc[:, 0], m.iloc[:, 1]).correlation)
                                      if len(m) > 10 else None)
    return out


def disagreement_check(sym, st, n=300, seed=20260906):
    """The plan's allocation-disagreement number, on this coin's own events.

    Needs no tick data, so it runs for all nine.  BTC's tick-truth run
    calibrates what it means (disagreement median 0.15 <-> true error 0.05).
    """
    b = pd.read_parquet(BARS / f"{sym}.parquet")
    bars = Bars(b.ts, b.open, b.high, b.low, b.close, b.volume)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(st), size=min(n, len(st)), replace=False)
    d = []
    for i in idx:
        r = st.iloc[int(i)]
        pu = build_profile(bars, int(r.t_sweep), LOOKBACKS[LB_MAIN], "uniform",
                           float(r.bin_size))
        pc = build_profile(bars, int(r.t_sweep), LOOKBACKS[LB_MAIN], "close",
                           float(r.bin_size))
        if pu is None or pc is None:
            continue
        d.append(abs(pu.poc - pc.poc) / float(r.atr_h14))
    d = np.array(d)
    return dict(n=int(len(d)), median=float(np.median(d)),
                q90=float(np.percentile(d, 90)))


def tick_truth_gate(sym, st):
    """Only BTC has spot prints on disk, so only BTC can be measured against
    truth.  The other eight are UNMEASURED -- said, not hidden."""
    p = QUALITY / f"stage1_tick_truth_{sym}.json"
    if not p.exists():
        return None
    tt = json.loads(p.read_text())
    out, ok = {}, True
    for name in LOOKBACKS:
        if name not in tt:
            continue
        err = tt[name]["err_uniform"]["median"]
        iqr = float(st[f"poc_dist_{name}"].quantile(.75)
                    - st[f"poc_dist_{name}"].quantile(.25))
        lim = ERR_IQR_FRAC * iqr
        out[name] = dict(err_median=err, iqr=iqr, limit=lim, ok=bool(err < lim))
        ok &= out[name]["ok"]
    out["pass"] = ok
    return out


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
        st = build(s)
        st.to_parquet(OUT / f"{s}.parquet", index=False)
        fails = run_asserts(st, s)
        dist = distribution_report(s, st)
        rho = correlation_table(st)
        dis = disagreement_check(s, st)
        if dis["median"] > DISAGREE_MAX:
            fails = fails + [f"uniform-vs-close disagreement median "
                             f"{dis['median']:.3f} > {DISAGREE_MAX}"]
        tk = tick_truth_gate(s, st)
        if tk and not tk["pass"]:
            fails = fails + [f"tick-truth error too large vs IQR: {tk}"]
        allok &= not fails
        summary[s] = dict(n=int(len(st)), asserts=fails, poc_dist_L2=dist,
                          spearman_reported_not_gated=rho,
                          l1_voided=int(st.l1_voided.sum()),
                          alloc_disagreement=dis, tick_truth=tk)
        d = dist.get("sellside", {})
        e = dist.get("buyside", {})
        print(f"{s:5s} n={len(st):5,}  poc_dist_L2 IQR sell={d.get('iqr',float('nan')):.3f} "
              f"buy={e.get('iqr',float('nan')):.3f}  frac>0 "
              f"{d.get('frac_positive',float('nan')):.3f}/{e.get('frac_positive',float('nan')):.3f}"
              f"  asserts={'ok' if not fails else fails}")
    (QUALITY / "stage4_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    print("\nStage 4 gate:", "ALL PASS" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
