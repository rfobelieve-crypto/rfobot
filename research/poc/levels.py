# -*- coding: utf-8 -*-
"""Stage 2 — liquidity levels, made point-in-time.

    data/levels/{COIN}.parquet
        level_id, side ('buyside' | 'sellside'), price,
        formed_at, confirmed_at, invalidated_at        (int64 ms, NaN = alive)
        pivot_idx, hour_ts

buyside  = liquidity ABOVE price (stops of shorts, breakout buys) = swing high
sellside = liquidity BELOW price (stops of longs, breakdown sells) = swing low

The job here is NOT to invent a level definition.  It is to take the frozen
one (sweep_core: PIVOT=10 swings on 1-HOUR bars) and express it as a table
that answers "which levels existed at time t" without ever consulting a bar
later than t.

Why the pivots stay on 1h
    Every number this research line owns -- Gate F, variant B, the ICT
    verdicts -- comes from 1h pivots.  Stage 3's regression gate demands >85%
    agreement with that event table, which is only meaningful if the level
    definition is unchanged.  The 1-minute bars are used for the things 1h
    cannot give: the minute the level is actually breached, and the volume
    profile.

Timing, and why each field is knowable when it is stamped
    formed_at       close of the pivot bar.  The price (that bar's high/low)
                    is known then -- but that it IS a pivot is not.
    confirmed_at    close of the bar PIVOT bars later.  The frozen engine
                    starts scanning strictly after this, so this is the first
                    moment the level can be acted on.
    invalidated_at  close of the first 1-MINUTE bar after confirmed_at whose
                    high exceeds (buyside) / low undercuts (sellside) the
                    price.  Plain touch, no tick buffer -- Stage 3 applies the
                    registered k=2 tick threshold on top and keeps its own
                    t_sweep, so the two are never confused.

Two independent implementations
    `pivots_vectorised` is what builds the table.  `pivots_reference` is a
    literal scalar transcription of sweep_core's condition.  The replay gate
    runs the reference over prefixes of history and demands set equality.
    Agreement between two implementations tests the vectorisation AND the
    causality claim; re-running the same function would test neither.
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
OUT = HERE / "data" / "levels"
QUALITY = HERE / "data" / "quality"
CACHE_1H = HERE.parent / "sweep_failure" / ".cache"
PIVOT = 10                      # frozen, sweep_core.PIVOT
HOUR_MS = 3_600_000
MIN_MS = 60_000
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


# ------------------------------------------------------------------ 1h bars
def to_hourly(df):
    """Resample the 1-minute table to UTC-aligned 1h bars.

    Built from the same source as everything else so there is one truth; the
    result is cross-checked against the frozen engine's own 1h cache below.
    """
    idx = pd.to_datetime(df["ts"], unit="ms", utc=True)
    g = df.set_index(idx).resample("1h")
    h = g.agg(open=("open", "first"), high=("high", "max"),
              low=("low", "min"), close=("close", "last"),
              volume=("volume", "sum"), n=("close", "count"))
    h = h[h["n"] > 0].copy()
    h["hour_ts"] = (h.index.view("int64") // 1_000_000).astype(np.int64)
    # Drop a trailing hour that has not finished.  A partial bar must never
    # define a pivot: its high/low are provisional, and the frozen 1h cache
    # (refreshed on its own schedule) holds a different partial for the same
    # hour.  Verified 2026-09-06: this was the ONLY disagreeing bar out of
    # 22,420 for BTC -- everything else matched exactly.
    last_min = int(df["ts"].iloc[-1])
    h = h[h["hour_ts"] + HOUR_MS <= last_min + MIN_MS]
    return h.reset_index(drop=True)


def crosscheck_hourly(sym, h):
    """Our resampled 1h vs the frozen engine's cached 1h — same instrument?"""
    p = CACHE_1H / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c = c.rename(columns={"time": "sec"})
    c["hour_ts"] = (c["sec"].astype(np.int64) * 1000)
    m = h.merge(c, on="hour_ts", suffixes=("_ours", "_theirs"))
    if m.empty:
        return dict(n=0)
    out = {}
    for col in ("high", "low", "close"):
        d = (m[col + "_ours"] - m[col + "_theirs"]).abs() / m[col + "_theirs"]
        out[col] = float(d.max())
    out["n"] = int(len(m))
    return out


# ------------------------------------------------------------------- pivots
def pivots_vectorised(high, low, pivot=PIVOT):
    """Indices of swing highs / lows under sweep_core's exact condition.

        all(h[i] >= h[k] for k in window)  <=>  h[i] == rolling max
        any(h[i] >  h[k] for k != i)       <=>  h[i] >  rolling min
    """
    n = len(high)
    w = 2 * pivot + 1
    sh = pd.Series(high)
    sl = pd.Series(low)
    hmax = sh.rolling(w, center=True).max().to_numpy()
    hmin = sh.rolling(w, center=True).min().to_numpy()
    lmin = sl.rolling(w, center=True).min().to_numpy()
    lmax = sl.rolling(w, center=True).max().to_numpy()
    idx = np.arange(n)
    inner = (idx >= pivot) & (idx < n - pivot)
    ph = inner & (high == hmax) & (high > hmin)
    pl = inner & (low == lmin) & (low < lmax)
    return np.flatnonzero(ph), np.flatnonzero(pl)


def pivots_reference(high, low, pivot=PIVOT):
    """Literal scalar transcription of sweep_core.detect_sweeps' pivot test."""
    n = len(high)
    ph, pl = [], []
    for i in range(pivot, n - pivot):
        seg = range(i - pivot, i + pivot + 1)
        if all(high[i] >= high[k] for k in seg) and \
           any(high[i] > high[k] for k in seg if k != i):
            ph.append(i)
        if all(low[i] <= low[k] for k in seg) and \
           any(low[i] < low[k] for k in seg if k != i):
            pl.append(i)
    return np.array(ph, dtype=np.int64), np.array(pl, dtype=np.int64)


# ------------------------------------------------------------------- build
def first_breach_minute(m_ts, m_high, m_low, after_ms, price, side):
    """Close time of the first 1-minute bar strictly after `after_ms` that
    touches through `price`.  Returns None if never."""
    i = int(np.searchsorted(m_ts, after_ms, side="left"))
    if side == "buyside":
        j = np.flatnonzero(m_high[i:] > price)
    else:
        j = np.flatnonzero(m_low[i:] < price)
    if len(j) == 0:
        return None
    k = i + int(j[0])
    return int(m_ts[k] + MIN_MS)


def build(sym):
    df = pd.read_parquet(BARS / f"{sym}.parquet",
                         columns=["ts", "open", "high", "low", "close", "volume"])
    h = to_hourly(df)
    xc = crosscheck_hourly(sym, h)

    hh = h["high"].to_numpy(float)
    hl = h["low"].to_numpy(float)
    hts = h["hour_ts"].to_numpy(np.int64)
    ph, pl = pivots_vectorised(hh, hl)

    m_ts = df["ts"].to_numpy(np.int64)
    m_high = np.nan_to_num(df["high"].to_numpy(float), nan=-np.inf)
    m_low = np.nan_to_num(df["low"].to_numpy(float), nan=np.inf)

    rows = []
    for side, idxs, px in (("buyside", ph, hh), ("sellside", pl, hl)):
        for i in idxs:
            formed = int(hts[i] + HOUR_MS)              # close of the pivot bar
            conf = int(hts[i + PIVOT] + HOUR_MS)        # close of bar i+PIVOT
            price = float(px[i])
            inval = first_breach_minute(m_ts, m_high, m_low, conf, price, side)
            rows.append((side, price, formed, conf, inval, int(i), int(hts[i])))
    lv = pd.DataFrame(rows, columns=["side", "price", "formed_at", "confirmed_at",
                                     "invalidated_at", "pivot_idx", "hour_ts"])
    lv = lv.sort_values(["confirmed_at", "side", "price"]).reset_index(drop=True)
    lv.insert(0, "level_id", [f"{sym}-{i:06d}" for i in range(len(lv))])
    lv["invalidated_at"] = lv["invalidated_at"].astype("Int64")
    OUT.mkdir(parents=True, exist_ok=True)
    lv.to_parquet(OUT / f"{sym}.parquet", index=False)
    return lv, h, xc


# -------------------------------------------------------------------- gates
def run_asserts(lv):
    f = []
    if not (lv["confirmed_at"] >= lv["formed_at"]).all():
        f.append("confirmed_at < formed_at")
    alive = lv["invalidated_at"].notna()
    if not (lv.loc[alive, "invalidated_at"] > lv.loc[alive, "confirmed_at"]).all():
        f.append("invalidated_at <= confirmed_at")
    if (lv["confirmed_at"] - lv["formed_at"] != PIVOT * HOUR_MS).any():
        f.append("confirmation lag is not exactly PIVOT hours")
    if lv["level_id"].duplicated().any():
        f.append("duplicate level_id")
    return f


def replay_gate(lv, h, n_trials=50, seed=20260906):
    """The gate.  For random cut-offs t, re-derive the live level set with the
    INDEPENDENT scalar implementation over bars[:t] only, and demand equality.

    Deviation from the plan, stated: the plan's expected set is
    `formed_at <= t`, but a system fed bars[:t] cannot return a pivot that is
    not yet confirmed -- it needs PIVOT bars after the extreme.  The
    comparison therefore uses `confirmed_at <= t`, which is what
    "point-in-time knowable" means here.  Using formed_at would fail by
    construction for every level younger than PIVOT hours.
    """
    hh = h["high"].to_numpy(float)
    hl = h["low"].to_numpy(float)
    hts = h["hour_ts"].to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    lo, hi = 3 * PIVOT, len(hts) - 1
    picks = rng.choice(np.arange(lo, hi), size=n_trials, replace=False)
    bad = []
    for m in sorted(picks):
        t = int(hts[m] + HOUR_MS)                       # cut-off = close of bar m
        rh, rl = pivots_reference(hh[:m + 1], hl[:m + 1])
        expect = set()
        for side, idxs, px in (("buyside", rh, hh), ("sellside", rl, hl)):
            for i in idxs:
                if int(hts[i + PIVOT] + HOUR_MS) > t:   # not confirmed yet
                    continue
                expect.add((side, round(float(px[i]), 10), int(i)))
        sub = lv[(lv["confirmed_at"] <= t)
                 & (lv["invalidated_at"].isna() | (lv["invalidated_at"] > t))]
        got = {(r.side, round(float(r.price), 10), int(r.pivot_idx))
               for r in sub.itertuples()}
        # the replay knows nothing about invalidation, so compare only the
        # pivot set it can produce; alive-ness is checked separately below
        expect_alive = set()
        for side, price, i in expect:
            arr_h = hh if side == "buyside" else hl
            j0 = i + PIVOT + 1
            if side == "buyside":
                hit = np.flatnonzero(hh[j0:m + 1] > price)
            else:
                hit = np.flatnonzero(hl[j0:m + 1] < price)
            if len(hit) == 0:
                expect_alive.add((side, price, i))
        if expect_alive != got:
            bad.append((t, len(expect_alive), len(got),
                        len(expect_alive - got), len(got - expect_alive)))
    return picks, bad


def report(sym, lv, h, xc, fails, n_trials, bad):
    ts = pd.to_datetime(lv["confirmed_at"], unit="ms", utc=True)
    per_month = lv.assign(m=ts.dt.to_period("M")).groupby("m").size()
    life = (lv["invalidated_at"] - lv["confirmed_at"]).dropna() / HOUR_MS
    alive = int(lv["invalidated_at"].isna().sum())
    L = [f"# Stage 2 levels — {sym}", ""]
    L.append(f"- levels: **{len(lv):,}**  (buyside {int((lv.side=='buyside').sum()):,} / "
             f"sellside {int((lv.side=='sellside').sum()):,})")
    L.append(f"- never breached (alive at the end): **{alive:,}** "
             f"({alive/len(lv)*100:.2f}%)")
    L.append(f"- lifetime hours (confirmed -> breached): median **{life.median():.1f}**, "
             f"q25 {life.quantile(.25):.1f}, q75 {life.quantile(.75):.1f}, "
             f"max {life.max():.0f}")
    if xc:
        L.append(f"- resampled 1h vs frozen 1h cache: n={xc['n']:,}, max rel diff "
                 f"high {xc['high']:.2e} / low {xc['low']:.2e} / close {xc['close']:.2e}")
    L.append("")
    L.append("## gate")
    L.append(f"- asserts: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    L.append(f"- replay (independent scalar implementation over prefixes): "
             f"**{n_trials - len(bad)}/{n_trials}** equal "
             f"-> {'PASS' if not bad else 'FAIL'}")
    if bad:
        L.append("")
        L.append("| cut-off ms | expected | got | missing | extra |")
        L.append("|---|---|---|---|---|")
        for r in bad[:10]:
            L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")
    L.append("## levels confirmed per month")
    L.append("")
    L.append("| month | n |")
    L.append("|---|---|")
    for m, v in per_month.items():
        L.append(f"| {m} | {v} |")
    QUALITY.mkdir(parents=True, exist_ok=True)
    (QUALITY / f"{sym}_levels.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return per_month


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    allok = True
    summary = {}
    for s in [x.strip().upper() for x in a.syms.split(",") if x.strip()]:
        if not (BARS / f"{s}.parquet").exists():
            print(f"{s:5s} no bars, skipped")
            continue
        lv, h, xc = build(s)
        fails = run_asserts(lv)
        picks, bad = replay_gate(lv, h, n_trials=a.trials)
        report(s, lv, h, xc, fails, a.trials, bad)
        xok = (xc is None) or (max(xc["high"], xc["low"], xc["close"]) < 1e-9)
        if not xok:
            fails = fails + [f"resampled 1h disagrees with the frozen cache "
                             f"(max rel {max(xc['high'], xc['low'], xc['close']):.2e})"]
        ok = not fails and not bad
        allok &= ok
        xs = f" 1h-xcheck_max={max(xc['high'], xc['low'], xc['close']):.1e}" if xc else ""
        print(f"{s:5s} levels={len(lv):6,} alive={int(lv.invalidated_at.isna().sum()):5,}"
              f" replay={a.trials-len(bad)}/{a.trials}{xs}"
              f"  asserts={'ok' if not fails else fails}  -> {'PASS' if ok else 'FAIL'}")
        summary[s] = dict(levels=int(len(lv)), replay_ok=int(a.trials - len(bad)),
                          trials=int(a.trials), asserts=fails,
                          hourly_xcheck=xc)
    (QUALITY / "stage2_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    print("\nStage 2 gate:", "ALL PASS" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
