# -*- coding: utf-8 -*-
"""Implements PREREG_passive_markout.md (frozen 2026-09-05) verbatim.

Every threshold, rule and horizon here is copied from that file; none may be
tuned in response to what this prints.

Run: python research/subhourly/passive_markout_test.py
Out: research/results/passive_markout_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "research" / "results" / "passive_markout_test.json"
HS = (1, 5, 15, 60)
TS = (60, 15)
Q_WIN, Q = 10_080, 0.95
TAKER, MAKERS = 5.0, (2.0, 0.0, -1.0)


def load():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ts_ms, mid_price, bid_l1_price, ask_l1_price FROM orderbook_snapshots_1m "
                        "WHERE canonical_symbol='BTC-USD' AND exchange='binance' ORDER BY ts_ms")
            d = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    d["m"] = (d["ts_ms"] // 60000) * 60000
    d = d.sort_values("ts_ms").groupby("m", as_index=False).last()
    for c in ("mid_price", "bid_l1_price", "ask_l1_price"):
        d[c] = d[c].astype(float)
    # regular minute grid; gaps become NaN (no fill / no markout possible there)
    idx = pd.RangeIndex(int(d["m"].min()), int(d["m"].max()) + 60000, 60000)
    d = d.set_index("m").reindex(idx)
    return d


def first_fill(p, fav, T):
    """p: order price per event index; fav: the book series that must reach p
    (buy: bid ≤ p ; sell: ask ≥ p, passed already-signed). Returns first k in
    1..T where fav[t+k] crosses, else -1. Vectorised over offsets."""
    n = len(p); k_fill = np.full(n, -1, np.int64)
    for k in range(1, T + 1):
        f = np.full(n, False)
        f[:-k] = fav[k:] & (k_fill[:-k] < 0)
        k_fill[f] = k
    return k_fill


def run_side(d, s, mask, T):
    mid = d["mid_price"].values; bid = d["bid_l1_price"].values; ask = d["ask_l1_price"].values
    n = len(d); idx = np.arange(n)
    p = np.where(s > 0, bid, ask)
    # crossing series evaluated at t+k against p[t]: build via offsets in first_fill
    # fav_k[t] = (bid[t+k] <= p[t]) for buy ; (ask[t+k] >= p[t]) for sell  -- need per-k; do inline
    k_fill = np.full(n, -1, np.int64); k_fill_strict = np.full(n, -1, np.int64)
    for k in range(1, T + 1):
        if s > 0:
            c = bid[k:] <= p[:-k]; cs = ask[k:] <= p[:-k]
        else:
            c = ask[k:] >= p[:-k]; cs = bid[k:] >= p[:-k]
        c = np.nan_to_num(c, nan=False).astype(bool); cs = np.nan_to_num(cs, nan=False).astype(bool)
        f = np.zeros(n, bool); f[:-k] = c & (k_fill[:-k] < 0); k_fill[f] = k
        fs = np.zeros(n, bool); fs[:-k] = cs & (k_fill_strict[:-k] < 0); k_fill_strict[fs] = k
    ev = idx[mask & np.isfinite(p) & (idx < n - T - max(HS) - 1)]
    tf = ev + k_fill[ev]; filled = k_fill[ev] > 0; filled_s = k_fill_strict[ev] > 0
    out = {"n": int(len(ev)), "fill": float(filled.mean()), "fill_strict": float(filled_s.mean())}
    rows = []
    for e, t_, ok in zip(ev, tf, filled):
        r = {"t": int(e), "day": int(d.index[e] // 86_400_000), "filled": bool(ok), "side": int(s)}
        take = ask[e] if s > 0 else bid[e]
        for h in HS:
            r[f"m{h}"] = s * (mid[e + h] - mid[e]) / mid[e] * 1e4                       # (3b) from posting mid
            r[f"take{h}"] = s * (mid[e + h] - take) / take * 1e4                        # active / (3a)
            if ok and t_ + h < n:
                r[f"mo{h}"] = s * (mid[t_ + h] - p[e]) / p[e] * 1e4                     # markout from fill
                r[f"half{h}"] = s * (mid[t_] - p[e]) / p[e] * 1e4
                r[f"drift{h}"] = s * (mid[t_ + h] - mid[t_]) / p[e] * 1e4
        rows.append(r)
    return out, pd.DataFrame(rows)


def dblock(v, days, B=2000, seed=3):
    rng = np.random.default_rng(seed); g = {}
    for x, dd in zip(v, days):
        if np.isfinite(x): g.setdefault(dd, []).append(x)
    ks = np.array(list(g))
    out = [np.concatenate([g[dd] for dd in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def report(tag, df, T):
    f = df[df.filled]; u = df[~df.filled]
    print(f"\n  [{tag}] T={T}m  事件 {len(df)}  成交率 {df.filled.mean()*100:.1f}%  "
          f"(成交 {len(f)} / 未成交 {len(u)})")
    print(f"    {'h':>4}{'markout|成交':>14}{'CI':>20}{'半價差':>8}{'漂移':>8}{'(3a)未成交機會成本':>18}{'(3b) m_h 成交/未成交':>22}")
    res = {}
    for h in HS:
        mo = f[f"mo{h}"].dropna(); lo, hi = dblock(mo.values, f.loc[mo.index, "day"].values)
        res[h] = {"markout": float(mo.mean()), "ci": [lo, hi], "half": float(f[f"half{h}"].mean()),
                  "drift": float(f[f"drift{h}"].mean()), "opp_unfilled": float(u[f"take{h}"].mean()) if len(u) else None,
                  "m_filled": float(f[f"m{h}"].mean()), "m_unfilled": float(u[f"m{h}"].mean()) if len(u) else None}
        r = res[h]
        print(f"    {h:>3}m{r['markout']:>+13.2f} [{lo:+7.2f},{hi:+7.2f}]{r['half']:>+8.2f}{r['drift']:>+8.2f}"
              f"{(r['opp_unfilled'] or 0):>+18.2f}{r['m_filled']:>+11.2f}/{(r['m_unfilled'] or 0):>+9.2f}")
    return res


def main():
    d = load()
    mid = d["mid_price"]
    ret60 = mid / mid.shift(60) - 1
    thr = ret60.abs().rolling(Q_WIN, min_periods=Q_WIN // 2).quantile(Q)
    sig = (ret60.abs() > thr) & thr.notna()
    side = -np.sign(ret60)
    # dedupe: first minute of a same-direction run within 60 min
    sig_idx = np.where(sig.values)[0]; keep = np.zeros(len(d), bool); last = {1: -10**9, -1: -10**9}
    for i in sig_idx:
        s = int(side.values[i])
        if i - last[s] > 60:
            keep[i] = True; last[s] = i
    print("=" * 100)
    print(f"  被動側 markout 測試 · Binance 現貨 BTC L1 1m · {pd.to_datetime(d.index[0],unit='ms').date()} → "
          f"{pd.to_datetime(d.index[-1],unit='ms').date()} · 分鐘 {len(d):,} · 訊號事件 {keep.sum():,}")
    print("=" * 100)
    out = {"n_minutes": int(len(d)), "n_signals": int(keep.sum()), "S": {}, "U": {}}
    for T in TS:
        parts = []
        for s in (1, -1):
            m = keep & (side.values == s)
            st, df = run_side(d, s, m, T); parts.append(df)
        S = pd.concat(parts, ignore_index=True)
        res = report("S 訊號組", S, T)
        # EV comparison at h=60 (only meaningful with T=60; report for both)
        active = S["take60"] - TAKER - TAKER
        ev = {"active": float(active.mean()), "active_ci": dblock(active.values, S["day"].values)}
        print(f"    主動 EV/訊號 (h=60, 進出各 5 taker): {ev['active']:+.2f} bps  CI [{ev['active_ci'][0]:+.2f},{ev['active_ci'][1]:+.2f}]")
        for mf in MAKERS:
            pas = np.where(S.filled, S["mo60"].fillna(0) - mf - TAKER, 0.0)
            diff = pas - active.values
            lo, hi = dblock(pas, S["day"].values); dlo, dhi = dblock(diff, S["day"].values)
            half = len(S) // 2; h1, h2 = diff[:half].mean(), diff[half:].mean()
            ev[f"passive_mf{mf:g}"] = {"ev": float(pas.mean()), "ci": [lo, hi], "diff": float(diff.mean()),
                                       "diff_ci": [dlo, dhi], "halves": [float(h1), float(h2)]}
            print(f"    被動 EV/訊號 maker={mf:+.0f}: {pas.mean():+.2f} [{lo:+.2f},{hi:+.2f}]  "
                  f"被動−主動 {diff.mean():+.2f} [{dlo:+.2f},{dhi:+.2f}]  兩半 {h1:+.2f}/{h2:+.2f}")
        out["S"][T] = {"stats": res, "ev": ev, "fill": float(S.filled.mean())}
        if T == 60:
            # U: every minute both sides (sampled every 3rd minute to bound runtime, both sides)
            um = np.zeros(len(d), bool); um[::3] = True
            parts = []
            for s in (1, -1):
                st, df = run_side(d, s, um, T); parts.append(df)
            U = pd.concat(parts, ignore_index=True)
            resU = report("U 無條件組（每 3 分鐘雙邊）", U, T)
            out["U"][T] = {"stats": resU, "fill": float(U.filled.mean())}
    # verdict on T=60, maker=2
    e = out["S"][60]["ev"]; p2 = e["passive_mf2"]; st = out["S"][60]["stats"]
    c1 = p2["diff"] > 0 and p2["diff_ci"][0] > 0
    c2 = p2["ev"] > 0 and p2["ci"][0] > 0
    c3 = st[15]["markout"] > 0 and st[15]["ci"][0] > 0 and st[60]["markout"] > 0 and st[60]["ci"][0] > 0
    c4 = np.sign(p2["halves"][0]) == np.sign(p2["halves"][1]) and p2["halves"][0] > 0
    u60 = out["U"][60]["stats"][60]; c5_toxic = u60["ci"][1] < 0
    verdict = "換邊 GO" if all((c1, c2, c3, c4)) else "不換邊"
    print(f"\n  判準 (1) 被動>主動 CI離零: {'過' if c1 else '不過'}  (2) 被動>0: {'過' if c2 else '不過'}  "
          f"(3) markout15&60>0: {'過' if c3 else '不過'}  (4) 兩半同號: {'過' if c4 else '不過'}  "
          f"(5) U 組有毒: {'是' if c5_toxic else '否'}")
    print(f"  ==> {verdict}")
    out["verdict"] = {"go": verdict, "c": [bool(x) for x in (c1, c2, c3, c4)], "venue_toxic": bool(c5_toxic)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
