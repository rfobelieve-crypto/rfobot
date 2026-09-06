# -*- coding: utf-8 -*-
"""TODO 1.00 — 流動性獵取 x 部位分布：事件表建構（成交量分布 + 狀態變數 + 標籤）。

**預註冊在 341a523（本體）與 7e1bb79（修訂 A/B），先 commit 再跑。**

這一支只負責「把每一筆掃單事件變成一列」，不做任何判定。判定在
poc_analyze.py，判準原文在 TODO 1.00，這裡不重述、也不得放寬。

凍結的定義（改任何一條 = 換一個母體，必須重註冊）
--------------------------------------------------
事件      sweep_core.detect_sweeps，PIVOT=10，1h，core9。一個字都不改。
t_sweep   1h 事件 bar 之內、第一根穿越 level 的 5m bar 的**收盤時刻**。
          基準價 = 那根的收盤價。兩端都是 bar 收盤（修訂 A）。
分布      只吃 close_time <= t_sweep - 5m 的 5m bar（穿越 bar 本身排除）。
          bin = max(tick, ATR_1h/20)；每根 bar 的量在它觸及的 bin 之間
          均勻攤開；替代法（敏感度）全部給 close 所在 bin。
lookback  L1 固定量 / L2 24h（主判定）/ L3 72h。
標籤      r_norm_tau = side_sign_cont x (close(t_sweep+tau) - base) / ATR_1h
          side_sign_cont = -1 (sellside) / +1 (buyside) -> **延續為正**。

前視守衛（會 assert，不靠人記得）
  · 分布的每一根 bar 的 close_time 必須 < t_sweep
  · trend/ER/RV 只用 t_sweep 之前已收的 bar
  · pierce 深度用 **5m 穿越 bar** 的極值，不是 1h bar 的（後者要等整點才
    知道，而 t_sweep 可能落在該小時的第 5 分鐘 — mistake.md 2026-09-03
    的同一個坑）

唯讀研究碼：不 import 生產模組、不寫 DB。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc  # noqa: E402

CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
CACHE = HERE / ".cache"
OUT = HERE.parent / "results" / "poc_profile"
TAUS = [900, 1800, 3600, 7200, 14400]          # 15m 30m 1h(主) 2h 4h
LOOKBACKS = ["L1", "L2", "L3"]
HORIZON_4H = 14400
M5 = 300


# ---------------------------------------------------------------- tick sizes
def tick_sizes(syms):
    p = CACHE / "ticks.json"
    if p.exists():
        d = json.loads(p.read_text())
        if all(s in d for s in syms):
            return d
    url = "https://api.binance.com/api/v3/exchangeInfo"
    req = urllib.request.Request(url, headers={"User-Agent": "poc-study/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.loads(r.read().decode())
    d = {}
    for s in info["symbols"]:
        base = s["symbol"]
        if not base.endswith("USDT"):
            continue
        for f in s["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                d[base[:-4]] = float(f["tickSize"])
    p.write_text(json.dumps({k: d[k] for k in syms if k in d}, indent=1))
    return d


# ------------------------------------------------------------------ 5m frame
class M5Frame:
    """5m bars, close-time indexed.  close_time = open_time + 300."""

    def __init__(self, path):
        rows = sc.load_csv(str(path))
        rows.sort(key=lambda r: r[0])
        self.ot = [r[0] for r in rows]
        self.ct = [r[0] + M5 for r in rows]
        self.o = [r[1] for r in rows]
        self.h = [r[2] for r in rows]
        self.l = [r[3] for r in rows]
        self.c = [r[4] for r in rows]
        self.v = [r[5] for r in rows]
        self._ot_ix = {t: i for i, t in enumerate(self.ot)}
        self._ct_ix = {t: i for i, t in enumerate(self.ct)}

    def __len__(self):
        return len(self.ot)

    def by_open(self, t):
        return self._ot_ix.get(t)

    def by_close(self, t):
        return self._ct_ix.get(t)

    def idx_at_or_before_close(self, t):
        """Last bar whose close_time <= t; -1 if none."""
        return bisect.bisect_right(self.ct, t) - 1


# -------------------------------------------------------------------- profile
def build_profile(fr, hi_idx, lo_ts, bin_size, alt_close_only=False):
    """Volume histogram over bars [.., hi_idx] with close_time >= lo_ts.

    hi_idx is INCLUSIVE and must already exclude the pierce bar.
    Returns (bins, total, vwap, n_bars) or None if empty.
    """
    if hi_idx < 0:
        return None
    lo_idx = bisect.bisect_left(fr.ct, lo_ts)
    if lo_idx > hi_idx:
        return None
    bins = {}
    tot = 0.0
    pv = 0.0
    for i in range(lo_idx, hi_idx + 1):
        vol = fr.v[i]
        if vol <= 0:
            continue
        tot += vol
        pv += vol * (fr.h[i] + fr.l[i] + fr.c[i]) / 3.0
        if alt_close_only:
            cb = int(fr.c[i] // bin_size)
            bins[cb] = bins.get(cb, 0.0) + vol
            continue
        b0 = int(fr.l[i] // bin_size)
        b1 = int(fr.h[i] // bin_size)
        share = vol / (b1 - b0 + 1)
        for b in range(b0, b1 + 1):
            bins[b] = bins.get(b, 0.0) + share
    if not bins or tot <= 0:
        return None
    return bins, tot, pv / tot, hi_idx - lo_idx + 1


def profile_nodes(bins, bin_size, side_sign, lvl):
    """POC / HVN / next_HVN / concentration / depth-to-level."""
    vals = sorted(bins.values())
    poc_bin = max(bins, key=lambda b: bins[b])
    poc_px = (poc_bin + 0.5) * bin_size
    tot = sum(bins.values())
    q80 = vals[min(len(vals) - 1, int(0.8 * (len(vals) - 1)))]
    hvn = [b for b, v in bins.items() if v >= q80]
    # next_HVN: sellside(side_sign=+1) looks BELOW the POC, buyside ABOVE.
    if side_sign > 0:
        cand = [b for b in hvn if b < poc_bin]
        nxt = max(cand) if cand else None
    else:
        cand = [b for b in hvn if b > poc_bin]
        nxt = min(cand) if cand else None
    lvl_bin = int(lvl // bin_size)
    a, b = (lvl_bin, poc_bin) if lvl_bin <= poc_bin else (poc_bin, lvl_bin)
    between = sum(v for k, v in bins.items() if a < k < b)
    return dict(
        poc_px=poc_px,
        poc_conc=bins[poc_bin] / tot,
        poc_depth=between / tot,
        next_hvn_px=None if nxt is None else (nxt + 0.5) * bin_size,
        n_bins=len(bins),
    )


# --------------------------------------------------------------------- events
def daily_vol(fr, hi_idx, t_sweep, days=30):
    lo = bisect.bisect_left(fr.ct, t_sweep - days * 86400)
    if lo > hi_idx:
        return None
    s = sum(fr.v[lo:hi_idx + 1])
    return s / days if s > 0 else None


def l1_window_start(fr, hi_idx, target_vol):
    """Walk back until cumulative volume >= target.  Returns lo_ts or None."""
    cum = 0.0
    i = hi_idx
    while i >= 0:
        cum += fr.v[i]
        if cum >= target_vol:
            return fr.ct[i]
        i -= 1
    return None


def build_symbol(sym, ticks, alt=False, verbose=True):
    b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    atr = sc.atr14(b1)
    fr = M5Frame(CACHE / "m5" / (sym + "_5m.csv"))
    if len(fr) == 0:
        raise SystemExit(sym + ": no 5m data")
    tick = ticks.get(sym, 0.0)
    c1 = [x[sc.C] for x in b1]

    rows = []
    skip = dict(no_atr=0, no_5m=0, no_pierce=0, no_profile=0)
    for e in sc.detect_sweeps(b1):
        j, lvl, kind = e["j"], e["level"], e["kind"]
        A = atr[j]
        if A is None or A <= 0 or j < 25:
            skip["no_atr"] += 1
            continue
        hour_open = b1[j][0]
        i0 = fr.by_open(hour_open)
        if i0 is None:
            skip["no_5m"] += 1
            continue
        pierce = None
        for k in range(i0, min(i0 + 12, len(fr))):
            if fr.ot[k] >= hour_open + 3600:
                break
            if kind == "buy" and fr.h[k] > lvl:
                pierce = k
                break
            if kind == "sell" and fr.l[k] < lvl:
                pierce = k
                break
        if pierce is None:
            skip["no_pierce"] += 1
            continue

        t_sweep = fr.ct[pierce]
        base = fr.c[pierce]
        side = "sellside" if kind == "sell" else "buyside"
        side_sign = 1.0 if kind == "sell" else -1.0        # POC 在反方向 = +
        cont = -1.0 if kind == "sell" else 1.0             # 延續 = +
        pierce_atr = ((fr.h[pierce] - lvl) if kind == "buy"
                      else (lvl - fr.l[pierce])) / A

        hi_idx = pierce - 1                                # strictly before t_sweep
        if hi_idx < 0:
            skip["no_profile"] += 1
            continue
        assert fr.ct[hi_idx] < t_sweep, "look-ahead: profile bar closes at/after t_sweep"

        bin_size = max(tick, A / 20.0)
        dv = daily_vol(fr, hi_idx, t_sweep)
        wins = {"L2": t_sweep - 86400, "L3": t_sweep - 3 * 86400,
                "L1": l1_window_start(fr, hi_idx, 0.5 * dv) if dv else None}

        rec = dict(sym=sym, side=side, t_sweep=t_sweep, j=j, lvl=lvl, atr=A,
                   base=base, pierce_atr=pierce_atr, bin_size=bin_size,
                   utc_hour=time.gmtime(t_sweep).tm_hour,
                   day=time.strftime("%Y-%m-%d", time.gmtime(t_sweep)))

        ok_any = False
        for name, lo_ts in wins.items():
            if lo_ts is None:
                continue
            pr = build_profile(fr, hi_idx, lo_ts, bin_size, alt_close_only=alt)
            if pr is None:
                continue
            bins, tot, vwap, nb = pr
            nd = profile_nodes(bins, bin_size, side_sign, lvl)
            rec["poc_dist_" + name] = side_sign * (nd["poc_px"] - lvl) / A
            rec["vwap_dist_" + name] = side_sign * (vwap - lvl) / A
            rec["poc_conc_" + name] = nd["poc_conc"]
            rec["poc_depth_" + name] = nd["poc_depth"]
            rec["poc_px_" + name] = nd["poc_px"]
            rec["next_hvn_" + name] = nd["next_hvn_px"]
            rec["nbars_" + name] = nb
            ok_any = True
        if not ok_any:
            skip["no_profile"] += 1
            continue

        # ---- controls (all strictly before t_sweep) ----
        rec["trend_24h"] = (c1[j - 1] - c1[j - 24]) / A
        path = sum(abs(c1[k] - c1[k - 1]) for k in range(j - 23, j))
        rec["er_24h"] = abs(c1[j - 1] - c1[j - 24]) / path if path > 0 else 0.0
        rets = [(c1[k] - c1[k - 1]) / c1[k - 1] for k in range(j - 23, j)]
        mu = sum(rets) / len(rets)
        rec["rv_24h"] = math.sqrt(sum((x - mu) ** 2 for x in rets) / (len(rets) - 1))
        lo12 = max(1, pierce - 12)
        r5 = [(fr.c[k] - fr.c[k - 1]) / fr.c[k - 1] for k in range(lo12, pierce)]
        if len(r5) > 2:
            m5m = sum(r5) / len(r5)
            rec["rv_1h"] = math.sqrt(
                sum((x - m5m) ** 2 for x in r5) / (len(r5) - 1)) * math.sqrt(12)
        else:
            rec["rv_1h"] = ""

        # ---- labels ----
        for tau in TAUS:
            k = fr.by_close(t_sweep + tau)
            rec["r_" + str(tau)] = "" if k is None else cont * (fr.c[k] - base) / A
        k0 = pierce + 1
        k1 = fr.idx_at_or_before_close(t_sweep + HORIZON_4H)
        if k1 >= k0:
            seg = range(k0, k1 + 1)
            if kind == "sell":
                ti = min(seg, key=lambda z: fr.l[z])
                term = fr.l[ti]
                adverse = max(fr.h[z] for z in seg)
            else:
                ti = max(seg, key=lambda z: fr.h[z])
                term = fr.h[ti]
                adverse = min(fr.l[z] for z in seg)
            rec["terminal_px"] = term
            rec["mfe"] = cont * (term - base) / A
            rec["mae"] = cont * (adverse - base) / A
            rec["t_extreme"] = fr.ct[ti] - t_sweep
        else:
            rec["terminal_px"] = rec["mfe"] = rec["mae"] = rec["t_extreme"] = ""
        rows.append(rec)

    if verbose:
        print("%-5s events=%5d  skips=%s" % (sym, len(rows), skip), flush=True)
    return rows


COLS = (["sym", "side", "t_sweep", "day", "utc_hour", "j", "lvl", "atr", "base",
         "pierce_atr", "bin_size", "trend_24h", "er_24h", "rv_24h", "rv_1h"]
        + [p + "_" + lb for lb in LOOKBACKS
           for p in ("poc_dist", "vwap_dist", "poc_conc", "poc_depth",
                     "poc_px", "next_hvn", "nbars")]
        + ["r_" + str(t) for t in TAUS]
        + ["terminal_px", "mfe", "mae", "t_extreme"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    ap.add_argument("--alt", action="store_true",
                    help="sensitivity: all of a bar volume into its close bin")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.syms.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    ticks = tick_sizes(syms)
    allrows = []
    for s in syms:
        allrows += build_symbol(s, ticks, alt=a.alt)
    name = a.out or ("events_alt.csv" if a.alt else "events.csv")
    p = OUT / name
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in allrows:
            w.writerow(r)
    print("\n%d events -> %s" % (len(allrows), p))


if __name__ == "__main__":
    main()
