# -*- coding: utf-8 -*-
"""§1.01 — is the two-venue premium a SESSION event? (exploration, pre-registered)

Why
---
Without Entropy's rebate the recording family needs an 18 bps band on four
taker crossings (6 bps with one resting leg) and the all-day p90 is 2-15.
A hour-of-day cut on 2026-09-04 showed the SNDK/NBIS band is 3-5x wider in
the two hours after the US cash open than at any other time, while ANTH
(private-company perp, no cash session) has no such shape. That fits the
mechanism the line was registered on ("盤後預言機制度不同" is the premium
source for stock perps): the two venues' pricing regimes collide at the
open. If true, the strategy is not "trade the pair" but "trade the window".

THIS IS EXPLORATION ON THE SELECTION WINDOW. Every row here was recorded
before the hypothesis existed, so nothing below is a verdict. It decides
only whether a session-window variant deserves a clock of its own. Rows
recorded from the registration timestamp forward are the validation set.

FROZEN WINDOWS (by mechanism, not by the best hour in the table)
  OPEN   13:30-15:30 UTC   first two hours of US regular trading
  RTH    15:30-20:00 UTC   rest of the cash session
  PRE    08:00-13:30 UTC   European hours / US pre-market
  OFF    20:00-08:00 UTC   cash market closed
  (No DST handling: this sample is entirely inside US daylight time.)

PRE-REGISTERED PREDICTIONS (written before the script ran)
  P1  OPEN p90 band (wider side) > 2x OFF p90 band for SNDK and NBIS
  P2  ANTH shows NO such ratio (< 1.5x) -- the control for "it's just
      more volatility everywhere at that hour"
  P3  OPEN-window episodes still converge >= 70% within 240 min (a wide
      band that is a persistent open-hour offset is worth nothing)
  P4  the OPEN advantage holds on >= 4 of the recorded days, not one

SURVIVAL: all four, else the window idea is dead here and does not get a
clock. Passing means one thing only: register variant W in TODO 1.00 with
window-specific p90 as its band and score it on rows from today forward.

Fees are the no-rebate schedule via fees.py (the question is whether the
line can live WITHOUT the promotion).

Run: python research/arb/session_window.py
Out: research/results/arb_session_window.json
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import premium_verdict as PV      # noqa: E402  frozen loader + convergence
import fees as FEES               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = HERE.parents[1]
OUT = ROOT / "research" / "results" / "arb_session_window.json"
WINDOWS = {"OPEN": (13.5, 15.5), "RTH": (15.5, 20.0), "PRE": (8.0, 13.5)}
PAIRS = [p for p in PV.PAIRS if p[0] in ("SNDK", "NBIS", "ANTH")]


def win_of(ts: int) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc)
    h = d.hour + d.minute / 60
    for k, (a, b) in WINDOWS.items():
        if a <= h < b:
            return k
    return "OFF"


def p90(xs):
    xs = sorted(xs)
    return xs[int(0.9 * len(xs))] if xs else float("nan")


def intraminute(csv_path: Path):
    """premium_high - premium_low per minute (what a faster sampler sees),
    read straight from the raw file because the frozen loader drops it."""
    out = {}
    for fp in sorted(glob.glob(str(csv_path) + "*.old")) + [str(csv_path)]:
        if not Path(fp).exists():
            continue
        with open(fp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    out[int(r["minute_ts"])] = (float(r["premium_high_bps"])
                                                - float(r["premium_low_bps"]))
                except (ValueError, KeyError):
                    continue
    return out


def main() -> int:
    print("=" * 96)
    print("  §1.01 時段條件化——溢價是不是美股開盤的事件？（探索，選擇窗；判準見檔頭）")
    print("=" * 96)
    res = {"windows": WINDOWS, "pairs": {}}
    for pid, sub, a, b, note in PAIRS:
        rows = PV.load(PV.LOGS / sub)
        if not rows:
            print(f"\n[{pid}] 無資料")
            continue
        va, vb = PV.VENUE_KEYS[pid]
        req_mt = FEES.required_band_bps(va, vb, "maker_taker", rebate=False)
        req_tt = FEES.required_band_bps(va, vb, "taker_taker", rebate=False)
        rng = intraminute(PV.LOGS / sub)
        byw = defaultdict(list)
        for r in rows:
            byw[win_of(r["ts"])].append(r)
        print(f"\n[{pid}] {a} vs {b}  n={len(rows)} 分鐘｜無返佣門檻：掛單 {req_mt:.0f} / 吃單 {req_tt:.0f} bps")
        print(f"  {'窗':<5}{'n':>6}{'sell p90':>10}{'buy p90':>9}{'分鐘內幅度中位':>12}{'肥刻深度中位$':>12}")
        wstats = {}
        for w in ("OPEN", "RTH", "PRE", "OFF"):
            rs = byw.get(w, [])
            if not rs:
                continue
            s90, b90 = p90([x["sell_max"] for x in rs]), p90([x["buy_max"] for x in rs])
            ir = st.median([rng[x["ts"]] for x in rs if x["ts"] in rng] or [float("nan")])
            ins = [x for x in rs if x["sell_ntl"] is not None and x["buy_ntl"] is not None]
            fat = sorted(ins, key=lambda x: -max(x["sell_max"], x["buy_max"]))[:max(1, len(ins) // 10)]
            dep = st.median([min(x["sell_ntl"], x["buy_ntl"]) for x in fat]) if fat else float("nan")
            wstats[w] = {"n": len(rs), "sell_p90": round(s90, 2), "buy_p90": round(b90, 2),
                         "intraminute_range_med": round(ir, 2), "fat_depth_med_usd": dep}
            print(f"  {w:<5}{len(rs):>6}{s90:>+10.2f}{b90:>+9.2f}{ir:>12.2f}{dep:>12,.0f}")
        # P1/P2 ratio
        o, f = wstats.get("OPEN"), wstats.get("OFF")
        ratio = None
        if o and f:
            best_o = max(o["sell_p90"], o["buy_p90"])
            best_f = max(f["sell_p90"], f["buy_p90"])
            ratio = best_o / best_f if best_f > 0 else float("inf")
            print(f"  OPEN/OFF 帶寬比（取較寬側）：{ratio:.2f}x")
        # P3: convergence of episodes that START inside OPEN, on the full-series midline
        conv_open = {}
        for key, lab in (("sell_max", "sell"), ("buy_max", "buy")):
            band = max(p90([x[key] for x in byw.get("OPEN", [])]), PV.NET_BPS_MIN)
            c = PV.convergence(rows, band, with_starts=True)
            eps = [(t, m) for t, m in c.get("starts", []) if win_of(t) == "OPEN"]
            ok = sum(1 for _, m in eps if m is not None and m <= PV.CONV_MAX_MIN)
            conv_open[lab] = {"band": round(band, 2), "episodes": len(eps), "converged": ok,
                              "frac": round(ok / len(eps), 2) if eps else None}
            pct = (ok / len(eps) * 100) if eps else float("nan")
            print(f"  OPEN {lab}: 用 OPEN 自己的 p90 帶 {band:.2f} → 窗內起始的偏離 {len(eps)} 次、"
                  f"4h 內收斂 {ok}（{pct:.0f}%）")
        # P4: per-day OPEN p90 vs that day's OFF p90
        byday = defaultdict(lambda: {"OPEN": [], "OFF": []})
        for r in rows:
            w = win_of(r["ts"])
            if w in ("OPEN", "OFF"):
                d = datetime.fromtimestamp(r["ts"], timezone.utc).date().isoformat()
                byday[d][w].append(max(r["sell_max"], r["buy_max"]))
        days = []
        for d in sorted(byday):
            oo, ff = byday[d]["OPEN"], byday[d]["OFF"]
            if len(oo) >= 60 and len(ff) >= 60:
                days.append((d, round(p90(oo), 2), round(p90(ff), 2)))
        wins = sum(1 for _, x, y in days if x > 2 * y)
        print("  逐日 OPEN p90 / OFF p90：" + "  ".join(f"{d[5:]} {x:.1f}/{y:.1f}" for d, x, y in days)
              + f"  → OPEN>2×OFF 的天數 {wins}/{len(days)}")
        # economics if trading the window only, no rebate, one resting leg
        econ = {}
        for lab in ("sell", "buy"):
            band = conv_open[lab]["band"]
            net = FEES.net_per_trade_bps(band, va, vb, "maker_taker", rebate=False)
            per_day = conv_open[lab]["converged"] / max(1, len(days))
            dep = (wstats.get("OPEN") or {}).get("fat_depth_med_usd") or 0
            usd = per_day * (net / 1e4) * dep if net > 0 else 0.0
            econ[lab] = {"net_bps_maker_norebate": round(net, 2),
                         "conv_events_per_day": round(per_day, 2),
                         "usd_per_day": round(usd, 2)}
            print(f"  只做 OPEN 窗·掛單·無返佣：{lab} 每筆 {net:+.2f} bps × {per_day:.1f} 次/天"
                  f" × 深度 ${dep:,.0f} ≈ ${usd:.2f}/天")
        res["pairs"][pid] = {"windows": wstats, "open_off_ratio": ratio, "conv_open": conv_open,
                             "days": days, "days_open_gt_2x_off": wins, "econ_open_only": econ,
                             "required_band_norebate": {"maker_taker": req_mt, "taker_taker": req_tt}}
    # verdict against the four pre-registered bars
    print("\n  預註冊四關：")
    P = res["pairs"]

    def r_(p):
        return (P.get(p) or {}).get("open_off_ratio") or 0

    bars = {
        "P1 SNDK 與 NBIS 的 OPEN 帶 > 2× OFF": r_("SNDK") > 2 and r_("NBIS") > 2,
        "P2 ANTH 對照組 < 1.5×": 0 < r_("ANTH") < 1.5,
        "P3 OPEN 窗內偏離收斂 ≥70%（SNDK、NBIS 各至少一側，≥5 次）": all(
            any((P[p]["conv_open"][s]["frac"] or 0) >= 0.7 and P[p]["conv_open"][s]["episodes"] >= 5
                for s in ("sell", "buy")) for p in ("SNDK", "NBIS") if p in P),
        "P4 OPEN>2×OFF 的天數 ≥4（SNDK、NBIS 各自）": all(
            P[p]["days_open_gt_2x_off"] >= 4 for p in ("SNDK", "NBIS") if p in P),
    }
    for k, v in bars.items():
        print(f"    {'✅' if v else '❌'} {k}")
    res["bars"] = bars
    res["verdict"] = ("存活：登記變體 W，從今天起錄" if all(bars.values())
                      else "陣亡：時段條件化在這批資料上不成立，不開時鐘")
    print(f"  → {res['verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
