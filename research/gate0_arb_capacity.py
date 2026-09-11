# -*- coding: utf-8 -*-
"""Gate 0 的第三關：§0.75 的容量（2026-09-11）

===========================================================================
為什麼是獨立的一關
===========================================================================
`.claude/rules/factor-research.md` 第 10 條寫死：

> **容量是獨立的一關，不是成本的附註。** 帶再寬，對手簿頂檔只有 $264
> 就只值零錢；容量的上限是**對手的深度**，不是自己的本金。

`gate0_arb_unhedged.py` 量到「掛單進 + 不跨場館」可以讓 6/8 個配對的
每筆淨值翻正（NBIS +2.56、ANTH +2.92 bps）。但那是在 **size_usd=200**
上算的，而 $200 × 2.56 bps = **$0.05 一筆**。所以現在的綁束不是邊際的
符號，是**能放多大**。

===========================================================================
先修一個會決定答案的儀器問題
===========================================================================
`arblib/cost_model.py` 第 231 行：

    depth_1bps=top * 4.5, depth_3bps=top * 4.5,   # recorder has top only
    # 4.5x = the median top->3bps ratio measured in TODO 1.00 v5. ASSUMED.

兩個分檔深度都是**同一個假設值**。後果有兩個，而且都直接決定容量：

  1. `slippage_bps` 的「1→3 bps 那一段」在 `d1 == d3` 之下**永遠走不到**，
     所以 2 bps 那一檔的成本從來沒有被計到；
  2. 超過 `d3` 就回 `inf`，於是**容量的懸崖位置＝一個假設乘出來的數**。

**但我們有真的量測。** `engine/tools/scanner.py` 的 `DEPTH_BPS = (1.0, 3.0)`
逐檔累加雙邊簿口，輸出在 `engine/logs/scan/scan_v5_*.csv` 的
`{a,b}_{bid,ask}_usd_{1bps,3bps}`，而它的宇宙**涵蓋 §0.75 這八個配對**
（`NBIS@IO-lighter`、`ANTH@IO-lighter-rh`、…）。

所以這一支做的事是：**把 top×4.5 換成量出來的分檔深度，再掃 size。**
凍結的判準、帶寬、毛利算式（`band/2`）一個字都沒有動——這是儀器修正。

===========================================================================
兩股反向的力量，所以容量不是一條單調線
===========================================================================
    size 變大 -> 滑價變大（吃掉更深的檔位）          淨值往下
    size 變大 -> 閒置資金攤薄                         淨值往上
                 （桶 3：小場館各壓 $300 閒置，
                  在 size=200 時那是名目的 3.0 倍）

所以淨值對 size 是一個**有峰的曲線**，而「容量」有兩個答案，兩個都要報：

    size_max    淨值還 > 0 的最大 size —— 那是**邊界**，而邊界上的淨值
                趨近零，所以它乘出來的金額也趨近零。它不是容量。
    size$       `size × 每筆淨` 最大的那個格子 —— **這個才是容量**，
                因為要決定的是「放多少錢」而目標函數是金額不是 bps。
    $/年        size$ × 該格淨值 × 該配對一年的**事件數**。

    第三項是最容易被忽略的那一項：深度決定一筆能放多大，
    **事件數決定一年能做幾筆**，而兩者是獨立的綁束。

===========================================================================
自曝檢查（答案已知，錯了就不解讀）
===========================================================================
D1  用**假設深度**在 size=200 重算，必須重現 `arb_cost_model.json` 的
    `net_bps`（容差 0.01 bps）。重現不了就是我接錯了 cost_model，
    下面全部不解讀（mistake.md 2026-08-26：第二份實作會安靜地不同意）。
D2  量出來的 1bps 深度必須 ≥ 頂檔深度、3bps 必須 ≥ 1bps。違反＝欄位接錯。
D3  `size <= top` 的滑價必須是 0（`slippage_bps` 的定義），用它驗串接。

    python research/gate0_arb_capacity.py
    python research/gate0_arb_capacity.py --mode taker_taker
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import arb_home as AH                          # noqa: E402
AH.add_to_path()

OUT = ROOT / "research" / "results" / "gate0_arb_capacity.json"
TOL = 0.01

# recorder 的 pid -> 掃描器的配對名。掃描器用的標的代號跟錄製器不同
# （GOLD_LL 的錄製對是 lighter vs lighter-rh，掃描器叫 GOLD_IDX），
# 所以這張表是手寫的，而 D2/D3 會抓到對錯了的列。
SCAN_PAIR = {
    "NBIS":    "NBIS@IO-lighter",
    "ANTH":    "ANTH@IO-lighter-rh",
    "BTC":     "BTC@HL-lighter-rh",
    "HYPE":    "HYPE@HL-lighter-rh",
    "ZEC":     "ZEC@HL-lighter-rh",
    "NEAR":    "NEAR@HL-lighter-rh",
    "GOLD_LL": "GOLD_IDX@lighter-lighter-rh",
    "NVDA_LL": "NVDA@lighter-lighter-rh",
}

SIZES = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000,
         7500, 10000, 15000, 25000, 50000]


def measured_depth() -> dict:
    """每個配對的 (頂檔, 1bps, 3bps) 可成交深度，中位數。

    一筆套利要**兩腿都成交**，所以每一檔的綁束是**兩腿的較小者**。
    方向上取**較差的那一側**（賣側 = 在 A 的 bid 賣、B 的ask 買；
    買側相反）——刻意悲觀，不讓它奉承任何配對。
    """
    files = sorted(glob.glob(str(AH.HOME / "engine" / "logs" / "scan"
                                 / "scan_v5_*.csv")))
    if not files:
        return {}
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = {}
    for pid, pair in SCAN_PAIR.items():
        m = d[d.pair == pair]
        if len(m) < 50:
            out[pid] = None
            continue
        r = {}
        for side, ca, cb in (("sell", "a_bid", "b_ask"), ("buy", "a_ask", "b_bid")):
            lv = []
            for suf in ("", "_1bps", "_3bps"):
                x = np.minimum(m[ca + "_usd" + suf].to_numpy(float),
                               m[cb + "_usd" + suf].to_numpy(float))
                x = x[np.isfinite(x)]
                lv.append(float(np.median(x)) if len(x) else np.nan)
            r[side] = lv                                   # [top, d1, d3]
        worse = min(("sell", "buy"), key=lambda s: r[s][2])   # 3bps 較淺的那側
        out[pid] = dict(n_rows=int(len(m)), per_side=r, side_used=worse,
                        top=r[worse][0], d1=r[worse][1], d3=r[worse][2])
    return out


def spec_for(pid, base, size, mode, dep, assumed_depth=False):
    from arblib.cost_model import TradeSpec
    sp = base["spec"]
    if assumed_depth or dep is None:
        top = sp["depth_top"]
        d1 = d3 = top * 4.5
    else:
        top, d1, d3 = dep["top"], dep["d1"], dep["d3"]
    return TradeSpec(leg_a=sp["leg_a"], leg_b=sp["leg_b"],
                     band_bps=sp["band_bps"], size_usd=float(size),
                     depth_top=top, depth_1bps=d1, depth_3bps=d3,
                     hold_minutes=sp["hold_minutes"],
                     fund_diff_bps_8h=sp["fund_diff_bps_8h"],
                     trades_per_year=sp["trades_per_year"],
                     mode=mode, rebate=sp.get("rebate", True))


def net_at(pid, base, size, mode, dep, no_transfer, assumed_depth=False):
    """每筆淨值 bps。`no_transfer` = §0.75 新構造的「不跨場館」那一項。"""
    from arblib.cost_model import cost_breakdown
    t = spec_for(pid, base, size, mode, dep, assumed_depth)
    r = cost_breakdown(t)
    if r["total_cost_bps"] is None:
        return float("-inf"), r
    cost = r["total_cost_bps"]
    if no_transfer:
        cost -= (r["buckets_bps"].get("5_transfer") or 0.0)
    return r["gross_bps"] - cost, r


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="maker_taker",
                    choices=("taker_taker", "maker_taker", "maker_maker"))
    ap.add_argument("--keep-transfer", action="store_true",
                    help="不把桶 5 歸零（預設歸零，照 gate0_arb_unhedged 的誠實版）")
    a = ap.parse_args()
    no_transfer = not a.keep_transfer

    cm = json.loads((AH.RESULTS / "arb_cost_model.json")
                    .read_text(encoding="utf-8"))
    dep = measured_depth()
    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), mode=a.mode,
               transfer_zeroed=no_transfer, depth=dep, pairs={})

    # ---------- D1 自曝：假設深度 + 原模式必須重現凍結的 net ----------
    print("=== D1 已知答案對照（假設深度、size=200、原模式，必須重現）===")
    d1_ok = True
    for pid, v in cm["pairs"].items():
        got, _ = net_at(pid, v, cm["size_usd"], v["spec"]["mode"], None,
                        no_transfer=False, assumed_depth=True)
        ref = v["net_bps"]
        ok = math.isfinite(got) and abs(got - ref) < TOL
        d1_ok &= ok
        print("  %-8s 重算 %+8.3f   凍結 %+8.3f   %s"
              % (pid, got, ref, "PASS" if ok else "**FAIL**"))
    res["D1"] = bool(d1_ok)
    if not d1_ok:
        print("\n**D1 FAIL —— 我接錯了 cost_model，以下全部不解讀。**")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        return 2
    print("  -> PASS\n")

    # ---------- D2 自曝：深度的單調性 ----------
    print("=== D2 量出來的深度（單調性必須成立）===")
    print("%-8s %6s %5s %10s %10s %10s %7s"
          % ("配對", "列數", "側", "頂檔$", "1bps$", "3bps$", "vs假設"))
    d2_ok = True
    for pid in SCAN_PAIR:
        w = dep.get(pid)
        if not w:
            print("%-8s %6s  掃描器資料不足" % (pid, "—"))
            d2_ok = False
            continue
        mono = w["top"] <= w["d1"] + 1e-6 <= w["d3"] + 1e-6
        d2_ok &= bool(mono)
        ratio = w["d3"] / w["top"] if w["top"] else float("nan")
        print("%-8s %6d %5s %10.0f %10.0f %10.0f %6.1fx%s"
              % (pid, w["n_rows"], w["side_used"], w["top"], w["d1"], w["d3"],
                 ratio, "" if mono else "  **非單調**"))
    res["D2"] = bool(d2_ok)
    print("  （假設值一律是 4.5x —— 右邊那一欄就是那個假設離真實多遠）")
    if not d2_ok:
        print("\n**D2 FAIL —— 深度欄位接錯或樣本不足，以下不解讀。**")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        return 2
    print("  -> PASS\n")

    # ---------- D3 自曝：size <= top 的滑價必須是 0 ----------
    from arblib.cost_model import slippage_bps
    s0, _ = slippage_bps(50.0, 100.0, 450.0, 900.0)
    s1, _ = slippage_bps(200.0, 100.0, 450.0, 900.0)
    s2, _ = slippage_bps(2000.0, 100.0, 450.0, 900.0)
    print("=== D3 滑價函式串接（頂檔內=0、檔內>0、超過3bps=inf）===")
    print("  size<=top %.4f   top<size<=d1 %.4f   size>d3 %s  -> %s"
          % (s0, s1, s2, "PASS" if (s0 == 0 and s1 > 0 and
                                    not math.isfinite(s2)) else "**FAIL**"))
    res["D3"] = bool(s0 == 0 and s1 > 0 and not math.isfinite(s2))
    if not res["D3"]:
        print("\n**D3 FAIL。以下不解讀。**")
        return 2
    print()

    # ---------- 容量掃描 ----------
    label = ("掛單進/吃單出" if a.mode == "maker_taker" else a.mode) + \
            ("、不跨場館" if no_transfer else "")
    print("=== 容量掃描（%s、**量出來的深度**）===" % label)
    print("每格是每筆淨值 bps；`—` = size 超過 3bps 深度（模型回 inf）\n")
    hdr = "%-8s" % "配對" + "".join("%8s" % ("%dk" % (s // 1000) if s >= 1000
                                            else str(s)) for s in SIZES)
    print(hdr)
    summary = []
    for pid, v in cm["pairs"].items():
        w = dep.get(pid)
        nets, line = [], "%-8s" % pid
        for s in SIZES:
            n, _ = net_at(pid, v, s, a.mode, w, no_transfer)
            nets.append(n)
            line += ("%8.2f" % n) if math.isfinite(n) else "%8s" % "—"
        print(line)
        fin = [(s, n) for s, n in zip(SIZES, nets) if math.isfinite(n)]
        pos = [(s, n) for s, n in fin if n > 0]
        best = max(fin, key=lambda x: x[1]) if fin else (None, None)
        smax = max(pos, key=lambda x: x[0]) if pos else (None, None)
        tpy = v["spec"]["trades_per_year"]
        # **目標函數是金額不是 bps。** size_max 是「還沒虧」的邊界，
        # 不是賺最多的地方——邊界上的淨值趨近零，乘上去也趨近零。
        # 所以另外報一個「size × 每筆淨」最大的格子。
        dstar = max(pos, key=lambda x: x[0] * x[1]) if pos else (None, None)
        summary.append(dict(
            size_dollar=dstar[0], net_at_dollar=dstar[1],
            usd_per_trade_best=(dstar[0] * dstar[1] / 1e4
                               if dstar[0] else None),
            usd_per_year_best=(dstar[0] * dstar[1] / 1e4 * tpy
                               if dstar[0] else None),
            pid=pid, side=w["side_used"] if w else None,
            top=w["top"] if w else None, d3=w["d3"] if w else None,
            size_star=best[0], net_star=best[1],
            size_max=smax[0], net_at_max=smax[1], trades_per_year=tpy,
            usd_per_trade_at_max=(smax[0] * smax[1] / 1e4
                                  if smax[0] else None),
            usd_per_year_at_max=(smax[0] * smax[1] / 1e4 * tpy
                                 if smax[0] else None)))
    res["pairs"] = {x["pid"]: x for x in summary}

    print("\n=== 容量結論（兩個答案都報）===")
    print("%-8s %9s %8s %8s %9s %9s %8s %8s %9s"
          % ("配對", "3bps深度", "size_max", "淨@max",
             "**size$**", "淨@size$", "$/筆", "次/年", "**$/年**"))
    for x in summary:
        print("%-8s %9s %8s %8s %9s %9s %8s %8.0f %9s"
              % (x["pid"],
                 "%.0f" % x["d3"] if x["d3"] else "—",
                 str(x["size_max"] or "**無**"),
                 "%+.2f" % x["net_at_max"] if x["net_at_max"] is not None else "—",
                 str(x["size_dollar"] or "—"),
                 "%+.2f" % x["net_at_dollar"] if x["net_at_dollar"] is not None else "—",
                 "%.3f" % x["usd_per_trade_best"] if x["usd_per_trade_best"] else "—",
                 x["trades_per_year"],
                 "%.2f" % x["usd_per_year_best"]
                 if x["usd_per_year_best"] else "—"))
    print("  size_max = 淨值還沒翻負的邊界（邊界上賺不到錢）；"
          "size$ = 金額最大的格子，**那個才是容量**")
    noev = [x["pid"] for x in summary
            if x["size_dollar"] and not x["trades_per_year"]]
    if noev:
        print("  %s 有正的格子但 `次/年 = 0` —— 那是**沒有量到收斂事件**，"
              % "/".join(noev))
        print("  不是「沒有機會」。在錄到事件數之前，它們的 $/年 是未知不是零。")

    live = [x for x in summary if x["usd_per_year_best"]]
    tot = sum(x["usd_per_year_best"] for x in live)
    res["total_usd_per_year"] = tot
    res["n_pairs_positive"] = len(live)
    print("\n**全家族合計 $%.2f / 年**（%d 個配對有正的格子）" % (tot, len(live)))
    print("毛利沿用凍結模型的 `band/2`，**沒有重新推導**；")
    print("桶 3 的閒置資金、桶 5/6/7 的機率仍是 ASSUMED —— 見 cost_model 的 tags。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
