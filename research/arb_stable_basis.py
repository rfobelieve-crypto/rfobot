# -*- coding: utf-8 -*-
"""§1.21：把穩定幣基差從 §0.75 的 premium 裡扣掉（2026-09-11）

===========================================================================
為什麼
===========================================================================
Quant Arb〈Small Trader Alpha #6: Perpetual Arbitrage〉把穩定幣正規化列為
**第一步**，理由不是精確度，是**你會賭到別的東西**：

> 「除非我們針對**交易所各自的**穩定幣價格做調整，而不是套用某個全球價格，
> 否則我們看到的永續之間的基差會是**一個對 USDC 的賭注**。」

他在〈Small Trader Alpha #4 - Funding Arbitrage〉的最後一句又講一次：
「報價資產不一樣。有些是 USDT、有些 USDC、有些 USD。**不要假設它們可互換。**」

而我們寫過這件事**四次**，沒有一次扣掉它：

    arblib/fees.py:62          「quotes in USDG, so part of any premium is the
                                stablecoin basis」   <- 逐字
    arblib/cost_model.py:18    「7 tail ... stablecoin depeg (USDG!)」
    arblib/cost_model.py:96    ("stablecoin depeg USDG/USDC", ...)
    arblib/premium_verdict.py:66「USDC vs USDG basis dressed up as premium」

===========================================================================
誰受影響（查證過的，不是推論）
===========================================================================
基差只存在於**有 lighter-rh 那一腿**的配對（它報價在 USDG，其餘 USDC）：

    NBIS      IO - lighter          **沒有基差**（兩腿都不是 USDG）
    ANTH      IO - lighter-rh       有，**但沒有參考標的可量**（見下）
    BTC/HYPE/ZEC/NEAR   HL - lighter-rh        有，可量
    GOLD_LL/NVDA_LL     lighter - lighter-rh   有，可量

**掃描器的宇宙決定了可量性**：`IO` 那兩個場館對上面只有四檔股票／私司永續
（io:SNDK / io:ANTH / io:NBIS / io:OAI）——**沒有任何流動幣**可以當參考。
所以 ANTH 只能用「轉運」（見 D3），而轉運是一個**假設**，要標明。

===========================================================================
怎麼量（重點是它必須是「共同因子」）
===========================================================================
用掃描器 v5 的逐筆雙邊簿口，對**參考標的**（流動幣，我們不交易它們）算

    offset(va, vb, day) = median over 參考標的 of (midA / midB − 1) × 1e4

**這量到的是「場館對的共同偏移」**，而穩定幣基差是它的一部分（另外還有結構與
合約差異）。**這樣反而更保守**：扣掉共同偏移等於做了一次中性化
（`factor-research.md` 第 9 條），留下的是標的自己的偏離。

**不可以用配對自己的標的去估 offset** —— 那會把訊號一起扣掉。

===========================================================================
自曝檢查（答案已知，錯了就不解讀）
===========================================================================
D1  用 minutes.csv 重算**未修正**的 band，必須重現 `arb_premium_verdict.json`
    的 `band_bps`（逐側、容差 0.05 bps）。重現不了就是我接錯了口徑。
D2  **穩定幣基差必須是跨標的共同的。** 同一個場館對、不同參考標的算出來的
    offset 必須接近（報逐標的值與離散度）。離散度大 = 它不是穩定幣基差，
    是標的自己的東西，**那就不該扣**。
D3  `HL - lighter-rh` 與 `lighter - lighter-rh` 都是「某個 USDC 場館 →
    lighter-rh」，所以**兩者的 offset 應該接近**（差的是 HL vs lighter 本身）。
    接近才允許把它轉運到 ANTH；不接近就明寫「ANTH 無法修正」。

    python research/arb_stable_basis.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import arb_home as AH                          # noqa: E402
AH.add_to_path()

OUT = ROOT / "research" / "results" / "arb_stable_basis.json"
NET_BPS_MIN = 1.0          # 抄自 premium_verdict.py:83，不是重新決定
TOL_D1 = 0.05
TOL_D2 = 3.0               # 逐標的 offset 的 IQR 上限（bps）
TOL_D3 = 3.0               # 兩個場館對的 offset 差上限（bps）

# 參考標的：**我們不交易的流動幣**。故意不含任何 §0.75 的標的。
REF = ("BTC", "ETH", "SOL", "XRP", "SUI")

# pid -> (leg_a, leg_b)，抄自 premium_verdict.VENUE_KEYS（唯一真相源）
PAIR_CSV = {
    "SNDK": "minutes.csv", "NBIS": "NBIS/minutes.csv", "ANTH": "ANTH/minutes.csv",
    "BTC": "BTC/minutes.csv", "HYPE": "HYPE/minutes.csv", "ZEC": "ZEC/minutes.csv",
    "NEAR": "NEAR/minutes.csv", "GOLD_LL": "GOLD_LL/minutes.csv",
    "NVDA_LL": "NVDA_LL/minutes.csv",
}


def venue_keys():
    # arblib 在 sys.path 上（AH.add_to_path），但 premium_verdict 住在 arblib/ 裡
    from arblib import premium_verdict as PV
    return dict(PV.VENUE_KEYS)


def load_rows(sub):
    """逐分鐘列，含輪替檔。**口徑抄 premium_verdict.load()**，不自己發明。"""
    base = AH.LOGS / sub
    rows = []
    for fp in [Path(x) for x in sorted(glob.glob(str(base) + "*.old"))] + [base]:
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        need = {"minute_ts", "sell_edge_max_bps", "buy_edge_max_bps"}
        if not need.issubset(d.columns):
            continue
        d = d[["minute_ts", "sell_edge_max_bps", "buy_edge_max_bps"]].dropna()
        rows.append(d)
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True).sort_values("minute_ts")
    d["day"] = pd.to_datetime(d.minute_ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return d


def band_of(vals):
    """p90 然後 floor —— 逐行抄 premium_verdict.side_stats。"""
    v = sorted(float(x) for x in vals)
    if not v:
        return None, None
    p90 = v[int(0.9 * len(v))]
    return round(p90, 3), round(max(p90, NET_BPS_MIN), 3)


def scan_offsets():
    """逐場館對、逐日、逐參考標的的中價偏移（bps）。"""
    files = sorted(glob.glob(str(AH.HOME / "engine" / "logs" / "scan"
                                 / "scan_v5_*.csv")))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["day"] = pd.to_datetime(d.ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    mida = (d.a_bid + d.a_ask) / 2.0
    midb = (d.b_bid + d.b_ask) / 2.0
    d["off_bps"] = (mida / midb - 1.0) * 1e4
    d = d[np.isfinite(d.off_bps)]
    # 參考標的：sym_a 去掉交易所後綴之後剛好等於參考名
    d["ref"] = d.sym_a.astype(str).str.replace("io:", "", regex=False)
    d = d[d.ref.isin(REF)]
    out = {}
    for (va, vb), g in d.groupby(["leg_a", "leg_b"]):
        per_ref = g.groupby("ref").off_bps.median().to_dict()
        per_day = g.groupby("day").off_bps.median().to_dict()
        # **剔除離群的參考標的，用規則不用手挑**：一個參考標的若自己就是
        # 離群（|x − med| > 3 × MAD），它量到的不是場館的共同因子，是它自己
        # 的東西 —— 那正是 D2 要偵測的。實測 SUI 在**每一個**場館對上都是
        # +6~+9 bps，也就是「SUI 在 HL/lighter 上就是貴」，不是場館基差。
        v_all = np.array(list(per_ref.values()), float)
        med = float(np.median(v_all))
        mad = float(np.median(np.abs(v_all - med))) or 1e-9
        keep = {k: x for k, x in per_ref.items() if abs(x - med) <= 3 * mad}
        dropped = sorted(set(per_ref) - set(keep))
        v = np.array(list(keep.values()), float)
        out[(va, vb)] = dict(
            n_rows=int(len(g)),
            per_ref={k: round(x, 3) for k, x in per_ref.items()},
            dropped_refs=dropped,
            per_day={k: round(x, 3) for k, x in per_day.items()},
            median=float(np.median(v)) if len(v) else float("nan"),
            median_all_refs=med,
            iqr=float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) > 1 else 0.0)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    vk = venue_keys()
    pv = json.loads((AH.RESULTS / "arb_premium_verdict.json")
                    .read_text(encoding="utf-8"))
    # D1 要對資料窗免疫：截到凍結 JSON 自己的 asof（見檔頭）
    asof_ts = int(pd.Timestamp(pv["asof_utc"], tz="UTC").timestamp())
    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), ref_symbols=list(REF))

    # ---------- D1：重現未修正的 band ----------
    print("=== D1 已知答案對照（未修正的 band 必須重現凍結值）===")
    print("%-9s %10s %10s %10s %10s %6s"
          % ("配對", "賣側本支", "賣側凍結", "買側本支", "買側凍結", "判"))
    rows_by_pid, d1_ok, d1_seen = {}, True, 0
    for pid, sub in PAIR_CSV.items():
        d = load_rows(sub)
        if d is None or not len(d):
            print("%-9s  資料不足" % pid)
            continue
        rows_by_pid[pid] = d
        dz = d[d.minute_ts <= asof_ts]          # 只有 D1 用截斷的窗
        _, bs = band_of(dz.sell_edge_max_bps)
        _, bb = band_of(dz.buy_edge_max_bps)
        # 凍結值的路徑是 pairs[pid].sides.{sell,buy}.full.band_bps
        # （不是 interim —— 那是 gate 未滿時的欄位名，第一版我寫錯了）
        sd = ((pv.get("pairs") or {}).get(pid) or {}).get("sides") or {}
        rs = ((sd.get("sell") or {}).get("full") or {}).get("band_bps")
        rb = ((sd.get("buy") or {}).get("full") or {}).get("band_bps")
        oks = rs is None or abs(bs - rs) < TOL_D1
        okb = rb is None or abs(bb - rb) < TOL_D1
        if rs is not None or rb is not None:
            d1_seen += 1
            d1_ok &= bool(oks and okb)
        print("%-9s %10.3f %10s %10.3f %10s %6s"
              % (pid, bs, ("%.3f" % rs) if rs is not None else "—",
                 bb, ("%.3f" % rb) if rb is not None else "—",
                 "OK" if (oks and okb) else "**不符**"))
    res["D1"] = bool(d1_ok and d1_seen >= 3)
    if not res["D1"]:
        print("\n**D1 未過（或可對照的配對不足 3 個）—— 我接錯了口徑，以下不解讀。**")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        return 2
    print("  -> PASS（%d 個配對對得上）\n" % d1_seen)

    # ---------- D2：基差必須跨標的共同 ----------
    off = scan_offsets()
    if not off:
        print("掃描器資料不足，停。")
        return 2
    print("=== D2 場館對的共同偏移，逐參考標的（必須接近）===")
    print("%-22s %7s %9s %8s   逐標的" % ("場館對", "列數", "中位bps", "IQR"))
    d2 = {}
    for (va, vb), w in sorted(off.items()):
        flat = w["iqr"] <= TOL_D2
        d2["%s-%s" % (va, vb)] = dict(w, cross_asset_consistent=bool(flat))
        print("%-22s %7d %+9.3f %8.3f   %s%s%s"
              % ("%s-%s" % (va, vb), w["n_rows"], w["median"], w["iqr"],
                 ", ".join("%s %+.2f" % (k, v) for k, v in
                           sorted(w["per_ref"].items())),
                 ("   剔除 " + "/".join(w["dropped_refs"])) if w["dropped_refs"] else "",
                 "" if flat else "   **仍離散，不該扣**"))
    res["offsets"] = d2

    # ---------- D3：兩條「→ lighter-rh」必須接近，才能轉運到 ANTH ----------
    a = off.get(("HL", "lighter-rh"), {}).get("median")
    b = off.get(("lighter", "lighter-rh"), {}).get("median")
    print("\n=== D3 轉運可行性（ANTH 沒有參考標的）===")
    if a is None or b is None:
        print("  兩條之一沒有資料 -> **ANTH 不修正**")
        transport = None
    else:
        gap = abs(a - b)
        transport = (a + b) / 2.0 if gap <= TOL_D3 else None
        print("  HL→lighter-rh %+.3f   lighter→lighter-rh %+.3f   差 %.3f bps -> %s"
              % (a, b, gap,
                 "可轉運（取兩者平均 %+.3f）" % transport if transport is not None
                 else "**差太大，ANTH 不修正**"))
    res["D3_transport_bps"] = transport

    # ---------- 套用：扣掉共同偏移之後的 band ----------
    # 正號的 premium 偏移讓賣側邊際 +c、買側 −c，所以修正是反向。
    print("\n=== 扣掉共同偏移之後的 band ===")
    print("%-9s %-22s %8s %9s %9s %9s %9s %8s"
          % ("配對", "場館對", "offset", "賣側原", "賣側修", "買側原", "買側修", "來源"))
    applied = {}
    for pid, d in rows_by_pid.items():
        va, vb = vk.get(pid, (None, None))
        key = (va, vb)
        src, c = "—", None
        if vb != "lighter-rh":
            src, c = "無基差(兩腿非USDG)", 0.0
        elif key in off and off[key]["iqr"] <= TOL_D2:
            src, c = "量測", off[key]["median"]
        elif transport is not None:
            src, c = "**轉運(假設)**", transport
        _, bs0 = band_of(d.sell_edge_max_bps)
        _, bb0 = band_of(d.buy_edge_max_bps)
        if c is None:
            print("%-9s %-22s %8s %9.3f %9s %9.3f %9s %8s"
                  % (pid, "%s-%s" % (va, vb), "—", bs0, "**不可**", bb0,
                     "**不可**", "無參考"))
            applied[pid] = dict(venues=[va, vb], offset_bps=None,
                                band_sell=bs0, band_buy=bb0,
                                band_sell_adj=None, band_buy_adj=None, source="無參考")
            continue
        _, bs1 = band_of(d.sell_edge_max_bps - c)
        _, bb1 = band_of(d.buy_edge_max_bps + c)
        applied[pid] = dict(venues=[va, vb], offset_bps=round(c, 3),
                            band_sell=bs0, band_buy=bb0,
                            band_sell_adj=bs1, band_buy_adj=bb1, source=src)
        print("%-9s %-22s %+8.3f %9.3f %9.3f %9.3f %9.3f %8s"
              % (pid, "%s-%s" % (va, vb), c, bs0, bs1, bb0, bb1, src))
    res["applied"] = applied

    print("\n=== 讀法 ===")
    print("  **這支不下判決。** 它只把一個已知的污染量化出來，")
    print("  並把『哪些配對修正不了』寫清楚——後者跟修正本身一樣重要。")
    print("  band 的 floor 是 1.0 bps（抄 premium_verdict），所以修正後")
    print("  碰到 floor 的那一側代表『扣掉基差之後那一側沒有空間了』。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
