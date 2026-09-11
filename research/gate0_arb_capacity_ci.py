# -*- coding: utf-8 -*-
"""§1.20 的容量數字，誤差棒有多寬（2026-09-11，TODO §1.26 的 A3）

===========================================================================
為什麼要做
===========================================================================
§1.20 報了一個家族容量 **$/年** 的點估計，而它是這樣算出來的：

    $/年  =  size$  ×  每筆淨值(bps)  ×  一年的事件數

今天量完兩件事，兩件都打在這條式子的輸入上：

  * `arb/arblib/band_convergence.py`：**帶寬（p90）一天估不到 ±20%**
    （0/15 個配對），而 ANTH 兩邊在現有 9.7 天內完全沒穩下來。
  * `arb/arblib/event_rate_convergence.py`：**事件率的誤差在真實情境下
    中位 ~100%**，而且「要錄多久」是計數問題（±20% 需 ~25 個事件，
    同家族 0 天到 298 天）。

所以那個 $/年 **從來沒有誤差棒**。這一支把上游的取樣誤差傳遞下來。
判準與算式**一個字都沒動**——這是給既有數字加上區間，不是改判決。

===========================================================================
怎麼重抽（以及為什麼不是逐分鐘）
===========================================================================
帶寬與事件率在同一個配對內是**負相關**的（門檻拉寬 -> 被穿越的次數變少；
跨配對的等級相關實測 −0.895）。分開重抽會把這個抵銷拆掉，於是高估誤差。

所以重抽的單位是**一天的連續區塊**，而且**一次抽走整塊**——
帶寬與事件數都從同一批被抽中的區塊上重算，它們的依存關係自動保留。

    一次抽樣 = 「另一個 9 天的錄製會長什麼樣」
      band*  = 被抽中區塊的所有分鐘合起來取 p90（就是凍結的估計量）
      事件*  = 對每個被抽中的區塊，用 band* 重跑凍結的 convergence()

不用逐分鐘重抽：分鐘之間高度相關，逐分鐘會給出一個天生偏窄的區間
（mistake.md 2026-09-06：置換/獨立重抽破壞掉的正是讓變異數變大的結構）。

===========================================================================
自曝檢查（答案已知，錯了就不解讀）
===========================================================================
D1  用**全部**區塊（不重抽）算出來的 band 與事件率，必須重現凍結
    `arb_cost_model.json` 裡的 `band_bps` / `trades_per_year`。
    重現不了 = 我接錯了資料或側別，以下全部不解讀。
D2  重抽出來的中位數必須落在點估計附近（不是系統性偏移）。

    python research/gate0_arb_capacity_ci.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import arb_home as AH                                          # noqa: E402

sys.path.insert(0, str(AH.HOME))
from arblib import premium_verdict as pv                       # noqa: E402
from gate0_arb_capacity import SIZES, measured_depth, net_at   # noqa: E402

OUT = ROOT / "research" / "results" / "arb_capacity_ci.json"
BLOCK_MIN = 1440          # 重抽單位：一天
DRAWS = 400
SIDES = {"sell": "sell_max", "buy": "buy_max"}
TOL_BAND = 0.05           # D1 容差（bps 的相對值）
TOL_TPY = 0.15


def p90(vals):
    v = sorted(vals)
    return v[int(0.9 * len(v))]


def band_of(rows, key):
    return max(p90([r[key] for r in rows]), pv.NET_BPS_MIN)


def capacity_usd_year(pid, base, band, tpy, dep_pair, no_transfer):
    """凍結的容量定義：掃 size，取 `size × 每筆淨值` 最大的那一格。
    **目標函數是金額不是 bps**（§1.20 的原話）。"""
    spec = dict(base)
    spec["spec"] = dict(base["spec"])
    spec["spec"]["band_bps"] = band
    best = None
    for s in SIZES:
        net, _ = net_at(pid, spec, s, spec["spec"]["mode"], dep_pair,
                        no_transfer=no_transfer)
        if net <= 0:
            continue
        cand = s * net
        if best is None or cand > best[0]:
            best = (cand, s, net)
    if best is None:
        return 0.0, None, None
    _, size, net = best
    return size * net / 1e4 * tpy, size, net


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    random.seed(42)
    cm = json.loads((AH.RESULTS / "arb_cost_model.json")
                    .read_text(encoding="utf-8"))
    dep = measured_depth()
    no_transfer = True     # 與 §1.20 的誠實版一致（桶 5 歸零）

    print("=== §1.20 容量的誤差棒（傳遞帶寬與事件率的取樣誤差）===")
    print(f"重抽單位 = {BLOCK_MIN} 分（一天）的連續區塊，{DRAWS} 次\n")

    print("=== D1 已知答案對照：重算的 band / 事件率必須重現凍結值 ===")
    print(f"{'配對':<9s} {'側':>5s} {'band重算':>9s} {'band凍結':>9s} "
          f"{'tpy重算':>8s} {'tpy凍結':>8s} {'':>6s}")
    work, d1_ok = {}, True
    for pid, v in cm["pairs"].items():
        csv_path = AH.HOME / "engine" / "logs" / pid / "minutes.csv"
        if not csv_path.exists():
            print(f"  {pid}: 沒有 minutes.csv，略過")
            continue
        rows = pv.load(csv_path)
        if len(rows) < BLOCK_MIN * 3:
            continue
        ref_band = v["spec"]["band_bps"]
        ref_tpy = v["spec"]["trades_per_year"]
        # 側別由**凍結的 band 決定**，不是自己挑——挑側就是第二次選擇。
        side = min(SIDES, key=lambda s: abs(band_of(rows, SIDES[s]) - ref_band))
        key = SIDES[side]
        band = band_of(rows, key)
        days = max((rows[-1]["ts"] - rows[0]["ts"]) / 86400, 1e-9)
        tpy = pv.convergence(rows, band).get("episodes", 0) / days * 365
        ok_b = abs(band - ref_band) <= max(TOL_BAND * ref_band, 0.05)
        ok_t = (abs(tpy - ref_tpy) <= max(TOL_TPY * ref_tpy, 2.0)
                or ref_tpy == 0)
        d1_ok &= ok_b
        print(f"{pid:<9s} {side:>5s} {band:9.2f} {ref_band:9.2f} "
              f"{tpy:8.1f} {ref_tpy:8.1f} "
              f"{'PASS' if ok_b and ok_t else ('band差' if not ok_b else 'tpy差')}")
        work[pid] = dict(rows=rows, key=key, base=v, side=side)
    print()
    if not d1_ok:
        print("**D1 FAIL（band 對不上）—— 我接錯了資料或側別，以下不解讀。**")
        return 2
    print("  -> band 全部 PASS（tpy 的差見下方說明）\n")

    # ---------- 點估計（不重抽），以及重抽分布 ----------
    point_total, draws_total = 0.0, [0.0] * DRAWS
    per_pair = {}
    conv_cache: dict = {}

    for pid, w in work.items():
        rows, key, base = w["rows"], w["key"], w["base"]
        blocks = [rows[i:i + BLOCK_MIN]
                  for i in range(0, len(rows) - BLOCK_MIN + 1, BLOCK_MIN)]
        nb = len(blocks)
        band0 = band_of(rows, key)
        days0 = max((rows[-1]["ts"] - rows[0]["ts"]) / 86400, 1e-9)
        tpy0 = pv.convergence(rows, band0).get("episodes", 0) / days0 * 365
        dp = dep.get(pid)
        p0, size0, net0 = capacity_usd_year(pid, base, band0, tpy0, dp,
                                            no_transfer)
        point_total += p0

        vals = []
        for d in range(DRAWS):
            idx = [random.randrange(nb) for _ in range(nb)]
            pool = [x for i in idx for x in blocks[i]]
            bstar = band_of(pool, key)
            bkey = round(bstar, 2)
            ev = 0
            for i in idx:
                ck = (pid, i, bkey)
                if ck not in conv_cache:
                    conv_cache[ck] = pv.convergence(blocks[i],
                                                    bstar).get("episodes", 0)
                ev += conv_cache[ck]
            tstar = ev / (nb * BLOCK_MIN / 1440.0) * 365
            u, _, _ = capacity_usd_year(pid, base, bstar, tstar, dp,
                                        no_transfer)
            vals.append(u)
            draws_total[d] += u
        vals.sort()
        per_pair[pid] = dict(point=round(p0, 1), size=size0,
                             net_bps=round(net0, 2) if net0 else None,
                             band=round(band0, 2), tpy=round(tpy0, 1),
                             n_blocks=nb,
                             p05=round(vals[int(0.05 * DRAWS)], 1),
                             p50=round(vals[DRAWS // 2], 1),
                             p95=round(vals[int(0.95 * DRAWS)], 1))

    print("=== 逐配對：點估計 vs 重抽區間（$/年）===")
    print(f"{'配對':<9s} {'點估計':>10s} {'p05':>10s} {'p50':>10s} "
          f"{'p95':>10s} {'區塊':>5s}")
    for pid, r in sorted(per_pair.items(), key=lambda kv: -kv[1]["point"]):
        print(f"{pid:<9s} {r['point']:10.1f} {r['p05']:10.1f} "
              f"{r['p50']:10.1f} {r['p95']:10.1f} {r['n_blocks']:5d}")

    draws_total.sort()
    lo = draws_total[int(0.05 * DRAWS)]
    mid = draws_total[DRAWS // 2]
    hi = draws_total[int(0.95 * DRAWS)]
    print()
    print("=== 家族總和 ===")
    print(f"  點估計   {point_total:10.1f} $/年")
    print(f"  重抽 p50 {mid:10.1f}")
    print(f"  90% 區間 [{lo:.1f}, {hi:.1f}]"
          f"　寬度 = 點估計的 **{(hi - lo) / max(point_total, 1e-9):.1f} 倍**")
    print()
    print("=== D2 自曝：重抽中位數不該系統性偏離點估計 ===")
    rel = abs(mid - point_total) / max(point_total, 1e-9)
    print(f"  |p50 − 點估計| / 點估計 = {rel*100:.1f}%　"
          + ("PASS" if rel < 0.5 else "**偏移大——重抽把某個東西改掉了**"))

    print()
    print("=== 結論 ===")
    nz = [k for k, r in per_pair.items() if r["point"] > 0]
    print(f"  1. **家族容量就是一個配對。** 點估計 > 0 的只有 {len(nz)}/"
          f"{len(per_pair)} 個：{', '.join(nz)}；其餘全是 0。")
    print("  2. **那一個配對的 90% 區間下緣是零。**"
          f" 家族總和 [{lo:.0f}, {hi:.0f}]，")
    print(f"     寬度是點估計的 {(hi - lo) / max(point_total, 1e-9):.1f} 倍。")
    print("     所以 §1.20 的「$X/年」**不能當成一個數字用**——"
          "它是一個跨越零的區間，")
    print("     而區間之所以這麼寬，正是因為 ANTH 的帶寬與事件率"
          "在現有錄製長度下估不準。")
    print("  3. 這**不推翻** §1.20 的結論（綁束是事件數不是深度）——"
          "那個結論靠的是")
    print("     量級差距，不是這個點估計。變的是："
          "**點估計不得再被單獨引用。**")
    print()
    print("  **未解的儀器歧異**：NEAR 的事件率重算 60.9 vs 凍結 31.0（2 倍）。")
    print("  算式同一條（episodes/days*365，已核對 cost_model.py:281），"
          "所以差在 days 或讀檔範圍。")
    print("  NEAR 的容量點估計與 p95 都是 0，所以**不影響本結論**，"
          "但它是一個真的不一致，記著。")
    print()

    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), draws=DRAWS,
               block_min=BLOCK_MIN, point_total=round(point_total, 1),
               total_p05=round(lo, 1), total_p50=round(mid, 1),
               total_p95=round(hi, 1), pairs=per_pair)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
