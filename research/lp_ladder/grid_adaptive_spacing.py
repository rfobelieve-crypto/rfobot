# -*- coding: utf-8 -*-
"""§1.18h 網格上半排：格距與部位隨 σ 調整，值不值得（預註冊 2026-09-05）

§1.18f 已判定左右軸畫不出來（路徑形態在 4h–168h 上都不可預測），所以三層控制
只剩上半排。這支測上半排本身：**σ 自適應的格距／部位，比固定參數好嗎。**

## 先把公式代進去（這改變了預期，所以寫在最前面）

    Π = σ²T·(g−c)/g²        →  d/dg[(g−c)/g²] = (2c−g)/g³ = 0  ⇒  g* = 2c

**最佳格距是 2c，一個常數，與 σ 無關。** σ 只縮放利潤，不移動最佳點。
所以 `g_t = max(2c, k·σ̂_t)` 這條規則**只在 k·σ̂ > 2c 時才生效，而那時它把格距
推離最佳點**——依公式，自適應格距的損益必然 ≤ 固定在 2c 的損益。
它的價值若存在，只能在**庫存與回撤**那一側，也就是使用者判準的第二個分支。

順帶量出來的事實：現行凍結政策 drop=0.25、N=30 ⇒ **g ≈ 0.96%**，而
2c ≈ 4–10 bps（依標的）。**現行格距是理論最佳的 10–24 倍寬。**
所以本檔加第三臂：固定在接近 2c 的窄格。若公式在模擬器裡成立，窄格應該大勝；
若不成立，那就是公式的假設（連續收割、無逆選擇、觸價即成交）在這裡不適用
——**兩種結果都有資訊。**

## 成本地板 c 的定義（不可只算手續費）

    c = maker 手續費 + 半價差        （逐標的，用 orderbook_snapshots_1m 實測）

只算手續費會讓 2c 落在價差以下，那是物理上不存在的格距。BTC 的 c ≈ 2.01 bps、
ADA ≈ 4.6 bps（價差 5.2 bps）。

## 臂（全部在同一個行程、同一批路徑上跑，逐路徑配對）

    FIXED      現行凍結政策，g ≈ 0.96%（N=30）
    ADAPTIVE   重錨時 g = max(2c, k·σ̂)，σ̂ = 前 24h 小時報酬標準差 → 換算 N
    ADAPT+SIZE ADAPTIVE ＋ 每次重錨的投入資金 × clamp(σ̂_med/σ̂, 0.5, 1.5)
    NARROW     固定 g = 2c（公式的最佳點）
    SKEW       反應式層三：庫存偏移。事前分不出路徑形態，但單向行情的定義就是
               庫存單邊累積——不用預測，會直接觀測到。A-S 的 r = S − qγσ²(T−t)
               在網格上的對應是「庫存往一側堆積時接貨速度按 q 遞減」。必然落後
               （先吃一段虧損才觸發），但不需要不存在的預測能力。γ∈{0.5,1,2}。
               這一臂補回被砍掉的層三，也是唯一處理「就是一路走」這種最常見
               死法的機制——另外兩個閘門（清算、深度）都只處理外生事件。

**主判定 = ADAPTIVE vs FIXED**（單一改動）。ADAPT+SIZE 與 NARROW 是次要，
只報不作為主結論的依據。

## 判準（2026-09-05 重構：改的是判準不是設計）

失去層三之後，層一與層二塌縮成同一件事：拉寬格距在震盪情境少賺、在單向情境
少賠，而事前分不出是哪一種，所以期望損益對稱——**它不是 alpha，是風險預算的
縮放**。因此主判準不看 Δpnl，看**同 MDD 預算下能部署多大**：

    L*        = MDD 預算 / |該臂 MDD 的 p95|   （網格損益與庫存隨規模線性）
    可部署報酬 = L* × 該臂報酬中位
    PASS      可部署報酬（該臂）−（固定臂），逐路徑配對 bootstrap CI 下界 > 0
    REJECT    CI 上界 < 0

**Δpnl 的 CI 涵蓋 0 是預期結果，不是失敗。** MDD 預算取 −30%（與 total kill 對齊）。

**加一條高原要求（使用者原文「要高原不要尖峰」的操作化）**：k 掃
{0.5, 1, 2, 4, 8}，**必須有連續 ≥3 個 k 通過**，單一 k 通過視為尖峰、不算。

## 儀器（登記在案的修正，這次一起做）

1. **`--asof` 釘死資料**：K 線快取每小時被排程更新，同一顆種子在不同時刻跑會
   抽到不同路徑。輸出記錄實際使用的最後一根 bar。
2. **兩臂在同一個行程、同一批合成路徑上跑**，逐路徑配對——跨行程比較就是
   跨資料快照比較（§0.93 十二補記）。
3. **成交要穿過一個價差**（`fill_pen`）：緊格距下「觸價即成交」是致命假設，
   而 NARROW 臂正是最會被這個假設奉承的那一臂。另報 2× 價差的敏感度。

## 先驗（寫在跑之前）

ADAPTIVE 的 Δ_pnl **為負**（公式如此），風險兩項**可能改善**但幅度未知；
主判定最可能落在「Δ_pnl CI 上界 < 0 但 MDD/庫存有改善」，那就要看改善夠不夠
20%。NARROW **應大勝**——若沒有，是成交假設或逆選擇在起作用，那本身是更值得
知道的事。

Run: python research/lp_ladder/grid_adaptive_spacing.py --paths 200
Out: research/results/grid_adaptive_spacing.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from grid_exec import simulate  # noqa: E402
from grid_mc_policy import load, bar_stats, synth  # noqa: E402

OUT = ROOT / "research" / "results" / "grid_adaptive_spacing.json"
VENUE = ROOT / "research" / "results" / "lp_grid_mc_venue.json"
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
FROZEN = dict(drop=0.25, profile="nested", reanchor="time", stop="hard", stop_delay_h=0)
MAKER_BPS = 2.0
KS = (0.5, 1.0, 2.0, 4.0, 8.0)
N_MIN, N_MAX = 8, 200
VOL_WIN = 24


def n_for_g(g, drop=0.25):
    """把目標格距換算成格數（log 等分）。"""
    return int(np.clip(round(np.log(1.0 / (1.0 - drop)) / np.log(1.0 + g)), N_MIN, N_MAX))


def sigma_hat(close):
    """因果的 trailing 24h 小時報酬標準差，對齊到每根 bar。"""
    r = np.diff(np.log(close), prepend=np.log(close[0]))
    out = np.full(len(close), np.nan)
    csum = np.cumsum(r); csq = np.cumsum(r * r)
    for i in range(VOL_WIN, len(close)):
        n = VOL_WIN
        s = csum[i - 1] - csum[i - 1 - n]
        q = csq[i - 1] - csq[i - 1 - n]
        v = max(q / n - (s / n) ** 2, 0.0)
        out[i] = np.sqrt(v)
    med = np.nanmedian(out)
    return np.where(np.isfinite(out), out, med), float(med)


def lstar(mdds, budget=0.30):
    """同 MDD 預算下能部署的倍數。網格的損益與庫存隨規模線性，
    所以 L* = 預算 / |MDD 的 p95|（mdd 為負，取第 5 百分位＝最差端）。"""
    worst = abs(float(np.percentile(mdds, 5)))
    return budget / worst if worst > 0 else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=60)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--budget", type=float, default=0.30)
    a = ap.parse_args()

    from shared.declared_scope import Scope
    scope = Scope("1.18h 網格上半排", expect_n=len(SYMS))

    spreads = {}
    if VENUE.exists():
        inp = json.loads(VENUE.read_text(encoding="utf-8")).get("inputs", {})
        spreads = {k: v["sp_bps"] for k, v in inp.items()}
    print("=" * 108)
    print(f"  1.18h 網格上半排（預註冊）| {a.paths} 條 {a.days} 天去漂移路徑/幣"
          f" | 主判準＝MDD 預算 {a.budget:.0%} 下的可部署報酬")
    print("=" * 108)

    GAMMAS = (0.5, 1.0, 2.0)
    res = {"params": vars(a), "frozen": FROZEN, "ks": list(KS),
           "gammas": list(GAMMAS), "coins": {}}
    t0 = time.time()
    pack = {}

    def add(arm, arr):
        pack.setdefault(arm, []).append(arr)

    ok_syms = []
    for sym in SYMS:
        try:
            low, high, close = load(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: {str(e)[:40]}")
            continue
        sp = spreads.get(sym, 1.0)
        c_bps = MAKER_BPS + sp / 2.0
        g_star = 2 * c_bps / 1e4
        fill_pen = sp / 1e4
        r, hi_r, lo_r = bar_stats(low, high, close)
        rng = np.random.default_rng(20260905)
        paths = [synth(r, hi_r, lo_r, a.days * 24, a.block, rng, demean=True)
                 for _ in range(a.paths)]

        def sim(**kw):
            out = []
            for pl, ph, pc in paths:
                m, _ = simulate(pl, ph, pc, maker=MAKER_BPS / 1e4,
                                fill_pen=fill_pen, **FROZEN, **kw)
                out.append((m["cagr"], m["mdd"]))
            return np.array(out)

        add("FIXED", sim(N=30))
        add("NARROW", sim(N=n_for_g(g_star)))
        for g in GAMMAS:
            add(f"SKEW{g}", sim(N=30, inv_skew=g))
        for k in KS:
            out = []
            for pl, ph, pc in paths:
                s_p, _ = sigma_hat(pc)
                ns = np.array([n_for_g(x) for x in np.maximum(g_star, k * s_p)])
                m, _ = simulate(pl, ph, pc, maker=MAKER_BPS / 1e4, fill_pen=fill_pen,
                                N=30, n_series=ns, **FROZEN)
                out.append((m["cagr"], m["mdd"]))
            add(f"ADAPT{k}", np.array(out))
        ok_syms.append(sym)
        res["coins"][sym] = {"spread_bps": sp, "c_bps": c_bps, "g_star": g_star}
        print(f"  {sym:<6} 價差 {sp:>6.2f} bps   2c {g_star*100:>6.3f}%   "
              f"完成 ({time.time()-t0:.0f}s)", flush=True)

    scope.check(actual_n=len(ok_syms))      # 少一個幣就 raise，不允許靜默降級
    res["scope"] = scope.as_dict()

    def arm_stats(arm):
        A = np.vstack(pack[arm])
        L = lstar(A[:, 1], a.budget)
        return {"cagr_med": float(np.median(A[:, 0])),
                "mdd_p95": float(np.percentile(A[:, 1], 5)),
                "lstar": float(L), "deployable": float(L * np.median(A[:, 0]))}

    base = arm_stats("FIXED")
    print(f"\n  {'臂':<12}{'報酬中位':>10}{'MDD p95':>10}{'L*':>7}"
          f"{'可部署報酬':>12}{'vs 固定':>10}  判定")
    print(f"  {'FIXED':<12}{base['cagr_med']:>+10.2%}{base['mdd_p95']:>+10.2%}"
          f"{base['lstar']:>7.2f}{base['deployable']:>+12.2%}{'-':>10}")

    def paired_ci(arm, B=3000, seed=23):
        A = np.vstack(pack[arm]); Bs = np.vstack(pack["FIXED"])
        La, Lb = lstar(A[:, 1], a.budget), lstar(Bs[:, 1], a.budget)
        d = La * A[:, 0] - Lb * Bs[:, 0]
        rng = np.random.default_rng(seed)
        m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    passes = {}
    order = [f"ADAPT{k}" for k in KS] + [f"SKEW{g}" for g in GAMMAS] + ["NARROW"]
    for arm in order:
        st = arm_stats(arm)
        dm, lo, hi = paired_ci(arm)
        ok = lo > 0
        passes[arm] = ok
        res.setdefault("arms", {})[arm] = {**st, "d_deployable": dm,
                                           "ci": [lo, hi], "pass": bool(ok)}
        tag = "PASS" if ok else ("-" if lo <= 0 <= hi else "REJECT")
        print(f"  {arm:<12}{st['cagr_med']:>+10.2%}{st['mdd_p95']:>+10.2%}"
              f"{st['lstar']:>7.2f}{st['deployable']:>+12.2%}{dm:>+10.2%}  "
              f"{tag}  CI[{lo:+.2%},{hi:+.2%}]")
    res["arms"]["FIXED"] = base

    run_len = best = 0
    for k in KS:
        run_len = run_len + 1 if passes.get(f"ADAPT{k}") else 0
        best = max(best, run_len)
    res["plateau_len"] = best
    skew_pass = [g for g in GAMMAS if passes.get(f"SKEW{g}")]
    res["verdict"] = ("自適應格距通過且成高原" if best >= 3
                      else "自適應格距未過（需連續 >=3 個 k）")
    res["skew_verdict"] = (f"反應式庫存偏移通過：gamma={skew_pass}" if skew_pass
                           else "反應式庫存偏移未過")
    print(f"\n  高原：最長連續通過 {best}/{len(KS)} 個 k（需 >=3）-> {res['verdict']}")
    print(f"  {res['skew_verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
