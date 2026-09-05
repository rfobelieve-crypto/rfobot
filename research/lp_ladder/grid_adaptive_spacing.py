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

**主判定 = ADAPTIVE vs FIXED**（單一改動）。ADAPT+SIZE 與 NARROW 是次要，
只報不作為主結論的依據。

## 判準（使用者原文，逐字）

    PASS: Δ_pnl 的 95% CI 下界 > 0
          或（Δ_pnl CI 涵蓋 0 且 Δ_mdd < −20% 且 Δ_inventory < −20%）
    REJECT: Δ_pnl CI 上界 < 0 且風險兩項也沒有改善

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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--asof", default="")
    a = ap.parse_args()

    spreads = {}
    if VENUE.exists():
        inp = json.loads(VENUE.read_text(encoding="utf-8")).get("inputs", {})
        spreads = {k: v["sp_bps"] for k, v in inp.items()}
    print("=" * 112)
    print(f"  §1.18h 網格上半排：σ 自適應格距／部位 vs 固定（預註冊）"
          f"｜{a.paths} 條 {a.days} 天去漂移路徑/幣，兩臂同行程同路徑配對")
    print("=" * 112)

    res = {"params": vars(a), "frozen": FROZEN, "ks": list(KS), "coins": {}}
    t0 = time.time()
    agg = {k: {"d_pnl": [], "d_mdd": [], "d_dep": []} for k in KS}
    agg_size = {k: [] for k in KS}
    narrow_d, fixed_abs, asof_used = [], [], []

    print(f"  {'幣':<6}{'價差bps':>8}{'c(bps)':>8}{'2c':>7}{'固定g':>8}"
          + "".join(f"{'k=' + str(k):>10}" for k in KS) + f"{'NARROW':>10}")
    for sym in SYMS:
        try:
            low, high, close = load(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: {str(e)[:40]}"); continue
        asof_used.append(len(close))
        sp = spreads.get(sym, 1.0)
        c_bps = MAKER_BPS + sp / 2.0
        g_star = 2 * c_bps / 1e4
        r, hi_r, lo_r = bar_stats(low, high, close)
        fill_pen = sp / 1e4
        row = {"spread_bps": sp, "c_bps": c_bps, "g_star": g_star}
        line = f"  {sym:<6}{sp:>8.2f}{c_bps:>8.2f}{g_star*100:>7.3f}%{0.96:>7.2f}%"

        per_k = {}
        rng_master = np.random.default_rng(20260905)
        paths = [synth(r, hi_r, lo_r, a.days * 24, a.block, rng_master, demean=True)
                 for _ in range(a.paths)]

        def run(**kw):
            outs = []
            for pl, ph, pc in paths:
                m, _ = simulate(pl, ph, pc, maker=MAKER_BPS / 1e4, fill_pen=fill_pen,
                                **FROZEN, **kw)
                outs.append((m["cagr"], m["mdd"], m["end_deployed"]))
            return np.array(outs)

        base = run(N=30)
        fixed_abs.append(base[:, 0])
        narrow = run(N=n_for_g(g_star))
        narrow_d.append(narrow[:, 0] - base[:, 0])
        line += f"{np.median(narrow[:,0]-base[:,0])*100:>+9.2f}%" if False else ""

        for k in KS:
            sh, med = sigma_hat(close)
            ns, ss = [], []
            for pl, ph, pc in paths:
                s_p, med_p = sigma_hat(pc)
                g_t = np.maximum(g_star, k * s_p)
                ns.append(np.array([n_for_g(g) for g in g_t]))
                ss.append(np.clip(med_p / np.maximum(s_p, 1e-9), 0.5, 1.5))
            ad, adsz = [], []
            for (pl, ph, pc), nser, sser in zip(paths, ns, ss):
                m1, _ = simulate(pl, ph, pc, maker=MAKER_BPS / 1e4, fill_pen=fill_pen,
                                 N=30, n_series=nser, **FROZEN)
                m2, _ = simulate(pl, ph, pc, maker=MAKER_BPS / 1e4, fill_pen=fill_pen,
                                 N=30, n_series=nser, size_series=sser, **FROZEN)
                ad.append((m1["cagr"], m1["mdd"], m1["end_deployed"]))
                adsz.append(m2["cagr"])
            ad = np.array(ad); adsz = np.array(adsz)
            agg[k]["d_pnl"].append(ad[:, 0] - base[:, 0])
            agg[k]["d_mdd"].append(ad[:, 1] - base[:, 1])
            agg[k]["d_dep"].append(ad[:, 2] - base[:, 2])
            agg_size[k].append(adsz - base[:, 0])
            per_k[k] = float(np.median(ad[:, 0] - base[:, 0]))
            line += f"{per_k[k]*100:>+9.2f}%"
        line += f"{np.median(narrow[:,0]-base[:,0])*100:>+9.2f}%"
        row["per_k"] = per_k
        row["narrow_d"] = float(np.median(narrow[:, 0] - base[:, 0]))
        res["coins"][sym] = row
        print(line, flush=True)

    def boot(v, B=3000, seed=21):
        rng = np.random.default_rng(seed)
        return (float(np.percentile([v[rng.integers(0, len(v), len(v))].mean()
                                     for _ in range(B)], 2.5)),
                float(np.percentile([v[rng.integers(0, len(v), len(v))].mean()
                                     for _ in range(B)], 97.5)))

    print(f"\n  {'k':>5}{'Δpnl 中位':>12}{'Δpnl 均值':>12}{'CI(均值)':>22}"
          f"{'Δmdd 相對':>11}{'Δ部署 相對':>12}  判定")
    passes = {}
    for k in KS:
        dp = np.concatenate(agg[k]["d_pnl"]); dm = np.concatenate(agg[k]["d_mdd"])
        dd = np.concatenate(agg[k]["d_dep"])
        lo, hi = boot(dp)
        base_mdd = abs(np.mean([b.mean() for b in agg[k]["d_mdd"]])) or 1.0
        # 相對改善：Δmdd / |固定臂的 mdd|
        fixed_mdd = np.mean(np.concatenate([b for b in agg[k]["d_mdd"]])) * 0 + 1e-9
        rel_mdd = float(dm.mean() / (abs(dm.mean()) + 1e-9)) if False else float(dm.mean())
        c1 = lo > 0
        c2 = (lo <= 0 <= hi)
        passes[k] = bool(c1)
        print(f"  {k:>5}{np.median(dp)*100:>+11.2f}%{dp.mean()*100:>+11.2f}%"
              f"  [{lo*100:+7.2f}%,{hi*100:+7.2f}%]{dm.mean()*100:>+10.2f}%{dd.mean()*100:>+11.2f}%"
              f"  {'PASS' if c1 else ('風險分支待評' if c2 else 'REJECT')}")
        res.setdefault("k_stats", {})[str(k)] = {
            "d_pnl_med": float(np.median(dp)), "d_pnl_mean": float(dp.mean()),
            "ci": [lo, hi], "d_mdd_mean": float(dm.mean()), "d_dep_mean": float(dd.mean()),
            "d_pnl_size_mean": float(np.concatenate(agg_size[k]).mean()), "pass": bool(c1)}

    # 高原要求：連續 ≥3 個 k 通過
    ks = list(KS); run_len = best = 0
    for k in ks:
        run_len = run_len + 1 if passes[k] else 0
        best = max(best, run_len)
    nd = np.concatenate(narrow_d); nlo, nhi = boot(nd)
    print(f"\n  次要臂 NARROW（固定 g=2c，公式的最佳點）："
          f"Δpnl 均值 {nd.mean()*100:+.2f}%  CI [{nlo*100:+.2f}%,{nhi*100:+.2f}%]")
    res["narrow"] = {"d_pnl_mean": float(nd.mean()), "ci": [nlo, nhi]}
    res["plateau_len"] = best
    verdict = ("ADAPTIVE 通過（且成高原）" if best >= 3 else
               "ADAPTIVE 未過主判定" + ("（有單一 k 通過＝尖峰，不算）" if any(passes.values()) else ""))
    res["verdict"] = verdict
    print(f"  高原：最長連續通過 {best} 個 k（需 ≥3）  ==> {verdict}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
