# -*- coding: utf-8 -*-
"""§0.93 十二 —— 逐幣執行現實進 MC：場館費率、點差、深度、資金費（預註冊 2026-09-05）

十一（X1）用同一個凍結政策跑九個幣，9/9 去漂移中位為正、BTC 排第 7。但那個引擎
的成本是常數（maker 2、停損 10 bps），沒有場館、沒有點差、沒有資金費、沒有容量。
使用者的問題「概念拓展到小幣有沒有肉」——答案的第一關就是 rule #10：逐標的真實
成本。這支把它們放進去，**政策一個參數都不動**。

**逐幣輸入（全部量測，不是假設）**
  點差 sp     orderbook_snapshots_1m（Binance 現貨，2026-07-28 起）該幣平均 (ask−bid)/mid
  深度 L5     同表 bid_depth_usd_l5 平均
  資金費 f    Binance 永續 fundingRate 近 365 天平均（8h），public REST，快取 research/results/funding_cache.json
  AVAX 不在簿口表 → 用一次即時 depth 快照（標「單次快照」）
**場館情境**
  SPOT  maker 10、taker 10（Bitget／Binance 現貨 VIP0）、無資金費
  PERP  maker 2、taker 5、資金費按持倉市值逐小時扣 f/8
  兩者共同：成交要求穿過水位一個點差（fill_pen = sp，佇列位置代理）；
  停損成本 = taker + 5（快市滑價，沿用）+ sp/2 + 衝擊 5 bps × min(停損名目/L5, 1)，
  停損名目取 $500（$1k 本金、半倉）——本金小到衝擊≈0，這一項在小資本下是容量問題不是損益問題。
**容量（只報不判）**  capital_cap = 0.10 × L5 / 最大單格佔比（單格掛單不超過該檔 10%）
**判準（寫死）**
  C1 SPOT 去漂移中位為正 ≥ 5/9
  C2 PERP 去漂移中位為正 ≥ 5/9
  C3 PERP 排名 vs X1 排名 Spearman ≥ 0.5（成本不重排；重排本身是資訊：哪個幣由成本決定）
  C4 容量表：報 capital_cap；任一幣 < $5k 標「零錢容量」
**先驗**  格寬 ≈ 0.96%（−25%／30 格 log 等分），現貨來回 20 bps 吃掉 ~20% 的格利潤
  → C1 過但中位下滑 2–3 個百分點；資金費 alts 年均 +5~15% × 部署 20–40% → 拖 1–4%
  → C2 過、但 DOGE／ADA 這種本來就低的可能翻負；C3 過。容量：SOL/ADA/XRP 六位數，AAVE 級四位數。

Run: python research/lp_ladder/grid_mc_venue.py --paths 100
Out: research/results/lp_grid_mc_venue.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from grid_exec import simulate  # noqa: E402
from grid_mc_policy import load, bar_stats, synth  # noqa: E402
from nested_martingale import nested_alloc  # noqa: E402
from shared.db import get_db_conn  # noqa: E402

OUT = ROOT / "research" / "results" / "lp_grid_mc_venue.json"
FCACHE = ROOT / "research" / "results" / "funding_cache.json"
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
FROZEN = dict(drop=0.25, N=30, profile="nested", reanchor="time", stop="hard", stop_delay_h=0)
VENUES = {"SPOT": dict(maker=10.0, taker=10.0, funding=False),
          "PERP": dict(maker=2.0, taker=5.0, funding=True)}
STOP_NOTIONAL = 500.0


def book_stats():
    conn = get_db_conn(); out = {}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT canonical_symbol s, AVG((ask_l1_price-bid_l1_price)/mid_price*1e4) sp, "
                        "AVG(bid_depth_usd_l5) d5 FROM orderbook_snapshots_1m WHERE ts_ms>1785219925731 GROUP BY 1")
            for r in cur.fetchall():
                out[r["s"].split("-")[0]] = {"sp": float(r["sp"]), "d5": float(r["d5"]), "src": "38d 平均"}
    finally:
        conn.close()
    for sym in SYMS:
        if sym not in out:
            d = requests.get("https://api.binance.com/api/v3/depth", params={"symbol": f"{sym}USDT", "limit": 5}, timeout=10).json()
            bid, ask = float(d["bids"][0][0]), float(d["asks"][0][0])
            out[sym] = {"sp": (ask - bid) / ((ask + bid) / 2) * 1e4,
                        "d5": sum(float(p) * float(q) for p, q in d["bids"][:5]), "src": "單次快照"}
    return out


def funding():
    if FCACHE.exists():
        return json.loads(FCACHE.read_text(encoding="utf-8"))
    out = {}; end = int(time.time() * 1000); start = end - 365 * 86400 * 1000
    for sym in SYMS:
        rates, t = [], start
        while t < end:
            r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                             params={"symbol": f"{sym}USDT", "startTime": t, "limit": 1000}, timeout=15).json()
            if not r:
                break
            rates += [float(x["fundingRate"]) for x in r]
            t = int(r[-1]["fundingTime"]) + 1
            if len(r) < 1000:
                break
        out[sym] = {"mean_8h": float(np.mean(rates)), "n": len(rates)}
    FCACHE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--days", type=int, default=365); ap.add_argument("--block", type=int, default=48)
    a = ap.parse_args()
    bk, fd = book_stats(), funding()
    x1 = json.load(io.open(ROOT / "research/results/lp_grid_mc_xcoin.json", encoding="utf-8"))["coins"]["去漂移"]
    max_share = float(nested_alloc(1.0, 1.5, 5, 6).max())
    print("=" * 110)
    print(f"  §0.93 十二 逐幣執行現實 MC（預註冊）｜{a.paths} 條 {a.days} 天去漂移路徑/幣/場館｜政策凍結不動")
    print("=" * 110)
    print(f"  {'幣':<6}{'點差bps':>8}{'L5深度':>11}{'資金費/年':>10}{'容量cap':>10}  來源")
    inputs = {}
    for s in SYMS:
        b = bk[s]; f = fd[s]["mean_8h"] * 3 * 365
        cap = 0.10 * b["d5"] / max_share
        inputs[s] = {"sp_bps": b["sp"], "l5_usd": b["d5"], "fund_ann": f, "cap_usd": cap, "src": b["src"]}
        print(f"  {s:<6}{b['sp']:>8.2f}{b['d5']:>11,.0f}{f:>+10.1%}{cap:>10,.0f}  {b['src']}")
    res = {"params": vars(a), "inputs": inputs, "venues": {}}
    t0 = time.time()
    for vn, v in VENUES.items():
        print(f"\n  [{vn}: maker {v['maker']:.0f} / taker {v['taker']:.0f}{' / 資金費' if v['funding'] else ''}]"
              f"  {'幣':<6}{'年化中位':>10}{'X1中位':>9}{'p5':>9}{'虧損':>7}{'MDDp95':>9}{'停損成本':>9}")
        res["venues"][vn] = {}
        for s in SYMS:
            low, high, close = load(s); r, hi_r, lo_r = bar_stats(low, high, close)
            sp = inputs[s]["sp_bps"] / 1e4
            stop_cost = (v["taker"] + 5.0 + inputs[s]["sp_bps"] / 2 + 5.0 * min(STOP_NOTIONAL / inputs[s]["l5_usd"], 1.0)) / 1e4
            fund_h = inputs[s]["fund_ann"] / (365 * 24) if v["funding"] else 0.0
            rng = np.random.default_rng(20260904); rets, mdds = [], []
            for _ in range(a.paths):
                pl, ph, pc = synth(r, hi_r, lo_r, a.days * 24, a.block, rng, demean=True)
                m, _ = simulate(pl, ph, pc, maker=v["maker"] / 1e4, stop_cost=stop_cost, fill_pen=sp, fund_hourly=fund_h, **FROZEN)
                rets.append(m["cagr"]); mdds.append(m["mdd"])
            rets, mdds = np.array(rets), np.array(mdds)
            o = {"med": float(np.median(rets)), "p5": float(np.percentile(rets, 5)), "loss": float((rets < 0).mean()),
                 "mdd_p95": float(np.percentile(mdds, 5)), "stop_cost_bps": stop_cost * 1e4}
            res["venues"][vn][s] = o
            print(f"          {s:<6}{o['med']:>+10.2%}{x1[s]['med']:>+9.2%}{o['p5']:>+9.2%}{o['loss']:>7.0%}{o['mdd_p95']:>+9.1%}{o['stop_cost_bps']:>9.1f}", flush=True)
    sp_ = res["venues"]["SPOT"]; pp_ = res["venues"]["PERP"]
    pos_s = [s for s in SYMS if sp_[s]["med"] > 0]; pos_p = [s for s in SYMS if pp_[s]["med"] > 0]
    rk = lambda d: {s: i for i, s in enumerate(sorted(SYMS, key=lambda s: -d[s]["med"]))}
    r1, r2 = rk(x1), rk(pp_); n = len(SYMS)
    rho = 1 - 6 * sum((r1[s] - r2[s]) ** 2 for s in SYMS) / (n * (n * n - 1))
    small = [s for s in SYMS if inputs[s]["cap_usd"] < 5000]
    bars = {"C1 SPOT 去漂移中位為正 ≥5/9": len(pos_s) >= 5, "C2 PERP 去漂移中位為正 ≥5/9": len(pos_p) >= 5,
            "C3 PERP 排名 vs X1 Spearman ≥0.5": rho >= 0.5}
    print(f"\n  SPOT 為正 {len(pos_s)}/9 {pos_s}\n  PERP 為正 {len(pos_p)}/9 {pos_p}\n  PERP 排名 vs X1 ρ = {rho:+.2f}"
          f"\n  排名 PERP：{sorted(SYMS, key=lambda s: -pp_[s]['med'])}\n  零錢容量（cap < $5k）：{small or '無'}")
    for k, b in bars.items():
        print(f"    {'✅' if b else '❌'} {k}")
    res.update({"bars": bars, "rho_vs_x1": rho, "rank_perp": sorted(SYMS, key=lambda s: -pp_[s]["med"]),
                "rank_spot": sorted(SYMS, key=lambda s: -sp_[s]["med"]), "small_capacity": small})
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
