# -*- coding: utf-8 -*-
"""λ(清算主導分鐘) vs λ(一般高波動分鐘)——C 鏡像存亡的單一數字（預註冊，2026-09-06）

## 張力

§1.18i 的 13 格全部 λ > 0，**高波動格最毒（+6.80）**。清算級聯期間必然落在
高波動格。所以表面上 13 格的結論是在判 C 鏡像（在叢集下方掛單讓清算引擎打到你）
的死刑。

反駁只有一條路，而且可測：高波動格池化了**所有**高波動分鐘，絕大多數是知情流
推動的——那正是 λ 該高的地方。清算主導的分鐘是它的子集，而清算引擎按定義
α ≈ 0。所以真正該問的是子集 vs 母體：

    Δλ = λ(清算主導分鐘，站在被迫流的對側) − λ(高波動但非清算的分鐘)

## 這支量的是什麼、不是什麼

- 量的是「**被強制流打到的成交，之後 60 分鐘價格怎麼走**」。這就是鏡像掛單成交
  那一刻的 λ。鏡像的設計是持續掛在下方，所以「被迫流來的時候你在」是構造保證，
  不需要預測 t。
- **不是**鏡像的總 EV。總 EV 還要加上「掛著的時候被非清算流打到」那一半
  （λ ≈ +3.1，§1.17），那是第二個計算，這裡先答第一個數字。
- 場館錯配要明講：清算來自 OKX/Bybit 永續，成交量在 Binance 現貨的簿口。
  分鐘尺度兩者價格同步（套利在秒級完成），是代理，不是同一本簿。

## 定義（沿用 §1.18i 的成交機制，一字不改）

  掛單    分鐘 t 開始時，買在 bid_l1(t)／賣在 ask_l1(t)，有效期 60 分鐘
  成交    主規則 bid(t') ≤ p（觸及）；嚴格規則 ask(t') ≤ p（穿過）。**兩種都報**
  markout s × (mid(t'+60) − p)/p × 1e4；λ = −markout
  LIQ     (幣, 分鐘) 內清算單邊 ≥ 80% 且名目 ≥ $50k（OKX+Bybit 合計）
          → **站對側**：SELL 主導（多單被平、被迫賣）→ 我們掛 BUY；BUY 主導 → 掛 SELL
  LIQ-錯側  同一批分鐘、站在被迫流的**同側**（對照：這一組應該最毒）
  HIVOL   trailing 60 分鐘實現波動在該幣前三分位、且該分鐘**不是** LIQ → 雙邊各一張
  ALL     同期所有分鐘、每 3 分鐘雙邊（§1.17 的無條件組，當基準）

## 判準（跑之前寫死）

  n 閘門   LIQ 對側成交 ≥ 50，否則 INCONCLUSIVE，滿 7 天再跑
  M1       Δλ = λ(LIQ 對側) − λ(HIVOL) 的事件 bootstrap 95% CI **上界 < 0**
           （兩種成交規則都要）→ **鏡像活**：它是 13 格結論的例外
  M2       CI 含零 → INCONCLUSIVE（一天的資料最可能落在這裡，先寫明）
  M3       CI 下界 > 0 → **鏡像死**，C 只剩吃單版（衰竭後主動進場，成本結構不同）
  對照      LIQ 錯側的 λ 必須 > LIQ 對側（物理：站在被迫流同側必然更毒），
           否則儀器有問題，M1–M3 一律不採信

**資料限制（誠實寫）**：清算地面真值只有 ~21 小時（2026-09-05 起錄）。
單一天、無法做日區塊 bootstrap，序列相關沒有被處理。**這一輪的任何結論都是
初判**，正式判決要 ≥ 7 天並用日區塊。

**先驗**：對照證明量到 SELL 主導分鐘的分鐘內移動是 −42 bps。若其中哪怕三成
在 60 分鐘內回吐，λ(LIQ 對側) 就會落到負值。**我的先驗是 Δλ < 0 但 CI 含零
（樣本不夠）**。

Run: python research/exit_paths/lambda_liq_vs_hivol.py [--thr 50000]
Out: research/results/lambda_liq_vs_hivol.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

OUT = ROOT / "research" / "results" / "lambda_liq_vs_hivol.json"
SYMS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
        "LINK-USD", "AVAX-USD", "SUI-USD", "UNI-USD", "AAVE-USD"]
EX = "binance"
T_HOLD, H = 60, 60


def load_book(sym, t0, t1):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts_ms, mid_price, bid_l1_price, ask_l1_price FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol=%s AND exchange=%s AND ts_ms BETWEEN %s AND %s ORDER BY ts_ms",
                (sym, EX, t0, t1))
            d = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    if d.empty:
        return None
    d["m"] = (d["ts_ms"] // 60000) * 60000
    d = d.sort_values("ts_ms").groupby("m", as_index=False).last()
    for c in ("mid_price", "bid_l1_price", "ask_l1_price"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    idx = pd.RangeIndex(int(d["m"].min()), int(d["m"].max()) + 60000, 60000)
    return d.set_index("m").reindex(idx)


def load_liq(base):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT FLOOR(ts_event/60000)*60000 m, "
                "SUM(CASE WHEN side='BUY' THEN notional_usd ELSE 0 END) b, "
                "SUM(notional_usd) tot FROM liq_events WHERE symbol=%s GROUP BY 1", (base,))
            rows = cur.fetchall()
            cur.execute("SELECT MIN(ts_event) a, MAX(ts_event) z FROM liq_events")
            span = cur.fetchone()
    finally:
        conn.close()
    return {int(r["m"]): (float(r["b"]), float(r["tot"])) for r in rows}, int(span["a"]), int(span["z"])


def fills(bid, ask, mid, e, s):
    """回傳 (markout_main, markout_strict)，未成交為 nan。"""
    p = bid[e] if s > 0 else ask[e]
    n = len(mid)
    if not np.isfinite(p):
        return np.nan, np.nan
    out = []
    for strict in (False, True):
        k_fill = -1
        for k in range(1, T_HOLD + 1):
            if e + k >= n:
                break
            if s > 0:
                hit = (ask[e + k] <= p) if strict else (bid[e + k] <= p)
            else:
                hit = (bid[e + k] >= p) if strict else (ask[e + k] >= p)
            if hit:
                k_fill = k; break
        if k_fill < 0 or e + k_fill + H >= n or not np.isfinite(mid[e + k_fill + H]):
            out.append(np.nan)
        else:
            out.append(s * (mid[e + k_fill + H] - p) / p * 1e4)
    return out[0], out[1]


def boot_diff(a, b, B=4000, seed=31):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
         for _ in range(B)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=50_000)
    a = ap.parse_args()
    groups = {k: {"main": [], "strict": []} for k in ("LIQ_對側", "LIQ_錯側", "HIVOL", "ALL")}
    per = {}
    print("=" * 100)
    print(f"  λ(清算主導分鐘) vs λ(高波動分鐘)——C 鏡像的單一數字｜LIQ 門檻 ${a.thr:,.0f}、單邊 ≥80%")
    print("=" * 100)
    print(f"  {'幣':<10}{'清算分鐘':>9}{'LIQ對側成交':>12}{'HIVOL成交':>11}{'λ LIQ對側':>11}{'λ HIVOL':>10}")
    span = None
    for sym in SYMS:
        base = sym.split("-")[0]
        liq, t0, t1 = load_liq(base)
        span = (t0, t1)
        d = load_book(sym, t0 - 7 * 86_400_000, t1 + 2 * 3_600_000)   # 前 7 天算波動基線
        if d is None or len(d) < 600:
            continue
        mid = d["mid_price"].values; bid = d["bid_l1_price"].values; ask = d["ask_l1_price"].values
        r = np.diff(np.log(mid), prepend=np.nan)
        vol60 = pd.Series(r).rolling(60, min_periods=30).std().values
        vq = np.nanquantile(vol60, 2 / 3)
        idx_of = {int(m): i for i, m in enumerate(d.index)}
        in_span = [i for m, i in idx_of.items() if t0 <= m <= t1]
        cnt = {"liq": 0, "lo": 0, "hi": 0}
        lam = {"LIQ_對側": [], "HIVOL": []}
        for i in in_span:
            m = int(d.index[i])
            b_usd, tot = liq.get(m, (0.0, 0.0))
            is_liq = tot >= a.thr and (b_usd / tot >= 0.8 or b_usd / tot <= 0.2)
            if is_liq:
                cnt["liq"] += 1
                forced_sell = (b_usd / tot <= 0.2)          # SELL 主導 = 多單被平 = 被迫賣
                right, wrong = (+1, -1) if forced_sell else (-1, +1)
                mo = fills(bid, ask, mid, i, right)
                groups["LIQ_對側"]["main"].append(mo[0]); groups["LIQ_對側"]["strict"].append(mo[1])
                if np.isfinite(mo[0]): lam["LIQ_對側"].append(-mo[0])
                mw = fills(bid, ask, mid, i, wrong)
                groups["LIQ_錯側"]["main"].append(mw[0]); groups["LIQ_錯側"]["strict"].append(mw[1])
            else:
                if np.isfinite(vol60[i]) and vol60[i] > vq:
                    cnt["hi"] += 1
                    for s in (+1, -1):
                        mo = fills(bid, ask, mid, i, s)
                        groups["HIVOL"]["main"].append(mo[0]); groups["HIVOL"]["strict"].append(mo[1])
                        if np.isfinite(mo[0]): lam["HIVOL"].append(-mo[0])
                if i % 3 == 0:
                    for s in (+1, -1):
                        mo = fills(bid, ask, mid, i, s)
                        groups["ALL"]["main"].append(mo[0]); groups["ALL"]["strict"].append(mo[1])
        per[sym] = {"liq_minutes": cnt["liq"], "hivol_minutes": cnt["hi"],
                    "n_liq_fill": len(lam["LIQ_對側"]), "n_hivol_fill": len(lam["HIVOL"]),
                    "lam_liq": float(np.mean(lam["LIQ_對側"])) if lam["LIQ_對側"] else float("nan"),
                    "lam_hivol": float(np.mean(lam["HIVOL"])) if lam["HIVOL"] else float("nan")}
        p = per[sym]
        print(f"  {sym:<10}{cnt['liq']:>9}{p['n_liq_fill']:>12}{p['n_hivol_fill']:>11}"
              f"{p['lam_liq']:>+11.2f}{p['lam_hivol']:>+10.2f}", flush=True)

    hrs = (span[1] - span[0]) / 3.6e6 if span else 0
    print(f"\n  清算地面真值跨度 {hrs:.1f} 小時（單一天，無日區塊——本輪只能是初判）\n")
    res = {"thr": a.thr, "hours": hrs, "per_coin": per, "groups": {}}
    print(f"  {'組':<10}{'n成交(主/嚴)':>14}{'λ 主':>9}{'λ 嚴':>9}")
    for g, v in groups.items():
        mm = np.array([x for x in v["main"] if np.isfinite(x)])
        ss = np.array([x for x in v["strict"] if np.isfinite(x)])
        res["groups"][g] = {"n_main": int(len(mm)), "n_strict": int(len(ss)),
                            "lam_main": float(-mm.mean()) if len(mm) else float("nan"),
                            "lam_strict": float(-ss.mean()) if len(ss) else float("nan")}
        r = res["groups"][g]
        print(f"  {g:<10}{len(mm):>7}/{len(ss):<6}{r['lam_main']:>+9.2f}{r['lam_strict']:>+9.2f}")

    L = groups["LIQ_對側"]; Hh = groups["HIVOL"]; W = groups["LIQ_錯側"]
    n_liq = sum(np.isfinite(L["main"]))
    verdict = "INCONCLUSIVE（n 不足，滿 7 天再跑）"
    if n_liq >= 50:
        out = {}
        for rule in ("main", "strict"):
            la = [-x for x in L[rule] if np.isfinite(x)]; hb = [-x for x in Hh[rule] if np.isfinite(x)]
            dm, lo, hi = boot_diff(la, hb)
            out[rule] = {"d_lambda": dm, "ci": [lo, hi]}
            print(f"\n  Δλ({rule}) = λ(LIQ對側) − λ(HIVOL) = {dm:+.2f} bps  CI [{lo:+.2f},{hi:+.2f}]")
        wrong = np.nanmean([-x for x in W["main"]]) if any(np.isfinite(W["main"])) else np.nan
        right = res["groups"]["LIQ_對側"]["lam_main"]
        ctrl_ok = np.isfinite(wrong) and wrong > right
        print(f"  對照：LIQ 錯側 λ {wrong:+.2f} vs 對側 {right:+.2f} → {'物理正確' if ctrl_ok else '**儀器存疑**'}")
        res["delta"] = out; res["control_ok"] = bool(ctrl_ok)
        if not ctrl_ok:
            verdict = "儀器對照未過，不採信"
        elif all(out[r]["ci"][1] < 0 for r in out):
            verdict = "M1 鏡像活——λ(清算對側) 顯著低於高波動母體，是 13 格的例外"
        elif all(out[r]["ci"][0] > 0 for r in out):
            verdict = "M3 鏡像死——面對清算流的成交不比一般高波動好"
        else:
            verdict = "M2 INCONCLUSIVE（CI 含零）"
    res["verdict"] = verdict; res["n_liq_fills"] = int(n_liq)
    print(f"\n  ==> {verdict}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
