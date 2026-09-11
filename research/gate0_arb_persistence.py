# -*- coding: utf-8 -*-
"""機會的「年齡」預不預測它的存活（2026-09-11）

===========================================================================
為什麼要測這個
===========================================================================
外部閱讀〈Ultimate Crypto Arbitrage Guide〉：

> 已經存在幾秒的機會，很可能再存在幾秒；剛出現的很可能被人搶走。
> **「如果你在延遲上跟不上，就去交易比較慢的那些機會。」**

這對我們有兩個直接用途：

1. **延遲補償**。我們量到 HL 的反應往返 **231 ms**，在延遲上是絕對劣勢。
   如果「已存活 k 分鐘」真的預測「還會再活著」，那我們可以**只做慢的**，
   用一個我們有的東西（耐心）換一個我們沒有的東西（速度）。
2. **§0.75 新構造裡唯一還沒驗的那一件**。`gate0_arb_unhedged` 的尾部桶
   （未完成損失）我**維持了原值**，因為「只做存活久的機會會降低未完成率」
   是他的主張而我們沒量過。它值 ANTH 從 +0.84 到 +2.71 的差距。

===========================================================================
量法（凍結）
===========================================================================
對每個配對，把「可成交邊際 ≥ 門檻」的連續分鐘視為一個**機會**。
對機會中的每一分鐘，記下它此刻的**年齡**（已經持續幾分鐘）與
**它是否再活過下一分鐘**。然後看：

    P(再活 1 分鐘 | 已存活 k 分鐘)  隨 k 怎麼變

**這是條件存活率，不是存活時間分布**——後者會被長機會主導，
前者才回答「我現在看到一個已經活了 k 分鐘的機會，它還會在嗎」。

對照組（**必須有，否則沒有分辨力**）：把同一條邊際序列**在幣內打亂**之後
重算。打亂會破壞時序結構，所以對照組的條件存活率應該**對 k 平坦**。
若真實資料也平坦 -> 年齡沒有資訊；若真實上升而對照平坦 -> 他的說法成立。

**這支不算損益。** 它只回答「等待有沒有用」。

    python research/gate0_arb_persistence.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import arb_home as AH                    # noqa: E402

OUT = ROOT / "research" / "results" / "gate0_arb_persistence.json"
EDGE_COLS = ("sell_edge_mean_bps", "buy_edge_mean_bps")
THRESH_BPS = 3.0          # 「有機會」的門檻；下面會做敏感度
AGES = (1, 2, 3, 5, 8, 13, 21)
SEED = 20260911


def cond_survival(alive: np.ndarray, ages=AGES):
    """alive 是 0/1 的逐分鐘序列。回傳 P(再活一分鐘 | 已存活 k 分鐘)。"""
    age = 0
    hit = {k: [0, 0] for k in ages}          # k -> [存活次數, 出現次數]
    for i in range(len(alive)):
        if alive[i]:
            age += 1
            nxt = bool(alive[i + 1]) if i + 1 < len(alive) else False
            for k in ages:
                if age == k:
                    hit[k][1] += 1
                    hit[k][0] += int(nxt)
        else:
            age = 0
    return {k: (hit[k][0] / hit[k][1] if hit[k][1] else np.nan) for k in ages}, \
           {k: hit[k][1] for k in ages}


def run_pair(pid, thresh=THRESH_BPS, rng=None):
    f = AH.LOGS / pid / "minutes.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if not set(EDGE_COLS).issubset(d.columns):
        return None
    e = d[list(EDGE_COLS)].max(axis=1).fillna(-1e9).values     # 兩側取較好的
    alive = (e >= thresh).astype(int)
    if alive.sum() < 50:
        return None
    real, n = cond_survival(alive)
    # 對照：幣內打亂，破壞時序結構
    rng = rng or np.random.default_rng(SEED)
    sh = alive.copy()
    rng.shuffle(sh)
    ctrl, _ = cond_survival(sh)
    return dict(minutes=int(len(d)), alive_frac=float(alive.mean()),
                real=real, ctrl=ctrl, n_at_age=n)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    pids = [p.name for p in sorted(AH.LOGS.iterdir())
            if p.is_dir() and (p / "minutes.csv").exists()
            and "delisted" not in p.name]
    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"),
               threshold_bps=THRESH_BPS, ages=list(AGES), pairs={})

    print("=== P(再活 1 分鐘 | 已存活 k 分鐘)，門檻 %.1f bps ===" % THRESH_BPS)
    print("真實 vs **幣內打亂的對照**（對照應該對 k 平坦）\n")
    hdr = "%-10s %6s " % ("配對", "有機會") + " ".join("%7s" % ("k=%d" % k) for k in AGES)
    print(hdr)
    agg_real = {k: [] for k in AGES}
    agg_ctrl = {k: [] for k in AGES}
    for pid in pids:
        r = run_pair(pid)
        if not r:
            continue
        res["pairs"][pid] = r
        print("%-10s %5.1f%% " % (pid, 100 * r["alive_frac"])
              + " ".join(("%7.3f" % r["real"][k]) if np.isfinite(r["real"][k])
                         else "      —" for k in AGES))
        print("%-10s %6s " % ("  對照", "")
              + " ".join(("%7.3f" % r["ctrl"][k]) if np.isfinite(r["ctrl"][k])
                         else "      —" for k in AGES))
        for k in AGES:
            if np.isfinite(r["real"][k]):
                agg_real[k].append(r["real"][k])
            if np.isfinite(r["ctrl"][k]):
                agg_ctrl[k].append(r["ctrl"][k])

    print("\n%-10s %6s " % ("**合計中位**", "")
          + " ".join("%7.3f" % np.median(agg_real[k]) if agg_real[k] else "      —"
                     for k in AGES))
    print("%-10s %6s " % ("  對照中位", "")
          + " ".join("%7.3f" % np.median(agg_ctrl[k]) if agg_ctrl[k] else "      —"
                     for k in AGES))
    res["agg_real_median"] = {str(k): (float(np.median(agg_real[k]))
                                       if agg_real[k] else None) for k in AGES}
    res["agg_ctrl_median"] = {str(k): (float(np.median(agg_ctrl[k]))
                                       if agg_ctrl[k] else None) for k in AGES}

    r1 = res["agg_real_median"].get("1")
    rk = res["agg_real_median"].get(str(AGES[-1]))
    c1 = res["agg_ctrl_median"].get("1")
    print("\n=== 讀法 ===")
    if r1 and rk:
        print("  真實：k=1 時 %.3f -> k=%d 時 %.3f（%s）"
              % (r1, AGES[-1], rk, "**上升，年齡有資訊**" if rk > r1 + 0.05
                 else ("下降" if rk < r1 - 0.05 else "大致平坦")))
    if c1:
        print("  對照：k=1 時 %.3f（打亂之後應該對 k 平坦）" % c1)
    print("\n**本支不算損益。** 它只回答「等待有沒有用」——")
    print("而那決定 §0.75 新構造的尾部桶能不能降，以及我們能不能用耐心換速度。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
