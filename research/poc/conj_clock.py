# -*- coding: utf-8 -*-
"""交會事件 — 凍結定義與前瞻時鐘（註冊書本體）

這個檔案**就是**註冊書。判準寫在會被執行的程式碼裡，不寫在散文裡
（`.claude/rules/factor-research.md` 總則 1）。

===========================================================================
凍結日 2026-09-07。以下每一項在此日之後不得修改；要改就是新註冊、
證據從零重新累積。
===========================================================================

假設
    價格幾何事件（掃單）與強制流事件**同時發生**時，事後 60 分鐘的價格
    延續，顯著大於「同日、同事前移動幅度」的對照分鐘。

    這不是「掃單有 edge」（那條已知很薄：凍結引擎 +0.0709 ATR，
    因果門檻下純掃單格 CI 含零），也不是「強制流有 edge」（單獨發生時
    +0.0208 ≈ 配對偏差底）。假設的對象是**交會本身**。

宇宙（事前凍結，不得增刪）
    core9 = BTC ETH SOL BNB XRP DOGE ADA LINK AVAX

事件定義（全部因果：只用嚴格更早或當下已完成的資料）
    掃單 S     `sweep_core.detect_sweeps` 的掃單 bar，樞紐需前後各 10 根
               確認，掃描從確認之後才開始。t_S = 穿越那一分鐘的收盤。
    主動量極端 D  |delta| 的 5 分鐘後向和 >= **滾動 30 日 p99**
    量能爆發 V    5 分鐘量 / 前 30 日同時段均值 >= **滾動 30 日 p99**
    OI 崩落 O     5 分鐘 OI 變化 <= **滾動 30 日 p1**，且只用
                  create_time <= t - 5min 的列（metrics 的一列不是當下快照）

    滾動門檻**每個 UTC 日更新一次**，訓練窗 = 前 30 日、嚴格早於本日，
    且至少 10,000 分鐘；不足則該日不產生事件（暖機期不回退全樣本門檻——
    回退就是把前視放回來）。

    **清算爆發 L 不納入**：只覆蓋 BTC/ETH、77 天，且 Binance forceOrder
    每秒只推一筆、漏失隨強度惡化。它在 in-sample 只貢獻 2,988 筆交會中的
    約 25 筆。納入一個殘缺的流會讓定義依賴一個修不好的資料源。

事件時刻與簽名
    把 (分鐘, 類型) 併起來排序，相鄰間隔 <= 5 分鐘併為同一時刻；
    錨點 = 群內**最早**那一分鐘；時刻之間 60 分鐘冷卻。
    **交會 = 簽名同時含 S 與至少一個 {D, V, O}。**

標籤（方向只用 t 之前，三個窗不重疊）
    impulse = sign(close(t) - close(t-5m))
    r_60    = impulse x (close(t+60m) - close(t)) / ATR_h14(t)

對照（與 in-sample 完全相同，不另立）
    同幣、**同一 UTC 日**、非任何事件、離任何事件 > 30 分鐘，
    且事前 5 分鐘移動幅度 |dP|/ATR 在 **±20% caliper** 內取最接近的一根。
    對照的方向定義與事件相同。**判準量 = 配對差（事件 - 對照）。**

===========================================================================
判準（寫在 CI 上，不寫在點估計上；mistake.md 2026-09-04）
===========================================================================
    PASS      n >= 300  且  配對差 60m 的日聚類 CI **下緣 > 0**
              且  逐幣為正 >= 6/9
    REJECT    n >= 300  且  CI **上緣 < 0**
    提前止損  任何時候 CI 上緣 < 0 且 n >= 150 -> 可提前判 REJECT。
              **只准往保守方向提前停手**；往有利方向提前停是違規。
    其餘      累積中（INCONCLUSIVE），不得引用為證據

功效（註冊當下就算，不是事後才問）
    in-sample SE = 0.0452（n=2,335，日聚類 bootstrap）
    事件頻率 ≈ 2.5 / 天（九幣合計）
        n=76 （1 個月）  SE≈0.250  MDE 0.49  >  效應 0.35  -> 不可能有答案
        n=150（2 個月）  SE≈0.178  MDE 0.35  ≈  效應       -> 邊緣
        n=300（4 個月）  SE≈0.126  MDE 0.25  <  效應 0.35  -> 有測量能力
    **所以樣本門檻是 300 不是 60。** 註冊 60 筆等於花四個月累積一個
    註定沒有答案的樣本（地形扳機那個病）。

in-sample 參考值（**不是判準**，只是它該長什麼樣）
    因果門檻、配對後 60m：+0.3485  CI [+0.2696, +0.4277]，9/9 幣為正
    純掃單對照組：+0.0388  CI [-0.0080, +0.0835]

已知會讓這個時鐘失效的事
    · 分鐘 bar 或 OI 停止更新 -> 事件數凍住。判準是**產物新鮮度**，
      本檔每次都印資料截止日；超過 48 小時舊就標 STALE 不出數字
      （mistake.md 2026-07-05：資料過期時寧可輸出「無法判定」）。
    · 配對本身有 ~+0.02~0.03 的系統偏差（in-sample 的純強制流對照組
      測到的）。它對事件與對照兩邊一致，不影響 CI 下緣是否 > 0 的判定，
      但引用點估計時要記得扣。

用法
    python research/poc/conj_clock.py            # 計分並寫時鐘
    python research/poc/conj_clock.py --insample # 用 in-sample 重跑一次核對
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import triage_matched as tm  # noqa: E402
import conj_causal as cc  # noqa: E402

FREEZE_DAY = "2026-09-07"
FREEZE_MS = int(datetime.strptime(FREEZE_DAY, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000)
N_TARGET = 300
N_EARLY_STOP = 150
COINS_MIN = 6
FLOW = ("delta_ext", "vol_burst", "oi_crash")     # L 刻意不納入，見檔頭
STALE_H = 48
OUT = HERE / "data" / "results"


def frozen_cand(sym, liq, events_dir=None):
    """凍結的事件偵測：sweep 原樣 + 三個流事件用因果門檻。

    `events_dir`（2026-09-10）：見 `event_census.detect_all`。None = 1h 樞紐。
    """
    cand, ts, cl, at, day, q = ec.detect_all(sym, liq, events_dir)
    caus = cc.causal_flags(q, day)
    out = {"sweep": cand["sweep"]}
    for k in FLOW:
        out[k] = caus.get(k, np.array([], np.int64))
    return out, ts, cl, at, day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insample", action="store_true",
                    help="用凍結日之前的資料重跑，核對參考值")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = []
    tried = matched = 0
    last_ts = 0
    for sym in ec.CORE9:
        cand, ts, cl, at, day = frozen_cand(sym, liq)
        last_ts = max(last_ts, int(ts[-1]))
        # 只保留凍結日之後（或 --insample 時只保留之前）的事件
        keep = (ts < FREEZE_MS) if a.insample else (ts >= FREEZE_MS)
        cand = {k: v[keep[v]] if len(v) else v for k, v in cand.items()}
        n0, n1 = tm.collect_symbol(sym, cand, ts, cl, at, rows)
        tried += n0
        matched += n1

    age_h = (datetime.now(timezone.utc).timestamp() * 1000 - last_ts) / 3_600_000
    asof = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    tag = "in-sample" if a.insample else "forward"
    print(f"=== 交會事件時鐘（{tag}）===")
    print(f"凍結日 {FREEZE_DAY}   資料截止 {asof} UTC（{age_h:.1f} 小時前）")
    if age_h > STALE_H and not a.insample:
        print(f"\n**STALE-DATA — 資料超過 {STALE_H} 小時未更新，不出判定。**")
        print("先跑 fetch_bars.py / bars.py / fetch_oi.py 補資料再重跑。")
        (OUT / "conj_clock.json").write_text(json.dumps(
            {"verdict": "STALE-DATA", "asof": asof, "age_h": age_h,
             "freeze_day": FREEZE_DAY}, indent=2), encoding="utf-8")
        return

    d = pd.DataFrame(rows)
    conj = d[d.lane == "掃單+強制流"] if len(d) else d
    n = len(conj)
    print(f"交會事件 {n:,} / {N_TARGET}  （配對 {matched:,} / {tried:,}）")

    res = {"freeze_day": FREEZE_DAY, "asof": asof, "age_h": age_h,
           "mode": tag, "n": int(n), "n_target": N_TARGET}

    if n < 30:
        res["verdict"] = "累積中"
        print("\n樣本 < 30，無法計算日聚類 CI。累積中。")
    else:
        x = (conj["e60"] - conj["c60"]).to_numpy(float)
        days = conj["day"].to_numpy()
        m, lo, hi, se = _day_ci(x, days)
        pc = conj.assign(x=x).groupby("sym").x.mean()
        pos = int((pc > 0).sum())
        res.update(mean=m, ci=[lo, hi], se=se, coins_pos=pos,
                   coins_total=int(len(pc)))
        print(f"配對差 60m {m:+.4f}   日聚類 CI [{lo:+.4f}, {hi:+.4f}]   "
              f"逐幣為正 {pos}/{len(pc)}")
        if n >= N_TARGET and lo > 0 and pos >= COINS_MIN:
            v = "PASS"
        elif n >= N_TARGET and hi < 0:
            v = "REJECT"
        elif n >= N_EARLY_STOP and hi < 0:
            v = "REJECT（提前止損，保守方向）"
        else:
            v = "累積中"
        res["verdict"] = v
        print(f"判定 -> {v}")
        print(f"（門檻：n >= {N_TARGET} 且 CI 下緣 > 0 且逐幣 >= {COINS_MIN}/9）")
        print(f"（in-sample 參考：+0.3485 [+0.2696, +0.4277]，9/9 幣 — 不是判準）")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "conj_clock_insample" if a.insample else "conj_clock"
    (OUT / f"{stem}.json").write_text(json.dumps(res, indent=2, default=float),
                                      encoding="utf-8")
    print("\nwritten ->", OUT / f"{stem}.json")


def _day_ci(x, days, b=2000):
    rng = np.random.default_rng(20260907)
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    reps = np.empty(b)
    for i in range(b):
        p = rng.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return (float(x.mean()), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)), float(np.std(reps, ddof=1)))


if __name__ == "__main__":
    main()
