# -*- coding: utf-8 -*-
"""交會事件·「且」變體 — 凍結定義與前瞻時鐘（註冊書本體）

這個檔案**就是**註冊書。判準寫在會被執行的程式碼裡，不寫在散文裡。

===========================================================================
這條時鐘為什麼從零開始 —— 先讀這一段，它比判準更重要
===========================================================================
現行時鐘（`conj_clock.py`，凍結 2026-09-07）的簽名條件是
**S ∧ (D ∨ V ∨ O)**「至少一個流事件」。2026-09-08 把它的 in-sample 母體
做互斥分解，發現效應**幾乎全部集中在「兩個流事件都開」那一格**：

    全體（現行「或」）  n=2,526  +0.3485  CI [+0.2691,+0.4327]  9/9
    S ∧ D ∧ V（本檔）   n=1,075  +0.6574  CI [+0.5213,+0.7966]  9/9
    只開一個            n=1,451  +0.1196  CI [+0.0397,+0.1997]  8/9

**這個分解不是證據，是開一條新時鐘的理由。** 它是在**已經看過的資料**上
切出來的子集，而「事後挑統計量最好的子集」正是 §0.92 判掉變體 C/D 的那件事
——當時 C/D 的統計量比 B 好看，仍然隨 B 連坐作廢，理由就是子集的好看可能
只是母集內部的運氣切片。

所以本檔的紀律：

    · **證據從零起算**。上面那 1,075 筆 in-sample 事件**不是**本時鐘的
      樣本，永遠不會被計入 n，也不得被引用為「它有效」的證據。
    · **現行時鐘不作廢、不被取代**。兩條並行、各判各的。現行那條測的是
      被稀釋過的版本，若「且」為真它照樣會過——它是保守的，不是錯的。
    · 兩條時鐘**共用同一份偵測與配對機器**（本檔 import
      `conj_clock.frozen_cand` 與 `triage_matched.collect_symbol`），
      差別只有簽名那一個條件。不另寫第二份實作。

疑慮部分降低（未消除）：「且」佔母體 **42.6%**，是四成的一格不是挑出來的
小切片；而且它與「只開一個」那格的差（+0.54）遠大於任一格自己的 CI 寬度。
但那仍然是 in-sample 的觀察——**唯一能解決它的是前瞻樣本**，這就是本檔。

===========================================================================
凍結日 2026-09-08。以下每一項在此日之後不得修改。
===========================================================================
宇宙       core9 = BTC ETH SOL BNB XRP DOGE ADA LINK AVAX（同現行，不得增刪）

事件定義   **與 `conj_clock.py` 逐字相同**（掃單、三個因果門檻的流事件、
           滾動 30 日 p99/p1、每 UTC 日更新、暖機不回退），
           唯一的差別是簽名條件：

               現行  簽名含 S 且含 {D, V, O} 至少一個
               本檔  簽名含 S 且**同時含 D 與 V**

           O（OI 崩落）**不參與本檔的條件**——既不要求也不排除。理由與
           live 路徑一致：OI 的 metrics 粒度是 5 分鐘，結構上壓不進 2 分鐘
           的可執行窗口，而且對事件的 AUC 是 0.4996（零資訊）。

標籤與對照 **與 `conj_clock.py` 逐字相同**（impulse 方向、r_60、同幣同日
           非事件、離事件 >30 分、事前移動 ±20% caliper 最近鄰）。
           判準量 = 配對差（事件 − 對照）。

===========================================================================
判準（寫在 CI 上，不寫在點估計上；mistake.md 2026-09-04）
===========================================================================
    PASS      n >= 200  且  配對差 60m 的日聚類 CI **下緣 > 0**
              且  逐幣為正 >= 6/9
    REJECT    n >= 200  且  CI **上緣 < 0**
    提前止損  任何時候 CI 上緣 < 0 且 n >= 100 -> 可提前判 REJECT。
              **只准往保守方向提前停手。**
    其餘      累積中（INCONCLUSIVE），不得引用為證據

**PASS 不等於可以上線。** 它只回答「效應在前瞻樣本上還在嗎」。可執行性
是另一條線（2 分鐘死線、真實成交價），由 `conj_intents` 那條路自己驗。

===========================================================================
功效（註冊當下就算，且用判決那台機器的 SE，不手算 —— mistake.md 2026-09-06）
===========================================================================
    in-sample SE = 0.0699（n=1,075，日聚類 bootstrap，3000 次重抽）
    事件頻率 ≈ 1.16 / 天（九幣合計；現行「或」是 2.72，本檔約它的 43%）

        n=150（4.3 個月）  SE 0.187  MDE 0.367 = 效應的 56%
        n=200（5.7 個月）  SE 0.162  MDE 0.318 = 效應的 48%   <- 選這個
        n=300（8.6 個月）  SE 0.132  MDE 0.259 = 效應的 39%

    **n_target 的選法（先寫規則再代數字）**：取「MDE <= in-sample 效應的
    50%」的最小整百數。n=150 是 56% 不合格、n=200 是 48% 合格 -> **200**。
    這個規則比現行時鐘自己的餘裕還嚴（它在 n=300 時是 0.259/0.348 = 74%）。
    不是為了縮短等待而選小的：150 也「有測量能力」，但餘裕不到一半。

in-sample 參考值（**不是判準**，只是它該長什麼樣）
    配對差 60m：+0.6574  CI [+0.5213, +0.7966]，9/9 幣為正

已知會讓這個時鐘失效的事（與現行時鐘同）
    · 分鐘 bar 停止更新 -> 事件數凍住。判準是**產物新鮮度**，超過 48 小時
      舊就標 STALE 不出數字。
    · 配對本身有 ~+0.02~0.03 的系統偏差，對事件與對照兩邊一致，不影響
      「CI 下緣是否 > 0」的判定，但引用點估計時要記得扣。

用法
    python research/poc/conj_clock_and.py            # 計分並寫時鐘
    python research/poc/conj_clock_and.py --insample # 用 in-sample 核對參考值
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
sys.path.insert(0, str(HERE.parents[1]))
import event_census as ec  # noqa: E402
import triage_matched as tm  # noqa: E402
import conj_clock as ck  # noqa: E402

OUT = HERE / "data" / "results"
FREEZE_DAY = "2026-09-08"
FREEZE_MS = int(datetime.strptime(FREEZE_DAY, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000)
N_TARGET = 200
N_EARLY_STOP = 100
COINS_MIN = 6
STALE_H = 48.0
REF_INSAMPLE = 0.6574          # in-sample 參考，不是判準


def is_and(sig: str) -> bool:
    """簽名同時含 delta_ext 與 vol_burst（sweep 由 lane 保證）。

    `sig` 是 `triage_matched` 2026-09-08 附加的欄位（"+".join(sorted(sig))）。
    O（oi_crash）既不要求也不排除。
    """
    parts = set(sig.split("+"))
    return {"delta_ext", "vol_burst"} <= parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insample", action="store_true",
                    help="用凍結日之前的資料重跑，核對參考值")
    a = ap.parse_args()

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
        # **同一份偵測、同一套配對**——只 import，不重寫
        cand, ts, cl, at, day = ck.frozen_cand(sym, liq)
        last_ts = max(last_ts, int(ts[-1]))
        keep = (ts < FREEZE_MS) if a.insample else (ts >= FREEZE_MS)
        cand = {k: (v[keep[v]] if len(v) else v) for k, v in cand.items()}
        n0, n1 = tm.collect_symbol(sym, cand, ts, cl, at, rows)
        tried += n0
        matched += n1

    age_h = (datetime.now(timezone.utc).timestamp() * 1000
             - last_ts) / 3_600_000
    asof = datetime.fromtimestamp(last_ts / 1000,
                                  tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    tag = "in-sample" if a.insample else "forward"
    print("=== 交會事件·「且」變體時鐘（%s）===" % tag)
    print(f"凍結日 {FREEZE_DAY}   資料截止 {asof} UTC（{age_h:.1f} 小時前）")
    print("定義：簽名含 S **且同時含 D 與 V**（O 不參與條件）")

    if age_h > STALE_H and not a.insample:
        print(f"\n**STALE-DATA — 資料超過 {STALE_H} 小時未更新，不出判定。**")
        (OUT / "conj_clock_and.json").write_text(json.dumps(
            {"verdict": "STALE-DATA", "asof": asof, "age_h": age_h,
             "freeze_day": FREEZE_DAY}, indent=2), encoding="utf-8")
        return

    d = pd.DataFrame(rows)
    if len(d):
        d = d[(d.lane == "掃單+強制流") & d.sig.map(is_and)]
    n = len(d)
    print(f"「且」事件 {n:,} / {N_TARGET}  （全體配對 {matched:,} / {tried:,}）")

    res = {"freeze_day": FREEZE_DAY, "asof": asof, "age_h": age_h,
           "mode": tag, "n": int(n), "n_target": N_TARGET,
           "definition": "S AND delta_ext AND vol_burst"}

    if n < 30:
        res["verdict"] = "累積中"
        print("\n樣本 < 30，無法計算日聚類 CI。累積中。")
    else:
        x = (d["e60"] - d["c60"]).to_numpy(float)
        days = d["day"].to_numpy()
        m, lo, hi, se = ck._day_ci(x, days)
        pc = d.assign(x=x).groupby("sym").x.mean()
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
        print(f"（in-sample 參考 {REF_INSAMPLE:+.4f} — **不是判準**，"
              f"而且那批樣本永遠不計入 n）")

    print()
    print("提醒：本時鐘的 in-sample 分解是**開時鐘的理由，不是證據**"
          "（§0.92 的 C/D 陷阱）。現行 conj_clock.py 不作廢，兩條並行。")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "conj_clock_and_insample" if a.insample else "conj_clock_and"
    (OUT / f"{stem}.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("written ->", OUT / f"{stem}.json")


if __name__ == "__main__":
    main()
