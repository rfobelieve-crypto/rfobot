# -*- coding: utf-8 -*-
"""交會效應在**因果門檻**下還在嗎 —— 凍結定義之前的最後一道關。

問題
    `event_census` 的三個流事件門檻是這樣切的：

        cand["delta_ext"] = np.flatnonzero(ad >= np.nanpercentile(ad, 99))

    `np.nanpercentile(ad, 99)` 算的是**整段歷史**的 p99，**包含未來**。
    用未來的資料決定「今天算不算極端」。這是事件**選擇層**的前視：它不直接
    污染標籤，但它會系統性地改變哪些時刻被選中——在成交量成長的期間，
    全樣本 p99 由後期資料訂定，於是早期「以當時標準算極端」的時刻被漏掉、
    後期被過度選取。而交會格要求**同時**兩三個極端，這個偏差會被放大。

    第二個理由更硬：**這個定義根本沒辦法往前跑**（明天的 p99 是多少？
    不知道）。要開前瞻時鐘，定義必須先變成因果的。

因果版（本檔）
    門檻改成**滾動 30 日分位、每個 UTC 日更新一次、只用嚴格更早的分鐘**：

        threshold(d) = percentile( q[ 第 d-30 日 .. 第 d-1 日 ] )

    暖機：前 30 日沒有門檻，整段不產生事件（不是回退到全樣本門檻——
    回退就等於把前視放回來）。
    `sweep` 不動：凍結引擎的樞紐要確認 10 根之後才成立，本來就是因果的。
    量值 q 由 `event_census.detect_all` 回傳，**不另算一份**。
    配對與計分走 `triage_matched.collect_symbol` / `report`，**同一套**。

判準（跑之前寫死，commit 在看數字之前）
    C1 效應還在嗎（主判準）
        因果門檻下，交會格配對後 **60m** 的日聚類 CI **下緣 > 0**
        -> 效應不是門檻前視，SURVIVES，可以進入凍結與前瞻註冊
        CI 含零 -> **THRESHOLD-LOOKAHEAD**：全樣本門檻版的 +0.3827
        不得引用，前瞻註冊停止
        CI 上緣 < 0 -> 反向，停手查儀器
    C2 幅度不得靠攏到「無差別」
        因果版交會格 60m 配對差必須仍 > 只有掃單那條的兩倍。
        若兩條靠攏，代表交會的特殊性是門檻前視造出來的。
    C3 事件數的變化要報，不得解讀成好壞
        因果門檻必然改變事件數（暖機期損失 + 逐日門檻浮動）。
        全格報告三條路的 n 變化，作為「這是不同母體」的提醒，不作判準。
    C4 已知答案的對照組
        「只有強制流」那條在全樣本門檻下配對後是 +0.0268（配對偏差底）。
        因果版應該仍在同一量級（|差| < 0.05）。若它暴衝，代表因果門檻的
        實作本身有問題，C1/C2 一律不解讀。

**這是儀器關不是判決。** 通過只代表「不是門檻前視」，不代表可交易。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import triage_matched as tm  # noqa: E402

TRAIL_DAYS = 30
MIN_TRAIL_MIN = 10_000          # 暖機下限：訓練窗至少要有這麼多分鐘
PCTL = {"delta_ext": ("ge", 99.0),
        "vol_burst": ("ge", 99.0),
        "oi_crash": ("le", 1.0)}
OUT = HERE / "data" / "results"


def causal_flags(q, day):
    """滾動 30 日、逐日更新、只用嚴格更早分鐘的門檻。"""
    out = {}
    udays, first = np.unique(day, return_index=True)
    starts = np.append(first, len(day))
    for name, (side, p) in PCTL.items():
        x = q.get(name)
        if x is None:
            continue
        hit = np.zeros(len(day), dtype=bool)
        for i, d0 in enumerate(udays):
            j = np.searchsorted(udays, d0 - TRAIL_DAYS, side="left")
            lo, hi = starts[j], starts[i]          # 嚴格早於本日
            if hi - lo < MIN_TRAIL_MIN:
                continue                            # 暖機：不產生事件
            w = x[lo:hi]
            w = w[np.isfinite(w)]
            if len(w) < MIN_TRAIL_MIN // 2:
                continue
            thr = np.percentile(w, p)
            seg = x[starts[i]:starts[i + 1]]
            m = (seg >= thr) if side == "ge" else (seg <= thr)
            hit[starts[i]:starts[i + 1]] = np.nan_to_num(m, nan=False)
        out[name] = np.flatnonzero(hit)
    return out


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = []
    tried = matched = 0
    n_full = {"sweep": 0, "delta_ext": 0, "vol_burst": 0, "oi_crash": 0}
    n_caus = dict(n_full)
    for sym in ec.CORE9:
        cand, ts, cl, at, day, q = ec.detect_all(sym, liq)
        caus = causal_flags(q, day)
        new = {"sweep": cand["sweep"]}                 # sweep 本來就是因果的
        for k in PCTL:
            new[k] = caus.get(k, np.array([], np.int64))
        for k in n_full:
            n_full[k] += len(ec.cooldown_filter(np.sort(cand.get(k, []))))
            n_caus[k] += len(ec.cooldown_filter(np.sort(new.get(k, []))))
        a, b = tm.collect_symbol(sym, new, ts, cl, at, rows)
        tried += a
        matched += b

    print("=== C3 事件數變化（全樣本門檻 -> 因果門檻）===\n")
    print(f"{'類型':11s} {'全樣本':>9s} {'因果':>9s} {'變化':>9s}")
    for k in ("sweep", "delta_ext", "vol_burst", "oi_crash"):
        ch = (n_caus[k] / n_full[k] - 1) * 100 if n_full[k] else float("nan")
        print(f"{k:11s} {n_full[k]:9,d} {n_caus[k]:9,d} {ch:+8.1f}%")
    print("\n（暖機 30 日 + 逐日門檻浮動必然改變事件數。這是提醒不是判準："
          "兩版是不同母體。）\n")

    d = pd.DataFrame(rows)
    res = tm.report(d, tried, matched, OUT / "conj_causal")

    print("\n=== 預註冊判準（因果門檻）===\n")
    L = res.get("lanes", {})
    if "掃單+強制流" in L and "只有掃單" in L:
        c = L["掃單+強制流"]["60"]
        s = L["只有掃單"]["60"]
        v1 = ("SURVIVES" if c["ci"][0] > 0 else
              "**反向 — 停手查儀器**" if c["ci"][1] < 0 else
              "**THRESHOLD-LOOKAHEAD**")
        print(f"C1 交會格 60m 配對差 {c['diff']:+.4f}  "
              f"CI [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]  -> {v1}")
        print(f"   （全樣本門檻版為 +0.3827 [+0.2947,+0.4720]）")
        ok2 = np.isfinite(s["diff"]) and c["diff"] > 2 * abs(s["diff"])
        print(f"\nC2 交會 {c['diff']:+.4f} 需 > 只有掃單 {s['diff']:+.4f} 的兩倍"
              f"  -> {'PASS' if ok2 else '**FAIL — 兩條靠攏**'}")
    if "只有強制流" in L:
        f = L["只有強制流"]["60"]
        ok4 = abs(f["diff"]) < 0.05
        print(f"\nC4 已知答案對照組：只有強制流 60m 配對差 {f['diff']:+.4f}"
              f"（全樣本門檻版 +0.0268，需 |差| < 0.05）"
              f" -> {'PASS' if ok4 else '**FAIL — 因果門檻實作有問題，上面不解讀**'}")


if __name__ == "__main__":
    main()
