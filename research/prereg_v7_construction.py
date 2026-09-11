# -*- coding: utf-8 -*-
"""§1.11 V7 的構造法三臂比較 —— 預註冊（凍結 2026-09-11，門檻待 SE 算完才填）

===========================================================================
為什麼要測這個
===========================================================================
外部閱讀（`docs/external_reading.md` §6）裡，作者把「直接把特徵變成部位」的
ad-hoc 構造法分成五種，每一種有不同的**內建假設**。其中 **Quantile 法**
（取上下分位）的內建假設是 —— **edge 全在極端值**。

**V7 用的就是 Quantile 法**：rolling 200 根的 top 5% 才開火（`inference.py`）。

而 V7 自己的資料否定那個假設。2026-08-24 的五分位實測（全歷史 n=113）：

    |pred| 最強的 Q1   勝率 54.5%
    |pred| 最弱的 Q5   勝率 64.0%      <- 反的

CLAUDE.md 已經據此寫下「極端預測不是更好的預測，收緊門檻不會提高準度」，
但**沒有人回頭問：那我們為什麼還在用一個假設極端值最好的構造法？**

`docs/common_cause_scan.md` 的假說 3 把它放進一個更大的模式裡：
「我們反覆假設越極端越強，而資料反覆說不是」——V7 五分位、§1.05 止損與爆倉
反向排列、§0.88h 預測反了、§1.14c 方向與預測相反。

===========================================================================
這在測什麼、不在測什麼
===========================================================================
**在測**：同一條預測序列 `pred_return_4h` 之下，三種**構造法**的差異。
**不在測**：V7 的歷史實績。三臂**全部從原始 pred 重新推導**，用同一套凍結規則。

為什麼一定要重推而不是用歷史 tier 標籤：CLAUDE.md §核心原則 7 記了兩個
**語意分界**（`DECODE_EPOCH` 2026-08-12 16:00、`CONFIDENCE_EPOCH` 2026-08-13）
——tier 的定義在那兩天之後換過。用歷史 tier 等於在 A 臂裡混兩種定義，
而 B/C 臂不受影響，比較就壞了。重推讓三臂共用同一條規則。

代價要講清楚：**這個設計回答「構造法哪個好」，不回答「V7 實際賺了多少」。**

===========================================================================
資料與必須排除的那一段
===========================================================================
`indicator_history`，2026-09-11 實測三段：

    model_version              n      窗                       mean        std
    None                     593  2026-04-03 ~ 05-01      -0.069192     0.7722   <- 排除
    2026-05-01T07:42:43     2382  2026-05-01 ~ 08-08      +0.000549     0.001736
    2026-08-08T20:59:07      807  2026-08-08 ~ 09-11      +0.000225     0.001435

**第一段必須排除**：std 0.7722 是其他兩段的 **400 倍**，那是不同的刻度不是
不同的市場（mistake.md：數字跟同一條線既有量級差一個數量級，先當儀器/單位問題）。
用它會讓滾動 z 分數在交界處爆掉。

留下 **3,189 列、2026-05-01 ~ 2026-09-11**，逐小時。
**橫軸重疊**：pred 的 horizon 是 4h 而取樣是 1h，所以相鄰列高度自相關
-> 日聚類 bootstrap 是必須的，不是選配（`harness.boot_days`）。

===========================================================================
三臂（凍結）
===========================================================================
共用：`z = (pred - rolling_mean(W)) / rolling_std(W)`，W = 200（V7 現行解碼窗）。

    A  Quantile（現行）   z 在滾動窗 top 5% -> +1；bottom 5% -> -1；否則 0
    B  Clipped z-score   clip(z, ±3) / 3   -> 部位介於 -1..+1（連續）
    C  最弱格（儀器檢查） |z| 落在滾動窗**最低**五分之一 -> sign(z)；否則 0

**C 臂不是策略候選，是儀器檢查。** 它問的是：
「極端值假設是錯的」到底只是「沒有單調關係」，還是**真的反過來**。
C 打敗 A -> 後者；C 也輸 -> 前者。這一關的設計理由與 mistake.md 2026-09-09
的 E4 相同：預註冊裡要有一關問「它是不是在做我以為它在做的事」。

標的變數（用專案自己的定義，不自創）：
    y = TWAP path return = mean(close[t+1..t+4]) / close[t] - 1
    （CLAUDE.md §核心 target；**不是** close-to-close）

===========================================================================
成本（必跑，且必跑零成本對照）
===========================================================================
每次部位變動付 `TURNOVER_BPS`，以 |Δposition| 計。
**B 臂的換手率天生高很多**（連續部位），所以不含成本的比較對 B 有利 ——
這正是要擋的事。

而且收工前**必跑一次 cost=0 對照**：含成本結果 ≥ 零成本結果 = 成本模型壞了
（mistake.md 2026-07-28，那次 t=8.27 其實是零成本）。

===========================================================================
門檻：刻意留空（與 §1.10 同一條規矩）
===========================================================================
先用**判決那台機器**在**真實標籤**上算 SE（只看變異數、不看臂間均值差，
不算偷看 —— mistake.md 2026-09-06），確認 SE < 門檻，才准填。
手算解析近似在 §1.00 低估了 **9 倍**，不准用。

    python research/prereg_v7_construction.py --se-only     # 先算 SE
    python research/prereg_v7_construction.py               # 門檻填了才跑得動
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.harness import boot_days, day8            # noqa: E402

OUT = ROOT / "research" / "results" / "v7_construction.json"

# --- 凍結的設定 ---------------------------------------------------------
EXCLUDE_MODEL_VERSIONS = (None,)      # std 0.7722 = 400x，不同刻度
WINDOW = 200                          # V7 現行解碼窗
TOP_Q = 0.05                          # A 臂：上下 5%
CLIP = 3.0                            # B 臂：截在 ±3 個標準差
WEAK_Q = 0.20                         # C 臂：|z| 最低五分之一
HORIZON = 4                           # TWAP path return 的根數
TURNOVER_BPS = 13.0                   # CLAUDE.md 記的每筆淨成本量級
SEED = 20260911

# --- 門檻（2026-09-11 凍結，在 --se-only 之後、在看任何臂間均值差之前）------
#
# SE 實測（真實標籤、日聚類、只看變異數不看均值差）：
#
#   指標            比較    2SE        對照
#   淨值(連續)      B-A     6.60e-05   經濟錨(每列 5.47e-05)的 **1.21 倍**
#   淨值(連續)      C-A     1.32e-04   經濟錨的 **2.41 倍**
#   方向命中率      A       5.46 pp    n=385
#   方向命中率      C       3.88 pp    n=582
#   方向命中率      C-A     ~6.7 pp    兩子集不相交，sqrt 合成
#
# 經濟錨怎麼來的：A 臂換手總量 460 -> 約 230 筆來回（1.83 筆/天），
# 每筆用 CLAUDE.md 記錄的 +7.1 bps 淨值 -> 攤到 2,985 列 = 5.471e-05/列。
#
# **結論：這個設計的兩半功效差很多，所以判準分兩半寫。**
#
#   淨值那一半：最小可測差異比 V7 整條線的 edge 還大 -> **測不動**。
#               這就是 §0.86 的病（SE >= 門檻），但這次在**跑之前**被抓到。
#   命中率那一半：可測差異 6.7pp，而 2026-08-24 量到的 Q1 vs Q5 落差是
#               **9.5pp**（54.5% vs 64.0%）-> **有測量能力**。
#
# 這重現了 mistake.md 2026-09-04 的反直覺結論：厚尾會淹掉連續指標，
# 命中率反而更有功效（那次 t 從 2.53 掉到 1.20）。
THRESHOLDS = dict(
    # --- H1：主判準，唯一可以做出判決的那一關 ---
    primary_metric="hit_rate",
    h1_ci_low_gt=0.0,          # 命中率 C - A 的日聚類 CI 下緣 > 0
    h1_min_n_each=300,         # 兩臂各至少 300 個開火列（現況 385 / 582）
    h1_boot_n=4000,
    # H1 過 = **極端值假設是反的**（不只是「沒有單調關係」）。
    # H1 不過 = 停在「沒有單調關係」，不得升級成「反向」。
    # **兩種情況都不是「B 臂可以拿去交易」的結論** —— 見下。

    # --- 淨值：事先宣告測不動 ---
    pnl_inconclusive_by_design=True,
    pnl_mde_over_anchor=dict(B_minus_A=1.21, C_minus_A=2.41),
    # 照跑照報**全格**，但**不得用它的 null 結果宣稱「兩種構造沒有差別」**。
    # 無效判決 != FAIL（§0.86 結案時特地區分過這兩件事）。

    # --- 儀器關（不是判準，是「它有沒有在做我以為它在做的事」）---
    need_cost_control=True,    # 含成本結果 >= 零成本結果 -> 成本模型壞了，整批作廢
    c_arm_is_instrument=True,  # C 是儀器檢查不是策略候選
    # B 臂**不得因為本測試的結果被推上線**。它換手率 0.2135/列 vs A 的 0.1541，
    # 而淨值那一半測不動 -> 我們沒有能力說 B 的經濟性比較好。
    # 本測試唯一能回答的是「極端值假設對不對」。

    # --- 報告紀律 ---
    report_first_half_choice=True,   # 若掃了任何參數，報「只用前半會選到什麼」
)

MIN_SNAPSHOTS = None   # 本節不適用（樣本已固定），保留鍵以免與 §1.10 的檢查混用


def load():
    from shared.db import get_db_conn
    conn = get_db_conn()
    q = ("SELECT dt, close, pred_return_4h, model_version "
         "FROM indicator_history ORDER BY dt")
    d = pd.read_sql(q, conn)
    conn.close()
    d = d[~d.model_version.isin([v for v in EXCLUDE_MODEL_VERSIONS if v is not None])]
    if None in EXCLUDE_MODEL_VERSIONS:
        d = d[d.model_version.notna()]
    d = d.dropna(subset=["close", "pred_return_4h"]).reset_index(drop=True)
    d["dt"] = pd.to_datetime(d["dt"])
    return d


def build(d: pd.DataFrame) -> pd.DataFrame:
    """加上 z、三臂部位、標的變數。**只用嚴格更早的資訊算滾動統計。**"""
    p = d["pred_return_4h"].astype(float)
    mu = p.rolling(WINDOW, min_periods=WINDOW).mean().shift(1)
    sd = p.rolling(WINDOW, min_periods=WINDOW).std().shift(1)
    d = d.assign(z=(p - mu) / sd)

    # A：滾動窗內的分位（同樣 shift(1)，不看當根）
    hi = p.rolling(WINDOW, min_periods=WINDOW).quantile(1 - TOP_Q).shift(1)
    lo = p.rolling(WINDOW, min_periods=WINDOW).quantile(TOP_Q).shift(1)
    d["pos_A"] = np.where(p > hi, 1.0, np.where(p < lo, -1.0, 0.0))

    # B：clipped z-score
    d["pos_B"] = np.clip(d["z"], -CLIP, CLIP) / CLIP

    # C：|z| 落在最低五分之一（儀器檢查）
    az = d["z"].abs()
    weak = az.rolling(WINDOW, min_periods=WINDOW).quantile(WEAK_Q).shift(1)
    d["pos_C"] = np.where(az <= weak, np.sign(d["z"]), 0.0)

    # 標的：TWAP path return（專案自己的 target 定義）
    c = d["close"].astype(float)
    fwd = sum(c.shift(-k) for k in range(1, HORIZON + 1)) / HORIZON / c - 1.0
    d["y"] = fwd

    d["day"] = d["dt"].map(lambda x: day8(int(x.timestamp() * 1000)))
    return d.dropna(subset=["z", "y"]).reset_index(drop=True)


def arm_pnl(d: pd.DataFrame, col: str, cost_bps: float):
    """逐列淨報酬 = 部位 x 標的 - 換手成本。"""
    pos = d[col].astype(float).values
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pos * d["y"].values - turn * cost_bps / 10000.0


def se_only(d: pd.DataFrame):
    """**只報變異數，不報臂間均值差。** 這不算偷看（mistake.md 2026-09-06）。"""
    print("樣本：%d 列、%d 天、%s ~ %s"
          % (len(d), d["day"].nunique(), str(d.dt.min())[:16], str(d.dt.max())[:16]))
    print("\n%-6s %8s %10s %10s" % ("臂", "開火率", "SE(日聚類)", "換手/列"))
    ses = {}
    for a in ("A", "B", "C"):
        v = arm_pnl(d, "pos_" + a, TURNOVER_BPS)
        _, se, _, _ = boot_days(d["day"].values, v, n=4000, seed=SEED)
        fire = float((d["pos_" + a] != 0).mean())
        turn = float(np.abs(np.diff(d["pos_" + a].values, prepend=0.0)).mean())
        ses[a] = se
        print("%-6s %8.1f%% %10.6f %10.4f" % (a, 100 * fire, se, turn))
    # 差的 SE（配對，同一天同一列）
    print("\n%-10s %12s" % ("差", "SE(日聚類)"))
    for x, y in (("B", "A"), ("C", "A")):
        dv = arm_pnl(d, "pos_" + x, TURNOVER_BPS) - arm_pnl(d, "pos_" + y, TURNOVER_BPS)
        _, se, _, _ = boot_days(d["day"].values, dv, n=4000, seed=SEED)
        print("%-10s %12.6f" % (x + " - " + y, se))
        ses[x + "-" + y] = se
    print("\n下一步：把上面的 SE 代進去挑門檻。**SE >= 門檻 -> 這個設計不能做決定**，"
          "\n要回去重新設計（加樣本、換聚合單位），不是把門檻放寬。")
    return ses


def verdict(*_a, **_kw):
    """門檻已凍結（見 THRESHOLDS），計分器待接。

    **凍結的順序是重點**：SE 先算、門檻後寫、計分器最後接。
    這個順序讓「先看結果再寫門檻」在構造上做不到。
    """
    if THRESHOLDS is None:
        raise RuntimeError("門檻還沒凍結。先跑 --se-only。")
    raise NotImplementedError(
        "門檻已凍結於 2026-09-11（commit 見 git log）。計分器待接。\n"
        "接的時候只准實作 THRESHOLDS 裡寫的那些，**不准新增或放寬**。\n"
        "特別注意：淨值那一半已事先宣告 INCONCLUSIVE BY DESIGN，"
        "它的 null 結果不得被讀成『沒有差別』。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--se-only", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    d = build(load())
    if a.se_only:
        ses = se_only(d)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(
            stage="prereg-frozen-se-measured",
            _readme="§1.11 構造法三臂。門檻未凍結時不產生任何判決數字。",
            rows=int(len(d)), days=int(d["day"].nunique()),
            se={k: (None if not np.isfinite(v) else float(v)) for k, v in ses.items()},
            asof=time.strftime("%Y-%m-%d %H:%M:%S")),
            ensure_ascii=False, indent=2), encoding="utf-8")
        print("written -> " + str(OUT))
        return 0
    try:
        verdict()
    except RuntimeError as e:
        print("[預期] 判決被擋下：\n" + str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
