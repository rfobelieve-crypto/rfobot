# -*- coding: utf-8 -*-
"""交會事件的分鐘級偵測器（shadow 模式：只記錄，不發訊號）

規格來源 TODO §1.03 / `conj_pipeline_spec.py`：
    可執行窗口 **2 分鐘**（扣成本後 CI 下緣仍 > 0 的最大延遲）。
    現行每小時班車平均延遲 32.5 分，效應只剩 29.5%，扣成本後為負。
    拿掉 `oi_crash`（Binance metrics 粒度就是 5 分鐘，結構上壓不進 2 分鐘）
    之後覆蓋率仍有 **89.8%**，而且效應**更好**（2 分鐘延遲 +0.2278 vs
    +0.2064）—— OI 那一支本來就貼零（+0.0176，扣成本 CI 下緣 −0.21），
    與 `BRIDGE.md` 量到的「淨 OI 對事件 AUC 0.4996」一致。

所以這條管線**只吃 1 分鐘 kline**，不碰 OI。

===========================================================================
設計原則
===========================================================================
1. **不動現有的每小時班車。** 那條班車服務三個時鐘與多個記帳器，它們不需要
   低延遲，動它風險高。本檔是**獨立的第二條線**，只寫自己的表。

2. **偵測邏輯不重寫。** 價位來自 `sweep_core.detect_sweeps`（凍結）、
   門檻來自 `conj_causal.causal_flags`（因果、滾動 30 日）。本檔只做
   「把它們預先算好、每分鐘拿新資料去比對」這件事。
   **同一份偵測，不是第二份實作**（mistake.md 2026-08-26）。

3. **狀態預算，熱路徑只做比較。**
   價位表：每小時重算一次（樞紐要確認 10 根之後才成立，本來就慢）
   門檻表：每日重算一次（滾動 30 日分位，逐日更新）
   熱路徑：抓最後幾根 1 分鐘 K -> 算 5 分鐘後向和 -> 比門檻 -> 比價位

4. **shadow 模式**：偵測到就寫 `conj_events_live`，**不發任何訊號**。
   目的是累積**真實的端到端延遲**——那個數字才是「2 分鐘做不做得到」的
   證據，估算不算數。

5. 每輪自報 `{ok, reason}` 旗標給 freshness board（把「從未開始」翻譯成
   「某個數字不對」，mistake.md 2026-09-01）。

用法（由排程每分鐘呼叫）
    python research/poc/conj_watch.py
    python research/poc/conj_watch.py --rebuild     # 強制重算狀態
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import conj_redef as cr  # noqa: E402

ROOT = HERE.parents[1]
LIVE = HERE / "data" / "live"
BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
FLAG = HERE / "data" / "results" / "conj_watch_last.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
FLOW = ("delta_ext", "vol_burst")        # OI 不納入，見檔頭
W = 5                                    # 事件聚合窗（分鐘）
MERGE_GAP = 5
LEVEL_TTL_S = 3600                       # 價位表存活 1 小時
THR_TTL_S = 86400                        # 門檻表存活 1 天
COOLDOWN = 60                            # 分鐘：與 event_census / event_triage 同
# 熱路徑的回看長度。舊值 40 分鐘**不夠重現離線的冷卻**：離線對每個類型先做
# 60 分冷卻、時刻之間再做 60 分冷卻，所以要看得到至少兩層冷卻鏈才會給出同樣
# 的結果。240 分鐘留了四倍餘裕（Binance limit 上限 1000）。
LOOKBACK_MIN = 240
# 只發最近這幾根裡錨定的時刻。**不可以設成 MERGE_GAP+1**：live 組出來的錨點
# 會比離線早一兩分鐘（流量的冷卻鏈落點略不同），那個值剛好把真事件切掉
# （注入測試 77.8%，漏掉的清一色是「錨點在 n-7 而閘門要求 >= n-6」）。
# 跨輪去重本來就由 `recent_ok` 的 60 分冷卻 + UNIQUE KEY 負責，這裡不需要緊。
EMIT_RECENT = 15

# ── 執行參數（2026-09-08 第 7 次 override，CLAUDE.md 有完整記錄）──
# 本層**只算訂單意圖、不送單**。送單在 jarvis（沿用既有 src/exchange/bg），
# 在這裡再寫一份 Python 下單層是被禁止的——那是第二份實作，而且是下單層。
NOTIONAL_USD = 150.0        # 每筆名目。$300-500 本金、最多 3 筆同時 -> ~1x
MAX_CONCURRENT = 3          # 全域同時持倉上限（模擬用的那一格：2x/3 槽）
MAX_PER_SYMBOL = 1          # 同一幣不重複進場
MAX_DAILY = 8               # 每日意圖上限（事件率 2-3/天，8 是異常煞車）
# ── 出場參數（2026-09-09 重新校準；舊值 1.0 ATR / 60 分是錯的那一組）──
# 1 ATR 停損切掉左尾、60 分持有切掉右尾，兩邊各砍一半。誠實錨點下重跑
# hold x stop 網格，3 ATR / 480 分是唯一讓毛利站得住的那一格，而且持有
# 單調性樣本外 +0.907。**這兩個數字不是可調參數**，改它們要重跑 conj_hold。
STOP_ATR = 3.0              # 停損距離（ATR 倍數）
HOLD_MIN = 480              # 時間出場（分鐘）
# 進場延遲：成立分鐘 + DELAY 根的**開盤**。delay=0 是前視（open(ready) 早於
# 形成這個事件的那根收盤），所以最小可交易延遲是 1，驗證用的是 3。
ENTRY_DELAY_MIN = 3
# 只對 S+D+V 產生意圖。S+V 單獨為負、S+D 樣本薄；驗證過的母體是三者齊發
# 那一格（conj_backtest 的 sigk=="and"）。shadow 表照記全部簽章。
INTENT_SIGNATURE = "S+D+V"
INTENT_TTL_S = 180          # 意圖過期：超過就別送陳舊的單
# ───────────────────── 2026-09-09：意圖層停止產生 ─────────────────────
# `conj_redef.py` 查出這條線的進場定義是前視的：`et.cluster` 的錨點是群內
# **最早**那一分鐘，而交會事件要到最後一個成分到齊（ready）才成立。
#   · 事件成立晚於錨點 60.7%（中位 2 分鐘）
#   · 進場（錨點+2）落在事件成立**之前** 22.3% —— 那一刻事件還不存在
#   · 那 671 筆 +0.5064 -> 誠實後 +0.1094，全體 +0.2286 -> +0.1157
# 改成誠實錨點（ready）後逐格重跑，**沒有任何可交易延遲的淨值 CI 下緣 > 0**
# （delay 1/2/3/5/10 淨 -0.043/-0.027/-0.019/-0.048/-0.060，逐幣 2-3/9）。
#
# 恢復條件是三條，2026-09-09 的進度：
#   (a) 錨點改 ready                     **已做**（見 assemble）
#   (b) 重跑 conj_redef 有一格過閘        **未達成**——誠實錨點下 OOS 的
#       CI 下緣仍是 −0.151，沒有任何延遲格的 CI 下緣 > 0。後來重新校準
#       出場（3 ATR / 480 分）＋ Bitget 返佣 50% 之後，樣本外 S+D+V 是
#       +0.1833、CI [−0.135, +0.549]、9/9 幣、P(edge>0)=84.9%
#       —— **CI 仍然跨零**，所以這一條依原始措辭沒有過。
#   (c) 回頭改 CLAUDE.md 的 override #7   **未做**
#
# ── 2026-09-11 更正：(b) 這一條的措辭本身沒有指定「哪一半」 ──────────────
# 上面那段跟 CLAUDE.md §SDV 的警示框**互相矛盾**：那邊寫「(a) 要查、只滿足
# 了第二條」，這邊寫「(a) 已做、(b) 未達成」。兩份都是半對的，而會矛盾是因為
# (b) 寫的是「重跑 conj_redef 有一格過閘」，卻**沒有說是全期還是樣本外**。
#
# 今天用定案參數重跑（`conj_redef.py --decided --sig and`，D1 跨儀器對照
# +0.3317 vs conj_backtest +0.3310 PASS）：
#
#     delay        淨        淨CI下      幣+
#       1      +0.3652     +0.0979     9/9
#       2      +0.3210     +0.0541     9/9
#       3      +0.3317     +0.0714     9/9      <- 本檔在用的
#       5      +0.2477     +0.0003     9/9      <- 下緣貼零，不是餘裕
#      10      +0.2126     −0.0113     7/9
#
# **conj_redef 沒有樣本外切分——它量的是全期。** 所以：
#   · 依 conj_redef 自己的口徑（全期、日聚類），(b) **已達成**；
#   · 依樣本外那一半（+0.1833、CI [−0.132, +0.550]），**仍然跨零**。
# 照核心原則 9（樣本外放主句），誠實的講法是 **(b) 在樣本外沒有過**，
# 而「有一格過閘」這句話會在全期口徑下被讀成通過 —— 那是措辭的漏洞，
# 不是新證據。**所以 HALT 不動。**
#
# 另一個要記下來的數字：「最大可用 delay = 5 分鐘」這句話靠的是 +0.0003 的
# 下緣。**那不是餘裕，是剛好沒有跨零。** 穩健的是 delay 1–3（下緣 +0.05 以上），
# 而本檔的 ENTRY_DELAY_MIN 已經是 3 —— 不要因為「最大可用是 5」就把它放寬。
#
# 換句話說：真的要開，那是一次**知情的推翻**（我自己寫下的恢復條件沒有
# 達成），必須照 override 儀式寫進 CLAUDE.md，不能靠改這一行悄悄放行。
# CLAUDE.md 核心原則 #10 說明了為什麼「CI 不跨零」是研究的門檻而不是
# 下注的門檻——但那是使用者的決定，不是這個檔案的。
# ── 2026-09-12：使用者開了它。第 8 次 informed override，記在 CLAUDE.md ──
# 使用者原話：「研究端那邊開 CONJ_INTENTS=1……我用戶端那邊處理好了剩這一步」。
# 上面那句「必須照 override 儀式寫進 CLAUDE.md，不能靠改這一行悄悄放行」
# **已經照做**：CLAUDE.md §「SDV 意圖層開啟（2026-09-12，第 8 次 ...）」。
#
# 三件當天查清楚、會影響怎麼讀這個旗標的事：
#
# 1) **它不是 Railway 的環境變數。** 本檔只跑在操作者機器的 Windows 排程
#    `FlowBot_ConjWatch`（run_hidden.vbs -> research/ops/conj_watch.bat），
#    Railway 側沒有任何服務跑偵測器。旗標設在那支 .bat 的 setlocal 裡。
#    在 Railway 設它不會有作用，而且不會有任何東西報錯。
#
# 2) **消費端只有 paper**，而且是寫死的：`../jarvis/public/u.html:2640`
#    有刻意的註解說明為什麼不吃 uiMode，`tenants.js:608` 也是 mode:'paper'。
#    所以這次沒有違反「策略 #2 不得進 executor」—— paper 不是 executor。
#    要變真錢必須改程式碼 + 另開一次 override。
#
# 3) **寫入路徑在開它的那天之前從來沒有被執行過**（`conj_intents` 0 列，
#    註冊於 09-08）。而 `/public/conj-signals` 回 count=0 在這裡是**合法狀態**，
#    所以寫入若壞了不會有任何東西變紅，而事件率只有 2-3/天。
#    當天用本檔的 `make_intent` + `intent_gate` + 熱路徑那條 INSERT 實證過：
#    16 欄全部落地、閘門放行、**端點正確濾掉刻意設成過期的測試列**（反向證明
#    產品端不會收到假訊號），測試列已刪。腳本：scratchpad/prove_intent_write.py。
#
# **維持不變的**：ENTRY_DELAY_MIN = 3（「最大可用 5 分鐘」靠的是 +0.0003 的
# CI 下緣，那不是餘裕）、九幣等權不挑幣、**paper 的損益不得當 edge 證據**。
# 設 CONJ_INTENTS=1 可強制開啟（2026-09-12 起：操作者已開，見上）。
INTENTS_ENABLED = os.environ.get("CONJ_INTENTS", "") == "1"
UA = {"User-Agent": "conj-watch/1.0"}


# ───────────────────────── 狀態層（預算） ─────────────────────────
def pivot_table(sym):
    """所有樞紐 + 它們**第一次被穿越**的小時索引（未被穿越 = n）。

    2026-09-08 加：`levels_asof` 每個時間切點都要重掃一次樞紐，對照測試
    要跑幾百個切點時是 O(切點 x n)。樞紐條件與「第一次被穿越」都跟切點
    無關，只算一次就好；之後任何切點只是兩個比較：
        conf < T  且  first_pierce >= T   ->  在 T 時還活著
    這與 `sweep_core.detect_sweeps` 是同一組條件（同一份偵測）。
    回傳 (ts_ms, conf, level, is_hi, first_pierce)，全部是 numpy 陣列。
    """
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None
    bars = sc.load_csv(str(p))
    atr = sc.atr14(bars)
    ts = np.array([b[0] for b in bars], np.int64)
    if ts.max() < 1e12:                 # 小時 K 是秒，分鐘 K 是毫秒
        ts = ts * 1000
    h = np.array([b[sc.H] for b in bars], float)
    l = np.array([b[sc.L] for b in bars], float)
    n = len(bars)
    conf_, lvl_, ishi_, fp_ = [], [], [], []
    for i in range(sc.PIVOT, n - sc.PIVOT):
        conf = i + sc.PIVOT
        if conf >= n or atr[conf] is None:
            continue
        sh, sl = h[i - sc.PIVOT:conf + 1], l[i - sc.PIVOT:conf + 1]
        for is_hi, lvl, piv in (
                (True, h[i], h[i] >= sh.max() and (sh < h[i]).any()),
                (False, l[i], l[i] <= sl.min() and (sl > l[i]).any())):
            if not piv:
                continue
            w = (h[conf + 1:] > lvl) if is_hi else (l[conf + 1:] < lvl)
            k = np.flatnonzero(w)
            conf_.append(conf)
            lvl_.append(float(lvl))
            ishi_.append(bool(is_hi))
            fp_.append(conf + 1 + int(k[0]) if len(k) else n)
    return (ts, np.array(conf_, np.int64), np.array(lvl_, float),
            np.array(ishi_, bool), np.array(fp_, np.int64), n,
            float(atr[n - 1] or 0.0))


def levels_asof(sym, hi_ts=None):
    """某個幣在某個時點「還沒被穿越」的價位表。

    `hi_ts=None` 就是現在（熱路徑用）。帶時間切點是為了讓注入測試能重播
    歷史——**用今天的價位表去重播 20 天前的事件，結構上永遠找不到**，因為
    當時那個事件掃掉的價位今天已經被標記為穿越過了。第一版注入測試就是這樣
    命中率 2.8%，看起來像發射路徑壞了，其實是儀器拿錯了時點
    （[[mistake 2026-09-03 bar 的不同欄位屬於不同時刻]] 的同族：
    同一張表在不同時點內容不同，而表名把這件事藏起來了）。
    """
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return []
    bars = sc.load_csv(str(p))
    atr = sc.atr14(bars)
    ts = np.array([b[0] for b in bars], np.int64)
    # 小時 K 快取的時間戳是**秒**，分鐘 K parquet 是**毫秒**。不轉換的話
    # searchsorted(秒, 毫秒) 永遠回傳結尾 —— 切點靜默失效、每個時點都拿到
    # 同一張表（注入測試三個不同錨點拿到一模一樣的 46 個價位，那就是徵兆）。
    # 照 mistake.md 2026-04-12：時間戳 unit 一律偵測，不硬編碼。
    if ts.max() < 1e12:
        ts = ts * 1000
    h = np.array([b[sc.H] for b in bars], float)
    l = np.array([b[sc.L] for b in bars], float)
    T = len(bars) if hi_ts is None else int(np.searchsorted(ts, int(hi_ts)))
    if T <= 2 * sc.PIVOT + 1:
        return []
    # 反向累積極值：cmax[j] = max(h[j:T])，用來 O(1) 判「確認之後有沒有被穿越」
    cmax = np.concatenate([np.maximum.accumulate(h[:T][::-1])[::-1], [-np.inf]])
    cmin = np.concatenate([np.minimum.accumulate(l[:T][::-1])[::-1], [np.inf]])
    a_now = float(atr[T - 1] or 0.0)
    out = []
    for i in range(sc.PIVOT, T - sc.PIVOT):
        conf = i + sc.PIVOT
        if conf >= T or atr[conf] is None:
            continue
        sh, sl = h[i - sc.PIVOT:conf + 1], l[i - sc.PIVOT:conf + 1]
        hi_piv = h[i] >= sh.max() and (sh < h[i]).any()
        lo_piv = l[i] <= sl.min() and (sl > l[i]).any()
        for is_hi, lvl, piv in ((True, h[i], hi_piv), (False, l[i], lo_piv)):
            if not piv:
                continue
            if (cmax[conf + 1] > lvl) if is_hi else (cmin[conf + 1] < lvl):
                continue                     # 確認之後已經被穿越 = 不算活著
            out.append(dict(sym=sym, level=float(lvl),
                            kind="buy" if is_hi else "sell",
                            atr=a_now, origin_ts=int(ts[i])))
    return out


def build_levels():
    """每個幣的未消耗價位表：呼叫 `levels_asof`，不另寫一份。"""
    out = []
    for sym in CORE9:
        out.extend(levels_asof(sym))
    d = pd.DataFrame(out)
    LIVE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(LIVE / "levels.parquet", index=False)
    return d


class Thr:
    """一個幣在某個時點的門檻（欄位名與 thresholds.parquet 的列一致）。"""

    def __init__(self, sym, thr_delta, thr_vol, vol_base_tod, atr):
        self.sym, self.thr_delta, self.thr_vol = sym, thr_delta, thr_vol
        self.vol_base_tod, self.atr = vol_base_tod, atr


def thresholds_asof(ts, vol, delta, hi, sym="", atr=0.0, days=30):
    """用 [hi-30日, hi) 這個窗算門檻 —— **live 每天重算一次做的就是這件事**。

    抽出來是為了讓對照測試能逐日模擬 live 的行為。第一版對照拿一個固定的
    「最近 30 天」門檻去比 30 天前的事件，那分不出「實作有 bug」和「門檻窗
    依設計不同」——同一個病我修過一次又留了殘留，這次連根拔掉：兩邊都用
    「該事件當天往前 30 天」，任何不一致就只可能來自實作。
    """
    n = len(ts)

    def back5(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(n)
        return c[i + 1] - c[np.clip(i + 1 - W, 0, n)]

    ad = back5(np.abs(delta))
    v5 = back5(vol)
    lo = hi - days * 86_400_000
    m = (ts >= lo) & (ts < hi)
    if m.sum() < 10_000:
        return None
    tod = ((ts // 60_000) % 1440).astype(int)
    s = np.bincount(tod[m], weights=v5[m], minlength=1440)
    c = np.bincount(tod[m], minlength=1440)
    bt = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    bs = bt[tod]
    vs = np.where(np.isfinite(bs) & (bs > 0), v5 / np.where(bs > 0, bs, 1), np.nan)
    return Thr(sym,
               float(np.nanpercentile(ad[m], 99)),
               float(np.nanpercentile(vs[m], 99)),
               bt.tolist(), float(atr))


def build_thresholds():
    """滾動 30 日 p99 門檻：呼叫 `thresholds_asof`，不另寫一份。"""
    rows = []
    for sym in CORE9:
        p = BARS / f"{sym}.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p, columns=["ts", "volume", "delta", "atr_h14"])
        ts = b["ts"].to_numpy(np.int64)
        r = thresholds_asof(
            ts,
            np.nan_to_num(b["volume"].to_numpy(float), nan=0.0),
            np.nan_to_num(b["delta"].to_numpy(float), nan=0.0),
            # 切在**今天零點**，與 causal_flags 的「嚴格早於本日」一致，
            # 也與對照測試 B 臂模擬的行為一致 —— 被測的就是被部署的。
            # 日界自 2026-09-10 起是 UTC+8（ec.DAY_OFFSET_MS），這裡必須
            # 用同一個位移，否則 live 與離線會用不同的日切，安靜地不同意。
            hi=(int((ts[-1] + ec.DAY_OFFSET_MS) // 86_400_000) * 86_400_000
                - ec.DAY_OFFSET_MS), sym=sym,
            atr=float(b["atr_h14"].to_numpy(float)[-1]))
        if r is None:
            continue
        rows.append(dict(sym=r.sym, thr_delta=r.thr_delta, thr_vol=r.thr_vol,
                         vol_base_tod=r.vol_base_tod, atr=r.atr))
    d = pd.DataFrame(rows)
    LIVE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(LIVE / "thresholds.parquet", index=False)
    return d


def load_state(rebuild=False):
    now = time.time()
    lv = LIVE / "levels.parquet"
    th = LIVE / "thresholds.parquet"
    levels = (build_levels() if rebuild or not lv.exists()
              or now - lv.stat().st_mtime > LEVEL_TTL_S
              else pd.read_parquet(lv))
    thr = (build_thresholds() if rebuild or not th.exists()
           or now - th.stat().st_mtime > THR_TTL_S
           else pd.read_parquet(th))
    return levels, thr


# ───────────────────────── 組裝（與離線共用同一顆） ─────────────────────────
def assemble(cand_like):
    """把 {類型: 分鐘索引} 組成交會時刻 —— **共用 conj_redef 的那顆分群**。

    2026-09-09：錨點從「群內最早那一分鐘」改成 **ready = max(第一根掃單,
    第一根流量)**，也就是**事件真正成立**的那一分鐘。

    `et.cluster` 的最早錨點是事件研究的慣例（事件窗從事件開始算），拿它
    當交易訊號時刻是前視：流量在 t、掃單在 t+3 時，錨點是 t，而 t 那一刻
    交會事件還不存在。離線量到事件成立晚於錨點 60.7%、進場落在成立之前
    22.3%，那 22% 的毛利 +0.5064 誠實後只剩 +0.1094 —— **一半的 edge 是
    這個錨點造出來的**。

    live 這條線同樣受害，而且更直接：它會在事件成立之前就寫一列
    `conj_events_live`，並用一個當時還不存在的價格算意圖。
    `cr.groups_with_members` 逐行照抄 `et.cluster`，只是連群成員一起回傳
    —— 那正是算 ready 需要的東西（同一份偵測，不是第二份實作）。

    2026-09-07 的對照測試（`conj_watch_parity.py`）在門檻對齊之後仍然只有
    55.6% 重疊，因為**組裝規則是第二份實作**：

        離線   各類型先自己 60 分冷卻 -> 相鄰 <=5 分併成一個時刻
               -> 錨點取該群最早那一分鐘 -> 時刻之間再 60 分冷卻
        舊 live 要求掃單與流量落在**同一分鐘**，且完全沒有冷卻

    於是 live 漏掉「掃單在 t、放量在 t+3」那一類（離線獨有），又把離線已被
    冷卻吃掉的重複算成新事件（live 獨有）。現在兩邊呼叫同一個 `et.cluster`，
    任何不一致只可能來自門檻或資料，不可能來自組裝。

    注意錨點可能比當下早最多 MERGE_GAP 分鐘（先掃單、後放量）。那是**真實
    延遲的一部分**，shadow 模式要如實記錄，不可用當下時刻假裝沒有。
    """
    pairs = []
    for nm in ("sweep",) + FLOW:
        v = cand_like.get(nm)
        if v is None or len(v) == 0:
            continue
        for m in ec.cooldown_filter(np.sort(np.asarray(v, np.int64))):
            pairs.append((int(m), nm))
    flowm = set(FLOW)
    out = []
    for _a, mem in cr.groups_with_members(pairs):
        s = {x for _, x in mem}
        if "sweep" not in s or not (s & flowm):
            continue
        sw0 = min(m for m, x in mem if x == "sweep")
        ready = max(sw0, min(m for m, x in mem if x in flowm))
        # sw0 一起回傳：被掃的價位掛在**掃單那一分鐘**上，而 ready 可能是
        # 流量那一分鐘。不帶著它，`lvl_at.get(ready)` 會拿到 0.0 —— 一個
        # 靜默的空價位（欄位有預設值的那種病）。
        out.append((ready, s, sw0))
    return out


def flow_flags(ts, vol, delta, t):
    """流量旗標：與 `event_census.detect_all` 同一份量值定義。"""
    n = len(ts)

    def back5(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(n)
        return c[i + 1] - c[np.clip(i + 1 - W, 0, n)]

    d5 = back5(np.abs(delta))
    v5 = back5(vol)
    bt = np.asarray(t.vol_base_tod, float)
    bs = bt[((ts // 60_000) % 1440).astype(int)]
    ratio = np.where(np.isfinite(bs) & (bs > 0), v5 / np.where(bs > 0, bs, 1), 0.0)
    return {"delta_ext": np.flatnonzero(d5 >= t.thr_delta),
            "vol_burst": np.flatnonzero(ratio >= t.thr_vol)}


# ───────────────────────── 熱路徑 ─────────────────────────
def fetch_recent(sym, limit=LOOKBACK_MIN):
    url = (f"https://api.binance.com/api/v3/klines?symbol={sym}USDT"
           f"&interval=1m&limit={limit}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS conj_intents (
            id            BIGINT AUTO_INCREMENT PRIMARY KEY,
            canonical_symbol VARCHAR(20) NOT NULL,
            anchor_ts     BIGINT      NOT NULL,
            intent_ts     BIGINT      NOT NULL,
            expires_ts    BIGINT      NOT NULL,
            side          VARCHAR(6)  NOT NULL,
            ref_price     DECIMAL(24,8) NOT NULL,
            stop_price    DECIMAL(24,8) NOT NULL,
            exit_ts       BIGINT      NOT NULL,
            atr           DECIMAL(24,8) NOT NULL,
            notional_usd  DECIMAL(18,4) NOT NULL,
            size_base     DECIMAL(28,10) NOT NULL,
            signature     VARCHAR(40) NOT NULL,
            status        VARCHAR(12) NOT NULL DEFAULT 'NEW',
            stop_dist     DECIMAL(24,8) NULL,
            hold_ms       BIGINT NULL,
            intent_id     VARCHAR(48) NULL,
            UNIQUE KEY uniq_intent (canonical_symbol, anchor_ts),
            INDEX idx_status (status, intent_ts)
        )
        """)
        # 2026-09-08 實盤體檢加的三欄。CREATE TABLE IF NOT EXISTS 不會改既有表，
        # 所以逐欄 ALTER；欄已存在（errno 1060）就略過，其他錯照拋。
        for ddl in ("ALTER TABLE conj_intents ADD COLUMN stop_dist DECIMAL(24,8) NULL",
                    "ALTER TABLE conj_intents ADD COLUMN hold_ms BIGINT NULL",
                    "ALTER TABLE conj_intents ADD COLUMN intent_id VARCHAR(48) NULL"):
            try:
                cur.execute(ddl)
            except Exception as ex:  # noqa: BLE001
                if getattr(ex, "args", [None])[0] != 1060:
                    raise
        # 成交回報表。agent 也會建同一張（它的 agent_* 命名空間），這裡建是
        # 為了 intent_gate 的 LEFT JOIN 在 agent 從沒建過它之前也不會炸。
        # 兩邊 DDL 必須逐字相同（queries._CONJ_FILLS_DDL）。
        cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_conj_fills (
            id            BIGINT AUTO_INCREMENT PRIMARY KEY,
            intent_id     VARCHAR(48) NOT NULL,
            canonical_symbol VARCHAR(20) NOT NULL,
            side          VARCHAR(6)  NOT NULL,
            fill_price    DECIMAL(24,8) NOT NULL,
            fill_qty      DECIMAL(28,10) NOT NULL,
            sent_ts       BIGINT NOT NULL,
            fill_ts       BIGINT NOT NULL,
            venue         VARCHAR(12) NOT NULL,
            order_id      VARCHAR(64) NULL,
            reported_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_fill (intent_id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS conj_events_live (
            id            BIGINT AUTO_INCREMENT PRIMARY KEY,
            canonical_symbol VARCHAR(20) NOT NULL,
            event_ts      BIGINT      NOT NULL,
            detected_ts   BIGINT      NOT NULL,
            latency_ms    BIGINT      NOT NULL,
            signature     VARCHAR(40) NOT NULL,
            direction     TINYINT     NOT NULL,
            level         DECIMAL(24,8) NOT NULL,
            px            DECIMAL(24,8) NOT NULL,
            atr           DECIMAL(24,8) NOT NULL,
            UNIQUE KEY uniq_evt (canonical_symbol, event_ts),
            INDEX idx_ts (event_ts)
        )
        """)
    conn.commit()


def atr_h14_now(sym, hi_ts=None):
    """此刻的 atr_h14 —— 與 `bars.atr_hourly_wilder` **同一個配方**算在小時 K 快取上。

    2026-09-08 實盤體檢抓到：意圖的停損距離用的是 thresholds.parquet 的 ATR
    （每日重建一次），而研究驗證用的是逐分鐘的 atr_h14。實測差 2-8%、而
    24 小時內分鐘 ATR 本身波動 17-31% —— 停損距離會偏 ±10-30%。
    Wilder EWM(adjust=False) 依賴長歷史，240 分鐘的 live 視窗算不出來；
    小時 K 快取有 22k 根、每小時更新，正好是它的原料。shift(1) 保證只用
    **已收盤**的小時（與 bars.py 的「嚴格早於 t」語意一致）。
    已知答案對照：對 parquet 最後一根 atr_h14 誤差 < 1%（見 conj_intent_check）。
    """
    import bars as _bars
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None
    b = sc.load_csv(str(p))
    if hi_ts is not None:
        # 已知答案對照用：截到「含 hi_ts 的那個小時」為止，與 parquet 的
        # atr_h14 在同一時點比。小時 K 快取的時間戳是秒（mistake.md 2026-04-12）。
        cts = np.array([(x[0] * 1000 if x[0] < 1e12 else x[0]) for x in b],
                       np.int64)
        cut = int(np.searchsorted(cts, (int(hi_ts) // 3_600_000) * 3_600_000 + 1))
        b = b[:cut]
    h = np.array([x[sc.H] for x in b], float)
    l = np.array([x[sc.L] for x in b], float)
    c = np.array([x[sc.C] for x in b], float)
    tr = _bars.true_range(h, l, c)
    a = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    a = a.shift(1)                       # 小時 H 的 ATR 在 H 收盤才知道
    v = float(a.iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def detect_window(sym, ts, high, low, close, vol, delta, t, g, det,
                  atr_override=None):
    """一個幣、一段分鐘 K -> 要寫進 conj_events_live 的列。

    抽出來是為了**能被注入已知答案驗證**。交會事件約每幣每 2.5 天一次，
    所以正常跑一輪的結果永遠是 events=0 —— 那正是
    [[mistake 2026-08-26 只在清單非空時才看得見的輸出]] 的形狀：
    唯一看得見這條路徑的時刻，正是沒人在看的時刻。
    `conj_watch_inject.py` 拿歷史上真實發生過的交會時刻餵進來，要求它發得出。
    """
    if g is None or not len(g):
        return []
    n = len(ts)
    flags = flow_flags(ts, vol, delta, t)
    # 掃單：哪幾分鐘穿過了還活著的價位
    lb = g[g.kind == "buy"].level.to_numpy(float)
    ls = g[g.kind == "sell"].level.to_numpy(float)
    # **穿越測試，不是水準測試**（2026-09-07 由 conj_watch_inject 抓到）。
    # 第一版寫 `lb[lb < high[j]]` —— 任何低於現價的活價位都算命中，於是
    # **每一根 K 都是掃單**，60 分冷卻只留下窗內第 0/60/120/180 根，真正
    # 事件那根反而被吃掉（注入命中率 2.8%）。這跟 §1.02 判掉掃單主線的
    # 那一行是同一個錯（`l[f] <= lvl` 量的是「價位在哪」不是「有沒有穿過去」），
    # 我在新程式碼裡又寫了一次。
    # 正確：這一根的極值越過價位，而**前面每一根都還沒越過**（=第一次穿越）。
    rmax = np.maximum.accumulate(high)
    rmin = np.minimum.accumulate(low)
    sw, lvl_at = [], {}
    for j in range(n):
        pmax = rmax[j - 1] if j else -np.inf
        pmin = rmin[j - 1] if j else np.inf
        cb = lb[(lb < high[j]) & (lb >= pmax)]
        cs = ls[(ls > low[j]) & (ls <= pmin)]
        if len(cb) or len(cs):
            sw.append(j)
            lvl_at[j] = float(cb.max() if len(cb) else cs.min())
    flags["sweep"] = np.array(sw, np.int64)

    out = []
    for a, s, sw0 in assemble(flags):
        if a < W or a < n - EMIT_RECENT:
            continue                    # 只發最近幾根，舊的上一輪發過
        sig = "S+" + "+".join(
            [x for x, k in (("D", "delta_ext"), ("V", "vol_burst")) if k in s])
        # 方向與 conj_backtest 的 A 臂同一條式子，只是錨在 ready 不是 earliest。
        imp = 1 if close[a] > close[a - W] else -1
        out.append((sym + "-USD", int(ts[a]), det,
                    det - (int(ts[a]) + 60_000), sig, imp,
                    lvl_at.get(sw0, 0.0), float(close[a]),
                    float(atr_override) if atr_override else float(t.atr)))
    return out


def make_intent(e, det):
    """事件 -> 完整訂單意圖。**這裡不送單**，只把參數算齊。

    送不出你沒算過的單：張數、方向、停損價這三個算錯就是真賠錢，而它們
    在 2026-09-08 之前一次都沒被執行過。所以先讓它們每分鐘被算出來、
    寫進 DB、被人看得到，再接真錢。

    e = (sym, event_ts, det_ts, latency, sig, direction, level, px, atr)
    direction: +1 = LONG, -1 = SHORT（impulse 方向，本線是順著走）

    `event_ts` 現在是**成立**那一分鐘（ready），不是群內最早那一分鐘。
    回傳 None 代表這一筆不在驗證過的母體裡（見 INTENT_SIGNATURE）。
    """
    sym, ev_ts, _d, _lat, sig, d, _lvl, px, atr = e
    if sig != INTENT_SIGNATURE:
        # S+V 單獨為負、S+D 樣本薄。驗證過的是三者齊發那一格；其餘照樣
        # 記進 conj_events_live（shadow 要全格），但不產生訂單意圖。
        return None
    px = float(px)
    atr = float(atr)
    side = "LONG" if d > 0 else "SHORT"
    stop_dist = STOP_ATR * atr
    stop = px - d * stop_dist
    # 2026-09-08 實盤體檢：停損與出場都必須**相對成交**，不能錨在 close[a]。
    #   stop_price 是研究端的參考（錨在 close[a]）；產品端真正掛的是
    #   fill ∓ stop_dist —— 否則成交若已離 close[a] 半個 ATR，有效停損就是
    #   0.5 或 1.5 ATR，極端時停損價在送單當下已被穿過（交易所拒單或立即觸發）。
    #   exit 同理：研究是「進場後 60 分」，不是「錨點後 60 分」——
    #   產品端平倉時刻 = fill_ts + hold_ms。exit_ts 只是參考。
    #   intent_id 是去重鍵：端點在 TTL 內會重複吐同一筆，產品端必須以它去重。
    # 過期時刻錨在**進場時刻**（ready + ENTRY_DELAY_MIN），不是 ready：
    # 回測的進場是 open(ready+3)，所以 ready+1 產生的意圖本來就該活到那時。
    entry_ts = int(ev_ts) + ENTRY_DELAY_MIN * 60_000
    return dict(canonical_symbol=sym, anchor_ts=int(ev_ts), intent_ts=det,
                expires_ts=entry_ts + INTENT_TTL_S * 1000,
                side=side, ref_price=px, stop_price=float(stop),
                stop_dist=float(stop_dist),
                exit_ts=entry_ts + HOLD_MIN * 60_000,
                hold_ms=HOLD_MIN * 60_000, atr=atr,
                notional_usd=NOTIONAL_USD,
                size_base=NOTIONAL_USD / px if px > 0 else 0.0,
                signature=sig, status="NEW",
                intent_id=f"{sym}:{int(ev_ts)}")


def intent_gate(conn, intents):
    """全域煞車：同時持倉、單幣、單日上限。任一超過就不產生意圖。

    這三個上限是 override 記錄裡列的緩解措施（CLAUDE.md 2026-09-08），
    不是可調參數——要動必須回去改那份記錄。
    """
    if not intents:
        return []
    now = int(time.time() * 1000)
    # 每日上限也跟著 UTC+8 換日（2026-09-10 統一日界）
    day0 = now - ((now + ec.DAY_OFFSET_MS) % 86_400_000)
    with conn.cursor() as cur:
        # 「持倉中」的定義（2026-09-08 實盤體檢修正）：
        #   (a) NEW 且**尚未過期**——還可能被送出去
        #   (b) 已有成交回報（agent_conj_fills）且持有窗未到
        # 原版把「status=NEW 且 exit_ts 未到」全算持倉：一個從沒被送出、
        # 180 秒就過期的意圖會佔一個槽位 60 分鐘 —— 幽靈持倉擋真單。
        # 而 status 本來就沒有任何路徑會變（agent 唯讀、產品端不直連 DB），
        # 所以「有沒有成交」只能從產品端回報的 agent_conj_fills 推。
        cur.execute(
            "SELECT i.canonical_symbol AS sym, COUNT(*) AS n FROM conj_intents i "
            "LEFT JOIN agent_conj_fills f "
            "  ON f.intent_id = CONCAT(i.canonical_symbol, ':', i.anchor_ts) "
            "WHERE (f.intent_id IS NULL AND i.status='NEW' AND i.expires_ts > %s) "
            "   OR (f.intent_id IS NOT NULL AND f.fill_ts + i.hold_ms > %s) "
            "GROUP BY i.canonical_symbol", (now, now))
        # DictCursor：fetchall() 回 list[dict]，`dict(...)` 會炸。見 `_one`。
        open_by = {r["sym"]: int(r["n"]) for r in (cur.fetchall() or [])}
        cur.execute("SELECT COUNT(*) AS n FROM conj_intents "
                    "WHERE intent_ts >= %s", (day0,))
        n_today = int(_one(cur.fetchone()) or 0)
    out = []
    live = sum(open_by.values())
    for it in intents:
        s = it["canonical_symbol"]
        if n_today + len(out) >= MAX_DAILY:
            print(f"[GATE] 每日上限 {MAX_DAILY} 已滿，跳過 {s}")
            continue
        if live + len(out) >= MAX_CONCURRENT:
            print(f"[GATE] 同時持倉上限 {MAX_CONCURRENT} 已滿，跳過 {s}")
            continue
        if open_by.get(s, 0) >= MAX_PER_SYMBOL:
            print(f"[GATE] {s} 已有部位，跳過")
            continue
        out.append(it)
    return out


def _one(row):
    """取單欄查詢的那個值。

    `shared.db.get_db_conn()` 用的是 **DictCursor**，所以 `fetchone()` 回的是
    dict 不是 tuple，`row[0]` 會丟 `KeyError: 0`。這個專案的其他地方都寫
    dict 存取，只有這支腳本寫成位置索引 —— 而它踩到的那條路徑一分鐘只跑
    「有事件的時候」，所以錯了兩天沒人知道（2026-09-09 查出 conj_events_live
    從註冊以來 0 列）。用欄位名不可行（`MAX(event_ts)` 的鍵是整串運算式），
    所以取「唯一的那個值」。
    """
    if not row:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()), None)
    return row[0]


def recent_ok(conn, sym, ev_ts):
    """跨輪的 60 分鐘冷卻：`assemble` 的冷卻只在本輪的視窗內成立。"""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(event_ts) FROM conj_events_live "
                    "WHERE canonical_symbol=%s AND event_ts > %s",
                    (sym, ev_ts - COOLDOWN * 60_000))
        r = cur.fetchone()
    return _one(r) is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    t_start = time.time()
    ok, reason, found = True, "ok", 0
    try:
        levels, thr = load_state(rebuild=a.rebuild)
        thr_by = {r.sym: r for r in thr.itertuples()}
        lv_by = {s: g for s, g in levels.groupby("sym")} if len(levels) else {}

        sys.path.insert(0, str(ROOT))
        from shared.db import get_db_conn
        conn = get_db_conn()
        ensure_table(conn)

        events = []
        intents = []
        for sym in CORE9:
            if sym not in thr_by:
                continue
            t = thr_by[sym]
            try:
                kl = fetch_recent(sym)
            except Exception as e:
                print(f"[WARN] {sym} fetch failed: {e}")
                continue
            # 只用**已收盤**的 K（最後一根仍在進行中，丟掉）
            kl = [k for k in kl if int(k[6]) <= time.time() * 1000][:-0 or None]
            if len(kl) < W + 2:
                continue
            ts = np.array([int(k[0]) for k in kl], np.int64)
            close = np.array([float(k[4]) for k in kl])
            high = np.array([float(k[2]) for k in kl])
            low = np.array([float(k[3]) for k in kl])
            vol = np.array([float(k[5]) for k in kl])
            tbb = np.array([float(k[9]) for k in kl])
            delta = 2 * tbb - vol

            det = int(time.time() * 1000)
            for e in detect_window(sym, ts, high, low, close, vol, delta,
                                   t, lv_by.get(sym), det,
                                   atr_override=atr_h14_now(sym)):
                if recent_ok(conn, e[0], e[1]) is False:
                    continue
                events.append(e)
                if INTENTS_ENABLED:
                    it = make_intent(e, det)
                    if it is not None:
                        intents.append(it)

        if not INTENTS_ENABLED and events:
            print(f"[HALT] 意圖層已停止（2026-09-09 前視判決，見檔頭 "
                  f"INTENTS_ENABLED）——本輪 {len(events)} 個事件只記 shadow，"
                  f"不產生任何訂單意圖")
        keep = intent_gate(conn, intents)
        if keep:
            cols = ("canonical_symbol,anchor_ts,intent_ts,expires_ts,side,"
                    "ref_price,stop_price,exit_ts,atr,notional_usd,"
                    "size_base,signature,status,stop_dist,hold_ms,intent_id")
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT IGNORE INTO conj_intents ({cols}) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [tuple(i[k] for k in cols.split(",")) for i in keep])
            conn.commit()
            for i in keep:
                print(f"[INTENT] {i['canonical_symbol']} {i['side']} "
                      f"ref={i['ref_price']:.6g} stop={i['stop_price']:.6g} "
                      f"size={i['size_base']:.6g} (${i['notional_usd']:.0f})")

        if events:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT IGNORE INTO conj_events_live "
                    "(canonical_symbol, event_ts, detected_ts, latency_ms, "
                    " signature, direction, level, px, atr) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", events)
            conn.commit()
            found = len(events)
        conn.close()
    except Exception as e:
        ok, reason = False, f"{type(e).__name__}: {e}"
        # traceback 不可省略：`KeyError: 0` 這種訊息完全指不出位置，而這支
        # 每分鐘跑一次、錯誤只會累積在 log 裡沒人看得懂（mistake.md
        # 2026-08-01「渲染層的 try/except 永遠不可以是靜默的」的排程版）。
        print(f"[ERROR] {reason}")
        traceback.print_exc()

    took = (time.time() - t_start) * 1000
    lat = [e[3] for e in events] if ok and 'events' in dir() and events else []
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps({
        "ok": ok, "reason": reason, "events": found,
        "cycle_ms": round(took),
        "latency_ms_max": max(lat) if lat else None,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    # 意圖層狀態**無條件印**：交會事件約每幣 2.5 天一次，所以 events=0 是
    # 常態，而「if not INTENTS_ENABLED and events」那行在 0 的時候不會印
    # —— 那個 0 同時代表「這分鐘沒事件」與「HALT 訊息路徑壞了」，畫面上
    # 一模一樣（mistake.md 2026-08-26 的形狀，本 session 已踩三次）。
    print(f"conj_watch: events={found}  cycle={took:.0f}ms  ok={ok}"
          + (f"  max_latency={max(lat)}ms" if lat else "")
          + ("  intents=**HALTED**（2026-09-09 前視判決）"
             if not INTENTS_ENABLED else "  intents=ON"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
