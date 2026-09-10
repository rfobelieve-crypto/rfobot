# -*- coding: utf-8 -*-
"""SDV · 回測檢視器 —— 把現行實盤規則的每一筆交易畫在 K 線上

使用者 2026-09-08：「那個歷史回測應該要顯示交會的那個策略才對喔」

網站 /charts/backtest 一直畫的是舊線（掃單失敗，2026-09-07 結案）。
現在要上實盤的是新線 —— SDV（2026-09-09 使用者命名；TODO §1.03），本檔給它同一種檢視器：
每一條線、每一個標記，都取自**同一筆被計分的記錄**，不重算、不近似、
不另寫一份繪圖用的邏輯（mistake.md 2026-08-26：兩份實作會安靜地不同意）。

===========================================================================
交易規則（＝ conj_watch / jarvis 要執行的那一套，一個字不動）
===========================================================================
    母體    掃單 ∧ (delta_ext ∨ vol_burst)     ——「NO-OI」，2 分鐘管線的母體
            （時鐘 conj_clock 註冊的是含 oi_crash 的版本；OI 粒度 5 分鐘、
              結構上進不了 2 分鐘死線，所以**實盤跑的是 NO-OI**，本頁畫的
              也是它。conj_clock_and 的「且」變體 = 本頁的 S+D+V 那一格。）
    組裝    `conj_redef.groups_with_members`（與判決端同一顆）
    訊號時刻 **ready** = 最後一個必要成分到齊那一分鐘。**不是群內最早那一分鐘**
            —— 用最早那個會讓 22.3% 的單下在事件成立之前（前視，§1.03b）
    方向    ready 前 5 分鐘的動能（`close[ready] > close[ready-5]` 就做多）。
            **這是 A 臂，現行規格。**「等它走出來再跟」（C 臂）在同日稍後
            被判過擬合並撤回 —— 誠實地只用前半選門檻會選到不同的一組，
            那組在後半只有 5/9（見 `ledger()` 的 docstring）。
            C 保留為 `--arm C` 的可選臂，不是規格。
    進場    ready + 3 分那根的**開盤**（限價，成交率 97.8%）
    停損    3.0 × ATR_h14(ready)，分鐘高低價判定，從進場**下一根**起
    出場    停損，或進場後 480 分那根的收盤，先到者
    單位    ATR。+0.30 = 平均每筆賺 0.3 個小時 ATR（≈ 0.28% 名目）

成本（Bitget **返佣 50% 後的實付**，2026-09-09 使用者提供）
    進場 1 ／ 時間出場 1 ／ 停損出場 3 bps（牌價 maker 2 / taker 6）。
    停損率 18% 下混合 2.37 bps。限價成交率 97.8% 已量過。
    逐筆換算：cost_ATR = bps/1e4 × entry / ATR
    表上「淨」= 毛 − 這一筆自己那條腿的成本

    已知答案對照（`tests/test_conj_backtest_parity.py`）：A 臂與測試裡一支
    **獨立的逐筆 for 迴圈**逐位比對；R 與出場價互相反解；停損時序。

畫什麼（全部來自同一筆記錄）
    ┈┈  價位線     從樞紐形成（formed_at）延伸到被掃那一分鐘
    ▽▲  掃單       第一次穿越價位的那分鐘
    ●   進場       ready + 3 分開盤（A 臂＝現行規格）
    ✕   出場       停損價（停損）或 +480 分收盤（時間）
    K 線用 **5 分鐘**（顯示用；規則跑在 1 分鐘上）。標記對齊到所在的
    5 分鐘 K；點選單筆時畫出精確價位／進場／停損／出場四條水平線，
    上方文字給精確到分鐘的時刻。

為什麼要發佈到 DB 而不是讓雲端算
    分鐘 bar、OI、事件表全部在 `research/poc/data/`（gitignored，映像裡
    沒有）。這是「錄製器在本機」那一族的第 N 個實例（v7_veto_clock、
    raid_signals_live…）：本機的 conj_update 班車算好 HTML，寫進
    `conj_backtest_pages`，agent 只 SELECT（agent-boundary.md）。

用法
    python research/poc/conj_backtest.py --symbol BTC
    python research/poc/conj_backtest.py --all --last-days 90 --publish
出：research/results/conj_backtest_{SYM}.html（+ DB 一列/幣）
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
sys.path.insert(0, str(HERE.parents[1]))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import conj_clock as ck  # noqa: E402
import sweep_core as sc  # noqa: E402   只借 metrics()，同一把尺
# 2026-09-09：群成員與「事件成立時刻」的定義**只有一份**，在 conj_redef。
# 本檔曾自己用 et.cluster 的錨點當進場時刻 —— 那是前視（TODO §1.03b）。
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
EVENTS = HERE / "data" / "events"
LEVELS = HERE / "data" / "levels"
RES = HERE / "data" / "results"
OUT = HERE.parents[0] / "results"
CORE9 = list(ec.CORE9)

# 規則常數 —— 與 conj_watch.py 同值。這裡不 import conj_watch（它會連 DB、
# 抓 Binance），但 parity 測試會斷言兩邊的數字相等。
W = 5
# 2026-09-09 定案（TODO §1.03d/f）:先前的 DELAY=2 / STOP=1.0 / HOLD=60 是
# 錯的出場設定 —— 1 ATR 停損砍掉左尾、60 分持有砍掉右尾,那才是把這條線
# 壓在水面下的主因,不是成本也不是訊號。
#   停損 3 ATR:觸發率 18%,樣本外「停損放寬」的相關性只有 +0.120,
#              所以不需要更寬;3 ATR 保留風險單位,接得進 sizing 與 kill switch
#   持有 480 分:毛利隨持有單調上升,樣本外單調性 +0.907(比樣本內還強)
DELAY = 3
STOP = 3.0
HOLD = 480
# C 臂（使用者 2026-09-09:「我不用一定要知道方向,只要知道獵取後怎麼走」）
# 不在事件當下猜方向,等 K 分鐘、看它實際走了多少,超過門檻才順著跟。
# 樣本外:命中率 48.6% -> 51.0%、毛利 +0.216 -> +0.305、逐幣 7/9 -> 9/9,
# 而且資金曲線最大回落 47.0% -> 26.5%(2x/3 槽)。
C_WAIT = 10          # 事件成立後等幾分鐘
C_MOVE = 0.5         # 這段期間至少要走幾個 ATR 才進場
FLOW = ("delta_ext", "vol_burst")
MERGE_GAP = et.MERGE_GAP
# 分腿成本（bps）：sweep_forward.SCEN 情境 A
# Bitget **返佣 50% 後的實付**（2026-09-09，使用者提供）：
# 標準 maker 2 / taker 6 bps，返佣後 maker 1 / taker 3。
# 進場與時間出場掛限價（成交率 97.8%，conj_rescue C3），停損吃單。
# 停損率 18% 下混合成本 = 1 + 0.82x1 + 0.18x3 = **2.37 bps**。
# 用實付而不是牌價：牌價會讓公開頁面低估這條線的淨值。
COST_ENTRY, COST_TIME, COST_STOP = 1.0, 1.0, 3.0
CANDLE_MIN = 5          # 預設顯示週期
# 使用者 2026-09-10：圖上要能切 5 分 / 15 分 / 1 小時
CANDLE_MINS = (5, 15, 60)
TABLE_DDL = """
CREATE TABLE IF NOT EXISTS conj_backtest_pages (
    sym        VARCHAR(8)  NOT NULL PRIMARY KEY,
    html       MEDIUMTEXT  NOT NULL,
    asof_ts    BIGINT      NOT NULL,
    n_view     INT         NOT NULL,
    n_all      INT         NOT NULL,
    updated_at TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
                           ON UPDATE CURRENT_TIMESTAMP
)"""


# ── KPI 卡的大小字為什麼是這個順序（2026-09-09）─────────────────────────
# 原本顯示窗（90 天、單一幣、16~34 筆）是大字，全期（235~399 筆）是灰色
# 小字 —— **把最不可靠的數字放最大**。實際後果是有人在手機上讀到單一幣
# 一季的勝率，把它當成這條線的勝率，而同一張卡下面的全期差了一倍以上。
# 已對調，並在顯示窗 < 50 筆時另外印警語。
#
# 這段說明**刻意留在 Python 這一側**：HTML 模板裡的註解會原樣印進公開
# 頁面的原始碼，所以任何牽涉到操作者、內部討論、或未公開判決細節的文字
# 都不可以寫進模板（公開面規則見 CLAUDE.md §對外網站呈現面）。
# ────────────────────────────────────────────────────────────────────


def _empty_liq():
    """liq 只餵 liq_burst，而本線不用它 —— 離線也能跑。"""
    return pd.DataFrame({"s": pd.Series(dtype=str), "w": pd.Series(dtype="int64"),
                         "u": pd.Series(dtype=float), "sym": pd.Series(dtype=str)})


# 顯示時區（2026-09-10 使用者：「時間統一用 UTC+8」）。**只影響顯示**：
# 事件時刻、樞紐、進出場、PDH/PDL 的日界全部仍以 UTC 計算。
DISPLAY_TZ = timezone(timedelta(hours=8))
TZ_LABEL = "UTC+8"


def to_day(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")


def ledger(sym, liq=None, arm="A", scale="1h"):
    """一個幣的完整交易帳（全歷史）。回傳 (trades, bars_df)。

    arm="A"  事件成立 +3 分進場，方向 = 事件前 5 分鐘動能（**預設，現行**）
    arm="C"  成立後等 10 分、走超過 0.5 ATR 才順著它實際的方向進場
             （2026-09-09 使用者提出「不預測方向，等它走出來再跟」）

    **C 的門檻在同日稍後被判過擬合，所以預設回到 A**：
    誠實地只用前半資料選門檻，選到的是「等 5 分／走 1.0 ATR」而不是
    (10, 0.5)；那組在後半是 5/9、n=98、CI 下緣 −0.598。各折選到的門檻
    還在跳（10分/1.0、5分/1.0）。先前 (10,0.5) 的「樣本外 9/9」之所以
    好看，正因為它是看過後半才挑的。**C 保留為可選臂供對照，不當現行規格。**

    C 唯一站得住的部分是**不加門檻**時的穩定性：樣本外命中率 48.6% -> 50.9%、
    CI 下緣 −0.189 -> −0.122、逐幣 7/9 -> 8/9，但點估計反而降低
    （+0.151 -> +0.082）。等它走出來換到的是穩定，不是報酬。

    兩臂唯一的差異是「方向怎麼決定、什麼時候進場」；停損、持有、成本全同。
    `tests/test_conj_backtest_parity.py` 對 A 臂釘住 `conj_redef` 的誠實值。
    """
    liq = _empty_liq() if liq is None else liq
    # 2026-09-10：樞紐尺度。"1h" = 現行（data/events、data/levels），
    # "5m" = 5 分鐘樞紐（data/events_5m、data/levels_5m）。規則不變，只換來源。
    ev_dir = EVENTS if scale == "1h" else HERE / "data" / f"events_{scale}"
    lv_dir = LEVELS if scale == "1h" else HERE / "data" / f"levels_{scale}"
    cand, ts, cl, at, _day = ck.frozen_cand(sym, liq, None if scale == "1h" else ev_dir)
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    n = len(ts)

    sweeps = ec.cooldown_filter(np.sort(cand["sweep"]))
    sweep_set = np.asarray(sweeps, np.int64)
    pairs = [(int(m), "sweep") for m in sweeps]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))

    ev = pd.read_parquet(ev_dir / f"{sym}.parquet",
                         columns=["level_id", "side", "t_sweep", "sweep_lvl"])
    ev = ev.sort_values("t_sweep")
    ev_ts = ev["t_sweep"].to_numpy(np.int64)
    # hour_ts = 樞紐 bar 的**開盤**；formed_at 是它的收盤。畫線起點要用
    # 前者（線才會坐在那根 K 上），可知時刻仍是後者 —— 用途不同。
    lv = pd.read_parquet(lv_dir / f"{sym}.parquet",
                         columns=["level_id", "formed_at",
                                  "hour_ts"]).set_index("level_id")

    trades = []
    for a, mem in cr.groups_with_members(pairs):
        sig = {t for _, t in mem}
        if "sweep" not in sig or not (sig & set(FLOW)):
            continue
        # **一切錨在 ready，不是 a**（2026-09-09，TODO §1.03b）。
        # a 是群內最早那一分鐘；ready 是最後一個必要成分到齊、SDV 真正
        # 成立的那一分鐘。用 a 當進場錨點，22.3% 的單會下在事件存在之前。
        m_sw = min(m for m, t in mem if t == "sweep")
        m_fl = min(m for m, t in mem if t in FLOW)
        ready = max(m_sw, m_fl)
        if ready < W or ready + max(DELAY, C_WAIT + 1) + HOLD >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        if arm == "C":
            mv = (cl[ready + C_WAIT] - cl[ready]) / A
            if abs(mv) < C_MOVE:
                continue                      # 沒走出來就不進場
            d = float(np.sign(mv) or 1.0)
            j0 = ready + C_WAIT + 1
        else:
            d = float(np.sign(cl[ready] - cl[ready - W]) or 1.0)
            j0 = ready + DELAY
        ent = float(op[j0])
        end = j0 + HOLD
        adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
               else (hi[j0 + 1:end + 1] - ent)) / A
        hit = np.flatnonzero(adv >= STOP)
        stop_px = ent - d * STOP * A
        if len(hit):
            jx = j0 + 1 + int(hit[0])
            exit_px, R, stopped = stop_px, -STOP, True
        else:
            jx = end
            exit_px, stopped = float(cl[end]), False
            R = float(d * (cl[end] - ent) / A)
        a_early = a           # 群內最早那一分鐘（只拿來顯示前視延遲）
        a = ready             # 以下一律以成立時刻為錨點
        # 群內的掃單那一分鐘（只拿來畫價位線）。
        # 2026-09-09 收緊：原本容許錨點後 30 分鐘（6 x MERGE_GAP），但併窗
        # 就是 5 分鐘 —— 超出的那個掃單**不屬於這一群**，畫出來的價位線會是
        # 別的事件的。實測 1.2%（37/3,005）落在這個縫裡。寧可不畫也不畫錯：
        # 超窗就 m=a，找不到對應事件列 -> level=None -> 該筆不畫價位線。
        # 2026-09-09 修：**直接用這一群自己的掃單分鐘 `m_sw`**。
        #
        # 原本寫 `searchsorted(sweep_set, a)` 去「找一個掃單」，而 a 已經改成
        # ready。ready = max(掃單, 流量)，所以當流量比掃單晚到（母體的 74%），
        # ready > m_sw，這個搜尋會**跳過這一群自己的掃單**去找下一個：
        #   · 找不到夠近的 -> 退回 m=ready -> 對不到事件列 -> level=None
        #   · 找到一個夠近的 -> 那是**別的掃單**的價位，畫錯線
        # 實測（修之前）：掃單後到 100% 對得到，掃單先到只有 68.2%，
        # 全部 357 筆 SDV 的失敗都落在「掃單先到」那組。
        #
        # 這是今天把錨點從「群內最早」換成 ready 時漏掉的下游 —— 而群成員
        # 本來就帶著 m_sw，根本不需要再去搜一次。**已經有的東西不要重新推導**
        # （本 session 第 N 次同一個形狀）。
        m = m_sw
        t_sw = int(ts[m]) + ec.MIN_MS
        e_i = np.searchsorted(ev_ts, t_sw)
        level = origin = None
        lside = ""
        if e_i < len(ev_ts) and ev_ts[e_i] == t_sw:
            r = ev.iloc[e_i]
            level, lside = float(r["sweep_lvl"]), str(r["side"])
            if r["level_id"] in lv.index:
                origin = int(lv.loc[r["level_id"], "hour_ts"])
        leg = COST_ENTRY + (COST_STOP if stopped else COST_TIME)
        cost = leg / 1e4 * ent / A
        trades.append(dict(
            sym=sym, sig="+".join(sorted(sig)),
            sigk="and" if {"delta_ext", "vol_burst"} <= sig
            else ("d" if "delta_ext" in sig else "v"),
            side="LONG" if d > 0 else "SHORT",
            anchor_ts=int(ts[a]), sweep_ts=int(ts[m]),
            origin_ts=origin, level=level, level_side=lside,
            entry_ts=int(ts[j0]), entry=ent, atr=A, stop=float(stop_px),
            exit_ts=int(ts[jx]), exit_px=float(exit_px), stopped=stopped,
            held=int(jx - j0), R=float(R), cost_bps=leg,
            lag_min=int(ready - a_early),
            sweep_first=bool(m_sw <= m_fl),
            R_net=float(R - cost),
            forward=bool(ts[a] >= ck.FREEZE_MS)))
    return trades, b


def pdh_pdl(b, lo_ms, hi_ms):
    """前一日（日界＝顯示時區，現為 UTC+8）的高 / 低。

    **display-only，不參與 SDV 判定**（2026-09-10 使用者：「PDH/PDL 不要
    被蓋掉了這個也要顯示出來」—— 但它不是被蓋掉，是 SDV 的價位表從來
    只有 swing pivot 一種，前日高低沒有被計算過）。

    無前視：第 D 天用的是第 D−1 天**已收盤**的高低，那在 D 天開盤時就
    完全知道。這也是它跟 swing pivot 的結構差異 —— swing 要等右邊 10 根
    才確認，PDH/PDL 在日界一到就成立。

    第二版：**每天一條 series 會讓瀏覽器卡死**（90 天 x 2 = 180 個 series，
    加上價位線直接把渲染器凍住，實測 CDP 截圖 timeout）。改成**各一條
    階梯線**（lineType WithSteps），日界自然跳變，series 數 180 -> 2。
    """
    # 日界跟著顯示時區走（2026-09-10 使用者：「改全部統一」）。
    # 這會切出**不同的**前一日高低 —— 是定義變更，不是顯示變更。
    # 安全的原因只有一個：PDH/PDL 不參與 SDV 判定。
    v = b[["ts", "high", "low"]].copy()
    off = int(DISPLAY_TZ.utcoffset(None).total_seconds() * 1000)
    v["d"] = (v["ts"] + off) // 86_400_000
    g = v.groupby("d").agg(hi=("high", "max"), lo=("low", "min"))
    g["ph"] = g["hi"].shift(1)          # 前一日高 = 今日的 PDH
    g["pl"] = g["lo"].shift(1)
    out = {"PDH": [], "PDL": []}
    for d, r in g.iterrows():
        t0 = int(d) * 86_400_000 - off     # 換回 UTC 毫秒
        if t0 + 86_400_000 < lo_ms or t0 > hi_ms:
            continue
        t = _snap(max(t0, lo_ms))
        for key, val in (("PDH", r["ph"]), ("PDL", r["pl"])):
            if np.isfinite(val):
                out[key].append(dict(time=t, value=float(val)))
    # 同一個 5 分鐘桶只能有一個點（顯示窗左緣被裁切時會撞在一起）
    res = []
    for k, pts in out.items():
        seen, clean = set(), []
        for x in pts:
            if x["time"] in seen:
                continue
            seen.add(x["time"])
            clean.append(x)
        if clean:
            res.append(dict(k=k, pts=clean))
    return res


def candles_at(b, lo_ms, hi_ms, minutes):
    v = b[(b["ts"] >= lo_ms) & (b["ts"] <= hi_ms)]
    if v.empty:
        return []
    w = minutes * 60_000
    g = v.assign(bk=(v["ts"] // w) * w) \
         .groupby("bk").agg(open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last"))
    return [dict(time=int(t // 1000), open=float(r.open), high=float(r.high),
                 low=float(r.low), close=float(r.close))
            for t, r in g.iterrows() if np.isfinite(r.open)]


def _snap(ms):
    """原始秒。**不再在這裡對齊到 K 線桶** —— 顯示週期可切換（5/15/60 分），
    對齊必須在 JS 端依當前週期做，寫死在這裡就只有一種週期的標記會落對。"""
    return int(ms // 1000)


def build(sym, t_from_ms, t_to_ms, liq=None, arm="A"):
    trades, b = ledger(sym, liq, arm=arm)
    last_ts = int(b["ts"].iloc[-1])
    lo = t_from_ms or int(b["ts"].iloc[0])
    hi = t_to_ms or last_ts
    # 只存最細的那一份；15 分／1 小時由 JS 端聚合（都是 5 的倍數，
    # 聚合精確）。存三份會讓 HTML 2.4MB -> 5.6MB，而 iframe 要載入它。
    candles = candles_at(b, lo, hi, CANDLE_MIN)
    if len(candles) < 30:
        raise SystemExit(f"{sym}: 顯示窗只有 {len(candles)} 根 K，範圍給錯了？")
    tr = [t for t in trades if lo <= t["entry_ts"] <= hi]

    markers, rows, eq = [], [], []
    # 2026-09-10：掃單價位獨立成兩組（`levels` / `sweeps`），**預設不畫**。
    # 使用者要「只要進場出場」，但同一天稍後又要「確認 sweep 位置跟 TV 一樣」
    # —— 兩個需求都真實，所以做成開關而不是二選一。
    levels, sweeps = [], []
    cum = 0.0
    for t in tr:
        cum += t["R"]
        eq.append(dict(time=_snap(t["exit_ts"]), value=round(cum, 4)))
    for i, t in enumerate(tr):
        win = t["R"] > 0
        col = "#0ecb81" if win else "#f6465d"
        lc = "#f0b90b" if t["level_side"] == "buyside" else "#7b61ff"
        if t["level"] is not None:
            o = t["origin_ts"] if t["origin_ts"] is not None else t["sweep_ts"] - 3_600_000
            # **起點必須夾在顯示窗內**：樞紐可能形成於顯示窗之前（實測最早
            # 早 28 天），而 lightweight-charts 收到超出 K 線範圍的時間點
            # 會去擴展時間軸 —— 5 分鐘桶 x 28 天 = 8,000 個空白桶，渲染器
            # 當場凍住（2026-09-10 實測 CDP 截圖 timeout）。
            a, z = max(_snap(o), _snap(lo)), _snap(t["sweep_ts"])
            # 每條價位一段，段與段之間插一個 whitespace 點（只有 time 沒有
            # value）把線斷開 —— 這樣兩個 side 各一條 series 就夠，不必
            # 一條價位一個 series（那會把瀏覽器凍住）。
            levels.append(dict(id=i, side=t["level_side"], c=lc,
                               a=a, z=max(z, a + CANDLE_MIN * 60),
                               v=t["level"], swept=True, traded=True))
        sweeps.append(dict(id=i, time=_snap(t["sweep_ts"]),
                           position="aboveBar" if t["level_side"] == "buyside" else "belowBar",
                           color=lc,
                           shape="arrowDown" if t["level_side"] == "buyside" else "arrowUp",
                           text=""))
        # 2026-09-10 使用者：「畫面很亂我就只要顯示進場出場就好了其他不用」。
        # 圖上只留這兩個標記 —— 價位虛線與掃單箭頭已移除（價位數字仍在下表
        # 與單筆說明裡，資料沒有消失，只是不畫在 K 線上）。
        #
        # 每個標記自帶 `id`。原本是「三個一組、JS 用 Math.floor(i/3) 反推
        # 是第幾筆」—— 用位置推導身分，少一個標記整個對應就錯位而且不會報錯
        # （mistake.md 同族：已經有的東西不要重新推導）。現在直接帶 id。
        markers += [
            dict(id=i, time=_snap(t["entry_ts"]),
                 position="belowBar" if t["side"] == "LONG" else "aboveBar",
                 color=col, shape="circle", text=f"{i + 1}"),
            dict(id=i, time=_snap(t["exit_ts"]),
                 position="aboveBar" if t["side"] == "LONG" else "belowBar",
                 color=col, shape="square",
                 text=f"{t['R']:+.2f}" + ("!" if t["stopped"] else "")),
        ]
        rows.append(dict(
            id=i, side=t["side"], sig=t["sig"], sigk=t["sigk"],
            anchor=to_day(t["anchor_ts"]), sweep=to_day(t["sweep_ts"]),
            entry_t=to_day(t["entry_ts"]), exit_t=to_day(t["exit_ts"]),
            origin=to_day(t["origin_ts"]) if t["origin_ts"] else "—",
            level=t["level"], level_side=t["level_side"],
            entry=t["entry"], stop=t["stop"], exit_px=t["exit_px"],
            atr=t["atr"], held=t["held"], R=round(t["R"], 4),
            R_net=round(t["R_net"], 4), cost_bps=t["cost_bps"],
            stopped=bool(t["stopped"]), forward=bool(t["forward"]),
            t_anchor=t["anchor_ts"] // 1000, t_entry=t["entry_ts"] // 1000,
            t_exit=t["exit_ts"] // 1000,
            t_origin=(t["origin_ts"] or t["sweep_ts"] - 3_600_000) // 1000))

    # 顯示窗內的**每一個**價位，不只形成交易的那些，也不只被掃過的那些。
    #   已被獵取 -> 實線，從樞紐開盤畫到被掃那一刻
    #   還沒被獵取 -> 虛線，從樞紐開盤畫到圖的右緣（它還掛在那裡）
    # 掃單箭頭只留形成交易的那些（使用者：不要灰色箭頭）。
    try:
        _lv = pd.read_parquet(LEVELS / f"{sym}.parquet",
                              columns=["level_id", "side", "price", "hour_ts",
                                       "confirmed_at", "invalidated_at"])
        right = int(candles[-1]["time"])
        # 每一列的「下一個同側樞紐開盤時刻」——未被掃的線畫到那裡為止，
        # 同一側因此只有最新一條延伸到右緣（照 LuxAlgo 的 set_level）。
        _lv = _lv.sort_values("hour_ts")
        _next = {}
        for _side, _g in _lv.groupby("side"):
            _hs = _g["hour_ts"].tolist()
            _ids = _g["level_id"].tolist()
            for _k in range(len(_ids)):
                _next[_ids[_k]] = int(_hs[_k + 1]) if _k + 1 < len(_hs) else None
        used = {(t["sweep_ts"], round(float(t["level"]), 8))
                for t in tr if t["level"] is not None}
        nid = 10_000
        for r in _lv.itertuples():
            inval = None if pd.isna(r.invalidated_at) else int(r.invalidated_at)
            # 這條線在顯示窗裡有沒有一段可見？
            if inval is not None and inval < lo:
                continue                      # 窗開始前就被掃掉了
            if int(r.confirmed_at) > hi:
                continue                      # 窗結束後才確認
            a = max(_snap(int(r.hour_ts)), _snap(lo))
            if inval is not None:
                z = _snap(inval - 60_000)      # 被獵取 -> 終止在那一刻
            else:
                nx = _next.get(r.level_id)     # 還掛著 -> 到下一個同側樞紐
                z = _snap(nx) if nx else right
            z = min(z, right)
            if z <= a:
                continue
            swept = inval is not None
            if swept and (int(inval) - 60_000, round(float(r.price), 8)) in used:
                continue                      # 已由上面的交易線畫過
            levels.append(dict(id=nid, side=str(r.side),
                               c="#f0b90b" if r.side == "buyside" else "#7b61ff",
                               a=a, z=z, v=float(r.price),
                               swept=bool(swept), traded=False, plain=True))
            nid += 1
    except Exception as e:                        # 顯示層不可靜默
        print(f"[WARN] {sym}: 價位線畫不出來: {e}")

    # 價位線：**一條價位一個 series**。時間上重疊的價位（不同價格、同時期）
    # 沒辦法塞進單一 series —— 試過用 whitespace 斷點合併，結果 33 條只畫得出
    # 13 條，重疊的全被跳過。33 個 series 對 lightweight-charts 沒有問題。
    #
    # 真正讓渲染器凍住的**從頭到尾只有一件事**：起點超出 K 線範圍（樞紐可能
    # 形成於顯示窗之前，實測最早早 28 天），圖表會去擴展時間軸、生出數千個
    # 空白桶。已在上面 clamp。這裡在產出時斷言，不靠瀏覽器發現。
    for L in levels:
        assert candles[0]["time"] <= L["a"] and L["a"] <= L["z"],             f"{sym} 價位線 {L['id']} 超出 K 線範圍"

    def _m(src, pred=lambda t: True):
        rs = [x["R"] for x in src if pred(x)]
        ns = [x["R_net"] for x in src if pred(x)]
        if len(rs) < 2:
            return None
        g, nn = sc.metrics(rs), sc.metrics(ns)
        g.update(net_exp=nn["exp"], net_pf=nn["pf"],
                 stop_rate=sum(1 for x in src if pred(x) and x["stopped"]) / len(rs) * 100)
        return g
    groups = {}
    for key, pred in (("all", lambda t: True),
                      ("and", lambda t: t["sigk"] == "and"),
                      ("d", lambda t: t["sigk"] == "d"),
                      ("v", lambda t: t["sigk"] == "v")):
        groups[key] = dict(view=_m(tr, pred), all=_m(trades, pred))

    clock = {}
    for stem in ("conj_clock", "conj_clock_and"):
        p = RES / f"{stem}.json"
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                clock[stem] = dict(n=j.get("n"), n_target=j.get("n_target"),
                                   verdict=j.get("verdict"))
            except Exception as e:                       # 顯示層不可靜默
                print(f"[WARN] {stem}.json unreadable: {e}")

    n_fwd = sum(1 for t in trades if t["forward"])
    return dict(sym=sym, candles=candles, markers=markers,
                levels=levels, sweeps=sweeps, pdhl=pdh_pdl(b, lo, hi),
                trades=rows, equity=eq, groups=groups,
                span=[to_day(lo), to_day(hi)],
                span_all=[to_day(int(b["ts"].iloc[0])), to_day(last_ts)],
                asof_ts=last_ts, n_all=len(trades), n_view=len(tr),
                n_forward=n_fwd, freeze_day=ck.FREEZE_DAY, clock=clock,
                params=dict(DELAY=DELAY, STOP=STOP, HOLD=HOLD, W=W,
                            C_WAIT=C_WAIT, C_MOVE=C_MOVE,
                            COST=[COST_ENTRY, COST_TIME, COST_STOP],
                            CANDLE_MIN=CANDLE_MIN,
                            CANDLE_MINS=list(CANDLE_MINS)))


TPL = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__SYM__ SDV · 回測檢視</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0e11;--pan:#12161c;--line:#1e242d;--ink:#eaecef;--dim:#848e9c;
      --up:#0ecb81;--dn:#f6465d;--buy:#f0b90b;--sell:#7b61ff;--amb:#f0b90b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.55 -apple-system,"Segoe UI","Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:16px 16px 40px;
      display:flex;flex-direction:column;gap:14px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px}
h1{margin:0;font-size:19px;font-weight:600;letter-spacing:.01em}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);
     border-radius:3px;padding:2px 8px;font-variant-numeric:tabular-nums}
.stat{border:1px solid var(--dn);border-left:4px solid var(--dn);
      border-radius:4px;background:rgba(246,70,93,.06);padding:11px 14px;
      display:flex;flex-direction:column;gap:5px}
.stat b{color:var(--dn);font-size:13px}
.stat p{margin:0;font-size:12px;line-height:1.6}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:8px}
.kpi{background:var(--pan);border:1px solid var(--line);border-radius:4px;
     padding:9px 11px;display:flex;flex-direction:column;gap:2px}
.kpi b{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi span{font-size:11px;color:var(--dim)}
.kpi small{font-size:10.5px;color:var(--dim);font-variant-numeric:tabular-nums}
#cbox{position:relative}
#c{height:520px;border:1px solid var(--line);border-radius:4px;overflow:hidden}
/* 全螢幕：優先用原生 API（:fullscreen）；被 iframe 政策擋掉時退回 .fs，
   在頁面內用 position:fixed 撐滿——嵌在網站裡時那就是撐滿 iframe。 */
#cbox:fullscreen,#cbox.fs{background:var(--bg);padding:8px;
  display:flex;flex-direction:column}
#cbox.fs{position:fixed;inset:0;z-index:9999}
#cbox:fullscreen #c,#cbox.fs #c{flex:1;height:auto;border-radius:0}
.fsbar{position:absolute;top:8px;right:10px;z-index:10;display:flex;gap:6px;
  align-items:center;font-size:11px}
.fsbar button{font:inherit;background:rgba(11,14,17,.82);color:var(--dim);
  border:1px solid var(--line);border-radius:3px;padding:3px 9px;cursor:pointer}
.fsbar button:hover{color:var(--ink);border-color:#2b3542}
.fsbar .fsonly{display:none}
#cbox:fullscreen .fsbar .fsonly,#cbox.fs .fsbar .fsonly{display:inline-flex}
#cbox:fullscreen .fsbar span.fsonly,#cbox.fs .fsbar span.fsonly{
  display:inline-block;min-width:4.5em;text-align:center;color:var(--dim);
  font-variant-numeric:tabular-nums}
#eq{height:150px;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;
     font-size:11.5px;color:var(--dim)}
.bar b{color:var(--ink);font-weight:500}
.sw{display:inline-block;width:16px;height:0;border-top:2px solid;
    vertical-align:middle;margin-right:5px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
     vertical-align:middle;margin-right:5px}
.sq{display:inline-block;width:8px;height:8px;vertical-align:middle;margin-right:5px}
button{background:var(--pan);color:var(--ink);border:1px solid var(--line);
       border-radius:3px;padding:4px 10px;font:inherit;font-size:11.5px;cursor:pointer}
button:hover{border-color:#3b4552}
button.on{border-color:var(--buy);color:var(--buy)}
.tw{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:4px}
table{width:100%;border-collapse:collapse;font-size:11.5px;
      font-variant-numeric:tabular-nums}
th,td{padding:5px 9px;text-align:right;white-space:nowrap;
      border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--pan);color:var(--dim);
   font-weight:500;text-align:right;z-index:1}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),
th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){text-align:left}
tbody tr{cursor:pointer}
tbody tr:hover{background:#171d25}
tbody tr.sel{background:#1c2530}
.pos{color:var(--up)}.neg{color:var(--dn)}.fwd{color:var(--amb)}
.note{font-size:11.5px;color:var(--dim);max-width:96ch}
/* 嵌入模式（2026-09-10）：網站是把整頁塞進固定高度的 iframe，而表頭
   （標題＋標籤＋摺疊區＋KPI）在 390x520 的手機框裡就把圖表整個推出框外
   —— 使用者看到的是「根本什麼都沒有」。所以偵測到自己在 iframe 裡時，
   把圖表以外的東西全部收起來，讓圖直接坐在最上面。
   直接開這個檔案時一切照舊，什麼都沒少。 */
body.embed header, body.embed .fold, body.embed .stat, body.embed .kpis,
body.embed #smallwarn, body.embed #eq, body.embed .tw,
body.embed .note, body.embed details{display:none!important}
/* .bar 裡只留按鈕與計數，說明文字與圖例在這個尺寸下沒有空間 */
body.embed .bar > span:not(#dense):not(#navpos){display:none!important}
body.embed .bar{gap:4px 6px;font-size:10.5px}
body.embed .bar button{padding:2px 7px;font-size:10.5px}
body.embed .wrap{padding:5px;gap:5px}
body.embed #c{height:calc(100vh - 84px);min-height:240px}
/* 浮層按鈕在窄框裡要更小，否則兩行蓋住圖 */
body.embed .fsbar{gap:3px;font-size:10px}
body.embed .fsbar button{padding:2px 5px;font-size:10px}
/* 摺疊區：長篇判決與規格預設收起來（2026-09-10「畫面很亂」）。
   收起不是刪除——公開頁的狀態標示與判決紀錄一個字都沒少，點開就在。 */
.fold{font-size:11.5px;color:var(--dim);border:1px solid var(--line);
      border-radius:4px;padding:7px 11px;background:var(--pan)}
.fold p{margin:6px 0 0;max-width:96ch}
details.stat{padding:9px 13px}
details>summary{cursor:pointer;list-style:none;color:var(--dim)}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸ ";color:var(--dim)}
details[open]>summary::before{content:"▾ "}
details.stat>summary{color:var(--dn)}
</style></head><body><div class="wrap">

<header>
  <h1>__SYM__USDT · SDV 回測</h1>
  <span class="tag">__SPAN__ · 時間 __TZ__</span>
  <span class="tag">進場 <b>成立 +__DELAY__ 分</b> · 停損 __STOP__ ATR · 持有 __HOLD__ 分</span>
  <span class="tag" style="border-color:var(--amb);color:var(--amb)">執行暫停中 · 樣本外 CI 下緣仍含零</span>
</header>

<details class="fold" style="border-color:#f0b90b">
  <summary style="color:#f0b90b"><b>要跟 TradingView 對照的話先看這行</b>
    —— BINANCE:BTCUSDT 現貨 · 1小時 · PIVOT=10 · UTC（點開看完整說明）</summary>
  <p>本頁所有 K 線與價位來自 <b>BINANCE:BTCUSDT（現貨）</b>、UTC。
  在 TradingView 上要比對 sweep 位置，商品必須設成同一個 —— 用
  <code>OKX:BTCUSDT.P</code> 之類的永續合約，K 線的最高最低本來就不同，
  樞紐位置不可能對得上（2026-09-10 校正的第一項）。時間週期 <b>1 小時</b>，
  樞紐規則 <b>PIVOT=10</b>（左右各 10 根）。</p>
  <p>另外兩個已量過的差異，看圖時會用到：<b>(1)</b> 我們保留**所有**還沒被
  消耗的價位，常見的擺盪指標（如 LuxAlgo Liquidity Swings）只保留**最新
  一個**，所以我們會用到很舊的價位 —— 被掃價位年齡中位 1.7 天，但
  <b>22.5% 超過一週</b>、3.9% 超過三個月。<b>(2)</b> 穿越判定我們用
  <b>盤中觸價 +2 ticks</b>，收盤穿越的版本實測只差 0.1%（1,429 vs 1,428），
  差別在時機不在有無。</p>
</details>

<details class="fold"><summary>規格與成本（展開）</summary>
  <p><b>S</b> 掃單 · <b>D</b> 主動量極端 · <b>V</b> 量能爆發 —— 三者齊發才是 SDV
  （下方可切分頁看 S+D／S+V）。成立 = 最後一個成分到齊那一分鐘。</p>
  <p>成本 Bitget 返佣後實付 1/1/3 bps（混合 2.37）· 限價成交率 97.8%。
  K 線 __CM__ 分鐘（顯示用）· 規則跑在 1 分鐘。凍結規則 · 純回測 · 非訊號。</p>
</details>

<details class="stat"><summary><b>2026-09-09 定案：出場參數修正（本頁畫的就是它）· 執行仍暫停</b>
  —— 點開看完整判決</summary>
  <p><b>今天改對的是出場，不是訊號也不是成本。</b>先前的停損 1 ATR 砍掉左尾、
  持有 60 分砍掉右尾 —— 那才是把這條線壓在水面下的主因。現在是
  <b>停損 3 ATR、持有 480 分</b>：毛利隨持有單調上升，而且那個單調性在
  <b>樣本外比樣本內還強</b>（+0.907 vs +0.472），整張「持有 × 停損」網格
  在樣本外 <b>15 格全部為正</b> —— 沒有峰值可以過擬合。</p>
  <p><b>樣本外（後半 1.25 年）</b>：每筆淨 +0.15 ~ +0.24 ATR、逐幣 8-9/9。
  但<b>樣本外／樣本內只有 34~36%</b>，所以樣本內的數字一律要打三折。
  十倍槓桿在樣本外是負的（單槽 −99.7%），因為停損 3 ATR ≈ 2.5% 價格，
  十倍下就是權益的 25%／筆。</p>
  <p><b>回落：先前這裡寫的 26~47% 是低估，已更正。</b>舊值取自單一條模擬
  路徑；改成重抽交易日順序跑 1500 條路徑之後，樣本外在兩倍槓桿三槽下是
  <b>中位 43.3%、p95 59.7%、回落超過五成的機率 24.1%</b>（超過七成 0.1%）。
  四次裡有一次會腰斬。同一組重抽的另一面：<b>優勢大於零的機率 85.0%</b>、
  勝率 47.9%、獲利因子 1.19、年化夏普 +0.93。<b>勝率低於五成</b>——這條線
  不靠猜對方向賺錢，靠的是贏的時候比輸的時候大。活數字跑
  <code>research/poc/conj_bet.py</code>。</p>
  <p><b>流量條件就是那個判別器（2026-09-09 新證據）</b>：同一批掃單、同一套
  進出場，唯一差別是流量旗標有沒有開火 —— 兩個都開 <b>+0.331、9/9 幣</b>，
  兩個都沒開只有 <b>+0.056、6/9</b>，<b>差六倍</b>。而反著做 SDV 是
  <b>0/9 幣</b>全輸，確認它確實是延續交易。</p>
  <p><b>一個被判掉的改良，留檔</b>：「不預測方向，等它走出來再跟」
  （成立後等 10 分、走超過 0.5 ATR 才進）看起來很好，但誠實地只用前半選門檻
  會選到<b>不同</b>的一組，而那組在後半只有 5/9。先前那個「樣本外 9/9」
  是看過後半才挑的。<b>已撤回，不作為現行規格。</b>
  站得住的只有它不加門檻時的穩定性（命中率 48.6%→50.9%、CI 下緣
  −0.189→−0.122），代價是點估計降低。</p>
  <p><b>2026-09-09 下半天又死了三個改良，一併留檔</b>：從未平倉量推導的
  <b>清算位密度</b>（指標的建構通過完整驗證，但密度預測不了延續——最高密度
  那格反而平庸，而波動度的單調性比它強八倍）；<b>「沒帶量的掃單會反轉」</b>
  （反著做是負的、逐幣 2/9，真樣本外兩個方向都貼零＝什麼都沒發生）；
  <b>把 delta 換成 CVD</b>（五分鐘 CVD 變化就是現行的 delta 五分鐘和；
  帶符號當方向來源，四個窗口跟現行動能 96-97% 一致，全落在雜訊裡）。
  判決全文 TODO §1.03i／§1.03j。</p>
  <p><b>為什麼還是暫停</b>：樣本外單筆淨值的信賴區間下緣仍然含零
  （−0.19 ~ −0.12），而且<b>整張出場網格 15 格沒有任何一格的下緣越過零</b>，
  兩年半的資料釘不住它。前瞻紀錄 3/300。<code>conj_watch</code> 的下單
  意圖層維持停止。判決全文 TODO §1.03b~f。</p>
</details>

<div class="kpis" id="kpis"></div>
<div id="smallwarn" style="margin:8px 0 0;color:var(--amb);font-size:12px"></div>

<div class="bar">
  <span><span class="dot" style="background:var(--up)"></span><span class="dot" style="background:var(--dn)"></span>
    <b>●</b> 進場 —— <b>顏色＝這筆賺賠，不是方向</b>（綠賺／紅賠；做多畫在 K 棒下方、做空在上方）</span>
  <span><span class="sq" style="background:var(--dn)"></span><b>■</b> 出場（數字＝毛 ATR，「!」= 觸及停損）</span>
  <span><span class="sw" style="border-color:var(--buy)"></span>買側價位　<span class="sw" style="border-color:var(--sell)"></span>賣側價位
    ——<b>實線</b>＝已被獵取（終止在被掃那一刻），<b>虛線</b>＝還掛著。
    <b>加粗</b>＝這個掃單形成了 SDV 交易。虛線畫到下一個同側樞紐出現為止，
    所以每一側只有最新一條會延伸到最右邊（與 LuxAlgo 的畫法一致）。</span>
  <span><span class="sw" style="border-color:#26a69a"></span>PDH　<span class="sw" style="border-color:#ef5350"></span>PDL（前一日高／低，日界 UTC+8，僅顯示）</span>
  <span style="width:100%"></span>
  <details class="fold" style="width:100%"><summary>為什麼有時候「明明跌了一大段卻算虧錢」</summary>
  <p>這是<b>延續</b>交易：價格穿過價位就<b>順著穿越方向</b>跟，不等回踩。實測
  <b>97.5% 的交易與突破同向</b>（上方價位被掃 → 1,396 筆做多 vs 36 筆做空）。
  所以一根長上影針刺穿上方價位後暴跌，系統做的是<b>多單</b>，那一大段跌幅
  是虧的不是賺的。圖上的數字已逐筆用「方向 ×(出場−進場)/ATR」獨立反解驗過，
  3,005 筆<b>零筆對不上</b>。</p>
  <p>進場價也<b>不一定在價位外側</b>：進場是成立 +3 分的開盤，這三分鐘價格可能
  已退回價位內，<b>樣本外 34.4%</b> 是這樣 —— 那不是改抓反轉，是順著突破方向
  但買在回檔裡。樣本外「進場在突破側」每筆 <b>+0.3232、9/9</b>，「退回價位內」
  <b>−0.0830、5/9</b>；差值 +0.2405、CI 下緣 −0.0305（<b>尚未過閘</b>，如實標）。</p>
  </details>
  <button id="btnAll" class="on">全部交易</button>
  <button id="btnWin">只看賺</button>
  <button id="btnLose">只看賠</button>
  <span style="width:100%"></span>
  <b>逐筆看：</b>
  <button id="btnPrev">‹ 上一筆</button>
  <span id="navpos" style="min-width:5em;text-align:center"></span>
  <button id="btnNext">下一筆 ›</button>
  <button id="btnFit">全景（90 天）</button>
  <span style="width:100%"></span>
  <span style="color:var(--dim)">簽名：</span>
  <button id="btnGA">不分（含 S+D、S+V）</button>
  <button id="btnGand" class="on">S+D+V＝SDV</button>
  <button id="btnGd">只有主動量（S+D）</button>
  <button id="btnGv">只有量能（S+V）</button>
  <span id="dense"></span>
</div>

<div id="cbox">
  <div class="fsbar">
    <span id="tfbar"></span>
    <button id="btnSweep" title="顯示／隱藏掃單價位與穿越箭頭">🗺 掃單價位</button>
    <button id="btnPdhl" title="顯示／隱藏前一 UTC 日的高低（僅顯示，不參與判定）">📏 PDH/PDL</button>
    <button id="fsPrev" class="fsonly" title="上一筆">‹ 上一筆</button>
    <span id="fsPos" class="fsonly"></span>
    <button id="fsNext" class="fsonly" title="下一筆">下一筆 ›</button>
    <button id="btnFS">⛶ 全螢幕</button>
  </div>
  <div id="c"></div>
</div>
<div id="eq"></div>
<div class="note" id="sel">點下方任一列 —— 圖表跳到那一筆，並畫出它的進場與出場兩條線。</div>

<div class="tw"><table>
<thead><tr>
<th>#</th><th>方向</th><th>簽名</th><th>錨點（UTC）</th><th>持(分)</th>
<th>價位</th><th>進場</th><th>停損</th><th>出場</th>
<th>ATR</th><th>毛(ATR)</th><th>淨(ATR)</th><th>結束於</th><th>期</th>
</tr></thead><tbody id="tb"></tbody></table></div>

<div class="note">
每一列都直接來自 <code>research/poc/conj_backtest.py:ledger()</code> 的同一筆記錄。
出場價與 R 在同一筆裡互相反解（停損筆的出場價就是停損價；時間出場筆的 R =
方向 × (出場 − 進場) / ATR），由 <code>tests/test_conj_backtest_parity.py</code>
釘住：池化毛利必須重現 <code>flow_direction.py</code> 的 +0.2275（同一條規則的
另一份實作，兩份必須同意）。「持」= 進場到出場的分鐘數（上限 60）。
簽名：D = 主動量極端（|delta| 5 分後向和 ≥ 滾動 30 日 p99）、V = 量能爆發
（5 分量 / 前 30 日同時段均值 ≥ 滾動 p99）。「且」那一格就是
<code>conj_clock_and</code> 另開時鐘的母體。
</div>

</div>
<script>
const D = __DATA__;
// UTC+8 的顯示格式化。**只在印出來的時候位移**，圖表內部仍是 UTC 秒。
// 宣告在所有使用者之前（本檔 2026-09-10 已因暫時性死區整段中斷過一次）。
const TZ_OFF = 8 * 3600;
function fmtTs(t, withTime){
  const d = new Date((t + TZ_OFF) * 1000);
  const p = n => String(n).padStart(2,'0');
  const ymd = d.getUTCFullYear() + '-' + p(d.getUTCMonth()+1) + '-' + p(d.getUTCDate());
  return withTime ? ymd + ' ' + p(d.getUTCHours()) + ':' + p(d.getUTCMinutes()) : ymd;
}
function fmtTick(t, type){
  const d = new Date((t + TZ_OFF) * 1000);
  const p = n => String(n).padStart(2,'0');
  // 0=年 1=月 2=日 3=時分  （type 由 lightweight-charts 決定粒度）
  if(type <= 1) return d.getUTCFullYear() + '-' + p(d.getUTCMonth()+1);
  if(type === 2) return (d.getUTCMonth()+1) + '/' + d.getUTCDate();
  return p(d.getUTCHours()) + ':' + p(d.getUTCMinutes());
}
const fmtP = v => v >= 1000 ? v.toFixed(1) : v >= 1 ? v.toFixed(3) : v.toFixed(5);

// 位置文字有兩個出口（一般工具列 + 全螢幕浮層），統一走這一顆，
// 免得兩邊各寫一次然後安靜地不同意。
function setPos(txt){ const a=document.getElementById('navpos'),
                      b=document.getElementById('fsPos');
  if(a) a.textContent=txt; if(b) b.textContent=txt; }
// 篩選狀態宣告在 kpis() **被呼叫之前**——let/const 的暫時性死區會把整段
// script 打斷、圖一片空白而頁面其他部分照常渲染（mistake.md 2026-09-08）。
// 目前的顯示週期（分鐘）。標記與線在 Python 端輸出的是**原始秒**，
// 對齊到 K 線桶必須在這裡做 —— 週期一換，桶就換。
let TF = D.params.CANDLE_MIN || 5;
const bucket = t => Math.floor(t / (TF * 60)) * (TF * 60);
const SNAP = () => TF * 60;
// ^^ 這三個必須宣告在 drawMarkers()/drawLevels()/focus() **之前**：
// const 的暫時性死區一旦被踩到，整段 script 當場中斷，而頁面其他部分
// 照常渲染 —— 症狀是「圖空白但版面正常」（mistake.md 2026-09-08，
// 本檔 2026-09-10 又踩一次）。

let filt = 'all';       // 賺賠
let grp  = 'and';       // 簽名。**預設就是 SDV（三者齊發）**——
// 頁面叫 SDV,預設卻顯示整個母體的話,標題與內容不同意。
// S+V 單獨為負、S+D 樣本薄,只有 and 那格撐得住樣本外。
// `$` 也宣告在這裡：focus() 在定義處之後才會被呼叫，但把取用工具留在
// 檔案下半部正是上一次 TDZ 的形狀，不重複同一個佈局。
const $ = id => document.getElementById(id);
let cur = -1;           // 目前聚焦的交易 id（逐筆導航用）

function kpis(){
  const G = (D.groups && D.groups[grp]) || {};
  const a = G.view, b = G.all;
  if(!a){document.getElementById('kpis').innerHTML =
    '<div class="kpi"><b>0</b><span>這段期間這一格沒有交易</span></div>'; return;}
  const cell = (v,k,s,cls) => `<div class="kpi"><b class="${cls||''}">${v}</b>`+
    `<span>${k}</span><small>${s}</small></div>`;
  const g = x => x >= 0 ? 'pos' : 'neg';
  const B = b || a;
  // 大字是全期，小字才是顯示窗（見產生器裡的說明）。
  const win = `顯示窗 ${a.n} 筆`;
  document.getElementById('kpis').innerHTML = [
    cell(B.n, '交易筆數（全期）', win),
    cell(B.exp.toFixed(4), '每筆毛利（ATR）', `${win} ${a.exp.toFixed(4)}`, g(B.exp)),
    cell(B.net_exp.toFixed(4), '每筆淨利（ATR）', `${win} ${a.net_exp.toFixed(4)}`, g(B.net_exp)),
    cell(B.wr.toFixed(1)+'%', '勝率（全期）', `${win} ${a.wr.toFixed(1)}%`),
    cell(B.stop_rate.toFixed(1)+'%', '停損率', `${win} ${a.stop_rate.toFixed(1)}%`),
    cell(isFinite(B.pf)?B.pf.toFixed(2):'∞', '獲利因子（毛）', `${win} ${isFinite(a.pf)?a.pf.toFixed(2):'∞'}`),
    cell(B.t.toFixed(2), 't 值（未聚類）', `${win} ${a.t.toFixed(2)}`),
  ].join('');
  const w = document.getElementById('smallwarn');
  if(w) w.innerHTML = a.n < 50
    ? `⚠ 下面圖上這 <b>${a.n}</b> 筆是 90 天顯示窗，<b>樣本太小、不能拿來判斷這條線</b>`
      + `——它只是讓你看得到每一筆長什麼樣。大字是全期，判決看的是那個，`
      + `而且真正的判決在九幣合計的樣本外，不在單一幣。`
    : '';
}
kpis();

const dark = {layout:{background:{color:'#0b0e11'},textColor:'#848e9c',fontSize:11},
  grid:{vertLines:{color:'#151a21'},horzLines:{color:'#151a21'}},
  rightPriceScale:{borderColor:'#1e242d'},
  // 2026-09-10 使用者：「圖標虛線不要有磁鐵讓我可以自由活動」。
  // lightweight-charts 預設 CrosshairMode.Magnet 會把十字線吸附到最近的
  // 收盤價，量兩點之間的距離時會被它拉走。Normal = 跟著游標自由移動。
  crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
  // 時間軸與十字線都顯示 UTC+8。lightweight-charts 內部一律以 UTC 秒運算，
  // 這裡只在「印出來」的時候加 8 小時 —— 資料本身一秒都沒有被移動。
  localization:{timeFormatter: t => fmtTs(t, true)},
  timeScale:{timeVisible:true,secondsVisible:false,rightOffset:6,
    borderColor:'#1e242d',
    tickMarkFormatter:(t,type)=>fmtTick(t,type)}};

const chart = LightweightCharts.createChart(document.getElementById('c'), dark);
const cs = chart.addCandlestickSeries({upColor:'#0ecb81',downColor:'#f6465d',
  borderVisible:false,wickUpColor:'#0ecb81',wickDownColor:'#f6465d'});
cs.setData(D.candles);

// 掃單圖層（預設關）。價位線用 series 畫、掃單用 marker，兩者都只在
// 開啟時才建立 —— 關掉時要真的移除 series，不是設成透明，否則價格軸
// 的自動縮放仍然會把它們算進去，圖會被一條 112 天前的老價位壓扁。
let showSweep = false;
let lvlSeries = [];
function drawLevels(){
  for(const sx of lvlSeries) chart.removeSeries(sx);
  lvlSeries = [];
  if(!showSweep) return;
  const ids = new Set(D.trades.filter(keep).map(t=>t.id));
  for(const L of D.levels){
    // plain = 只是掃單、沒形成交易 —— 不受「只看賺/只看賠」等篩選影響
    if(!L.plain && !ids.has(L.id)) continue;
    // 0 = 實線（已被獵取）、2 = 虛線（還掛著）；形成交易的加粗
    const sx = chart.addLineSeries({color:L.c,
      lineWidth: L.traded ? 2 : 1,
      lineStyle: L.swept ? 0 : 2,
      lastValueVisible:false, priceLineVisible:false,
      crosshairMarkerVisible:false, autoscaleInfoProvider:()=>null});
    const a = bucket(L.a), z = Math.max(bucket(L.z), a + TF*60);
    sx.setData([{time:a, value:L.v}, {time:z, value:L.v}]);
    lvlSeries.push(sx);
  }
}

const keepWL = t => filt==='all' || (filt==='win' ? t.R>0 : t.R<=0);
const keepG  = t => grp==='all' || t.sigk===grp;
const keep = t => keepWL(t) && keepG(t);
function drawMarkers(){
  const vis = D.trades.filter(keep);
  const ids = new Set(vis.map(t=>t.id));
  const dense = vis.length > 45;
  let ms = D.markers.filter(m=>ids.has(m.id))
                     .map(m=>Object.assign({}, m, {time:bucket(m.time)}));
  // 箭頭只畫形成交易的那些（使用者 2026-09-10：不要灰色箭頭）
  if(showSweep) ms = ms.concat(D.sweeps.filter(m=>ids.has(m.id))
                     .map(m=>Object.assign({}, m, {time:bucket(m.time)})));
  // lightweight-charts 要求 markers 依時間遞增，否則整組安靜地不畫
  ms.sort((a,b)=>a.time-b.time);
  cs.setMarkers(ms.map(m => dense ? Object.assign({}, m, {text:''}) : m));
  document.getElementById('dense').textContent =
    dense ? `顯示 ${vis.length} 筆 —— 標記文字已關閉（>45 筆會疊住）。點下表任一列看單筆。`
          : `顯示 ${vis.length} 筆`;
}
// 週期切換：換掉 K 線資料，然後把所有標記／線依新的桶重畫一次。
const TFS = D.params.CANDLE_MINS || [5,15,60];
const tfLabel = m => m>=60 ? (m/60)+' 小時' : m+' 分';
function buildTfBar(){
  const bar = document.getElementById('tfbar');
  bar.innerHTML = '';
  for(const m of TFS){
    const b = document.createElement('button');
    b.textContent = tfLabel(m);
    b.title = 'K 線改用 ' + tfLabel(m) + '（只影響顯示，規則永遠跑在 1 分鐘上）';
    if(m===TF) b.classList.add('on');
    b.onclick = () => setTf(m);
    bar.appendChild(b);
  }
}
// 從 5 分鐘 K 聚合出更粗的週期。base 一定是 CANDLE_MIN(5)，而 15 與 60
// 都是它的整數倍，所以 open=首、high=最大、low=最小、close=末 是精確的。
const BASE = D.params.CANDLE_MIN || 5;
const aggCache = {};
function candlesAt(m){
  if(m === BASE) return D.candles;
  if(aggCache[m]) return aggCache[m];
  const w = m * 60, out = [];
  let cur = null;
  for(const c of D.candles){
    const b = Math.floor(c.time / w) * w;
    if(!cur || cur.time !== b){
      if(cur) out.push(cur);
      cur = {time:b, open:c.open, high:c.high, low:c.low, close:c.close};
    }else{
      cur.high = Math.max(cur.high, c.high);
      cur.low  = Math.min(cur.low,  c.low);
      cur.close = c.close;
    }
  }
  if(cur) out.push(cur);
  aggCache[m] = out;
  return out;
}

function setTf(m){
  const data = candlesAt(m);
  if(!data || !data.length) return;
  const vr = chart.timeScale().getVisibleRange();
  TF = m;
  cs.setData(data);
  buildTfBar();
  drawMarkers(); drawLevels(); drawPdhl();
  if(cur!==null && cur!==undefined){ const t=D.trades.find(x=>x.id===cur); if(t) focus(t); }
  else if(vr) chart.timeScale().setVisibleRange(vr);
}

drawMarkers();
drawLevels();
buildTfBar();

// 嵌在 iframe 裡就自動精簡（見 CSS .embed）。判斷用 window.self!==window.top，
// 不靠網站傳參數 —— 這樣 product-site 不用改任何一行。
try{ if(window.self !== window.top) document.body.classList.add('embed'); }
catch(e){ document.body.classList.add('embed'); }

// PDH/PDL 圖層。同樣 autoscaleInfoProvider:()=>null —— 前日高低常常遠在
// 顯示窗之外，讓它參與縮放會把 K 線壓成一條。
let showPdhl = false;
let pdhlSeries = [];
function drawPdhl(){
  for(const sx of pdhlSeries) chart.removeSeries(sx);
  pdhlSeries = [];
  if(!showPdhl || !D.pdhl) return;
  for(const L of D.pdhl){
    const sx = chart.addLineSeries({
      color: L.k==='PDH' ? '#26a69a' : '#ef5350',
      lineWidth:1, lineStyle:1, lineType:1,   // 1 = WithSteps，日界跳變
      lastValueVisible:false,
      priceLineVisible:false, crosshairMarkerVisible:false,
      autoscaleInfoProvider:()=>null});
    const seen = new Set(), pts = [];
    for(const p of L.pts){
      const t = bucket(p.time);
      if(seen.has(t)) continue;
      seen.add(t); pts.push({time:t, value:p.value});
    }
    sx.setData(pts);
    pdhlSeries.push(sx);
  }
}
drawPdhl();

document.getElementById('btnPdhl').onclick = function(){
  showPdhl = !showPdhl;
  this.classList.toggle('on', showPdhl);
  this.title = (showPdhl?'隱藏':'顯示') + ' PDH/PDL（僅顯示，不參與判定）';
  drawPdhl();
};

document.getElementById('btnSweep').onclick = function(){
  showSweep = !showSweep;
  this.classList.toggle('on', showSweep);
  this.title = (showSweep?'隱藏':'顯示') + '掃單價位與穿越箭頭';
  drawMarkers(); drawLevels();
};

const eqc = LightweightCharts.createChart(document.getElementById('eq'),
  Object.assign({}, dark, {layout:{background:{color:'#0b0e11'},textColor:'#848e9c',fontSize:10}}));
eqc.addLineSeries({color:'#0ecb81',lineWidth:2,title:'累積 ATR（毛）',
  priceLineVisible:false,crosshairMarkerVisible:false}).setData(D.equity);
chart.timeScale().subscribeVisibleLogicalRangeChange(r=>{if(r)eqc.timeScale().setVisibleLogicalRange(r);});

let lines = [], segs = [];
function clearLines(){
  for(const l of lines) cs.removePriceLine(l); lines=[];
  for(const s of segs) chart.removeSeries(s); segs=[];
}
// 這四條線**只畫在這一筆活著的那段時間上**，不橫跨整張圖。
// 無邊界的水平線（createPriceLine）會製造一個具體的誤讀：這一筆是時間
// 出場、停損從來沒被打到，但那條停損線一路延伸到右邊——而右邊後來價格
// 真的跌破它，看起來就像被停損掃掉了。線有沒有邊界，決定讀者以為
// 「這條線在講哪一段時間」。
function focus(t){
  clearLines();
  const g = SNAP();
  const t0 = bucket(t.t_anchor);
  const t1 = Math.max(t0 + g, bucket(t.t_exit) + g);
  const mk = (p,c,txt,st) => {
    const s = chart.addLineSeries({color:c, lineWidth:1,
      lineStyle:(st===undefined?2:st), lastValueVisible:true, title:txt,
      priceLineVisible:false, crosshairMarkerVisible:false});
    s.setData([{time:t0,value:p},{time:t1,value:p}]);
    segs.push(s);
  };
  // 2026-09-10 使用者：只要進場出場。價位線與停損線已移除（兩者的價格
  // 仍在下方單筆說明與表格裡，資料沒少，只是不畫在圖上）。
  mk(t.entry, t.R>0?'#0ecb81':'#f6465d','進場');
  mk(t.exit_px,'#f0b90b','出場 '+t.R.toFixed(3)+'ATR');
  // ±90 分鐘：一筆交易 62 分鐘，這個視野讓 5 分鐘 K 有 ~40 根、每根約 35
  // 像素，掃單／進場／出場才分得開（±3 小時時每根只剩 15 像素，還是擠）。
  const pad = 5400;
  chart.timeScale().setVisibleRange({from:t.t_anchor-pad, to:t.t_exit+pad});
  cur = t.id;
  const vis = D.trades.filter(keep), i = vis.findIndex(x=>x.id===t.id);
  if(i>=0) setPos(`${i+1} / ${vis.length}`);
  document.getElementById('sel').innerHTML =
    `<b>#${t.id+1} ${t.side}</b>${t.forward?' <span class="fwd">（前瞻）</span>':''}`+
    ` · ${t.sweep} 穿過${t.level_side==='buyside'?'買側':'賣側'}價位 ${t.level===null?'—':fmtP(t.level)}`+
    `（樞紐 ${t.origin}）+ ${t.sig.replace('sweep','S').replace('delta_ext','D').replace('vol_burst','V')}`+
    ` → 成立 ${t.anchor} → ${t.entry_t} 開盤進場 ${fmtP(t.entry)}`+
    ` → 停損掛 ${fmtP(t.stop)}（${D.params.STOP} × ATR ${fmtP(t.atr)}）`+
    ` → ${t.exit_t} ${t.stopped?'觸及停損':'持有 '+D.params.HOLD+' 分到期'} 出在 ${fmtP(t.exit_px)}，`+
    `毛 <b class="${t.R>0?'pos':'neg'}">${t.R.toFixed(4)} ATR</b>、`+
    `淨 <b class="${t.R_net>0?'pos':'neg'}">${t.R_net.toFixed(4)}</b>（成本 ${t.cost_bps} bps）`;
  for(const tr of document.querySelectorAll('#tb tr')) tr.classList.remove('sel');
  const row = document.getElementById('r'+t.id); if(row) row.classList.add('sel');
}

const sigShort = s => s.replace('sweep','S').replace('delta_ext','D').replace('vol_burst','V');
function table(){
  document.getElementById('tb').innerHTML = D.trades.filter(keep).map(t=>
    `<tr id="r${t.id}"><td>${t.id+1}</td>`+
    `<td class="${t.side==='LONG'?'pos':'neg'}">${t.side}</td>`+
    `<td>${sigShort(t.sig)}</td><td>${t.anchor}</td><td>${t.held}</td>`+
    `<td>${t.level===null?'—':fmtP(t.level)}</td><td>${fmtP(t.entry)}</td><td>${fmtP(t.stop)}</td>`+
    `<td>${fmtP(t.exit_px)}</td><td>${fmtP(t.atr)}</td>`+
    `<td class="${t.R>0?'pos':'neg'}">${t.R.toFixed(4)}</td>`+
    `<td class="${t.R_net>0?'pos':'neg'}">${t.R_net.toFixed(4)}</td>`+
    `<td>${t.stopped?'停損':'時間'}</td>`+
    `<td class="${t.forward?'fwd':''}">${t.forward?'前瞻':'樣本內'}</td></tr>`).join('');
  for(const tr of document.querySelectorAll('#tb tr'))
    tr.onclick = () => focus(D.trades[+tr.id.slice(1)]);
}
table();

// 不用「id 直接當全域變數」那種瀏覽器特有寫法——J2 的替身抓不到它，
// 而且它在真瀏覽器裡也只是碰巧能用。（`$` 宣告在檔案上方。）
function setF(f, btn){
  filt = f;
  for(const b of ['btnAll','btnWin','btnLose']) $(b).classList.toggle('on', b===btn);
  drawMarkers(); table(); nav(0);
}
$('btnAll').onclick=()=>setF('all','btnAll');
$('btnWin').onclick=()=>setF('win','btnWin');
$('btnLose').onclick=()=>setF('lose','btnLose');
function setG(g, btn){
  grp = g;
  for(const b of ['btnGA','btnGand','btnGd','btnGv']) $(b).classList.toggle('on', b===btn);
  kpis(); drawMarkers(); table(); nav(0);
}
$('btnGA').onclick=()=>setG('all','btnGA');
$('btnGand').onclick=()=>setG('and','btnGand');
$('btnGd').onclick=()=>setG('d','btnGd');
$('btnGv').onclick=()=>setG('v','btnGv');
$('btnFit').onclick=()=>{clearLines();chart.timeScale().fitContent();
  setPos('全景');
  $('sel').textContent='全景下一筆交易只有 62 分鐘 ≈ 12 根 K，標記會疊在一起 —— 用「逐筆看」或點下表任一列。';};

// 預設**不做 fitContent**：90 天 = 25,921 根 5 分 K 塞進一個畫面，一根 K
// 只有 0.05 像素，而一筆交易 62 分鐘 = 12 根 K = 0.67 像素 —— 掃單／進場／
// 出場三個標記在全景下疊成同一個點，看起來像「進出場位置不對」，
// 但計算層逐筆對回原始 1 分鐘 bar 全部吻合 —— 錯的是預設視野。
// 所以開頁就聚焦到最後一筆，並提供逐筆導航。（`cur` 宣告在檔案上方。）
function nav(step){
  const vis = D.trades.filter(keep);
  if(!vis.length){ setPos('0 筆'); return; }
  let i = vis.findIndex(t=>t.id===cur);
  i = (i<0) ? vis.length-1 : Math.min(vis.length-1, Math.max(0, i+step));
  cur = vis[i].id;
  setPos(`${i+1} / ${vis.length}`);
  focus(vis[i]);
}
$('btnPrev').onclick=()=>nav(-1);
$('btnNext').onclick=()=>nav(1);
$('fsPrev').onclick=()=>nav(-1);
$('fsNext').onclick=()=>nav(1);
nav(0);

// ── 全螢幕 ────────────────────────────────────────────────────────────
// 兩條路徑：原生 Fullscreen API（獨立開啟時、或 iframe 帶了 allowfullscreen
// 時可用），失敗就退回 .fs（position:fixed 撐滿）。嵌在網站裡而父層沒給
// 權限時，退路撐滿的是 iframe 那個框——仍然比 520px 高的圖好用很多。
// 圖表不會自己跟著容器變大：LightweightCharts 要被明確告知新尺寸，
// 所以每次切換都重算一次 #c 的實際框並 resize()。
const cbox=$('cbox'), btnFS=$('btnFS');
function fsSize(){
  const r=$('c').getBoundingClientRect();
  if(r.width>0 && r.height>0) chart.resize(r.width, r.height);
}
function fsLabel(on){ btnFS.textContent = on ? '⛶ 離開全螢幕' : '⛶ 全螢幕'; }
function fsOn(){ return !!(document.fullscreenElement===cbox
                           || cbox.classList.contains('fs')); }
function fsFallback(on){ cbox.classList.toggle('fs', on); fsLabel(on);
                         setTimeout(fsSize,0); }
btnFS.onclick=function(){
  if(fsOn()){
    if(document.fullscreenElement && document.exitFullscreen) document.exitFullscreen();
    else fsFallback(false);
    return;
  }
  if(cbox.requestFullscreen){
    const p=cbox.requestFullscreen();
    if(p && p.catch) p.catch(()=>fsFallback(true)); // 政策擋掉 -> 退路
  } else fsFallback(true);
};
document.addEventListener('fullscreenchange',function(){
  const on=!!document.fullscreenElement;
  if(on) cbox.classList.remove('fs');   // 原生生效就不要疊 CSS 那層
  fsLabel(on); setTimeout(fsSize,0);
});
document.addEventListener('keydown',function(e){
  if(e.key==='Escape' && cbox.classList.contains('fs')) fsFallback(false);
});

new ResizeObserver(()=>{chart.applyOptions({});eqc.applyOptions({});fsSize();})
  .observe(document.body);
</script></body></html>
"""


def render(d):
    p = d["params"]
    ck_ = d.get("clock", {})
    parts = []
    for stem, lab in (("conj_clock", "「或」時鐘"), ("conj_clock_and", "「且」時鐘")):
        c = ck_.get(stem)
        if c and c.get("n") is not None:
            parts.append(f"{lab} {c['n']}/{c['n_target']}（{c.get('verdict', '')}）")
    clock_txt = "現在 " + "、".join(parts) if parts else "進度見研究看板"
    return (TPL.replace("__DATA__", json.dumps(d, ensure_ascii=False, default=float))
            .replace("__SYM__", d["sym"])
            .replace("__TZ__", TZ_LABEL)
            .replace("__SPAN__", f'{d["span"][0]} → {d["span"][1]}')
            .replace("__DELAY__", str(p["DELAY"])).replace("__STOP__", str(p["STOP"]))
            .replace("__CW__", str(p["C_WAIT"])).replace("__CM__", str(p["C_MOVE"]))
            .replace("__HOLD__", str(p["HOLD"])).replace("__CM__", str(p["CANDLE_MIN"]))
            .replace("__FREEZE__", d["freeze_day"]).replace("__NFWD__", str(d["n_forward"]))
            .replace("__CLOCK__", clock_txt))


def publish(pages):
    """[(sym, html, asof_ts, n_view, n_all)] -> conj_backtest_pages（一列/幣）。"""
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(TABLE_DDL)
            for sym, html, asof, nv, na in pages:
                cur.execute(
                    "INSERT INTO conj_backtest_pages (sym, html, asof_ts, n_view, n_all) "
                    "VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                    "html=VALUES(html), asof_ts=VALUES(asof_ts), "
                    "n_view=VALUES(n_view), n_all=VALUES(n_all)",
                    (sym, html, int(asof), int(nv), int(na)))
        conn.commit()
    finally:
        conn.close()


def parse_day(s):
    return int(datetime.strptime(s, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--last-days", type=int, default=90)
    ap.add_argument("--arm", default="A", choices=("A", "C"),
                    help="A=現行（預設）；C=等它走出來（門檻已判過擬合，僅供對照）")
    ap.add_argument("--publish", action="store_true", help="寫進 conj_backtest_pages")
    a = ap.parse_args()

    syms = CORE9 if a.all else [a.symbol.upper()]
    OUT.mkdir(parents=True, exist_ok=True)
    pages = []
    for s in syms:
        t_from, t_to = parse_day(a.d_from), parse_day(a.d_to)
        if t_from is None:
            last = int(pd.read_parquet(BARS / f"{s}.parquet", columns=["ts"])["ts"].iloc[-1])
            t_from = last - a.last_days * 86_400_000
        d = build(s, t_from, t_to, arm=a.arm)
        html = render(d)
        f = OUT / f"conj_backtest_{s}.html"
        f.write_text(html, encoding="utf-8")
        m = d["groups"]["all"]["view"]
        ma = d["groups"]["all"]["all"]
        tail = (f"{m['n']} 筆  毛 {m['exp']:+.4f} 淨 {m['net_exp']:+.4f} ATR  "
                f"勝率 {m['wr']:.1f}%  停損率 {m['stop_rate']:.1f}%" if m else "0 筆")
        print(f"{s}: {d['span'][0]} → {d['span'][1]}  顯示窗 {len(d['candles']):,} 根 / {tail}")
        print(f"   全期 {ma['n']} 筆 毛 {ma['exp']:+.4f}   前瞻 {d['n_forward']} 筆   -> {f}")
        pages.append((s, html, d["asof_ts"], d["n_view"], d["n_all"]))
    if a.publish:
        publish(pages)
        print(f"published {len(pages)} page(s) -> conj_backtest_pages")


if __name__ == "__main__":
    main()
