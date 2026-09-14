# -*- coding: utf-8 -*-
"""Unified artifact-freshness board — one instrument for a disease that has
struck four times.

The recurring family (all in mistake.md / TODO):
  2026-07-05  DailyCollect pointed at a deleted path for 96 days, panel green
  2026-08-01  research-line Coinglass parquets rotted 5 days, scheduler green
  2026-08-19  CRLF-broken .bat killed the hourly train for 29h, State=Ready
  2026-08-20  v7-clock served a build-time snapshot (4/60 vs truth 34/60)

Every incident got its own ad-hoc freshness patch (published_utc,
upstream_live, per-file mtime rules) — one more stamp somebody must
remember to look at.  This file collapses the class: ONE frozen registry
of (artifact, expected cadence, observer), one table, one alert channel.

Design rules:
  - This script must NOT ride the hourly train it monitors — it runs on
    its own Windows schedule (every 6h) plus a line in the weekly clocks.
  - Alerts fire on TRANSITIONS only (new red, or recovery), deduped via a
    state file — a 6-hourly "still red" spam train teaches people to
    ignore the channel, which is how silent failures win.
  - Judging aliveness by PRODUCT freshness, never by scheduler panels
    (the 2026-08-19 rule).
  - A registry entry that cannot be measured (missing table/file) is RED,
    not skipped — absence of evidence is the failure mode here.

Known standing red: cg_fear_greed parquet (stale since 2026-04-13; pipeline
fix is a registered TODO).  It stays on the board and stays red — hiding a
known-bad line is how the next one hides too.

Run:  python research/freshness_board.py            # table + alerts
      python research/freshness_board.py --no-alert # table only
Exit: 0 all green, 1 any red.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

STATE = ROOT / "research" / "results" / "freshness_state.json"
OUT = ROOT / "research" / "results" / "freshness_board.json"

H = 3600.0


def _live_hmm_pair():
    """現在真正在跑的那支 HMM live 引擎的代號,或 None。

    **為什麼要算而不是寫死（2026-09-14）**：這一列原本寫死 `GMX`,而 GMX
    當天退場,於是看板掛著一盞永遠紅的燈。同一天這個形狀出現四次
    （AERO 的 Discord 看護、XPL 的看護、重啟 MON 的空窗、和這裡）,
    而代價不是噪音 —— **真的紅燈會被埋在假的紅燈裡**:scan_pull 連死
    31 次的那個下午,唯一在響的頻道報的是別的東西。

    真相源是看門狗的 `$Members` 減去 `logs/stop/*.stop`,也就是今天為
    「停止」建立的那個單一狀態。這裡是它的第四個讀者（.bat 的迴圈、
    arb_watchdog.ps1、account_budget.py，加上這支）。

    **解析不出來時回 None,而呼叫端會留一列紅的** —— 一列消失的守衛會被
    讀成「有人採用了它」,而不是「它壞了」（mistake.md 2026-09-04）。
    """
    import glob as _g
    import re as _re
    try:
        arb = os.path.join(ROOT, "..", "arb")
        wd = os.path.join(arb, "ops", "arb_watchdog.ps1")
        src = io.open(wd, encoding="utf-8").read()
        members = _re.findall(
            r"^\s*'([A-Za-z0-9_]+)'\s*=\s*@\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)",
            src, _re.M)
        if not members:
            return None
        stopped = {os.path.basename(f)[:-len(".stop")] + ".bat"
                   for f in _g.glob(os.path.join(arb, "engine", "logs",
                                                 "stop", "*.stop"))}
        for name, _sig, bat in members:
            if bat in stopped or not bat.startswith("run_hmm_"):
                continue
            body = io.open(os.path.join(arb, "engine", bat),
                           "rb").read().decode("ascii", "replace")
            # record-only / shadow 不是 live,而 live 才是這一列要盯的
            if "--record-only" in body or "--shadow" in body:
                continue
            return name
    except Exception:
        return None
    return None


_HMM = _live_hmm_pair()

# ── the frozen registry ──────────────────────────────────────────────────
# kind: file  = mtime of one file
#       glob  = mtime of the STALEST match (2026-04-12 lesson: health checks
#               must cover the weakest member, not the most reliable one)
#       db    = MAX(<col>) of <table>, DB clock (UTC)
# max_age_h picked from the cadence each producer already promises, plus
# slack for one late cycle — not tuned, and loosening one to silence a red
# is the anti-pattern this board exists to catch.
REGISTRY = [
    # -- the hourly local train (shadow_engine.bat) and its products --
    ("bat-train heartbeat", "file",
     "research/results/sweep_shadow_run.log", 2.5,
     "hourly train ran at all (29h outage family)"),
    ("sweep shadow log", "file",
     "research/results/sweep_shadow_log.csv", 2.5,
     "frozen shadow accounting"),
    ("kline caches", "glob",
     "research/sweep_failure/.cache/*_1h.csv", 3.5,
     "stalest coin of the 29; feeds every regime instrument"),
    ("weather station row", "db",
     "weather_station:updated_at", 2.5,
     "site survival card upstream"),
    ("raid signals row", "db",
     "raid_signals_live:updated_at", 2.5,
     "follow-bot signal surface"),
    ("v7 veto clock row", "db",
     "v7_veto_clock:updated_at", 2.5,
     "site trigger countdown (build-time-snapshot family)"),
    ("raid outcomes row", "db",
     "raid_outcomes:updated_at", 2.5,
     "skip-vs-taken scoring surface; silent staleness = consumer silently scores against stale outcomes"),
    # Cadence corrected 2026-09-03: the recorder rides the HOURLY train
    # (shadow_engine.bat), not a 10-min loop — with max 1.0h this row sat at
    # exactly 1.0h before every train and would have flapped red each hour
    # once the epoch bug above was fixed. Same limit as the other hourly rows.
    ("basis obs (§0.91)", "db",
     "basis_obs:ts_received", 2.5,
     "Bitget in-venue basis recorder (hourly, on shadow_engine.bat)"),
    # 2026-09-03: the V7 fill pipeline (TODO 0.81) sat broken for days
    # printing one skip line an hour -- MILL_EXPORT_UID was still the
    # account name after the product side went id-only. No artifact-age
    # rule could see it: "no fills yet" and "misconfigured" both produce
    # nothing. The producer now states its own health and this reads it.
    # 2026-09-05 (TODO 0.88d): GEX recorder. Two rows on purpose -- the DB
    # row says "a snapshot landed", the flag says "the recorder itself is ok"
    # (a failed Deribit call writes ok=false with the reason; the DB row alone
    # would just go stale, which is the shape that hid the §0.81 breakage).
    ("gex snapshots (§0.88d)", "db",
     "gex_snapshots:created_at", 2.5,
     "Deribit option OI+IV -> dealer gamma, hourly on shadow_engine.bat"),
    ("gex recorder flag (§0.88d)", "json_flag",
     "research/results/gex_last.json:ok", 2.5,
     "recorder self-report {ok, reason}; red = Deribit call or DB write failed"),
    # 2026-09-05 (出路研究線 A/C): 兩個 24/7 錄製器。judged by their own
    # self-report, not by artifact age -- a quiet market and a dead websocket
    # leave the same blank (mistake.md 2026-09-03).
    # 2026-09-06 看板覆核抓到兩個「從未被排程」的計分器：Q2 時鐘手動跑過一次就
    # 停在 09-01，Gate F 的月度計分器根本不在月度 cmd 裡。freshness 對「從未開始」
    # 是瞎的（mistake.md 2026-09-01），所以把它們的產物登記進來——下次再停會變紅。
    ("v7 Q2 clock json (§0.60)", "file",
     "research/results/v7_regime_q2_clock.json", 2.5,
     "每小時 shadow_engine.bat 的 v7_regime_q2_clock.py；紅 = 計分器停了"),
    ("sweep_forward gate json (Gate F)", "file",
     "research/results/sweep_forward_gate.json", 24 * 36,
     "每月 5 號 run_monthly_revalidation.cmd 的 sweep_forward.py；紅 = 月度沒跑"),
    # 2026-09-07 交會事件時鐘(TRIAGE.md)。它需要分鐘 bar + OI,而三個抓取器
    # 從來沒被任何排程叫到過(grep .bat/.ps1/.vbs 零命中)——時鐘會永遠停在
    # 0/300 而且不會有任何燈變紅。照 mistake.md 2026-09-01 的建議,把
    # 「從未開始」翻譯成「某個數字不對」:更新器自報旗標,這一列讀它。
    ("conj clock flag (交會前瞻)", "json_flag",
     "research/poc/data/results/conj_clock_last.json:ok", 30.0,
     "research/poc/conj_update.py 自報;紅 = 沒跑、抓取失敗、或資料 STALE"),
    # 2026-09-07 交會事件的分鐘級 shadow 偵測器(TODO 1.03)。每分鐘跑,
    # 目的是累積**真實端到端延遲**——2 分鐘的預算不能靠估算結案。
    # 它正常的輸出是 events=0(交會約每幣每 2.5 天一次),所以「有沒有產出
    # 資料」分不出死活;判準必須是它自報的 {ok,reason}(mistake.md 2026-09-03)。
    # 門檻 0.5h:每分鐘跑的東西超過半小時沒回報就是排程或啟動器死了。
    ("conj watch flag (交會 shadow)", "json_flag",
     "research/poc/data/results/conj_watch_last.json:ok", 0.5,
     "conj_watch.py 每分鐘自報;紅 = 排程沒跑、Binance 抓取失敗、或 DB 寫入失敗"),
    # 2026-09-10：研究端的回歸測試。這個 repo 至今沒有任何機制強制測試在改動
    # 後跑，而同日就發現一個釘死的 parity 測試早已變紅（基準釘在滾動資料窗上）
    # ——「守衛壞掉沒人發現」的第四次。run_guards.py 每日自報 {ok,...}，
    # 判準是產物不是退出碼。它也把「全部 skip」判成紅：一個都沒跑起來跟全過
    # 在輸出上長得一樣（mistake.md 2026-08-26）。
    # 2026-09-10：排班的每一個持有者。這條擋的是發生過三次的一整類事故
    # （07-05 排程指向改名前的路徑 96 天、09-04 搬線時 grep 抓不到排程的
    # action、09-10 daily_collect.bat 被 gitignore 所以排班內容沒有版控）。
    # 共同形狀是「關鍵狀態在版控與 grep 的範圍之外，失效時不報錯」。
    # 2026-09-10：研究資料的血統。擋的是「資料的形狀變了而消費者不知道」
    # ——.cache 是滾動 930 天窗（沒有任何檔案寫著）、輪替 CSV 讓下游從零重數、
    # 同一份資料兩份拷貝只有一份有人更新。登記簿裡每份資料要宣告 kind：
    # rolling 允許頭部前移、append 頭部前移就是紅、frozen 任何變動都是紅。
    # 2026-09-10：產品端還有沒有在成交。既有的 v7 export pipe 那一列綠著、
    # 旗標寫 HTTP 200，而三個 bot 的最後一筆 live 成交都在 08-30/31 ——
    # 管子通，裡面沒東西。門檻刻意寬（72h），因為磨坊有月磨損閘、單一策略
    # 閒置是正常的；紅的條件是**全部來源同時沉默**。
    # 2026-09-11：Hyperliquid 鏈上錄製。錄的四樣東西**都沒有歷史端點**
    # （market 的 OI、L2 簿口、清算價直方圖、觸發單），所以停一小時就永久
    # 少一小時。門檻 2.5h：每小時跑、一輪約 5 分鐘。
    # 2026-09-11：鏈上資料的**單位**驗證。使用者在開始累積歷史之前問
    # 「每筆的名目有沒有一樣」—— 那正是 mistake.md 2026-09-03 的坑
    # （把最小下單單位當成合約面值，一顆 BTC 的頂檔記成 $14.57）。
    # 十關每一關都對上一個**獨立發布的數字**，不是自我一致性。
    # 2026-09-13：累積快照自己也要被盯著。**衍生的判決檔沒有一個被盯著**
    # 是 mistake.md 2026-09-11 的原話，而那次有一份 json 凍了六天沒人發現，
    # 下游三份報告全跑在它上面。這一支是「監測監測的東西」，漏掉它就等於
    # 整個累積看板可以悄悄停掉而沒有任何燈變色。
    # 門檻 2.5h：它掛在每小時班車上，兩個半小時沒動就不是慢是死。
    ("累積快照 (accum)", "json_flag",
     "research/results/accum_snapshot_last.json:ok", 2.5,
     "accum_snapshot.py 每小時自報；紅 = 本支沒跑、或掃不到任何資料集。"
     "注意 ok 的語意是「儀器自己健康」不是「沒有洞」——洞在 hours_missing，"
     "那是被監測對象的事"),
    # 2026-09-13：**看板要監測自己的嘴。** 09-05~09-13 有 11 次狀態轉換
    # 全部沒送達，而唯一的痕跡是 log 裡一行 WARN —— 告警管道的失敗只有
    # 告警自己知道，那是一個完整的盲區（mistake.md 2026-07-05）。
    # 門檻 26h：心跳是每日的，所以超過一天沒有成功投遞就該紅。
    ("告警管道 (alert)", "json_flag",
     "research/results/alert_last.json:ok", 26.0,
     "notify.py 每次投遞後自報；紅 = 沒有設定任何管道、或投遞失敗。"
     "管道是 Discord（主）與 Telegram（備），設定走 env 或 .env"),
    ("hl 單位驗證", "json_flag",
     "research/results/hl_verify_last.json:ok", 26.0,
     "hl_verify.py 每日自報十關；紅 = 名目與 |數量|x價格 不符、szDecimals 違反、"
     "成交量對不上交易所公布值、時間戳單位錯、現貨混進永續、或止損方向反了"),
    ("hl onchain recorder", "json_flag",
     "research/results/hl_fuel_last.json:ok", 2.5,
     "hl_fuel_recorder.py 每小時自報；紅 = 覆蓋率掉到 1% 以下、清算價少於 100 筆、"
     "或幾何違反不為 0（多單清算價必在現價之下）"),
    # 2026-09-11：全市場成交帶。**沒有被註冊是怎麼被發現的**——它在
    # UTC 23:36 死掉，兩小時後我去查覆蓋率才看到，而看板全程 0 red。
    # hl_tape.py 的檔頭早就寫著「判準看旗標不看行程」，但那條線沒接上來。
    # 門檻 0.6h：落盤週期 5 分鐘，36 分鐘沒動就不是慢是死。
    # 它**不可回補**——WS 是串流，斷掉那段沒有任何端點補得回來。
    # 2026-09-11：分鐘級中價與佇列。**為什麼要另外錄** ——
    # 報酬目標用成交價算會被買賣價跳動污染，薄的標的會呈現比實際強得多的
    # 反轉（外部閱讀 HFT Alpha Research 101）。而我們的 §4.65 結論正是
    # 「分鐘級是反著做」，那是在 BTC/ETH 上量的；搬到 HL 長尾會中招。
    # 簿口沒有歷史端點 -> 不可回填 -> 停一小時永久少一小時。
    # 門檻 0.6h：取樣 60 秒、落盤 5 分鐘，36 分鐘沒動就是死了不是慢。
    ("hl mid/queue", "json_flag",
     "research/results/hl_mid_last.json:ok", 0.6,
     "hl_mid.py 常駐自報；紅 = 行程死了或 WS 卡住。宇宙是量能前 40 名"
     "（98.3% 的日成交額），重啟由 exit_paths_watchdog.ps1 負責"),
    ("hl trade tape", "json_flag",
     "research/results/hl_tape_last.json:ok", 0.6,
     "hl_tape.py 常駐自報；紅 = 行程死了或卡在 WS 讀取上。"
     "重啟由 exit_paths_watchdog.ps1 負責（每 5 分鐘），"
     "所以這一列紅超過一輪代表**重啟也失敗**，不只是剛好死掉"),
    # 2026-09-12：Lighter 全市場逐筆成交帶。**兩個開著的問題都卡在它身上**
    # （TODO §1.29 Lighter 上的做市毛利、§1.31 300ms 值幾 bps），而它跟
    # hl_tape 一樣是 WS 串流 -> **不可回填，停一小時永久少一小時**。
    # 門檻 0.6h：落盤週期 5 分鐘，36 分鐘沒動就不是慢是死。
    # 它比 HL 的成交帶多帶**雙方實付費率與成交前部位** —— 那是
    # `arblib/fee_receipts.py` 一直拿不到的收據查證，以及區分
    # 「在管庫存的人」與「在下方向的人」所需的欄位。
    ("lighter trade tape", "json_flag",
     "research/results/lighter_tape_last.json:ok", 0.6,
     "lighter_tape.py 常駐自報；紅 = 行程死了或卡在 WS 讀取上。"
     "宇宙是永續日成交額前 80 名（99.7% 的成交額，量測出來的）；"
     "重啟由 exit_paths_watchdog.ps1 負責"),
    # 2026-09-12：Lighter 分鐘級中價與深度。**它是成交帶的必要配套不是選配**
    # —— markout 必須用中價，成交價在薄標的上自帶負自相關、會偽裝成均值回歸
    # （mistake.md 2026-09-11）。簿口沒有歷史端點 -> 不可回填。
    # 門檻 0.6h：牆鐘 60 秒取樣、落盤 5 分鐘，36 分鐘沒動就是死了不是慢。
    # 它自己帶兩道自曝關：best_bid < best_ask（交錯就不寫那一列並計數）、
    # 以及 nonce 斷裂計數（斷了就清簿重訂閱，寧可空著不報假頂檔）。
    ("lighter mid/depth", "json_flag",
     "research/results/lighter_mid_last.json:ok", 0.6,
     "lighter_mid.py 常駐自報；紅 = 行程死了、WS 卡住、或所有市場都沒簿口。"
     "宇宙與 lighter_tape 同一組（永續前 80，刻意相同才 join 得起來）"),
    ("product live fills", "json_flag",
     "research/results/product_fills_last.json:ok", 26.0,
     "check_product_fills.py 每日自報；量的是**宣告與實際的落差**："
     "宣告在跑卻沉默 >72h -> 紅；宣告停用卻有成交 -> 紅。"
     "宣告在 research/product_expected.json（進版控，可稽核）"),
    ("research data manifest", "json_flag",
     "research/results/data_manifest.json:ok", 26.0,
     "data_manifest.py 每日自報；紅 = 凍結資料被動過、或 append 資料的頭部被吃掉"),
    ("scheduled task refs", "json_flag",
     "research/results/schedules_last.json:ok", 26.0,
     "check_schedules.py 每日自報；紅 = 排程引用的檔案不見了，或在 repo 內卻沒被追蹤"),
    ("research guards (回歸測試)", "json_flag",
     "research/results/guards_last.json:ok", 26.0,
     "research/run_guards.py 每日自報；紅 = 回歸測試有失敗、或一個都沒跑起來"),
    ("liq recorder flag (路徑C)", "json_flag",
     "research/results/liq_last.json:ok", 1.0,
     "OKX+Bybit 強平推送錄製器自報；紅 = WS 斷或 DB 寫入失敗"),
    ("lighter recorder flag (路徑A)", "json_flag",
     "research/results/lighter_last.json:ok", 1.0,
     "Lighter L2 × Binance 現貨 250ms 取樣；紅 = 任一側無幀"),
    ("v7 export pipe (§0.81)", "json_flag",
     "research/results/v7_product_trades_status.json:ok", 2.5,
     "product-side /export/v7 reachable AND configured (not: has rows)"),
    ("ops board row", "db",
     "ops_board:checked_at", 2.5,
     "operations surface (schedule + revalidation history) for the site"),
    ("arb status row", "db",
     "arb_status:checked_at", 2.5,
     "§0.75 family surface for the site (off-cloud recorder -> DB -> agent)"),
    ("prereg board row", "db",
     "prereg_clocks:updated_at", 2.5,
     "site research-progress board; a frozen board reads as 'no progress'"),
    # -- the cloud indicator service --
    # The degradation guard's own liveness (2026-09-01). It writes
    # checked_at every cycle even when nothing changed — a guard that only
    # stamps state CHANGES looks dead whenever the system is healthy, which
    # is exactly when you need to trust it.
    ("degradation guard", "db",
     "data_degradation_state:checked_at", 2.5,
     "§0.85 guard ran this cycle (not just: state last changed)"),
    ("indicator bars", "db",
     "indicator_history:dt", 2.5,
     "V7 inference alive (the honest liveness witness)"),
    # 2026-09-06 退役（不是刪除）：OKX 帳戶 2026-08-18 14:15:39 一秒內階躍到 $0
    # （提領，操作者 08-21 決定改用 Bitget），executor 對空帳戶報了 18 天 $0；
    # 09-05 的 Railway 重新部署讓它重跑 kill 檢查 → CAP-4 −100% → DEMOTE → WS 停。
    # 一個永遠紅的列跟壞掉的列一樣沒用（mistake.md 2026-09-03），所以門檻拉到
    # 「不會紅」但列保留——帳戶再入金、executor 重啟時把 1.5 改回來。
    ("okx balance snapshots (retired 09-06)", "db",
     "v7_okx_balance_snapshots:ts", 24 * 3650,
     "OKX 帳戶 08-18 起 $0、executor 09-05 DEMOTED；入金重啟後把門檻改回 1.5h"),
    ("cloud train parity", "db",
     "train_parity:updated_at", 2.5,
     "cloud recorder alive (weakness-#1 migration; RED until service up)"),
    ("arb universe recorder (§1.25)", "json_flag",
     "../arb/engine/logs/universe/_flag.json:ok", 0.6,
     "record_universe.py 常駐自報（150 個配對、264 個訂閱、三條 WS）。"
     "ok 的語意是「連得上且設定對」不是「有資料」——啟動那一瞬間就寫 ok=True，"
     "否則看門狗會殺掉剛起來的行程（mistake.md 2026-09-11）。"
     "重啟由 ../arb/ops/arb_watchdog.ps1 負責"),
    ("HMM 引擎 %s (§1.41)" % (_HMM or "**解析不出來**"), "json_flag",
     "../arb/engine/logs/%s/status.json:ok" % (_HMM or "__NO_LIVE_HMM__"), 0.2,
     # 標的**不寫死**（2026-09-14）：這一列原本寫 GMX,而 GMX 當天退場,
     # 於是看板掛著一盞永遠紅的燈 —— 而永遠紅的燈跟壞掉的燈一樣沒用
     # （mistake.md 2026-09-03）。現在從看門狗註冊表減去 STOP 檔算出來,
     # 見本檔的 _live_hmm_pair()。算不出來時路徑會指向一個不存在的目錄,
     # 那一列因此變紅 —— **那是要的**:沒有 live HMM 引擎、或解析壞了,
     # 兩者都該被看見,而不是讓這一列安靜消失。
     "HMM（對沖做市）的引擎自報。**盯的是引擎不是錄製器** —— 看板既有那幾列"
     "讀的是 minutes.csv，而那是 recorder 寫的，引擎的策略層死掉它照樣更新。"
     "ok 的語意是「連得上且設定對」不是「有成交」：只有 RED guard 會讓它 false，"
     "所以安靜的市場不會看起來像故障（mistake.md 2026-09-03）。"
     "帳戶級閘門（B6）擋住開倉時會在這裡以 ACCOUNT OUT OF MARGIN / "
     "ACCOUNT CAP HIT 現形 —— 那是「引擎跑著但什麼都不送」唯一看得見的地方。"
     "重啟由 ../arb/ops/arb_watchdog.ps1 的 HMM_GMX 負責。"
     "**另外九支錄製器還沒加**：它們跑的是沒有 ok 欄位的舊碼，現在加會誤報紅"),
    ("arb 帳戶額度加總 (B6)", "json_flag",
     "../arb/results/account_budget.json:ok", 0.5,
     "引擎的每一道風控閘門都是**逐行程**的（cap_usd 逐場館、max_gross_usd "
     "逐行程），而一個行程只跑一個 ticker —— N 個市場 = N 個行程共用同一個 "
     "Lighter/HL 帳號，五個各守 $1,000 的行程在帳戶層可以是 $5,000。"
     "`engine/tools/account_budget.py` 每 5 分鐘由 arb_watchdog 跑一次，"
     "把 live 行程的額度按資金池加總（B1 開關齊全 / B2 同池同天花板 / "
     "B3 Σ 不超過天花板 / B4 沒有不在註冊表裡的 live 啟動器）。"
     "執行期那一半在 entropy_arb/account.py（讀交易所回報的帳戶層曝險）"),
    ("arb recorder (§0.75)", "file",
     "../arb/engine/logs/minutes.csv", 1.0,
     "two-venue premium recording; silence = the week of data quietly stops"),
    ("arb recorder NBIS (§0.75)", "file",
     "../arb/engine/logs/NBIS/minutes.csv", 1.0,
     "§0.75 family 2026-08-30: io:NBIS vs lighter"),
    ("arb recorder ANTH (§0.75)", "file",
     "../arb/engine/logs/ANTH/minutes.csv", 1.0,
     "§0.75 family 2026-08-30: io:ANTH vs lighter-rh ANTHROPIC"),
    ("arb recorder BTC (§0.75)", "file",
     "../arb/engine/logs/BTC/minutes.csv", 1.0,
     "§0.75 family 2026-08-30: CONTROL pair — band must stay ~0"),
    ("arb recorder ZEC (§0.75)", "file",
     "../arb/engine/logs/ZEC/minutes.csv", 1.0,
     "§0.75 family 2026-08-30: HL vs lighter-rh, thin"),
    ("arb recorder NEAR (§0.75)", "file",
     "../arb/engine/logs/NEAR/minutes.csv", 1.0,
     "§0.75 family 2026-08-30: HL vs lighter-rh, thin"),
    ("arb recorder HYPE (§0.75)", "file",
     "../arb/engine/logs/HYPE/minutes.csv", 1.0,
     "§0.75 family 2026-09-01: largest funding gap in the snapshot"),
    # "glob" = stalest match (right when every file must stay fresh). The
    # scanner ROTATES daily, so yesterday's file is stale BY DESIGN — the
    # stalest rule makes this row permanently red, which trains the
    # operator to ignore the channel (the exact failure this board exists
    # to prevent). "glob_newest" asks the real question: is the CURRENT
    # file being written?
    ("arb recorder GOLD_LL (§1.02)", "file",
     "../arb/engine/logs/GOLD_LL/minutes.csv", 1.0,
     "zero-fee control: lighter XAU vs lighter-rh XAU"),
    ("arb recorder NVDA_LL (§1.02)", "file",
     "../arb/engine/logs/NVDA_LL/minutes.csv", 1.0,
     "zero-fee control: lighter NVDA vs lighter-rh NVDA"),
    ("arb recorder %s (§1.41b)" % (_HMM or "**解析不出來**"), "file",
     "../arb/engine/logs/%s/minutes.csv" % (_HMM or "__NO_LIVE_HMM__"), 1.0,
     # 同樣不寫死（2026-09-14）：原本是 MET,而 MET 當天退場 -> 永遠紅。
     "HMM 候選，2026-09-14 開錄。它要回答 GMX 死掉的那一關：premium 會不會"
     "震盪（GMX 173 分鐘裡 96% 為負 -> 只能單邊賣 -> 4 張滿了就卡住）。"
     "判準凍結在 arb/arblib/hmm_screen.py。**它的 samples 跟凍結的九支不可比**"
     "——MET 用 HMM 的條件錄（staleness 30s ＋ 5 秒心跳），因為用錄製家族的"
     "條件會跳過約 40% 的秒，而被跳過的正好是安靜的那些，那會讓 premium 的"
     "符號分佈偏向活躍時段。要跟那九支比的是 GMX，不是 MET"),
    ("arb scanner (§0.75b)", "glob_newest",
     "../arb/engine/logs/scan/scan_*.csv", 0.5,
     "跨場館 REST 掃描器。**2026-09-13 起它跑在 Railway 上**，本機這些檔是"
     "拉回來的（檔名帶 _rw，見 ../arb/tools/scan_pull.py）。搬家的理由是"
     "per-IP 的 WAF 預算：掃描器一支 65 次/分，是十支引擎合計的十六倍，"
     "而被擋住的是**引擎的重連**，那時引擎手上有部位（docs/DEPLOY.md §6）。"
     "節奏同時從 180 秒放寬到 600 秒 —— §1.39 量到這支掃描器自己的腿差就有"
     "53 秒，所以 180 秒的格本來就比它能分辨的細。"
     "**這一列答的是「本機有沒有收到新資料」**，斷掉的原因看下一列"),
    ("arb 掃描器拉取 (Railway)", "json_flag",
     "../arb/results/scan_pull_last.json:ok", 0.3,
     "每 5 分鐘由 arb_watchdog.ps1 拉一次 Railway 掃描器的產物。"
     "**判準是本機的位元組有沒有在長，不是遠端有沒有回 200** —— Railway "
     "活著而拉取斷了的話，本機資料會靜靜地停在昨天，而十個消費者一個都"
     "不會報錯（mistake.md 2026-08-29：計數類的下游對這種病的反應不是壞掉，"
     "是從零重數）。旗標每一輪都寫，所以 mtime 變舊本身就是拉取沒在跑。"
     "reason 會說拉了幾個位元組、或遠端讀不到"),
    # -- daily --
    # 2026-09-01 (§0.85): mtime answers "is the writer running"; content
    # age answers "is the data moving".  During an upstream outage those
    # diverge — the collector keeps rewriting files whose newest row never
    # advances (2026-08-01 precedent: schedule green, parquet stale).
    ("coinglass parquet CONTENT", "parquet_content",
     "market_data/raw_data/cg_*.parquet", 48.0,
     "last DATA row age of the stalest CG parquet — mtime lies in an outage"),
    ("coinglass parquets", "glob",
     "market_data/raw_data/cg_*.parquet", 48.0,
     "STALE-DATA guard threshold; stalest file reported"),
    ("daily collect log", "file",
     "research/results/daily_collect.log", 30.0,
     "04:00 daily task heartbeat"),
    # 2026-09-02: the board never watched ITSELF. If the 6-hourly task
    # stopped, every row below would freeze at its last good value and the
    # panel would keep showing green — the exact failure this file exists
    # to catch, applied to the file itself (quis custodiet). Its own JSON
    # output is the artifact.
    ("freshness board self", "file",
     "research/results/freshness_board.json", 7.0,
     "the board's own 6-hourly run — nothing else watches the watchman"),
    # -- monthly --
    # The revalidation is the only periodic check of the model's SCALE
    # (rank metrics are blind to level drift — mistake.md 2026-08-08, which
    # went unnoticed for three months). 35 days = one month plus slack.
    # glob_newest, not glob: reports ACCUMULATE, so the stalest match is
    # June's and always will be. Same trap as the arb scanner's daily
    # rotation (2026-09-01) — "stalest" is right only when every file must
    # stay fresh.
    ("revalidation report", "glob_newest",
     "research/results/dual_model/quarterly_revalidation_*.md", 840.0,
     "monthly model revalidation actually produced a report"),
    # -- weekly --
    ("portfolio clocks", "file",
     "research/results/portfolio_clocks.log", 195.0,
     "Monday 09:30 weekly report ran (8d + slack)"),
    # -- slow guards --
    ("tracked signals", "db",
     "tracked_signals:signal_time", 336.0,
     ">=14d without ANY signal = decode locked again (TODO rule 7)"),
]


def age_file(rel: str) -> float | None:
    p = ROOT / rel
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / H


def age_glob_newest(pattern: str):
    """Age of the NEWEST match — for rotating files (daily scan_YYYYMMDD.csv)
    where older members are stale by design."""
    import glob as _glob
    import os as _os
    files = _glob.glob(pattern)
    if not files:
        return None, "no files"
    newest = max(files, key=lambda f: _os.path.getmtime(f))
    return ((time.time() - _os.path.getmtime(newest)) / H,
            _os.path.basename(newest))


def age_parquet_content(pattern: str):
    """Age of the LAST DATA ROW in the stalest matching parquet, in hours.

    Reads the newest timestamp INSIDE each file rather than its mtime, so
    an upstream outage that keeps the writer alive (fresh mtime, frozen
    content) still goes red. Failure to read a file is skipped, not
    fatal — this is a monitor, it must not become the thing that breaks.
    """
    import glob as _glob
    import os as _os
    files = _glob.glob(pattern)
    if not files:
        return None, "no files"
    worst_age, worst_name = None, "?"
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                age = 1e9
            else:
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex):
                    cand = [c for c in df.columns
                            if pd.api.types.is_datetime64_any_dtype(df[c])]
                    if not cand:
                        continue
                    idx = pd.DatetimeIndex(df[cand[0]])
                last = idx.max()
                if last.tzinfo is None:
                    last = last.tz_localize("UTC")
                age = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600
        except Exception:
            continue
        if worst_age is None or age > worst_age:
            worst_age, worst_name = age, _os.path.basename(f)
    return worst_age, worst_name


def age_glob(pattern: str) -> tuple[float | None, str]:
    files = list(ROOT.glob(pattern))
    if not files:
        return None, "(no match)"
    stalest = min(files, key=lambda f: f.stat().st_mtime)
    return (time.time() - stalest.stat().st_mtime) / H, stalest.name


def age_json_flag(spec: str):
    """Age + a boolean health flag out of a small status artifact.

    For pipelines whose failure mode is "it never produced anything at all"
    (mistake.md 2026-09-01): an mtime rule cannot tell "no rows yet, which
    is legitimate" from "misconfigured, which is a bug", because both leave
    the same absence. So the producer writes {ok: bool, reason: str} every
    run and this reads the flag. Missing file = RED (absence of evidence is
    the failure mode here, per the module docstring).

    spec: "<path relative to repo root>:<boolean key>"
    """
    rel, key = spec.rsplit(":", 1)
    p = ROOT / rel
    if not p.exists():
        return None, "(no status file)", False
    age = (time.time() - p.stat().st_mtime) / H
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return age, f"unreadable: {exc}"[:60], False
    return age, str(d.get("reason") or "")[:60], bool(d.get(key))


def age_db(spec: str, conn) -> float | None:
    table, col = spec.split(":")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX({col}) m FROM {table}")   # noqa: S608
            row = cur.fetchone()
        m = row and row.get("m")
        if m is None:
            return None
        if isinstance(m, (int, float)) and not isinstance(m, bool):
            # Epoch columns (basis_obs.ts_received is BIGINT). Before
            # 2026-09-03 this fell into fromisoformat, threw, and the row
            # was RED forever -- a guard that cannot go green cannot detect
            # a real death either (the recorder was alive the whole time).
            # Seconds vs milliseconds: anything past ~2001 in seconds is
            # < 1e12; treat larger values as ms.
            secs = float(m) / (1000.0 if float(m) > 1e12 else 1.0)
            m = datetime.fromtimestamp(secs, tz=timezone.utc)
        elif not isinstance(m, datetime):
            m = datetime.fromisoformat(str(m))
        return (datetime.now(timezone.utc)
                - m.replace(tzinfo=timezone.utc)).total_seconds() / H
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-alert", action="store_true")
    args = ap.parse_args()

    conn = None
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
    except Exception:
        pass                      # every db row will go red, correctly

    rows, reds = [], []
    for name, kind, target, max_h, note in REGISTRY:
        detail, flag = "", True
        if kind == "file":
            age = age_file(target)
        elif kind == "glob":
            age, detail = age_glob(target)
        elif kind == "parquet_content":
            age, detail = age_parquet_content(target)
        elif kind == "glob_newest":
            age, detail = age_glob_newest(target)
        elif kind == "json_flag":
            age, detail, flag = age_json_flag(target)
        else:
            age = age_db(target, conn) if conn else None
        ok = age is not None and age <= max_h and flag
        rows.append({"name": name, "age_h": None if age is None
                     else round(age, 1), "max_h": max_h, "ok": ok,
                     "detail": detail, "note": note})
        if not ok:
            reds.append(name)
    if conn:
        conn.close()

    print(f"freshness board — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"{'artifact':22} {'age':>8} {'limit':>7}  status")
    for r in rows:
        a = "   MISSING" if r["age_h"] is None else f"{r['age_h']:7.1f}h"
        s = "ok" if r["ok"] else "RED  <-- " + r["note"][:60]
        print(f"{r['name']:22} {a:>9} {r['max_h']:6.1f}h  {s}"
              + (f"  [{r['detail']}]" if r["detail"] and not r["ok"] else ""))
    print(f"\n{len(reds)} red / {len(rows)} tracked"
          + (f": {', '.join(reds)}" if reds else ""))

    OUT.write_text(json.dumps({
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "rows": rows, "reds": reds}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # ── transition-only alerting ─────────────────────────────────────────
    prev, last_hb = set(), 0.0
    if STATE.exists():
        try:
            _st = json.loads(STATE.read_text())
            prev = set(_st["reds"])
            last_hb = float(_st.get("last_heartbeat", 0) or 0)
        except Exception:
            prev, last_hb = set(), 0.0
    cur = set(reds)
    new_red = sorted(cur - prev)
    recovered = sorted(prev - cur)

    # **每日心跳。** 沒有它，沉默分不出「沒事」與「管道又死了」—— 而那正是
    # 2026-09-05~09-13 那 8 天的形狀（每 6 小時跑、轉換有發生、投遞全失敗，
    # 而畫面上跟健康完全一樣）。心跳讓**沉默本身變成警報**。
    _now = time.time()
    # 2026-09-13：station_post 現在每小時推一張圖（使用者：「圖表每小時更新
    # 一次就好了」），所以這裡的每日心跳變成**備援**：station 最近 6 小時
    # 推過就不要再貼一次文字版。留著而不是刪掉，是因為兩者跑在**不同排程**
    # 上（station 在 SweepShadow、本支在 FreshnessBoard）—— 班車停了的時候
    # 這條還會出聲。
    _station_fresh = False
    try:
        _af = ROOT / "research" / "results" / "alert_last.json"
        if _af.exists():
            _aj = json.loads(_af.read_text(encoding="utf-8"))
            if _aj.get("source") == "station" and _aj.get("delivered"):
                _station_fresh = (_now - _af.stat().st_mtime) < 6 * 3600
    except Exception:                                   # noqa: BLE001
        _station_fresh = False
    _now = time.time()
    _send_hb = ((not args.no_alert) and (not _station_fresh)
                and (_now - last_hb > 20 * 3600))
    if _send_hb:
        try:
            sys.path.insert(0, str(ROOT))
            from research.ops import notify as _nt
            # 心跳的內容就是**資料監控站**（使用者 2026-09-13 把 Discord
            # 那個頻道從「V7 每小時圖表」改成這個用途）。station_text() 只讀
            # 現成的兩份 json，不重算任何東西 —— 第二份實作會安靜地跟第一份
            # 不一致（mistake.md 2026-08-26）。
            _txt = ("flowbot 資料監控站 — %d red / %d tracked\n"
                    % (len(reds), len(rows)))
            try:
                _txt += _nt.station_text()
            except Exception as _e2:            # noqa: BLE001
                _txt += "station_text 失敗：%s" % _e2
                if reds:
                    _txt += "\n  紅：" + ", ".join(reds[:8])
            # **回報用圖表**（使用者 2026-09-13）。圖畫不出來就退回純文字 ——
            # 一個只會用圖回報的管道，在畫圖壞掉的那天就完全沉默了。
            _sent = False
            try:
                from research.ops import accum_png as _ap
                _p, _sn, _age = _ap.build()
                _cap = ("flowbot 資料監控站 — %d red / %d tracked"
                        % (len(reds), len(rows)))
                if reds:
                    _cap += "  |  紅：" + ", ".join(reds[:6])
                _sent = _nt.send_image(_p, _cap, source="station")["delivered"]
            except Exception as _e3:            # noqa: BLE001
                print("[WARN] 監控站圖失敗，退回文字：%s" % _e3)
            if _sent or _nt.heartbeat(_txt)["delivered"]:
                last_hb = _now
                # log 不可以說謊：送的是圖的時候，別印一整段文字裝成是它送的。
                print("heartbeat DELIVERED (%s): %s"
                      % ("圖" if _sent else "文字", _txt.split("\n")[0]))
            else:
                print("[WARN] heartbeat NOT delivered")
        except Exception as _e:       # noqa: BLE001
            print("[WARN] heartbeat error: %s" % _e)

    STATE.write_text(json.dumps({"reds": sorted(cur),
                                 "last_heartbeat": last_hb}), encoding="utf-8")

    if not args.no_alert and (new_red or recovered):
        msg_lines = ["Freshness board transition:"]
        for n in new_red:
            r = next(x for x in rows if x["name"] == n)
            a = "MISSING" if r["age_h"] is None else f"{r['age_h']:.1f}h"
            msg_lines.append(f"  RED: {n} (age {a}, limit {r['max_h']:.0f}h)"
                             f" — {r['note']}")
        for n in recovered:
            msg_lines.append(f"  recovered: {n}")
        msg = "\n".join(msg_lines)
        try:
            # 2026-09-13：改走 research/ops/notify.py。舊版只讀 os.environ，
            # 而 chat id 只在 .env 裡 -> chat="" -> **連嘗試都沒嘗試**，
            # 於是 09-05 到 09-13 有 11 次轉換全部沒送達，唯一痕跡是一行
            # WARN。notify.cfg() 做 env -> .env 回退，並把投遞結果寫成
            # alert_last.json（本檔下方已把它註冊成一列，所以「告警送不出去」
            # 自己會變成一盞紅燈）。
            sys.path.insert(0, str(ROOT))
            from research.ops import notify
            sent = notify.send(msg, source="freshness")["delivered"]
            # Success must leave a trace too — during the 08-21/22 outage
            # the log could not answer "did the alert deliver?" because
            # success printed nothing. An alert channel whose delivery is
            # unverifiable is itself a silent-failure surface.
            print(("alert DELIVERED: " if sent
                   else "[WARN] freshness alert NOT delivered: ") + msg)
        except Exception as e:  # noqa: BLE001
            print("[WARN] freshness alert failed:", e)

    return 1 if reds else 0


if __name__ == "__main__":
    raise SystemExit(main())
