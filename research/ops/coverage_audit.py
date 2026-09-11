# -*- coding: utf-8 -*-
"""覆蓋率與缺口稽核（2026-09-11，使用者訂立四個軸）

===========================================================================
為什麼需要這一支，而既有的 freshness board 不夠
===========================================================================
`freshness_board` 量的是**最後一筆有多新**。它對三種病完全免疫：

  1. **標的少了一個**：11 檔變 10 檔，最後一筆一樣新。
     實際發生：`depth_deltas_1m` 只有 10 檔而 `orderbook_snapshots_1m`
     有 11 檔 —— 少的是 SOL，而這兩張表是撤單流的**成對輸入**。
  2. **中間有洞**：排程斷過幾小時，最後一筆還是新的。
  3. **存的是聚合不是原始**：桶子一樣每分鐘更新，但逐筆永遠回不來。

使用者 2026-09-11 訂立的四個軸，本檔逐一實作：

  二、**存原始事件還是聚合桶** —— `GRAIN` 欄位。不可逆，所以要早查。
  三、**覆蓋率與缺口分布** —— 不是總列數，是「應有 vs 實有」，而且要看
      缺口集中在哪。**偏誤方向比缺口大小更重要**。
  四、**靜默失敗防護** —— 每一項先宣告預期標的數與時間範圍，
      **實際低於宣告就紅**。不宣告就沒有東西會發現少了。

===========================================================================
宣告（EXPECT）的來源，以及它為什麼必須寫在這裡
===========================================================================
宣告不是我猜的，每一條都有出處（寫在 `why` 欄）。把它寫死在程式碼裡而不是
散文裡，是因為**程式碼會被代入數字，散文不會**（mistake.md 2026-08-26）。

宣告與現實不符時有兩種可能，而它們的處置相反：
  - 現實少於宣告 -> **管線有缺口**，要修管線
  - 宣告本身過時（例如標的清單縮編過）-> 要**改宣告並寫下理由**
所以紅燈的訊息一律同時印出「宣告是多少、出處是哪裡」，讓下一個人能判斷
該修哪一邊。不准為了變綠而默默改宣告。
"""
from __future__ import annotations

import datetime
import glob
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "research" / "results" / "coverage_audit.json"

# ===========================================================================
# 宣告表
#   name      顯示名
#   table     MySQL 表名（None = 檔案側）
#   sym_col   標的欄（None = 單標的或不分標的）
#   ts_col    時間欄
#   unit      "ms" = epoch 毫秒 / "dt" = DATETIME
#             **這一欄咬過**：同一個註冊表裡混著兩種型別，用錯會得到 0
#             而 0 看起來像「沒資料」（mistake.md 2026-09-03）
#   n_sym     宣告的標的數（None = 不檢查）
#   per_day   宣告的每標的每日列數（None = 事件驅動，不檢查）
#   grain     raw = 逐筆原始事件 / bucket = 聚合桶 / snapshot = 快照
#   why       宣告的出處
# ===========================================================================
EXPECT = [
    dict(name="清算逐筆 liq_events", table="liq_events", sym_col="symbol",
         ts_col="ts_event", unit="ms", n_sym=2, per_day=None, grain="raw",
         why="逐筆(ts_event/price/qty/notional)＝Hawkes 需要的形狀。"
             "n_sym 宣告的是**場館數≥2**(okx+bybit)，標的數是市場決定的不宣告"),
    dict(name="清算桶 liquidation_1m", table="liquidation_1m",
         sym_col="canonical_symbol", ts_col="window_start", unit="ms",
         n_sym=2, per_day=None, grain="bucket",
         why="BTC/ETH 兩檔。per_day 不宣告：沒有清算的分鐘不寫列，"
             "稀疏是正常的，宣告 1440 會製造永遠紅的燈"),
    dict(name="撤單流 depth_deltas_1m", table="depth_deltas_1m",
         sym_col="canonical_symbol", ts_col="created_at", unit="dt",
         n_sym=11, per_day=None, grain="bucket",
         why="cleanup.py 的註解寫「11 檔 × 1440 列/日」，而"
             "orderbook_snapshots_1m 實測就是 11 檔 —— 兩張表是成對輸入"),
    dict(name="簿口快照 orderbook_snapshots_1m", table="orderbook_snapshots_1m",
         sym_col="canonical_symbol", ts_col="created_at", unit="dt",
         n_sym=11, per_day=None, grain="snapshot",
         why="cleanup.py 2026-07-28 註解：追蹤標的從 2 檔擴到 11 檔"),
    dict(name="秒級深度 depth_events_1s", table="depth_events_1s",
         sym_col="canonical_symbol", ts_col="created_at", unit="dt",
         n_sym=1, per_day=None, grain="raw",
         why="實測只有 BTC。宣告 1 是**記錄現狀**不是背書 —— "
             "它佔全庫 34% 且沒有保留策略，擴標的之前要先決定上限"),
    dict(name="流量桶 flow_bars_1m", table="flow_bars_1m",
         sym_col="canonical_symbol", ts_col="window_start", unit="ms",
         n_sym=2, per_day=1440 * 3, grain="bucket",
         why="BTC/ETH x **三個 exchange_scope（all/bybit/okx）** x 每分鐘一列。"
             "data-model.md 寫「always combined」已過時（2026-09-11 實測三種）。"
             "唯一(標的,分鐘)組合 345,152 = 119.8 天 x 1440 x 2 幣，資料完整"),
    dict(name="未平倉 oi_snapshots", table="oi_snapshots",
         sym_col="canonical_symbol", ts_col="ts_received", unit="ms",
         n_sym=2, per_day=None, grain="snapshot", why="BTC/ETH"),
    dict(name="資金費 funding_rates", table="funding_rates",
         sym_col="canonical_symbol", ts_col="ts_received", unit="ms",
         n_sym=2, per_day=None, grain="snapshot", why="BTC/ETH"),
    dict(name="GEX gex_snapshots", table="gex_snapshots", sym_col=None,
         ts_col="created_at", unit="dt", n_sym=None, per_day=None,
         grain="snapshot",
         why="§0.88d。實測 288 列/6.2 天 = 46.5/日，不是每小時一列 —— "
             "兩個到期日還是 call/put 各一列還沒查清，**所以不宣告一個沒查證"
             "的數字**。活性由 freshness 的『gex recorder flag』那一列負責"),
    dict(name="站內資金費 basis_obs", table="basis_obs", sym_col=None,
         ts_col="ts_received", unit="ms", n_sym=None, per_day=240,
         grain="snapshot", why="§0.91 每輪 10 列 × 每小時 = 240/日"),
    dict(name="交會 shadow conj_events_live", table="conj_events_live",
         sym_col=None, ts_col="event_ts", unit="ms", n_sym=None, per_day=None,
         grain="raw",
         why="事件驅動，約每幣 2.5 天一次，**不宣告 per_day** —— "
             "宣告了就是一盞永遠紅的燈。它的活性由 conj_watch 旗標負責"),
    dict(name="撤單事件 cancel_playbook_events", table="cancel_playbook_events",
         sym_col="canonical_symbol", ts_col="created_at", unit="dt",
         n_sym=2, per_day=None, grain="raw", why="BTC/ETH"),
]

# 檔案側：宣告「應該有幾個檔」
EXPECT_FILES = [
    dict(name="九幣分鐘 bar", pat="research/poc/data/bars/*.parquet", n=9,
         grain="bucket", why="core9 凍結宇宙"),
    dict(name="九幣 OI", pat="research/poc/data/oi/*.parquet", n=9,
         grain="snapshot", why="core9"),
    dict(name="九幣樞紐價位 levels", pat="research/poc/data/levels/*.parquet",
         n=9, grain="derived",
         why="core9。2026-09-08 才接進 conj_update —— 之前它停在手動跑的那天，"
             "時鐘因此結構上不可能累積（mistake.md 同日）"),
    dict(name="九幣掃單事件 events", pat="research/poc/data/events/*.parquet",
         n=9, grain="derived", why="同上"),
    dict(name="九幣清算 liq", pat="research/poc/data/liq/*.parquet", n=9,
         grain="raw", why="core9"),
    dict(name="30 幣 1h 快取", pat="research/sweep_failure/.cache/*USDT_1h.csv",
         n=29, grain="bucket",
         why="實測 29 檔。宣告 29 是記錄現狀 —— 30 幣宇宙裡有一檔沒有快取，"
             "要嘛補抓要嘛把宇宙改成 29 並寫下理由"),
    dict(name="HL 歷史 K 線 1h", pat="research/hl/data/candles/*_1h.parquet",
         n=178, grain="bucket",
         why="HL universe 234 個，其中 **56 已下市**，main_coins() 回 178。"
             "下市標的沒有 K 線可抓，所以 178 就是滿分。"
             "**注意不對稱且它是對的**：部位與市場快照記全部 234（含下市），"
             "因為下市標的必須留在樣本裡（factor-research §2 存活者偏誤）。"
             "端點保留上限 5000 根 = 1h 只有 208 天"),
    dict(name="HL 逐部位快照", pat="research/hl/data/positions/*.parquet",
         n=1, grain="raw",
         why="每小時一檔，只宣告 >=1（它從 2026-09-11 才開始，"
             "宣告固定數量會永遠紅）"),
]


def g(r, k, i):
    return r[k] if isinstance(r, dict) else r[i]


def audit_db():
    from shared.db import get_db_conn
    conn = get_db_conn()
    cur = conn.cursor()
    res = []
    for e in EXPECT:
        row = dict(e)
        try:
            sc = e["sym_col"]
            if sc:
                cur.execute("SELECT COUNT(DISTINCT `%s`) AS s FROM `%s`"
                            % (sc, e["table"]))
                row["sym_actual"] = int(g(cur.fetchone(), "s", 0))
            else:
                row["sym_actual"] = None
            tc, unit = e["ts_col"], e["unit"]
            cur.execute("SELECT MIN(`%s`) AS a, MAX(`%s`) AS b, COUNT(*) AS n "
                        "FROM `%s`" % (tc, tc, e["table"]))
            r = cur.fetchone()
            a, b, n = g(r, "a", 0), g(r, "b", 1), int(g(r, "n", 2))
            row["rows"] = n
            if a is None:
                row["span_days"] = 0.0
                row["first"] = row["last"] = None
            elif unit == "ms":
                row["span_days"] = (int(b) - int(a)) / 86400000.0
                row["first"] = datetime.datetime.utcfromtimestamp(
                    int(a) / 1000).strftime("%Y-%m-%d %H:%M")
                row["last"] = datetime.datetime.utcfromtimestamp(
                    int(b) / 1000).strftime("%Y-%m-%d %H:%M")
            else:
                row["span_days"] = (b - a).total_seconds() / 86400.0
                row["first"], row["last"] = str(a)[:16], str(b)[:16]
            # 應有 vs 實有
            row["expected_rows"] = None
            if e["per_day"] and row["span_days"] > 0:
                ns = row["sym_actual"] or 1
                row["expected_rows"] = int(e["per_day"] * row["span_days"]
                                           * (ns if e["sym_col"] else 1))
                row["completeness"] = (n / row["expected_rows"]
                                       if row["expected_rows"] else None)
            else:
                row["completeness"] = None
            bad = []
            if e["n_sym"] is not None and row["sym_actual"] is not None \
                    and row["sym_actual"] < e["n_sym"]:
                bad.append("標的 %d < 宣告 %d" % (row["sym_actual"], e["n_sym"]))
            if row["completeness"] is not None and row["completeness"] < 0.9:
                bad.append("完整率 %.1f%% < 90%%" % (100 * row["completeness"]))
            row["bad"] = bad
            row["ok"] = not bad
        except Exception as ex:
            row.update(ok=False, bad=["查詢失敗: %s" % str(ex)[:80]])
        res.append(row)
    cur.close()
    conn.close()
    return res


def audit_files():
    res = []
    for e in EXPECT_FILES:
        row = dict(e)
        files = glob.glob(str(ROOT / e["pat"]))
        row["actual"] = len(files)
        if files:
            ages = [(time.time() - Path(f).stat().st_mtime) / 3600
                    for f in files]
            row["oldest_mtime_h"] = round(max(ages), 1)
            row["newest_mtime_h"] = round(min(ages), 1)
        bad = []
        if row["actual"] < e["n"]:
            bad.append("檔數 %d < 宣告 %d" % (row["actual"], e["n"]))
        row["bad"] = bad
        row["ok"] = not bad
        res.append(row)
    return res


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    db = audit_db()
    fs = audit_files()

    print("=== MySQL：宣告 vs 實有 ===")
    print("%-34s %5s %5s %9s %8s %9s %s"
          % ("項目", "宣告", "實有", "列數", "跨度天", "完整率", "grain"))
    for r in db:
        print("%-34s %5s %5s %9s %8s %9s %s"
              % (r["name"][:34], r.get("n_sym", "-"),
                 r.get("sym_actual", "-"), r.get("rows", "-"),
                 ("%.1f" % r["span_days"]) if r.get("span_days") else "-",
                 ("%.1f%%" % (100 * r["completeness"]))
                 if r.get("completeness") is not None else "-",
                 r["grain"]))
    print("\n=== 檔案：宣告 vs 實有 ===")
    print("%-30s %5s %5s %10s %s" % ("項目", "宣告", "實有", "最舊mtime", "grain"))
    for r in fs:
        print("%-30s %5s %5s %9sh %s"
              % (r["name"][:30], r["n"], r["actual"],
                 r.get("oldest_mtime_h", "-"), r["grain"]))

    reds = [r for r in db + fs if not r["ok"]]
    print("\n=== 紅燈 %d 項 ===" % len(reds))
    for r in reds:
        print("  [RED] %s" % r["name"])
        for b in r["bad"]:
            print("        %s" % b)
        print("        宣告出處: %s" % r["why"])

    # grain 盤點：不可逆的那一類要單獨列出來
    raw = [r["name"] for r in db + fs if r["grain"] == "raw"]
    bucket = [r["name"] for r in db + fs if r["grain"] in ("bucket", "snapshot")]
    print("\n=== grain 盤點（**這一類不可逆**）===")
    print("  raw 逐筆原始事件 (%d)：%s" % (len(raw), "、".join(raw)))
    print("  bucket/snapshot 聚合 (%d)：%s" % (len(bucket), "、".join(bucket)))
    print("  聚合的那些，原始事件回不來 —— 要改成存 raw 只能從今天開始。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        ok=not reds,
        reason=("; ".join("%s: %s" % (r["name"], "/".join(r["bad"]))
                          for r in reds)[:500] if reds
                else "%d 項全部達到宣告" % (len(db) + len(fs))),
        asof=time.strftime("%Y-%m-%d %H:%M:%S"),
        db=db, files=fs), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print("\ncoverage: %s  %s" % ("OK" if not reds else "RED",
                                  "%d red" % len(reds) if reds else ""))
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
