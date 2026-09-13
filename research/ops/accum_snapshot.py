# -*- coding: utf-8 -*-
"""資料累積快照 —— 回答「有沒有在累積」，而不是「現在活不活」。

===========================================================================
為什麼需要這一支（freshness board 答不了的那個問題）
===========================================================================
使用者 2026-09-13：「因為沒有可視化我不確定數據累積有沒有在跑」。

`freshness_board.py` 有 52 列、全綠，但它量的是**最後一筆有多舊**——
一個「現在」的量。它結構上答不了兩件事：

1. **有沒有斷過。** 2026-09-11 `hl_mid` 被看門狗殺了 70 次、近 6 小時沒錄到
   東西，而 freshness 全程綠的（它每 5 分鐘就被重啟一次，所以「最後一筆」
   永遠很新）。那三個小時的洞要到今天掃檔案才看得到。
2. **從未開始。** mistake.md 2026-09-01：freshness 監測的是心跳，而「一件
   該做的事從來沒被啟動」沒有心臟。

所以本支產出的是**逐小時的時間序列**，不是一排燈。空格就是洞，一眼看得到。

===========================================================================
設計決定（都有理由，不是風格）
===========================================================================
* **不讀資料，只讀 parquet 的 metadata。** `pq.ParquetFile(f).metadata.num_rows`
  是檔頭裡的數字，不用把 1.7M 列載進記憶體。掃 300 個檔是秒級的事。
* **「可回填」是第一級欄位。** 不可回填的空格是**永久損失**，可回填的只是
  待補，兩者不該長一樣（CLAUDE.md §鏈上錄製集：五樣裡四樣不可回填）。
* **本支自己要進 freshness。** 否則就重複 mistake.md 2026-09-11 那條
  「衍生的判決檔沒有一個被盯著」——而那次是一份 json 凍了六天沒人發現。
* **不碰交易路徑、不寫任何 quant 表。** 純讀。
* 照 `starting-your-quant-trading-business`（2024-02-18）那句
  「The dashboard is for the algorithm NOT the other way around」：
  本支只產資料，樣式留給 renderer，而 renderer 要刻意樸素。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "research", "results", "accum_snapshot.json")
FLAG = os.path.join(ROOT, "research", "results", "accum_snapshot_last.json")
DAYS = 14

# ── D 槽上的小時分檔錄製器 ────────────────────────────────────────────
# (顯示名, 根目錄, 可回填?, 為什麼)
PARQUET_SETS = [
    ("Lighter 逐筆成交帶", "D:/flowbot_data/lighter/trades", False,
     "常駐 WS；無歷史端點"),
    ("Lighter 中價與深度", "D:/flowbot_data/lighter/mid", False,
     "常駐 WS；簿口無歷史端點"),
    ("HL 逐筆成交帶", "D:/flowbot_data/hl/trades", False,
     "常駐 WS；無歷史端點"),
    ("HL 中價與佇列", "D:/flowbot_data/hl/mid", False,
     "常駐 WS；簿口無歷史端點"),
    # 2026-09-13：Lighter 頂檔事件（價變才記）。TODO §1.37 的 L3/L4 用逐筆帶
    # 永遠答不了（250ms 一格只有 0.8% 兩邊都有成交），只有簿口答得了。
    # 實測 57 事件/秒 = 246 MB/日（33 個共同標的；全 80 個會再多一些）。
    ("Lighter 頂檔事件", "D:/flowbot_data/lighter/tob", False,
     "常駐 WS，頂檔價變才記；簿口無歷史端點"),
]

# ── MySQL 裡還在成長的表 ──────────────────────────────────────────────
# (顯示名, 表, 時間欄, 可回填?, 註)
MYSQL_SETS = [
    ("撤單事件 1s", "depth_events_1s", "created_at", False, "WS 推送", False),
    ("簿口快照 1m", "orderbook_snapshots_1m", "ts_ms", False, "REST 輪詢", False),
    ("撤單深度差 1m", "depth_deltas_1m", "minute_start_ms", False,
     "十月 L2 檢查點還開著", False),
    ("未平倉量", "oi_snapshots", "ts_received", True, "REST 有歷史", False),
    ("資金費", "funding_rates", "ts_received", True, "REST 有歷史", False),
    # 清算是**事件驅動**：空一小時可能是「那小時沒有清算」，不是斷線。
    # 第 6 個欄位 = event_driven。
    ("清算事件", "liq_events", "created_at", False, "WS 推送；事件驅動", True),
    ("選擇權 GEX", "gex_snapshots", "created_at", False, "快照無歷史", False),
]


def hour_keys(days: int = DAYS):
    """回傳最近 days 天的每小時鍵，由舊到新。最後一個是**還在寫**的那小時。"""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [(now - timedelta(hours=h)).strftime("%Y%m%d%H")
            for h in range(days * 24 - 1, -1, -1)]


def scan_parquet(root: str, keys):
    """逐小時列數與位元組。**只讀檔頭，不載資料。**"""
    import pyarrow.parquet as pq
    want = set(keys)
    rows, byts = {}, {}
    for f in glob.glob(root + "/*/*.parquet"):
        d = os.path.basename(os.path.dirname(f))
        h = os.path.basename(f)[:2]
        k = d + h
        if k not in want:
            continue
        try:
            rows[k] = rows.get(k, 0) + pq.ParquetFile(f).metadata.num_rows
            byts[k] = byts.get(k, 0) + os.path.getsize(f)
        except Exception as e:                     # 半寫入的檔不該讓整支掛掉
            print("  [WARN] 讀不了 %s：%s" % (f, e))
    return rows, byts


def scan_mysql(keys):
    """逐小時列數。一張表一次 GROUP BY，不逐小時查（那是 336 次往返）。"""
    from shared.db import get_db_conn
    cn = get_db_conn()
    cur = cn.cursor()
    lo = datetime.strptime(keys[0], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    out = {}
    for lab, tbl, col, fill, note, ev in MYSQL_SETS:
        try:
            cur.execute("SELECT DATA_TYPE d FROM information_schema.columns "
                        "WHERE table_schema=DATABASE() AND table_name=%s "
                        "AND COLUMN_NAME=%s", (tbl, col))
            r = cur.fetchall()
            if not r:
                out[lab] = {"error": "欄位 %s.%s 不存在" % (tbl, col)}
                continue
            dtype = (dict(r[0]) if not isinstance(r[0], dict) else r[0])["d"]
            if dtype in ("datetime", "timestamp", "date"):
                cur.execute("SELECT DATE_FORMAT(`%s`, '%%%%Y%%%%m%%%%d%%%%H') k,"
                            " COUNT(*) c FROM `%s` WHERE `%s` >= %%s"
                            " GROUP BY k" % (col, tbl, col),
                            (lo.strftime("%Y-%m-%d %H:00:00"),))
            else:
                # epoch（秒或毫秒）。先探一筆決定除數。
                cur.execute("SELECT MAX(`%s`) v FROM `%s`" % (col, tbl))
                mx = (dict(cur.fetchall()[0]))["v"]
                if mx is None:
                    out[lab] = {"rows": {}, "note": note, "backfillable": fill,
                                "source": "mysql:" + tbl, "event_driven": ev}
                    continue
                div = 1000 if float(mx) > 1e11 else 1
                cur.execute(
                    "SELECT DATE_FORMAT(FROM_UNIXTIME(`%s`/%d),"
                    " '%%%%Y%%%%m%%%%d%%%%H') k, COUNT(*) c FROM `%s`"
                    " WHERE `%s` >= %%s GROUP BY k" % (col, div, tbl, col),
                    (lo.timestamp() * div,))
            got = {}
            for row in cur.fetchall():
                row = dict(row)
                got[str(row["k"])] = int(row["c"])
            out[lab] = {"rows": got, "note": note, "backfillable": fill,
                        "source": "mysql:" + tbl,
                        "event_driven": ev}
        except Exception as e:
            out[lab] = {"error": "%s" % e, "source": "mysql:" + tbl}
    cn.close()
    return out


def db_sizes():
    from shared.db import get_db_conn
    cn = get_db_conn()
    cur = cn.cursor()
    cur.execute("SELECT COUNT(*) n, COALESCE(SUM(data_length+index_length),0)"
                "/1048576 mb FROM information_schema.tables "
                "WHERE table_schema=DATABASE()")
    r = dict(cur.fetchall()[0])
    cn.close()
    return {"tables": int(r["n"]), "used_mb": round(float(r["mb"]), 1)}


def disk_free(drive: str = "D:"):
    import shutil
    try:
        t, u, f = shutil.disk_usage(drive + "\\")
        return {"total_gb": round(t / 1e9, 1), "free_gb": round(f / 1e9, 1)}
    except OSError as e:
        return {"error": str(e)}


def build(days: int = DAYS):
    keys = hour_keys(days)
    sets = []
    for lab, root, fill, note in PARQUET_SETS:
        if not os.path.isdir(root):
            sets.append({"name": lab, "source": root, "backfillable": fill,
                         "note": note, "error": "目錄不存在（D 槽連結？）"})
            continue
        rows, byts = scan_parquet(root, keys)
        sets.append({"name": lab, "source": root, "backfillable": fill,
                     "note": note, "rows": rows, "bytes": byts})
    for lab, info in scan_mysql(keys).items():
        info["name"] = lab
        sets.append(info)

    # 逐集摘要
    for s in sets:
        r = s.get("rows") or {}
        # **「上線」不能定義成「第一個有資料的小時」。** Lighter 的成交帶每次
        # 重訂閱會重播近期成交，而那些列的時戳很舊 -> 八月的小時檔裡會有 1-2
        # 列，於是「上線」被判成 08-31、然後報出 170 個假缺口。
        # 正確的判法：**從哪一小時開始，到現在的覆蓋率就一直 >= 90%**。
        # 取最早一個滿足它的小時。重播造成的零星舊小時另記 replay_hours，
        # 不丟掉（丟掉就看不出它存在）。
        span_all = keys[:-1]                        # 末格還在寫，不算
        # **量的地板**：只看「有沒有列」不夠——重播的 20 列與正常的 16,000 列
        # 都是「有資料」，所以光看存在性會把重播的八月小時判成「已上線」。
        # 地板 = 最近 48 個有料小時的中位數的 10%。
        # **用 p90 不用中位數。** 中位數會被「上線前那些稀疏小時」拉下去：
        # Lighter 成交帶穩態是 ~17,000 列/時，但最近 48 小時裡有一半是上線前
        # 重播的 20-30 列，中位數於是給出 49 —— 地板跟著塌掉，整個判法失效。
        # 穩態速率是穩定的，所以 p90 就是營運水準，而稀疏小時待在低尾。
        recent = sorted(r[k] for k in span_all[-48:] if r.get(k))
        lvl = recent[min(len(recent) - 1, int(0.9 * len(recent)))] if recent else 0
        # 地板取 p90 的 **1%** 不是 10%。10% 對**爆發型**的流太嚴：清算平時
        # 100-200 筆、爆發 1,438，地板 143 會把大半正常小時判成斷線 ->
        # 假紅燈，而假紅燈會訓練人忽略這個頻道。死掉的小時是 0 列、活的是
        # 幾百到幾萬列，中間的鴻溝很大，地板只要高過重播雜訊（20-30 列）就夠。
        floor = max(1, int(lvl * 0.01))
        s["hourly_level"] = lvl
        s["floor"] = floor

        def live(k):
            return r.get(k, 0) >= floor

        first = None
        for i, k in enumerate(span_all):
            tail = span_all[i:]
            # 起點自己必須有料，而且從它開始覆蓋率就 >= 90%
            if live(k) and len(tail) >= 3 \
                    and sum(1 for x in tail if live(x)) >= 0.9 * len(tail):
                first = k
                break
        span = [k for k in span_all if first and k >= first]
        gaps = [k for k in span if not live(k)]
        s["first_hour"] = first
        s["hours_live"] = len(span)
        # 事件驅動的資料（清算）空一小時可能是合法的「那小時沒事件」，
        # 不是斷線。把它記成 empty 而不是 missing —— 假紅燈會訓練人忽略
        # 這個頻道，那正是 transition-only 告警要避免的事（mistake.md）。
        ev = s.get("event_driven", False)
        s["hours_missing"] = 0 if ev else len(gaps)
        s["hours_empty"] = len(gaps) if ev else 0
        s["missing_hours"] = gaps[:24]
        s["replay_hours"] = sum(1 for k in span_all
                                if r.get(k) and not live(k)
                                and (not first or k < first))
        s["rows_24h"] = sum(r.get(k, 0) for k in keys[-25:-1])
        b = s.get("bytes") or {}
        s["mb_24h"] = round(sum(b.get(k, 0) for k in keys[-25:-1]) / 1e6, 1) \
            if b else None

    snap = {"asof_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hours": keys, "days": days, "sets": sets,
            "capacity": {"disk_D": disk_free("D:"), "disk_C": disk_free("C:"),
                         "mysql": db_sizes()}}
    return snap


def write_flag(snap, ok=True, reason=""):
    miss = sum(s.get("hours_missing", 0) for s in snap.get("sets", []))
    errs = [s["name"] for s in snap.get("sets", []) if s.get("error")]
    # ok 的語意是「本支跑完且掃得到東西」，不是「沒有洞」——洞是被監測對象的
    # 事，不是儀器的事（mistake.md 2026-09-03）。
    payload = {"ok": ok, "reason": reason or
               "%d 個資料集｜上線後缺 %d 小時｜錯誤 %d"
               % (len(snap.get("sets", [])), miss, len(errs)),
               "asof": snap["asof_utc"], "sets": len(snap.get("sets", [])),
               "hours_missing_total": miss, "errors": errs}
    with open(FLAG, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    snap = build(a.days)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=1)
    write_flag(snap)
    if a.quiet:
        return
    print("%-22s %4s %7s %6s %9s %7s %5s %s"
          % ("資料集", "回填", "上線時數", "缺口", "近24h列", "列/時",
             "重播", "上線於 / 缺在哪"))
    print("-" * 104)
    for s in snap["sets"]:
        if s.get("error"):
            print("%-22s  **錯誤：%s**" % (s["name"], s["error"]))
            continue
        f = s["first_hour"]
        det = "-" if not f else "%s-%s %sh" % (f[4:6], f[6:8], f[8:])
        if s["missing_hours"]:
            det += "  缺 " + ",".join("%s-%s %sh" % (k[4:6], k[6:8], k[8:])
                                      for k in s["missing_hours"][:4])
        print("%-22s %4s %7d %6d %9s %7s %5d %s"
              % (s["name"], "是" if s["backfillable"] else "否",
                 s["hours_live"], s["hours_missing"],
                 format(s["rows_24h"], ","),
                 format(s["hourly_level"], ","),
                 s["replay_hours"], det))
    c = snap["capacity"]
    print("\nD 槽 %.0f/%.0f GB 可用｜C 槽 %.0f/%.0f GB｜MySQL %.0f MB / %d 表"
          % (c["disk_D"].get("free_gb", 0), c["disk_D"].get("total_gb", 0),
             c["disk_C"].get("free_gb", 0), c["disk_C"].get("total_gb", 0),
             c["mysql"]["used_mb"], c["mysql"]["tables"]))
    print("寫出 %s" % OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
