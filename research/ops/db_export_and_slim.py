# -*- coding: utf-8 -*-
"""Railway MySQL 匯出與瘦身（2026-09-20，docs/INVENTORY_2026_09_20.md 的第 2 步）。

**背景**：使用者 2026-09-20 移除全部 Railway 服務要做全盤清點。MySQL 是停機
不是刪除，但本機**沒有任何 dump**（只有 migrations/*.sql 的 schema）。而裡面
56%% 的列（588 萬）在餵兩條已判決死亡的線。

**設計上最重要的一個決定：全部匯出，再 drop。**
D 槽有 662 GB，整個庫撐死幾 GB —— 所以沒有理由只匯出「不可回填」那幾張。
全部匯出之後，drop 就是**完全可逆**的，使用者不必做任何不可逆的判斷。

**順序不可以顛倒**：先關水龍頭（寫入端），再匯出，最後 drop。如果先 drop
但寫入端還開著，它們會立刻填回來。2026-09-20 已停用的寫入端見
research/sweep_failure/shadow_engine.bat 裡的 [DISABLED 2026-09-20] 區塊，
以及兩個不要再部署的 Railway 服務（marketdata、cloud_train）。

**用法**

    python research/ops/db_export_and_slim.py --report
    python research/ops/db_export_and_slim.py --export      # 可續跑
    python research/ops/db_export_and_slim.py --verify
    python research/ops/db_export_and_slim.py --drop --i-have-the-export

--drop 的四道自曝檢查（任何一道不過就中止，不 drop 任何東西）：

  D1  只允許 DROP_WHITELIST 裡的表名，其餘一律拒絕
  D2  manifest.json 必須存在，且該表 verified=True
  D3  **drop 前重查一次 DB 列數，跟 manifest 不符就中止** —— 列數還在長
      代表寫入端沒關乾淨，那正是「先 drop 會被填回來」那個錯的偵測器
  D4  parquet 檔必須存在、讀得開、列數對得上

判準是產物不是退出碼（mistake.md 2026-08-26）：--drop 跑完會把剩下的表與
總位元組再印一次，那個數字才是「瘦身成功了沒」的證據。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_ROOT = Path(r"D:\flowbot_data\db_export")

# 2026-09-20 判決：這三張表餵的線已經死了。列數是 docs/DB_REGISTRY.md 的
# 2026-08-21 快照，drop 當下會重查（D3）。
DROP_WHITELIST = {
    "depth_events_1s": "撤單流 F7 秒級事件流；計分器從沒寫過，檢查點 09-13 已過",
    "depth_deltas_1m": "撤單流方向性；2026-08-10 預註冊判決 FAIL 定案（三個檢定全滅）",
    "v7_okx_balance_snapshots": "OKX executor；2026-08-21 決定不再重啟，帳戶 08-18 起 $0",
}

# 匯出時這些表優先（不可回填），其餘照字母序。只影響順序，不影響範圍。
PRIORITY = [
    "tracked_signals", "indicator_history", "flow_bars_1m",
    "orderbook_snapshots_1m", "cancel_playbook_events",
    "v7_okx_positions", "v7_okx_kill_log", "v7_okx_reconciliation_log",
    "raid_outcomes", "raid_signals_live", "raid_pending_levels",
]

CHUNK = 100_000


def _conn():
    from shared.db import get_db_conn
    return get_db_conn()


def _tables(cur) -> list[str]:
    cur.execute("SHOW TABLES")
    return sorted(r[0] for r in cur.fetchall())


def _stats(cur) -> dict:
    """列數與實際位元組。information_schema.table_rows 是估計值，所以列數另外
    用 COUNT(*) 查 —— 估計值拿來判斷「要不要 drop」太危險。"""
    cur.execute(
        "SELECT table_name, data_length + index_length "
        "FROM information_schema.tables WHERE table_schema = DATABASE()"
    )
    return {r[0]: int(r[1] or 0) for r in cur.fetchall()}


def _count(cur, t: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM `{t}`")
    return int(cur.fetchone()[0])


def _out_dir() -> Path:
    d = OUT_ROOT / time.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return _out_dir() / "manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_manifest(m: dict) -> None:
    _manifest_path().write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- report

def cmd_report() -> int:
    c = _conn()
    cur = c.cursor()
    try:
        tabs = _tables(cur)
        byt = _stats(cur)
        rows = []
        for t in tabs:
            rows.append((t, _count(cur, t), byt.get(t, 0)))
        rows.sort(key=lambda r: -r[2])

        total_r = sum(r[1] for r in rows)
        total_b = sum(r[2] for r in rows)
        dead_r = sum(r[1] for r in rows if r[0] in DROP_WHITELIST)
        dead_b = sum(r[2] for r in rows if r[0] in DROP_WHITELIST)

        print(f"{'表':<34}{'列數':>12}{'大小':>12}   標記")
        print("-" * 78)
        for t, n, b in rows:
            mark = "<<< DROP 名單" if t in DROP_WHITELIST else ""
            print(f"{t:<34}{n:>12,}{b/1048576:>10.1f}MB   {mark}")
        print("-" * 78)
        print(f"{'合計':<34}{total_r:>12,}{total_b/1048576:>10.1f}MB")
        print(f"{'其中 DROP 名單':<34}{dead_r:>12,}{dead_b/1048576:>10.1f}MB"
              f"   = {100*dead_r/max(total_r,1):.0f}% 的列 / "
              f"{100*dead_b/max(total_b,1):.0f}% 的位元組")
        return 0
    finally:
        c.close()


# ---------------------------------------------------------------- export

def _export_one(c, t: str, dest: Path) -> int:
    """串流匯出一張表。用 SSCursor 避免把整張表讀進記憶體。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pymysql.cursors

    cur = c.cursor(pymysql.cursors.SSCursor)
    cur.execute(f"SELECT * FROM `{t}`")
    cols = [d[0] for d in cur.description]

    writer = None
    n = 0
    try:
        while True:
            batch = cur.fetchmany(CHUNK)
            if not batch:
                break
            # 全部轉成字串以外的型別交給 pyarrow 推斷；None 保持 None
            table = pa.Table.from_pydict(
                {col: [row[i] for row in batch] for i, col in enumerate(cols)})
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            n += len(batch)
            print(f"    {t}: {n:,}", end="\r", flush=True)
    finally:
        if writer is not None:
            writer.close()
        cur.close()
    return n


def cmd_export() -> int:
    d = _out_dir()
    man = _load_manifest()
    c = _conn()
    cur = c.cursor()
    try:
        tabs = _tables(cur)
        order = [t for t in PRIORITY if t in tabs] + \
                [t for t in tabs if t not in PRIORITY]
        print(f"匯出目的地：{d}\n表數：{len(order)}\n")
        for t in order:
            dest = d / f"{t}.parquet"
            want = _count(cur, t)
            prev = man.get(t, {})
            if dest.exists() and prev.get("rows") == want:
                print(f"  跳過 {t}（已匯出 {want:,} 列）")
                continue
            print(f"  匯出 {t}（{want:,} 列）…")
            if want == 0:
                # 空表也要留一個記號，否則「沒有檔」跟「沒匯出」分不開
                dest.write_bytes(b"")
                got = 0
            else:
                got = _export_one(c, t, dest)
            man[t] = {"rows": got, "db_rows_at_export": want,
                      "file": dest.name,
                      "asof": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "verified": False}
            _save_manifest(man)
            print(f"    {t}: {got:,} 列 -> {dest.name}"
                  f"{'  [!] 與 DB 不符' if got != want else ''}")
        print(f"\n完成。manifest：{_manifest_path()}")
        return 0
    finally:
        c.close()


# ---------------------------------------------------------------- verify

def cmd_verify() -> int:
    import pyarrow.parquet as pq
    d = _out_dir()
    man = _load_manifest()
    if not man:
        print("沒有 manifest —— 先跑 --export")
        return 1
    c = _conn()
    cur = c.cursor()
    bad = []
    try:
        for t, rec in sorted(man.items()):
            f = d / rec["file"]
            db_now = _count(cur, t)
            if not f.exists():
                bad.append((t, "檔案不存在")); rec["verified"] = False; continue
            if rec["rows"] == 0:
                ok = (db_now == 0)
                rec["verified"] = ok
                if not ok:
                    bad.append((t, f"空檔但 DB 有 {db_now:,} 列"))
                continue
            pf_rows = pq.ParquetFile(f).metadata.num_rows
            ok = (pf_rows == rec["rows"] == db_now)
            rec["verified"] = ok
            rec["db_rows_at_verify"] = db_now
            if not ok:
                bad.append((t, f"parquet {pf_rows:,} / manifest {rec['rows']:,} "
                               f"/ DB {db_now:,}"))
        _save_manifest(man)
    finally:
        c.close()

    n_ok = sum(1 for r in man.values() if r.get("verified"))
    print(f"通過 {n_ok}/{len(man)}")
    for t, why in bad:
        print(f"  FAIL {t}: {why}")
    if bad:
        print("\n有表沒過 —— 重跑 --export（它會跳過已對上的表）")
    return 1 if bad else 0


# ---------------------------------------------------------------- drop

def cmd_drop(confirmed: bool) -> int:
    import pyarrow.parquet as pq
    if not confirmed:
        print("拒絕：--drop 必須同時帶 --i-have-the-export")
        return 2
    d = _out_dir()
    man = _load_manifest()

    # ---- D1/D2/D4 完全不需要連線，先跑完 ——「還沒準備好」這件事不該
    # 要先連得上資料庫才知道。連線失敗會把它偽裝成別的問題。
    problems = []
    for t in DROP_WHITELIST:
        rec = man.get(t)
        if rec is None:                                        # D2
            problems.append(f"{t}: manifest 裡沒有它（先跑 --export）"); continue
        if not rec.get("verified"):                            # D2
            problems.append(f"{t}: verified=False（先跑 --verify）"); continue
        f = d / rec["file"]
        if not f.exists():                                     # D4
            problems.append(f"{t}: parquet 不見了"); continue
        try:
            pf_rows = pq.ParquetFile(f).metadata.num_rows if rec["rows"] else 0
        except Exception as e:                                 # D4
            problems.append(f"{t}: parquet 讀不開 {e}"); continue
        if pf_rows != rec["rows"]:                             # D4
            problems.append(f"{t}: parquet {pf_rows:,} != manifest {rec['rows']:,}")

    if problems:
        print("中止，一張表都沒有 drop：")
        for p in problems:
            print("  FAIL " + p)
        return 1

    c = _conn()
    cur = c.cursor()
    try:
        # ---- D3 需要連線：drop 前重查列數。列數變了 = 寫入端沒關乾淨 ----
        for t in DROP_WHITELIST:
            now = _count(cur, t)
            want = man[t]["rows"]
            if now != want:
                print(f"中止，一張表都沒有 drop：\n  FAIL {t}: DB 現在 {now:,} 列，"
                      f"匯出時 {want:,} 列 —— **列數變了代表還有寫入端在跑**，"
                      f"先把它關掉再回來")
                return 1

        before = _stats(cur)
        for t, why in DROP_WHITELIST.items():
            print(f"DROP `{t}`  ({before.get(t,0)/1048576:.1f} MB)  —— {why}")
            cur.execute(f"DROP TABLE `{t}`")
        c.commit()

        # ---- 判準是產物：再量一次 ----
        tabs = _tables(cur)
        after = _stats(cur)
        still = [t for t in DROP_WHITELIST if t in tabs]
        print(f"\n剩餘表數：{len(tabs)}")
        print(f"剩餘大小：{sum(after.values())/1048576:.1f} MB"
              f"（drop 前 {sum(before.values())/1048576:.1f} MB）")
        if still:
            print(f"[!] 這些還在：{still}")
            return 1
        print("三張表都不在了。")
        return 0
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--i-have-the-export", action="store_true")
    a = ap.parse_args()
    if a.report:
        return cmd_report()
    if a.export:
        return cmd_export()
    if a.verify:
        return cmd_verify()
    if a.drop:
        return cmd_drop(a.i_have_the_export)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
