# -*- coding: utf-8 -*-
"""Operations board — every scheduled check, and every verdict it produced.

Registered 2026-09-02 (user: "所有自己排程的複驗可以都顯示在系統網站上嗎
方便管理紀錄"). Until now the schedule existed in four disconnected places:
Windows Task Scheduler, a .bat with a dozen publishers inside it, the
freshness board, and a folder of monthly revalidation reports. Nothing
showed them together, so "is everything running, and what did it decide"
could only be answered by looking in four places on the operator's own
machine.

This publishes three things the site can render:

  1. SCHEDULE — the frozen registry of jobs (name, cadence, what dying
     would cost). Cadence is declared here, not scraped: a job that
     silently stopped being scheduled must still appear, red. That is the
     whole 2026-07-05 lesson (DailyCollect pointed at a deleted path for 96
     days while the panel read "Ready").
  2. FRESHNESS — the live product-age table the board already computes.
     Aliveness is judged by ARTIFACT age, never by a scheduler's status
     light (2026-08-19).
  3. REVALIDATION HISTORY — every monthly report's date + verdict + the
     headline numbers, parsed from the reports themselves. A verdict that
     only exists in a local .md is a record nobody can audit.

Public-surface safe: percentages, dates, states and counts only — no
dollars, no positions, no model internals (feature names, cutoffs,
weights). AUC/IC are performance metrics and are already public on the
writeups page.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRESHNESS_JSON = ROOT / "research/results/freshness_board.json"
REVAL_GLOB = str(ROOT / "research/results/dual_model/quarterly_revalidation_*.md")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── frozen job registry ──────────────────────────────────────────────────
# Declared, not discovered. A job missing from the machine still shows up
# here (and its artifact goes stale), which is the only way "it stopped
# being scheduled" is visible at all.
JOBS = [
    dict(id="shadow-train", name="每小時班車", cadence="每小時",
         what="獵取記帳、天氣站、各看板 publisher、V7 veto 時鐘、"
              "套利判決與發布、訊號雜湊鏈",
         artifact="sweep shadow log",
         cost="停掉 = 所有前瞻記帳與看板同時凍結（2026-08-19 曾靜默 29 小時）"),
    dict(id="daily-collect", name="每日資料收集", cadence="每天 04:00",
         what="Coinglass 生產線與研究線 parquet",
         artifact="daily collect log",
         cost="停掉 = 研究資料安靜腐爛（2026-07-05 曾指向已刪路徑 96 天）"),
    dict(id="freshness", name="產物新鮮度看板", cadence="每 6 小時",
         what="26 條產物的年齡巡檢，紅燈推 Telegram",
         artifact="freshness board",
         cost="它是其他所有排程的守門員"),
    dict(id="portfolio-clocks", name="組合時鐘週報", cadence="每週一 09:30",
         what="Gate B／Gate F／扳機／縮帆線／相關性預算",
         artifact="portfolio clocks",
         cost="停掉 = 判決日到了沒人知道"),
    dict(id="monthly-reval", name="模型月度復驗", cadence="每月 5 號 09:00",
         what="AUC/IC 天花板、SNR 與洗牌 null、生產輸出水平漂移、"
              "regime 拆解、Strong 勝率",
         artifact="revalidation report",
         cost="模型刻度歪掉的唯一定期檢查（2026-08-08 漂了 3 個月）"),
    dict(id="arb-recorders", name="套利錄價（七配對）", cadence="連續",
         what="兩場館逐分鐘盤口與資金費率",
         artifact="arb recorder",
         cost="靜默斷錄 = 整週資料白等"),
]

VERDICT_RE = re.compile(r"^\*\*(PASS|DRIFT|FAIL|STALE-DATA[^*]*)\*\*\s*—?\s*(.*)$",
                        re.M)
AUC_RE = re.compile(r"sign_AUC\s*=\s*([0-9.]+)")
IC_RE = re.compile(r"Spearman IC\s*=\s*([+\-0-9.]+)")
SNR_RE = re.compile(r"SNR\(Spearman\)\s*=\s*([0-9.]+)%")


def revalidation_history() -> list:
    out = []
    for f in sorted(glob.glob(REVAL_GLOB)):
        try:
            txt = Path(f).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        stem = Path(f).stem.split("_")[-1]
        date = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}" if len(stem) == 8 else stem
        m = VERDICT_RE.search(txt)
        auc = AUC_RE.search(txt)
        ic = IC_RE.search(txt)
        snr = SNR_RE.search(txt)
        out.append({
            "date": date,
            "verdict": (m.group(1) if m else "—"),
            "summary": (m.group(2)[:160] if m else ""),
            "auc": float(auc.group(1)) if auc else None,
            "ic": float(ic.group(1)) if ic else None,
            "snr_spearman_pct": float(snr.group(1)) if snr else None,
            "stale_guard_hit": "STALE-DATA" in txt,
            "push_failed": "TELEGRAM PUSH FAILED" in txt,
        })
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def freshness_rows() -> dict:
    try:
        d = json.loads(FRESHNESS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"asof_utc": None, "rows": [], "reds": []}
    rows = [{"name": r["name"], "age_h": r.get("age_h"),
             "max_h": r.get("max_h"), "ok": bool(r.get("ok")),
             "note": r.get("note", "")} for r in d.get("rows", [])]
    return {"asof_utc": d.get("asof_utc"), "rows": rows,
            "reds": d.get("reds", [])}


def build() -> dict:
    fresh = freshness_rows()
    by_name = {r["name"]: r for r in fresh["rows"]}
    jobs = []
    for j in JOBS:
        # a job's health is its ARTIFACT's age — never a scheduler light
        matches = [r for n, r in by_name.items() if j["artifact"] in n]
        ok = all(r["ok"] for r in matches) if matches else None
        worst = max((r["age_h"] for r in matches
                     if r.get("age_h") is not None), default=None)
        jobs.append({**j, "healthy": ok, "artifact_age_h": worst,
                     "artifacts_watched": len(matches)})
    hist = revalidation_history()
    return {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "jobs": jobs,
        "freshness": fresh,
        "revalidations": hist,
        "principle": "排程是否活著，一律以**產物的新鮮度**判斷，不看排程面板的"
                     "狀態燈——面板顯示的是「有沒有被觸發」，不是「有沒有做完事」。"
                     "每一次月度復驗的判決都留在這裡，包含資料過期而拒絕判決的那些。",
        "disclaimer": "Operational record — research status only, "
                      "not a live strategy, not financial advice.",
    }


def main() -> int:
    payload = build()
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ops_board (
                    id TINYINT PRIMARY KEY,
                    payload MEDIUMTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    checked_at DATETIME NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            try:
                cur.execute("SELECT checked_at FROM ops_board LIMIT 1")
            except Exception:
                cur.execute("ALTER TABLE ops_board ADD COLUMN checked_at DATETIME NULL")
            cur.execute(
                "INSERT INTO ops_board (id, payload, checked_at) "
                "VALUES (1,%s,NOW()) ON DUPLICATE KEY UPDATE "
                "payload=VALUES(payload), checked_at=NOW()",
                (json.dumps(payload, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()
    nred = len(payload["freshness"]["reds"])
    print(f"ops_board published: {len(payload['jobs'])} jobs, "
          f"{len(payload['revalidations'])} revalidations, "
          f"{nred} red artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
