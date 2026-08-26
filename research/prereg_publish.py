# -*- coding: utf-8 -*-
"""Pre-registration board publisher — every open hypothesis in one place.

Why (2026-08-26): the pre-registrations live in TODO.md sections and their
progress is only visible by running a script per clock. The operator asked
"是這樣我才知道進度怎麼樣" — the discipline is invisible from outside, so
from outside it looks like nothing is happening while five clocks run.

Design rule, and the reason this file adds almost no arithmetic: THE BOARD
SHOWS PROGRESS, NOT VERDICTS. Each verdict keeps exactly one owner (the
scorer that froze its criteria); duplicating a number here would create a
second implementation that can silently disagree — the failure mode of
mistake.md 2026-08-01 (two copies of the same data, one rots quietly).

So each entry sources its progress one of three ways, in order of
preference:
  json    read the clock file its owner already writes
  count   count rows in the frozen ledger (unambiguous, no scoring)
  date    days elapsed since the frozen registration date — for gates whose
          binding constraint IS wall time, this is exact from the date alone

Off-cloud-recorder pattern (fourth instance after raid-signals,
weather_station, v7_veto_clock): THIS machine has the caches and rides the
hourly train, so it computes and UPSERTs; the agent only SELECTs.

Public-surface rule (CLAUDE.md): percentages, R multiples, counts and dates
only. No contract sizes, no USD, no balances. Nothing here touches those.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "research" / "results"
LOG = RES / "sweep_shadow_log.csv"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
HOME = {"RANGING", "TREND_DOWN"}                      # §0.59b
SEP_59 = datetime(2026, 8, 26, tzinfo=timezone.utc)


def _days_since(d: datetime) -> float:
    return (datetime.now(timezone.utc) - d).total_seconds() / 86400


def _json(name: str):
    try:
        return json.loads((RES / name).read_text(encoding="utf-8"))
    except Exception:
        return None


def _shadow_rows():
    try:
        with open(LOG, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _n_gate_f(rows):
    """Gate F progress — MUST reproduce shadow_engine's "B (pierce) closed=".

    Deliberately NOT filtered to core9. The first version of this function
    added that filter and produced 346 against the published 1127 — the
    exact "second implementation quietly disagrees" failure this file's
    docstring warns about, committed inside the file that warns about it.
    If this number ever stops matching the hourly log's B-row, THIS is
    wrong, not the log.
    """
    return sum(1 for r in rows if r.get("variant_b") == "1"
               and r.get("status") == "CLOSED")


def _n_059(rows):
    """Fresh home-regime fills since the §0.59 floor.

    core9 HERE IS CORRECT and must not be "fixed" to match _n_gate_f: the
    §0.58/§0.59 evidence base is core9 throughout, so the clock counts the
    same population the verdict will score. Two different filters on the
    same CSV, each right for its own question.

    Counting only — no meanR here. The verdict is sweep_forward's job.
    """
    n = 0
    for r in rows:
        if (r.get("variant_b") != "1" or r.get("status") != "CLOSED"
                or r.get("universe") != "core9"):
            continue
        if r.get("regime_cell") not in HOME:
            continue
        try:
            ts = int(float(r["fill_ts"]))
        except (ValueError, TypeError, KeyError):
            continue
        if ts >= SEP_59.timestamp():
            n += 1
    return n


def build():
    rows = _shadow_rows()
    q2 = _json("v7_regime_q2_clock.json") or {}
    veto = _json("v7_veto_clock.json") or {}
    open_items = [
        {
            "id": "0.59", "line": "流動性獵取", "title": "regime 進場濾網",
            "hypothesis": "只在主場（震盪 ∪ 空頭趨勢）開火，其餘一律不進場",
            "why": "主場在兩批獨立樣本間穩定，崩的全在非主場（§0.58）",
            "registered": "2026-08-26", "source": "count",
            "n": _n_059(rows), "gate_n": 150,
            "days": round(_days_since(SEP_59), 1), "gate_days": 30,
            "note": "提出這條規則用掉的樣本已作廢，只採 08-26 之後的新成交",
        },
        {
            "id": "0.60 Q2", "line": "V7", "title": "上漲趨勢只收看空訊號",
            "hypothesis": "TREND_UP 格擋掉 UP 訊號，只接受 DOWN 訊號",
            "why": "該格做多 48.1%（八格最差），做空 65.7%（第二好）",
            "registered": "2026-08-26", "source": "json",
            "n": q2.get("n", 0), "gate_n": q2.get("gate_n", 60),
            "days": q2.get("days", 0), "gate_days": q2.get("gate_days", 30),
            "note": "已有一批 11 筆的反例（55%），寫進註冊書不得遺忘",
        },
        {
            "id": "地形扳機", "line": "V7", "title": "地形濾網上線扳機",
            "hypothesis": "保留 vs 否決的勝率差 ≥8pp 且新訊號 ≥60 筆",
            "why": "地形四維過了三關但仍是 display-only，扳機決定能否進場規則",
            "registered": "2026-08-02", "source": "json",
            "n": veto.get("strong_since_trigger", 0),
            "gate_n": veto.get("trigger_target", 60),
            "days": round(_days_since(datetime(2026, 8, 2, tzinfo=timezone.utc)), 1),
            "gate_days": None,
            "note": "看板主數字是已結算的；剛開火的訊號要等約 4 小時才會動",
        },
        {
            "id": "Gate F", "line": "流動性獵取", "title": "變體 B 前瞻驗證",
            "hypothesis": "掃單失敗的 edge 在前瞻樣本上仍為正",
            "why": "回測的 t=8.27 是滑價符號寫反造成的假象，修正後只剩 t=3.35",
            "registered": "2026-07-28", "source": "count",
            "n": _n_gate_f(rows), "gate_n": 1400,
            "days": round(_days_since(datetime(2026, 7, 28, tzinfo=timezone.utc)), 1),
            "gate_days": None,
            "note": "判決日附件必須分格報告——主場失效與非主場拖累的後續不同",
        },
        {
            "id": "0.52", "line": "縮帆（風控）", "title": "ADX 趨勢環境減碼",
            "hypothesis": "偵測到趨勢時把部位砍半，降低單邊行情的受傷幅度",
            "why": "不預測風暴，只收帆——趨勢環境是獵取與網格的逆風",
            "registered": "2026-08-17", "source": "date",
            "n": None, "gate_n": None,
            "days": round(_days_since(datetime(2026, 8, 17, tzinfo=timezone.utc)), 1),
            "gate_days": 30,
            "note": "數字見市場天氣站卡片；此處只顯示時間閘門的進度",
        },
        {
            "id": "0.53", "line": "縮帆（風控）", "title": "波動目標倉位",
            "hypothesis": "依近期波動連續縮放部位（波動大→下小注）",
            "why": "§0.52 的連續版；儀表排名裡 trailing vol 遠強於 ADX",
            "registered": "2026-08-20", "source": "date",
            "n": None, "gate_n": None,
            "days": round(_days_since(datetime(2026, 8, 20, tzinfo=timezone.utc)), 1),
            "gate_days": 30,
            "note": "成交在時間上高度集中，短窗等於一個行情段落不是一批樣本",
        },
    ]
    settled = [
        {"id": "0.59b", "line": "流動性獵取", "title": "主場定義修正",
         "verdict": "已定案", "tone": "ok",
         "text": "主場從「只有震盪」改為「震盪 ∪ 空頭趨勢」。原定義是從一個"
                 "不分方向的舊框架帶來的殘留；判準誠實套用會選出兩格。"},
        {"id": "0.58", "line": "流動性獵取", "title": "前瞻落差的來源",
         "verdict": "已定案", "tone": "ok",
         "text": "回測與前瞻的落差有 75% 是格內衰退、只有 7% 是行情組成。"
                 "但主場兩期一致（+0.098 → +0.093）——不是 edge 死了，"
                 "是非主場崩了。"},
        {"id": "0.60 Q1", "line": "V7", "title": "方向軸不成立",
         "verdict": "負結果", "tone": "warn",
         "text": "獵取那條線上「趨勢要分上下」成立，V7 這裡不成立"
                 "（上漲 58.0% vs 下跌 58.5%，差 0.6pp）。"
                 "一條線有的效應不能假設另一條也有。"},
        {"id": "0.60 Q4", "line": "V7", "title": "格內衰退成立",
         "verdict": "已定案", "tone": "warn",
         "text": "近期訊號準度下滑，拆解後組成效應僅 −0.1pp、格內效應 −6.3pp。"
                 "主場不但沒退還變好，崩的是非主場——與獵取同構。"},
        {"id": "撤單流", "line": "撤單流", "title": "方向性判決 FAIL",
         "verdict": "已陣亡", "tone": "dead",
         "text": "三個方向性檢定全滅，bar 級的方向領先主張到此為止。"
                 "唯一存活的是波動預測，但系統沒有旋鈕接得住。"},
    ]
    return {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "open": open_items, "settled": settled,
        "principle": "預註冊 = 先寫死規則與判準，再累積全新樣本。"
                     "提出假設用掉的樣本一律作廢，不得重複計入判決。",
        "disclaimer": "Forward validation in progress — research status "
                      "only, not a live strategy, not financial advice.",
    }


def main() -> int:
    payload = build()
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prereg_clocks (
                    id TINYINT PRIMARY KEY,
                    payload MEDIUMTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute(
                "INSERT INTO prereg_clocks (id, payload) VALUES (1, %s) "
                "ON DUPLICATE KEY UPDATE payload = VALUES(payload)",
                (json.dumps(payload, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()
    o = payload["open"]
    print(f"prereg_clocks published: {len(o)} open, "
          f"{len(payload['settled'])} settled | "
          + " | ".join(f"{x['id']}:{x['n']}/{x['gate_n']}"
                       for x in o if x["gate_n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
