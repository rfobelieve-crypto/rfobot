# -*- coding: utf-8 -*-
"""Publish the §0.75 arbitrage family to MySQL for the public site.

Why a publisher and not a cloud route (2026-09-01): the verdict lives in a
local JSON produced by `research/arb/premium_verdict.py`, which reads CSVs
written by a recorder running OFF-CLOUD on the operator's machine. A cloud
endpoint cannot see any of it — the fourth instance of the same fix family
as raid_signals / v7_veto / prereg_clocks (a route that cannot compute a
number silently serves a build-time snapshot instead).

PUBLIC-SAFE BY CONSTRUCTION. The site's rule is percentages, direction and
time only — never dollars. So this strips every money figure (book depth,
capturable USD/day) and republishes only:
  * progress (minutes recorded, days elapsed, gate)
  * band width in bps, fires/day, convergence fraction
  * funding carry as bps/8h and annualised %
  * a qualitative depth tier, because "the band is fat but the book is
    thin" is the single most important honest caveat on this line and it
    must survive the money-stripping.

Nothing here computes a verdict: `premium_verdict.py` is the single owning
scorer (the prereg-board discipline). This file only reshapes what that
scorer already wrote.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "research" / "results" / "arb_premium_verdict.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _depth_tier(usd) -> str | None:
    """Book depth as a tier, not a number (public pages carry no dollars).

    Thresholds are descriptive, not a gate: they exist so the site can say
    'thin' out loud. SNDK's fat prints sat around $200-400 — that is the
    fact the tiers are calibrated to communicate.
    """
    if usd is None:
        return None
    if usd < 250:
        return "thin"
    if usd < 1000:
        return "moderate"
    return "deep"


def _side(interim: dict, lab: str) -> dict | None:
    s = interim.get(lab)
    if not s:
        return None
    conv = interim.get(f"conv_{lab}") or {}
    depth = interim.get(f"depth_{lab}") or {}
    return {
        "band_bps": s.get("band_bps"),
        "fires_per_day": s.get("fires_per_day"),
        "converges": bool(conv.get("passed")) if conv.get("episodes") else None,
        "convergence_frac": conv.get("frac"),
        "convergence_episodes": conv.get("episodes"),
        "median_minutes_to_converge": conv.get("median_minutes"),
        "depth_tier": _depth_tier(depth.get("fat_median_notional_usd")),
        "stale_prints": depth.get("fat_stale_gt5s"),
    }


def build() -> dict:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    pairs = []
    for pid, p in (raw.get("pairs") or {}).items():
        interim = p.get("interim") or {}
        f = interim.get("funding") or {}
        pairs.append({
            "pair": pid,
            "legs": p.get("legs"),
            "note": p.get("note"),
            "status": p.get("status"),
            "minutes": p.get("minutes"),
            "days": p.get("days"),
            "gate_days": p.get("gate_days"),
            "start_utc": p.get("start_utc"),
            "is_control": pid == "BTC",
            "sell": _side(interim, "sell"),
            "buy": _side(interim, "buy"),
            "carry": ({
                "median_bps_8h": f.get("median_bps_8h"),
                "annualised_pct": f.get("annualised_pct_at_median"),
                "frac_positive": f.get("frac_positive"),
                "n": f.get("n"),
            } if f else None),
            "verdict": p.get("verdict"),
        })
    pairs.sort(key=lambda x: (-(x["minutes"] or 0), x["pair"]))
    return {
        "asof_utc": raw.get("asof_utc"),
        "gate_days": raw.get("gate_days"),
        "pairs": pairs,
        "principle": "判準 2026-08-28 凍結：扣費後 ≥1bps 的帶、日均 ≥10 次、"
                     "兩半皆成立、且偏離後要收斂。家族同判準、全部報告、"
                     "不挑好看的。BTC 是對照組——它若過閘代表儀器壞了。",
        "carry_note": "資金費率是第二條收益軸（持有就付，不需要收斂），"
                      "報告用、不進判準：凍結的四關是關於價差的，不因為"
                      "出現第二種收益就事後擴充。",
        "disclaimer": "Research recording only — not a live strategy, "
                      "not financial advice.",
    }


def main() -> int:
    if not SRC.exists():
        print("arb_publish: verdict json 不存在 — 先跑 premium_verdict.py")
        return 1
    payload = build()
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arb_status (
                    id TINYINT PRIMARY KEY,
                    payload MEDIUMTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    checked_at DATETIME NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            try:
                cur.execute("SELECT checked_at FROM arb_status LIMIT 1")
            except Exception:
                cur.execute("ALTER TABLE arb_status ADD COLUMN checked_at DATETIME NULL")
            cur.execute(
                "INSERT INTO arb_status (id, payload, checked_at) "
                "VALUES (1,%s,NOW()) ON DUPLICATE KEY UPDATE "
                "payload=VALUES(payload), checked_at=NOW()",
                (json.dumps(payload, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()
    n = len(payload["pairs"])
    live = sum(1 for p in payload["pairs"] if (p.get("carry") or {}).get("n"))
    print(f"arb_status published: {n} pairs, {live} with carry data, "
          f"asof {payload['asof_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
