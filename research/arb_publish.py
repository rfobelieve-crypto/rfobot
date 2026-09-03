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
SCAN = ROOT / "research" / "results" / "arb_scan_rank.json"

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


# ── the battlefield scan (2026-09-03, TODO §1.00) ──────────────────────────
# One weapon, many battlefields: the recording family is the weapon and it is
# fixed; this block is the search for where to point it. It is a SEARCH board,
# never a verdict board — `scan_rank.py` owns the promotion metric and
# `premium_verdict.py` owns every verdict, exactly as before.
#
# Same money-stripping as the rest of this file: band in bps, fires/day,
# depth as a TIER. `capturable_usd_per_day` (the ranking metric) never
# reaches the payload — only the ORDER it produces does.
CLASS_SETS = {
    "商品": {"GOLD", "SILVER", "COPPER", "PLATINUM", "PALLADIUM", "NATGAS",
             "CL", "BRENTOIL", "WTI", "OIL", "URANIUM", "ALUMINIUM", "URNM",
             "WHEAT", "SOY"},
    "指數": {"SP500", "XYZ100", "JP225", "KR200", "US500", "USA500", "USTECH",
             "USA100", "SMALL2000", "MAGS", "SMH", "SOXL", "XBI", "XLE",
             "TOTAL2", "OTHERS", "BTCD", "SEMI", "IGV"},
    "外匯": {"EUR", "GBP", "JPY"},
    "利率": {"2Y", "10Y", "30Y", "USBOND", "SGOV"},
}
# Venues that ONLY list non-crypto. Lighter's Robinhood chain is deliberately
# NOT here: it carries stocks AND crypto (BTC, ZEC, HYPE), so its presence
# says nothing about the asset (first version called ZEC a stock).
STOCK_VENUES = {"xyz", "IO", "para", "mkts"}


def _asset_class(pair: str) -> str:
    """Display-only bucket. Ticker sets first, then venue: a pair that
    touches a stock/commodity dex is not crypto even if we do not know
    the ticker."""
    head, _, venues = pair.partition("@")
    for label, members in CLASS_SETS.items():
        if head in members:
            return label
    legs = set(venues.split("-", 1)) if venues else set()
    legs |= {v for v in STOCK_VENUES if v in venues}
    return "股票" if legs & STOCK_VENUES else "加密"


def _scan_block() -> dict | None:
    if not SCAN.exists():
        return None
    d = json.loads(SCAN.read_text(encoding="utf-8"))
    ctrl = d.get("control_band_bps")
    promote = set(d.get("promote") or [])
    rows = []
    for r in (d.get("top") or [])[:20]:
        band = r.get("band_bps")
        rows.append({
            "pair": r.get("pair"),
            "asset_class": _asset_class(r.get("pair") or ""),
            "band_bps": band,
            # how far above the control pair's band — the instrument's own
            # noise floor. <=2x is "not distinguishable from spread".
            "band_vs_control": (round(band / ctrl, 2)
                                if band and ctrl else None),
            "fires_per_day": r.get("fires_per_day"),
            "depth_tier": _depth_tier(r.get("depth_usd")),
            "samples": r.get("n"),
            "stage": ("升格候選" if r.get("pair") in promote else
                      "掃描中" if d.get("gate_ok") else "資料未滿"),
        })
    return {
        "asof_utc": d.get("asof_utc"),
        "span_days": d.get("span_days"),
        "quotes": d.get("quotes"),
        "pairs": d.get("pairs"),
        "gate_ok": d.get("gate_ok"),
        "control_band_bps": ctrl,
        "rows": rows,
        # The honest ladder. A row near the top of this board has passed
        # exactly ONE of these steps.
        "ladder": [
            {"step": "掃描", "state": "本板", "means": "兩個場館的報價差得夠開、夠常發生"},
            {"step": "錄製 7 天", "state": "下一步",
             "means": "升格後從自己的第一分鐘起算，掃描期資料不進判決"},
            {"step": "收斂關", "state": "未做",
             "means": "偏離後要回得來；持續偏移看起來最肥卻永遠拿不到"},
            {"step": "費率查證", "state": "未做",
             "means": "零費率是獲客補貼，builder dex 不可假設也是 0"},
            {"step": "小額實盤", "state": "未做",
             "means": "回測與錄製都過了也只是紙上；斷腿與真實成交才算數"},
        ],
        "caveat": "本板是「找戰場」，不是「已賺到」。帶寬與次數是掃描期的"
                  "觀察，深度只給分級；排序用的可捕獲金額不出現在公開頁。"
                  "任何一列都還沒過收斂關、沒查費率、沒有實盤成交。",
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
        "scan": _scan_block(),
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
    sc = payload.get("scan") or {}
    print(f"arb_status published: {n} pairs, {live} with carry data, "
          f"asof {payload['asof_utc']}"
          + (f" | scan {sc.get('pairs')} 配對 / {sc.get('span_days')}天 "
             f"→ {len(sc.get('rows') or [])} 列上牆" if sc
             else " | scan MISSING (先跑 arb/scan_rank.py)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
