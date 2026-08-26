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
import json
import sys
import time
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
    ("prereg board row", "db",
     "prereg_clocks:updated_at", 2.5,
     "site research-progress board; a frozen board reads as 'no progress'"),
    # -- the cloud indicator service --
    ("indicator bars", "db",
     "indicator_history:dt", 2.5,
     "V7 inference alive (the honest liveness witness)"),
    ("okx balance snapshots", "db",
     "v7_okx_balance_snapshots:ts", 1.5,
     "executor WS alive — the aliveness signal per mistake 2026-07-28"),
    ("cloud train parity", "db",
     "train_parity:updated_at", 2.5,
     "cloud recorder alive (weakness-#1 migration; RED until service up)"),
    # -- daily --
    ("coinglass parquets", "glob",
     "market_data/raw_data/cg_*.parquet", 48.0,
     "STALE-DATA guard threshold; stalest file reported"),
    ("daily collect log", "file",
     "research/results/daily_collect.log", 30.0,
     "04:00 daily task heartbeat"),
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


def age_glob(pattern: str) -> tuple[float | None, str]:
    files = list(ROOT.glob(pattern))
    if not files:
        return None, "(no match)"
    stalest = min(files, key=lambda f: f.stat().st_mtime)
    return (time.time() - stalest.stat().st_mtime) / H, stalest.name


def age_db(spec: str, conn) -> float | None:
    table, col = spec.split(":")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX({col}) m FROM {table}")   # noqa: S608
            row = cur.fetchone()
        m = row and row.get("m")
        if m is None:
            return None
        if not isinstance(m, datetime):
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
        detail = ""
        if kind == "file":
            age = age_file(target)
        elif kind == "glob":
            age, detail = age_glob(target)
        else:
            age = age_db(target, conn) if conn else None
        ok = age is not None and age <= max_h
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
    prev = set()
    if STATE.exists():
        try:
            prev = set(json.loads(STATE.read_text())["reds"])
        except Exception:
            prev = set()
    cur = set(reds)
    new_red = sorted(cur - prev)
    recovered = sorted(prev - cur)
    STATE.write_text(json.dumps({"reds": sorted(cur)}), encoding="utf-8")

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
            import os
            from indicator.okx.alerter import send_critical
            chat = (os.environ.get("TG_ALERT_CHAT_ID")
                    or os.environ.get("TG_CRITICAL_CHAT_ID") or "")
            sent = False
            if chat:
                for _ in range(6):
                    if send_critical(chat, msg):
                        sent = True
                        break
                    time.sleep(60)
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
