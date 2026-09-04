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
    ("okx balance snapshots", "db",
     "v7_okx_balance_snapshots:ts", 1.5,
     "executor WS alive — the aliveness signal per mistake 2026-07-28"),
    ("cloud train parity", "db",
     "train_parity:updated_at", 2.5,
     "cloud recorder alive (weakness-#1 migration; RED until service up)"),
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
    ("arb scanner (§0.75b)", "glob_newest",
     "../arb/engine/logs/scan/scan_*.csv", 0.5,
     "cross-venue REST scanner, ~2 min cycle over ~110 pairs"),
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
