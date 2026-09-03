# -*- coding: utf-8 -*-
"""Pull jarvis V7Bot executions and publish them for the chart overlay.

Why (2026-08-31, user: 「V7 的進出場圖表上面應該要標示才對」): the charts'
live-execution overlay reads `v7_okx_positions`, frozen at 2026-08-11 when
execution migrated to jarvis/Bitget. jarvis has written every V7 entry/exit
to `{userDir}/v7_trades.jsonl` since 2026-08-25 (v7bot `_trade`), and the
research export channel (RESEARCH_EXPORT_TOKEN, TODO §0.78) is the
sanctioned product→research data path. This script:

  1. GET `<MILL_EXPORT_URL with /export/fills → /export/v7>` (same token,
     same uid — the user's single §0.78 setup step powers both pipelines)
  2. keeps a local replayable copy research/results/v7_product_trades.jsonl
  3. pairs entry/exit events into rounds (single-position bot: an entry is
     open until the next exit event; a flip entry closes nothing by itself
     because v7bot always emits the exit first)
  4. upserts rounds into MySQL `v7_product_trades` (PK = entry ms), which
     `indicator/v7_product_trades.py` serves to both charts

Honest notes:
  - this is a RECONSTRUCTION of rounds from the event stream, mirroring the
    mill pipeline's division of labour (product writes events, research
    pairs them). If counts ever disagree with jarvis's own stats, suspect
    the pairing here first.
  - rows before 2026-08-25 don't exist (the ledger didn't); the 08-11→08-25
    execution gap is real and stays unmarked.
  - env unset → exit 0 with a skip line (the hourly bat must not fail while
    the user hasn't set the token).

Self-test (no network): `python research/v7_product_trades_publish.py
--selftest` feeds a synthetic event stream with a known answer through the
pairing + upsert + chart-fetch path, then removes its rows (source =
'selftest'; the chart fetch excludes that source) — a new instrument gets
verified on known answers before its output is believed (mistake.md
2026-07-29).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "results" / "v7_product_trades.jsonl"
# Health of the PIPE, written every run (2026-09-03). The rows artifact
# cannot report this: "no fills yet" (legitimate) and "misconfigured"
# (a bug) both produce zero rows, which is exactly how this pipeline sat
# broken for days printing one skip line an hour. freshness_board reads
# the `ok` flag -- see mistake.md 2026-09-01 ("translate 'never started'
# into 'some number is wrong'").
STATUS = ROOT / "research" / "results" / "v7_product_trades_status.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if v:
        return v
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _export_url(base: str, leaf: str) -> str:
    """Normalise the export URL to the V1.62.0 location.

    2026-09-02 handover (產品端請求_研究端匯出搬家): the endpoint moved from
    /api/u/export/* to /api/research/export/*, the token is header-only
    (a token in the query string now returns 400 — it leaks into access logs
    and referers), and uid must be the 16-hex id, not the account name.
    Old URLs still in .env are rewritten here so the migration is one edit,
    not a scavenger hunt across machines.
    """
    base = base.replace("/api/u/export/", "/api/research/export/")
    if base.rstrip("/").endswith("/export"):
        base = base.rstrip("/") + "/" + leaf
    for other in ("fills", "v7"):
        if base.endswith("/export/" + other) and other != leaf:
            base = base[: -len(other)] + leaf
    return base


def _check_uid(uid: str) -> bool:
    """The product side stopped accepting account names (they are guessable)."""
    ok = len(uid) == 16 and all(c in "0123456789abcdefABCDEF" for c in uid)
    if not ok:
        print(f"MILL_EXPORT_UID='{uid}' 不是 16 位 hex 的使用者 id。"
              "產品端 V1.62.0 起只收 id（名字回 404）——請把 .env 裡的名字"
              "換成 id 後重跑。")
    return ok


def _status(ok: bool, reason: str) -> None:
    """State the pipe's own health. Never raises: a broken monitor must not
    break the hourly train, but it must not be silent either (the
    2026-08-01 rule about non-silent except blocks)."""
    try:
        STATUS.write_text(json.dumps(
            {"asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "ok": bool(ok), "reason": reason}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] v7_product_trades status write failed: {exc}")


def fetch_events() -> list:
    base = _env("MILL_EXPORT_URL")
    token = _env("MILL_EXPORT_TOKEN")
    uid = _env("MILL_EXPORT_UID")
    if not (base and token and uid):
        print("v7_product_trades: MILL_EXPORT_* not set — skip (token 未設，"
              "同 §0.78 的那一步)")
        _status(False, "MILL_EXPORT_* 未設")
        return []
    if not _check_uid(uid):
        _status(False, f"MILL_EXPORT_UID='{uid}' 不是 16 hex id（產品端 V1.62.0 起只收 id）")
        return []
    url = (_export_url(base, "v7") + "?"
           + urllib.parse.urlencode({"uid": uid}))
    req = urllib.request.Request(url, headers={"x-export-token": token})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        # 401 today = /export/v7 not shipped on the product side yet (unknown
        # research subs fall through to the login-required handler); network
        # errors likewise must not fail the hourly bat. One line, exit 0.
        print(f"v7_product_trades: fetch failed ({exc}) — skip "
              f"(product side /export/v7 pending, 規格請求 2026-08-31)")
        _status(False, f"fetch failed: {exc}")
        return []
    rows = data.get("rows") or []
    # 200 with zero rows is HEALTHY (the bot simply has not traded yet) --
    # the flag is about reachability + configuration, never about volume.
    _status(True, f"HTTP 200, {len(rows)} events for {data.get('user')}")
    print(f"v7_product_trades: fetched {len(rows)} events for {data.get('user')}")
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
                   + ("\n" if rows else ""), encoding="utf-8")
    return rows


def pair_rounds(events: list) -> list:
    """Entry/exit event stream → rounds. Single-position bot semantics."""
    rounds, open_row = [], None
    for e in sorted(events, key=lambda x: x.get("t") or 0):
        ev = e.get("event")
        ts = e.get("t")
        if ev == "entry" and ts:
            if open_row is not None:
                # two entries with no exit between them: product-side flip
                # emits exit first, so this means a lost exit event — close
                # the dangling row as unknown rather than inventing an exit.
                open_row["status"] = "CLOSED"
                open_row["exit_reason"] = "unknown_missing_exit"
                rounds.append(open_row)
            open_row = {
                "entry_ms": int(ts),
                "entry_time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                              .replace(tzinfo=None),
                "direction": "LONG" if e.get("side") == "buy" else "SHORT",
                "entry_price": float(e.get("entry") or 0) or None,
                "size_base": e.get("sizeBase"),
                "is_flip": 1 if e.get("isFlip") else 0,
                "status": "OPEN", "exit_time": None, "exit_price": None,
                "exit_reason": None, "pnl": None, "r": None, "win": None,
            }
        elif ev == "exit" and ts:
            if open_row is None:
                continue            # exit with no tracked entry (pre-history)
            open_row["status"] = "CLOSED"
            open_row["exit_time"] = (datetime
                                     .fromtimestamp(ts / 1000, tz=timezone.utc)
                                     .replace(tzinfo=None))
            open_row["exit_price"] = (float(e["exit"])
                                      if e.get("exit") is not None else None)
            open_row["exit_reason"] = e.get("reason")
            open_row["pnl"] = e.get("pnl")
            open_row["r"] = e.get("r")
            if e.get("pnl") is not None:
                open_row["win"] = 1 if float(e["pnl"]) > 0 else 0
            rounds.append(open_row)
            open_row = None
    if open_row is not None:
        rounds.append(open_row)
    return rounds


def upsert(rounds: list, source: str = "bitget") -> int:
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS v7_product_trades (
                    entry_ms BIGINT PRIMARY KEY,
                    entry_time DATETIME NOT NULL,
                    direction VARCHAR(8) NOT NULL,
                    entry_price DOUBLE NULL,
                    size_base DOUBLE NULL,
                    is_flip TINYINT NOT NULL DEFAULT 0,
                    status VARCHAR(8) NOT NULL,
                    exit_time DATETIME NULL,
                    exit_price DOUBLE NULL,
                    exit_reason VARCHAR(40) NULL,
                    pnl DOUBLE NULL,
                    r DOUBLE NULL,
                    win TINYINT NULL,
                    source VARCHAR(16) NOT NULL DEFAULT 'bitget',
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    KEY idx_entry_time (entry_time)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            for rr in rounds:
                cur.execute(
                    "INSERT INTO v7_product_trades (entry_ms, entry_time, "
                    " direction, entry_price, size_base, is_flip, status, "
                    " exit_time, exit_price, exit_reason, pnl, r, win, source) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON DUPLICATE KEY UPDATE status=VALUES(status), "
                    " exit_time=VALUES(exit_time), exit_price=VALUES(exit_price), "
                    " exit_reason=VALUES(exit_reason), pnl=VALUES(pnl), "
                    " r=VALUES(r), win=VALUES(win)",
                    (rr["entry_ms"], rr["entry_time"], rr["direction"],
                     rr["entry_price"], rr["size_base"], rr["is_flip"],
                     rr["status"], rr["exit_time"], rr["exit_price"],
                     rr["exit_reason"], rr["pnl"], rr["r"], rr["win"], source))
        conn.commit()
        return len(rounds)
    finally:
        conn.close()


def selftest() -> int:
    """Known-answer test of pairing + upsert + chart fetch. No network."""
    t0 = 946684800000   # 2000-01-01 — far outside any chart window's data,
                        # and source='selftest' is excluded by the fetch.
    ev = [
        {"t": t0 + 0, "event": "entry", "side": "buy", "entry": 100.0,
         "sizeBase": 0.01, "isFlip": False},
        {"t": t0 + 3_600_000, "event": "exit", "side": "buy", "reason": "opp",
         "entry": 100.0, "exit": 110.0, "pnl": 0.1, "r": 1.5},
        {"t": t0 + 7_200_000, "event": "entry", "side": "sell", "entry": 110.0,
         "sizeBase": 0.01, "isFlip": True},
    ]
    rounds = pair_rounds(ev)
    assert len(rounds) == 2, rounds
    a, b = rounds
    assert a["direction"] == "LONG" and a["status"] == "CLOSED" \
        and a["exit_price"] == 110.0 and a["win"] == 1 and a["r"] == 1.5, a
    assert b["direction"] == "SHORT" and b["status"] == "OPEN" \
        and b["is_flip"] == 1, b
    print("pairing: known answer OK (2 rounds, LONG closed win / SHORT open flip)")
    n = upsert(rounds, source="selftest")
    print(f"upsert: {n} rows (source=selftest)")
    from indicator.v7_product_trades import fetch_v7_product_trades_for_chart
    got = fetch_v7_product_trades_for_chart(datetime(1999, 1, 1),
                                            datetime(2001, 1, 1))
    assert got == [], f"chart fetch must exclude selftest rows, got {got}"
    print("chart fetch: selftest rows correctly excluded")
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM v7_product_trades WHERE source='selftest'")
        conn.commit()
    finally:
        conn.close()
    print("selftest rows removed — PASS")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    events = fetch_events()
    if not events:
        return 0
    rounds = pair_rounds(events)
    n = upsert(rounds)
    closed = sum(1 for r in rounds if r["status"] == "CLOSED")
    print(f"v7_product_trades: upserted {n} rounds ({closed} closed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
