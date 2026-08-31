# -*- coding: utf-8 -*-
"""Tamper-evident hash chain over the V7 signal history (TODO §0.83 #1).

Threat: "signal tampering is the most lethal risk" — and half of that
threat is retroactive: an attacker (or a bug, or a well-meaning backfill)
rewriting tracked_signals history so the public track record no longer
matches what was actually published. The project already has the RULE
("no signal overwrite on retrain", memory feedback_no_signal_overwrite);
this makes the rule machine-checkable.

Mechanism: every hour (shadow_engine bat) append new tracked_signals rows
to `signal_audit_chain`:

    payload_hash = sha256(canonical json of the row's immutable fields)
    chain_hash   = sha256(prev_chain_hash + payload_hash)

A row edited after being chained changes its payload hash and breaks every
later link. `--verify` recomputes the whole chain against the CURRENT
table contents and reports the first broken link — silent history edits
become loud.

Scope note (honest): `correct` / `actual_return_4h` are backfilled ~4h
after signal time by design, so the chain covers only the fields that are
immutable from the moment of publication: id, signal_time, direction,
strength, confidence, entry_price. The chain is appended only for rows
older than FINALIZE_MIN minutes so a row is never chained mid-write.

Writer: research side (this script). Reader: anyone. The agent service
never writes here (agent-boundary).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

FIELDS = ("id", "signal_time", "direction", "strength", "confidence",
          "entry_price")
FINALIZE_MIN = 10
GENESIS = "0" * 64

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _canon(row: dict) -> str:
    return json.dumps({k: str(row.get(k)) for k in FIELDS},
                      sort_keys=True, separators=(",", ":"))


def _payload_hash(row: dict) -> str:
    return hashlib.sha256(_canon(row).encode("utf-8")).hexdigest()


def _chain_hash(prev: str, payload_hash: str) -> str:
    return hashlib.sha256((prev + payload_hash).encode("utf-8")).hexdigest()


def _ensure_table(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signal_audit_chain (
            seq BIGINT AUTO_INCREMENT PRIMARY KEY,
            signal_id INT NOT NULL UNIQUE,
            signal_time DATETIME NOT NULL,
            payload_hash CHAR(64) NOT NULL,
            chain_hash CHAR(64) NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")


def append() -> int:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            _ensure_table(cur)
            cur.execute("SELECT signal_id, chain_hash FROM signal_audit_chain "
                        "ORDER BY seq DESC LIMIT 1")
            last = cur.fetchone()
            last_id = last["signal_id"] if last else 0
            prev = last["chain_hash"] if last else GENESIS
            cur.execute(
                "SELECT id, signal_time, direction, strength, confidence, "
                "entry_price FROM tracked_signals WHERE id > %s "
                "AND created_at < NOW() - INTERVAL %s MINUTE "
                "ORDER BY id ASC", (last_id, FINALIZE_MIN))
            rows = cur.fetchall()
            batch = []
            for r in rows:
                ph = _payload_hash(r)
                prev = _chain_hash(prev, ph)
                batch.append((r["id"], r["signal_time"], ph, prev))
            if batch:
                cur.executemany(
                    "INSERT INTO signal_audit_chain (signal_id, signal_time, "
                    "payload_hash, chain_hash) VALUES (%s,%s,%s,%s)", batch)
        conn.commit()
        print(f"signal_audit_chain: appended {len(rows)} (head id={last_id})")
        return 0
    finally:
        conn.close()


def verify() -> int:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            _ensure_table(cur)
            cur.execute("SELECT seq, signal_id, payload_hash, chain_hash "
                        "FROM signal_audit_chain ORDER BY seq ASC")
            chain = cur.fetchall()
            if not chain:
                print("signal_audit_chain: empty — nothing to verify")
                return 0
            ids = [c["signal_id"] for c in chain]
            fmt = ",".join(["%s"] * len(ids))
            cur.execute(
                f"SELECT id, signal_time, direction, strength, confidence, "
                f"entry_price FROM tracked_signals WHERE id IN ({fmt})", ids)
            cur_rows = {r["id"]: r for r in cur.fetchall()}
        prev = GENESIS
        for c in chain:
            row = cur_rows.get(c["signal_id"])
            if row is None:
                print(f"✗ BROKEN: signal id {c['signal_id']} chained but "
                      f"MISSING from tracked_signals (deleted?)")
                return 1
            ph = _payload_hash(row)
            if ph != c["payload_hash"]:
                print(f"✗ BROKEN: signal id {c['signal_id']} content changed "
                      f"after chaining (payload hash mismatch)")
                return 1
            prev = _chain_hash(prev, ph)
            if prev != c["chain_hash"]:
                print(f"✗ BROKEN: chain link at seq {c['seq']} "
                      f"(id {c['signal_id']}) — chain rewritten")
                return 1
        print(f"✓ signal_audit_chain intact: {len(chain)} links, "
              f"head={prev[:16]}…")
        return 0
    finally:
        conn.close()


def main() -> int:
    if "--verify" in sys.argv:
        return verify()
    return append()


if __name__ == "__main__":
    raise SystemExit(main())
