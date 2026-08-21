# -*- coding: utf-8 -*-
"""Cloud hourly train — moves the single most fragile piece of this system
(the laptop-scheduled shadow_engine.bat) onto Railway.

Why (2026-08-21, weakness #1 of the data-engineering audit): the hourly
recorder, 29-coin kline cache and all three DB publishers ride one Windows
laptop's Task Scheduler. It has already died three ways (29h CRLF outage,
bash-escaping corruption, path rot) and every death is silent-by-design of
the downstream fallbacks. The engine and every consumer already honor
SWEEP_DATA_DIR (volume) and SWEEP_KLINES_BASE (Binance mirror domain,
Railway IPs are 418-blocked on api.binance.com) — this file just adds the
scheduler and the cutover discipline.

Two phases, switched by TRAIN_PHASE env — because the shadow log is Gate F
evidence and MUST NOT be double-written by two recorders:

  parallel  (default) run ONLY shadow_engine into the volume copy and
            publish a parity row (train_parity, id=1: row count + key hash)
            so the laptop side can verify cloud==local for N days.
            No DB publishers — the laptop train stays the authority.
  authority run the FULL train (engine + weather/raid/veto publishers +
            pf mirror/intents). Flip this only after parity holds AND the
            laptop bat has been demoted to puller — never both.

Runs at :05 past each hour (the laptop bat's cadence), plus once at boot.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
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

PHASE = os.environ.get("TRAIN_PHASE", "parallel").strip().lower()
DATA_DIR = os.environ.get("SWEEP_DATA_DIR", "/data").strip()

PARALLEL_STEPS = ["research/sweep_failure/shadow_engine.py"]
AUTHORITY_STEPS = [
    "research/sweep_failure/shadow_engine.py",
    "research/weather_station_publish.py",
    "research/raid_signals_publish.py",
    "research/pf_mirror.py",
    "research/pf_dry_intents.py",
    "research/v7_veto_publish.py",
]


def log(msg: str) -> None:
    print(f"[train {datetime.now(timezone.utc):%m-%d %H:%M:%S}] {msg}",
          flush=True)


def run_step(rel: str) -> bool:
    r = subprocess.run([sys.executable, str(ROOT / rel)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900, cwd=str(ROOT))
    tail = (r.stdout or "").strip().splitlines()[-2:]
    for ln in tail:
        log(f"  {Path(rel).name}: {ln[:180]}")
    if r.returncode != 0:
        log(f"  {Path(rel).name} FAILED rc={r.returncode}: "
            f"{(r.stderr or '')[-300:]}")
        return False
    return True


def publish_parity() -> None:
    """Row count + key hash of the volume ledger, for local comparison.

    The ledger is deterministically recomputed from klines, so cloud and
    laptop must converge on the same (symbol, fill_ts, level_kind) key set
    for CLOSED rows. first_seen differs by construction and is excluded.
    """
    import csv
    from shared.db import get_db_conn
    p = Path(DATA_DIR) / "sweep_shadow_log.csv"
    if not p.exists():
        log("parity: no ledger yet")
        return
    keys = []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "CLOSED":
                keys.append(f"{r.get('symbol')}|{r.get('fill_ts')}|"
                            f"{r.get('level_kind')}|{r.get('net_r')}")
    keys.sort()
    digest = hashlib.md5("\n".join(keys).encode()).hexdigest()
    payload = {"rows_closed": len(keys), "key_hash": digest,
               "phase": PHASE,
               "asof_utc": datetime.now(timezone.utc).strftime(
                   "%Y-%m-%d %H:%M:%S")}
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS train_parity (
                    id TINYINT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute(
                "INSERT INTO train_parity (id, payload) VALUES (1, %s) "
                "ON DUPLICATE KEY UPDATE payload = VALUES(payload)",
                (json.dumps(payload),))
        conn.commit()
    finally:
        conn.close()
    log(f"parity published: {len(keys)} closed rows, hash {digest[:10]}")


def one_cycle() -> None:
    steps = AUTHORITY_STEPS if PHASE == "authority" else PARALLEL_STEPS
    log(f"cycle start (phase={PHASE}, {len(steps)} steps)")
    for rel in steps:
        try:
            run_step(rel)
        except Exception as e:  # noqa: BLE001
            log(f"  {rel} crashed: {e}")
    try:
        publish_parity()
    except Exception as e:  # noqa: BLE001
        log(f"parity failed: {e}")
    log("cycle done")


def main() -> None:
    log(f"cloud train boot — phase={PHASE}, data={DATA_DIR}")
    one_cycle()
    while True:
        # next :05 past the hour, pure epoch arithmetic (no calendar edges)
        now_s = time.time()
        nxt = (int(now_s) // 3600) * 3600 + 300
        if nxt <= now_s:
            nxt += 3600
        time.sleep(max(1.0, nxt - now_s))
        one_cycle()


if __name__ == "__main__":
    main()
