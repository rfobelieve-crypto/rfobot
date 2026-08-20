# -*- coding: utf-8 -*-
"""V7 raid-veto adoption clock publisher — third instance of the
"recorder runs off-cloud" fix (family: a2fd90f raid-signals, 2026-08-17
weather_station).

Why this exists (2026-08-20): the site's V7 filter card reads the agent,
the agent calls the indicator service's /research/v7-clock, and that route
shells out to v7_raid_veto.py — which needs the LOCAL kline cache
(research/sweep_failure/.cache/*.csv, not in git, not in the image).  On
Railway the script fails every time and the route silently serves the
JSON committed at build time: the card sat at asof 08-10 / trigger 4/60
while the truth was 34/60.  The user spotted it as "數據有問題".

Fix = the established pattern: THIS machine (which has the cache and
rides the hourly shadow_engine.bat train) computes the clock and UPSERTs
one row into `v7_veto_clock`; the agent reads the row.  The quant system
persists, the agent only SELECTs (agent-boundary.md).

Runs v7_raid_veto.py --clock as a subprocess (one implementation of the
numbers — this file adds zero arithmetic of its own) and publishes the
JSON it wrote.  A compute failure publishes nothing: a stale row with an
honest updated_at beats a fresh row of wrong numbers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLOCK_SCRIPT = ROOT / "research" / "v7_raid_veto.py"
CLOCK_JSON = ROOT / "research" / "results" / "v7_veto_clock.json"


def main() -> int:
    r = subprocess.run([sys.executable, str(CLOCK_SCRIPT), "--clock"],
                       capture_output=True, text=True, timeout=110,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    if r.returncode != 0:
        # Loud, not silent (mistake.md 2026-08-01): the hourly log must
        # show WHY there is no fresh row today.
        print("[WARN] v7_veto_publish: clock script failed:",
              (r.stderr or "")[-400:])
        return 1
    payload = json.loads(CLOCK_JSON.read_text(encoding="utf-8"))

    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS v7_veto_clock (
                    id TINYINT PRIMARY KEY,
                    payload MEDIUMTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute(
                "INSERT INTO v7_veto_clock (id, payload) VALUES (1, %s) "
                "ON DUPLICATE KEY UPDATE payload = VALUES(payload)",
                (json.dumps(payload, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()
    print("v7_veto_clock published: asof", payload.get("asof_utc"),
          "trigger", payload.get("strong_since_trigger"), "/",
          payload.get("trigger_target"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
