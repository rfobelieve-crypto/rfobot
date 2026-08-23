# -*- coding: utf-8 -*-
"""Laptop-side parity check for the cloud train migration (weakness #1).

Compares the LOCAL shadow ledger's closed-row key hash against the
train_parity row the cloud recorder publishes hourly. The ledger is
deterministically recomputed from klines, so once both sides have seen the
same closed bars their CLOSED key sets must converge — first_seen differs
by construction and is excluded from the hash on both sides.

Cutover rule (frozen here before the cloud service exists): the laptop bat
may be demoted to puller-only after **7 consecutive days** of MATCH.
A MISMATCH during parallel running is not an emergency — the sides may be
one bar apart — but a mismatch that PERSISTS across runs means revision
or divergence and blocks cutover until explained.

Rides shadow_engine.bat after the local engine step. Log-only, exit 0.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"


def local_state():
    keys, per_sym = [], {}
    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "CLOSED":
                keys.append(f"{r.get('symbol')}|{r.get('fill_ts')}|"
                            f"{r.get('level_kind')}|{r.get('net_r')}")
                sym = str(r.get("symbol"))
                per_sym[sym] = per_sym.get(sym, 0) + 1
    keys.sort()
    return (len(keys), hashlib.md5("\n".join(keys).encode()).hexdigest(),
            per_sym)


def main() -> int:
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT payload, updated_at FROM train_parity "
                            "WHERE id=1")
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"train-parity: cloud row unavailable ({e}) — cloud train "
              f"not up yet or DB unreachable")
        return 0
    if not row:
        print("train-parity: no cloud row yet")
        return 0
    cloud = json.loads(row["payload"])
    n, h, per_sym = local_state()
    same = h == cloud.get("key_hash")
    print(f"train-parity: local {n} closed (hash {h[:10]}) vs cloud "
          f"{cloud.get('rows_closed')} (hash {str(cloud.get('key_hash'))[:10]},"
          f" {cloud.get('asof_utc')}, phase {cloud.get('phase')}) -> "
          f"{'MATCH' if same else 'MISMATCH'}")
    if not same:
        print("  (one run apart is normal; a PERSISTENT mismatch blocks "
              "cutover — see cutover rule in this file's docstring)")
        cps = cloud.get("per_symbol") or {}
        if cps:
            diffs = [f"{s} local {per_sym.get(s, 0)} vs cloud {cps.get(s, 0)}"
                     for s in sorted(set(per_sym) | set(cps))
                     if per_sym.get(s, 0) != cps.get(s, 0)]
            print("  per-symbol diff: " + ("; ".join(diffs) if diffs
                  else "counts identical -> same rows, differing net_r or "
                       "level_kind (data revision, not a missing recorder)"))
        else:
            print("  per-symbol map not published yet (cloud on pre-08-24 "
                  "payload) — next cloud cycle carries it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
