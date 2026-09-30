"""Freeze the website's data panels into static JSON (2026-09-30).

The public site (../product-site) used to read these payloads live from
agent-mcp `/public/*`, which SELECTed rows that the publishers below wrote to
MySQL every hour. Railway (agent-mcp + MySQL) was retired when the project
became a portfolio, so the site now reads `content/archive/<name>.json`
instead. This script produces those files by calling the SAME builder each
publisher calls — no second implementation, so a panel cannot drift from the
number its publisher would have shown.

Not produced here (their source table died with the MySQL):
  live-status / track-record  -> recovered by hand from the Vercel ISR cache,
                                 see D:/flowbot_data/site_cache_rescue/
  signal-feed / signal-history / cancel-flow-stats -> none; the site hides them

Run: python research/site_export.py      (writes into ../product-site)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "product-site" / "content" / "archive"
for p in (str(ROOT), str(ROOT / "research"), str(ROOT / "research" / "sweep_failure")):
    if p not in sys.path:
        sys.path.insert(0, p)

DISCLAIMER = ("Informational and analytical output only. Not financial advice. "
              "Past performance does not guarantee future results.")


def _sweep_status() -> dict:
    from indicator.agent.server import _sweep_status_payload
    return _sweep_status_payload()


def _arb() -> dict:
    import arb_publish
    return arb_publish.build()


def _ops() -> dict:
    import ops_board_publish
    return ops_board_publish.build()


def _weather() -> dict:
    import weather_station_publish
    return weather_station_publish.build_payload()


def _prereg() -> dict:
    import prereg_publish
    return prereg_publish.build()


JOBS = {
    "sweep-status": _sweep_status,
    "arb-status": _arb,
    "ops-board": _ops,
    "weather-station": _weather,
    "prereg-clocks": _prereg,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed = []
    for name, fn in JOBS.items():
        try:
            data = fn()
            if not isinstance(data, dict) or "error" in data:
                raise RuntimeError(f"builder returned {str(data)[:200]}")
            data.setdefault("disclaimer", DISCLAIMER)
            data["published_utc"] = now
            data["_source"] = "archive"
            (OUT / f"{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"OK   {name:16s} {len(json.dumps(data)):>8,d} B")
        except Exception as e:  # noqa: BLE001 — report every job, then fail
            failed.append(name)
            print(f"FAIL {name:16s} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
