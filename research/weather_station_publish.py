"""Weather-station snapshot publisher — the survival layer's public face.

Computes every gauge/sensor state from the local kline caches (the hourly
SweepShadow task keeps them fresh) and UPSERTs ONE row into the
`weather_station` table.  The agent-mcp /public/weather-station endpoint
reads that row; the product site reads the endpoint.  This is the
agent-boundary-compliant path (2026-08-17): the quant system PERSISTS,
the agent only SELECTs — no compute inside the agent, no site access to
the trading path.

Public-surface rules apply to the payload: states, ratios and evidence
tiers only.  No sizes, no dollar figures, no model internals.  Every gauge
carries its verification tier so the site can label 已驗證/顯示 honestly
(研究結論上牆必須標狀態, CLAUDE.md).

Runs hourly, appended to shadow_engine.bat.  Read-only on market data;
writes only its own snapshot table.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.crowd_battery import paid_states  # noqa: E402
from research.crowd_battery2 import adx_state  # noqa: E402
from research.crowd_battery3 import (  # noqa: E402
    fetch_funding, paid_states_from_pos, pos_funding, pos_grid, pos_psar)
from research.survival_cards import CACHE, CORE9, SC  # noqa: E402


def _last(states: dict) -> str:
    if not states:
        return "UNKNOWN"
    v = list(states.values())[-1]
    if isinstance(v, str):
        return v
    return "PAID" if v > 0 else "STARVED"


def build_payload() -> dict:
    btc = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    v1 = paid_states(btc)                       # trend(SMA), mr, breakout
    adx_btc = _last(adx_state(btc))
    psar = _last(paid_states_from_pos(btc, pos_psar(btc)))
    grid = _last(paid_states_from_pos(btc, pos_grid(btc)))
    try:
        fund = _last(paid_states_from_pos(
            btc, pos_funding(btc, fetch_funding("BTC"))))
    except Exception:
        fund = "UNKNOWN"

    trending = paid_bo = n = 0
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        a = adx_state(bars)
        b = paid_states(bars)["breakout"]
        if a and b:
            n += 1
            trending += 1 if _last(a) == "TRENDING" else 0
            paid_bo += 1 if _last(b) == "PAID" else 0

    return {
        "updated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"),
        # tier: verified-2 = CI-grade (alert-eligible), verified-1 = sign+
        # breadth (display), sensor = zero registered predictions
        "gauges": [
            {"id": "adx", "label_zh": "ADX 市場狀態", "label_en": "ADX regime",
             "value": adx_btc, "detail": f"TRENDING {trending}/{n} coins",
             "tier": "verified-2",
             "note_zh": "盤整=獵取順風", "note_en": "ranging favors raids"},
            {"id": "breakout", "label_zh": "突破派", "label_en": "Breakout crowd",
             "value": _last(v1.get("breakout", {})),
             "detail": f"PAID {paid_bo}/{n} coins", "tier": "verified-1",
             "note_zh": "挨餓=獵取順風", "note_en": "starved favors raids"},
            {"id": "psar", "label_zh": "趨勢派 (PSAR)",
             "label_en": "Trend crowd (PSAR)", "value": psar,
             "tier": "verified-1",
             "note_zh": "盛宴=V7 逆風", "note_en": "feasting is V7 headwind"},
        ],
        "sensors": [
            {"id": "mr", "label_zh": "回歸派", "label_en": "Mean-reversion",
             "value": _last(v1.get("mr", {}))},
            {"id": "grid", "label_zh": "網格", "label_en": "Grid bots",
             "value": grid},
            {"id": "funding", "label_zh": "資金費率反向",
             "label_en": "Funding contrarian",
             "value": fund if fund != "STARVED" else "DORMANT"},
        ],
        "cadence": "hourly",
        "disclaimer": "Research display only. Not financial advice.",
    }


def publish(payload: dict) -> None:
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weather_station (
                    id TINYINT PRIMARY KEY,
                    payload MEDIUMTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute(
                "INSERT INTO weather_station (id, payload) VALUES (1, %s) "
                "ON DUPLICATE KEY UPDATE payload = VALUES(payload)",
                (json.dumps(payload, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    payload = build_payload()
    publish(payload)
    print("weather_station published:",
          ", ".join(f"{g['id']}={g['value']}" for g in payload["gauges"]))


if __name__ == "__main__":
    main()
