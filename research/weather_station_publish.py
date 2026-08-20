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


def _carry_state() -> str:
    try:
        from research.crowd_battery4 import carry_states
        st = carry_states()
        return list(st.values())[-1] if st else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _liq_bounce_state(btc_bars) -> str:
    try:
        from research.crowd_battery3 import paid_states_from_pos
        from research.crowd_battery4 import liq_burst_hours, pos_liq_bounce
        st = paid_states_from_pos(
            btc_bars, pos_liq_bounce(btc_bars, liq_burst_hours()))
        return _last(st)
    except Exception:
        return "UNKNOWN"


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
            # v4 (2026-08-18, §0.49g): carry richness is the most macro
            # sensor on the board — THIN has held for the ENTIRE 12-month
            # sample (7d-mean funding never crossed baseline), i.e. no
            # crowded-long mania era yet; RICH appearing would be news.
            {"id": "carry", "label_zh": "carry 農夫",
             "label_en": "Carry farmers", "value": _carry_state()},
            # liq-bounce data rides DailyCollect (daily 04:00) — up to a
            # day stale by design; the gauge is regime-scale anyway.
            {"id": "liq_bounce", "label_zh": "清算搶反彈",
             "label_en": "Liq-bounce hunters", "value": _liq_bounce_state(btc)},
        ],
        # ── reefing (縮帆, §0.52/§0.53, added 2026-08-20) ────────────────
        # The station's RESPONSE arm: the gauges above measure the wind,
        # this block says how much sail the pre-registered rules would
        # carry right now.  Status is "preregistered" until the forward
        # verdicts land — the site must render it as 預註冊·forward累積中,
        # never as an active rule (研究結論上牆必須標狀態, CLAUDE.md).
        # Ratios and dates only; no sizes, no dollars.
        "reefing": _reefing_block(btc, adx_btc),
        "cadence": "hourly",
        "disclaimer": "Research display only. Not financial advice.",
    }


def _reefing_block(btc_bars, adx_btc: str) -> dict:
    """Current would-be weights of the frozen reefing rules + verdict clocks.

    Weights come from the same frozen §0.53 machinery the scorer replays
    (vol_target_shadow.weight_series) — one implementation, no drift
    between what the site shows and what the verdict will be judged on.
    """
    from datetime import date

    try:
        from research.vol_target_shadow import weight_series
        w_btc = None
        ws = weight_series(btc_bars)
        if ws:
            w_btc = list(ws.values())[-1]
        w_all, n = 0.0, 0
        for sym in CORE9:
            fp = CACHE / f"{sym}USDT_1h.csv"
            if not fp.exists():
                continue
            s = weight_series(SC.load_csv(str(fp)))
            if s:
                w_all += list(s.values())[-1]
                n += 1
        w_core9 = round(w_all / n, 3) if n else None
    except Exception:
        w_btc = w_core9 = None

    try:
        from research.crowd_battery import pos_mr
        from research.crowd_battery2 import pos_bb_mr
        from research.crowd_battery3 import pos_grid as _pg
        w = btc_bars[-1440:]
        c = [b[SC.C] for b in w]

        def ser(pos):
            o, pv = [], 0.0
            for i in range(len(c) - 1):
                p = float(pos[i])
                o.append(p * (c[i + 1] / c[i] - 1) - abs(p - pv) * 5e-4)
                pv = p
            return o

        def cor(a, b):
            k = min(len(a), len(b))
            ma, mb = sum(a[:k]) / k, sum(b[:k]) / k
            va = sum((x - ma) ** 2 for x in a[:k]) ** .5
            vb = sum((x - mb) ** 2 for x in b[:k]) ** .5
            return 0.0 if va == 0 or vb == 0 else sum(
                (a[i] - ma) * (b[i] - mb) for i in range(k)) / (va * vb)

        fam = [ser(pos_mr(w)), ser(_pg(w)), ser(pos_bb_mr(w))]
        prs = [cor(fam[0], fam[1]), cor(fam[0], fam[2]), cor(fam[1], fam[2])]
        corr60 = round(sum(prs) / 3, 2)
    except Exception:
        corr60 = None

    today = date.today()
    return {
        "status": "preregistered",          # site: 預註冊·forward累積中
        "label_zh": "縮帆", "label_en": "Reefing",
        "note_zh": "不預測風暴，只在風大時收帆——風控研究線，未生效",
        "note_en": "reduce sail when the wind is up; pre-registered, not live",
        "vol_target_w_btc": round(w_btc, 3) if w_btc is not None else None,
        "vol_target_w_core9": w_core9,
        "adx_desize_now": "x0.5" if adx_btc == "TRENDING" else "x1.0",
        "mr_family_corr_60d": corr60,
        "clocks": [
            {"id": "adx_desize", "label_zh": "ADX 減碼判決",
             "due": "2026-09-16",
             "days_left": max(0, (date(2026, 9, 16) - today).days)},
            {"id": "vol_target", "label_zh": "vol targeting 判決",
             "due": "2026-09-19",
             "days_left": max(0, (date(2026, 9, 19) - today).days)},
        ],
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
