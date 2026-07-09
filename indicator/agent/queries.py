"""Read-only DB queries for the MCP agent.

Every SELECT the agent runs lives HERE, in one file, so the boundary is
auditable at a glance: this module contains only SELECTs against the
quant system's existing tables. No writes, no imports of any trading /
executor / inference module. See .claude/rules/agent-boundary.md.

If OKX_AGENT_SEED=1, all queries return canned demo data instead of
touching MySQL — lets a reviewer clone the repo and run the MCP server
with zero infrastructure.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

DISCLAIMER = (
    "Informational and analytical output only. Not financial advice. "
    "Past performance does not guarantee future results."
)


def _seed_mode() -> bool:
    return os.environ.get("OKX_AGENT_SEED", "").lower() in ("1", "true", "yes")


def _conn():
    # Imported lazily so seed mode needs no DB driver / no env at all.
    from shared.db import get_db_conn
    return get_db_conn()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Tool 1: current signal ─────────────────────────────────────────────

def latest_signal() -> dict[str, Any]:
    if _seed_mode():
        return {
            "signal_time": "2026-07-07T00:00:00+00:00",
            "direction": "UP",
            "tier": "Strong",
            "confidence": 87.3,
            "regime": "TRENDING_BULL",
            "entry_price": 63120.5,
            "top_drivers": [
                {"feature": "cg_bfx_margin_delta", "shap": 0.0412},
                {"feature": "impact_asymmetry", "shap": -0.0233},
                {"feature": "post_absorb_breakout", "shap": 0.0198},
            ],
            "model_version": "v7-dual-2026-06",
            "disclaimer": DISCLAIMER,
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, strength, confidence, "
                "       regime, entry_price, model_version, shap_top "
                "FROM tracked_signals ORDER BY signal_time DESC LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return {"error": "no signals recorded yet", "disclaimer": DISCLAIMER}

    drivers: list[dict] = []
    raw = row.get("shap_top")
    if raw:
        try:
            parsed = json.loads(raw)
            # shap_top may be a list of [feature, value] or dicts
            for item in (parsed or [])[:3]:
                if isinstance(item, dict):
                    drivers.append({"feature": item.get("feature"),
                                    "shap": item.get("shap") or item.get("value")})
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    drivers.append({"feature": item[0], "shap": item[1]})
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "signal_time": row["signal_time"].isoformat() if row.get("signal_time") else None,
        "direction": row.get("direction"),
        "tier": row.get("strength"),
        "confidence": _f(row.get("confidence")),
        "regime": row.get("regime"),
        "entry_price": _f(row.get("entry_price")),
        "top_drivers": drivers,
        "model_version": row.get("model_version"),
        "disclaimer": DISCLAIMER,
        "_source": "live",
    }


# ── Tool 2: order-flow snapshot ────────────────────────────────────────

def orderflow_snapshot() -> dict[str, Any]:
    if _seed_mode():
        return {
            "timestamp": "2026-07-07T00:00:00+00:00",
            "mid_price": 63118.0,
            "spread_bps": 0.8,
            "bid_depth_usd_l20": 4_820_000,
            "ask_depth_usd_l20": 3_110_000,
            "imbalance_l20": 0.216,   # bid-heavy
            "note": "positive imbalance = more resting bid liquidity (L20)",
            "disclaimer": DISCLAIMER,
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts_ms, mid_price, spread_bps, bid_depth_usd_l20, "
                "       ask_depth_usd_l20, imbalance_l20 "
                "FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol='BTC-USD' "
                "ORDER BY ts_ms DESC LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return {"error": "no orderbook snapshot yet", "disclaimer": DISCLAIMER}
    ts = row.get("ts_ms")
    return {
        "timestamp": (datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                      .isoformat() if ts else None),
        "mid_price": _f(row.get("mid_price")),
        "spread_bps": _f(row.get("spread_bps")),
        "bid_depth_usd_l20": _f(row.get("bid_depth_usd_l20")),
        "ask_depth_usd_l20": _f(row.get("ask_depth_usd_l20")),
        "imbalance_l20": _f(row.get("imbalance_l20")),
        "note": "positive imbalance = more resting bid liquidity (L20)",
        "disclaimer": DISCLAIMER,
        "_source": "live",
    }


# ── Tool 3: track record ───────────────────────────────────────────────

def track_record(window_days: Optional[int] = None) -> dict[str, Any]:
    if _seed_mode():
        return {
            "gate_a_signal_layer": {
                "n": 739, "win_rate_pct": 59.5,
                "ci95": [56.0, 63.2], "note": "Strong-tier tracked signals",
            },
            "trade_layer": {
                "n_closed": 34, "avg_net_bps": 6.2,
                "win_rates": {"gross_pct": 61.8, "net_pct": 52.9,
                              "equity_pct": 47.1},
            },
            "caveat": "signal accuracy != trading profit after costs/stops",
            "disclaimer": DISCLAIMER,
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            where = "WHERE strength='Strong' AND correct IS NOT NULL"
            params: list = []
            if window_days:
                where += " AND signal_time >= (NOW() - INTERVAL %s DAY)"
                params.append(int(window_days))
            cur.execute(
                f"SELECT COUNT(*) n, AVG(correct) wr FROM tracked_signals {where}",
                params)
            sig = cur.fetchone() or {}

            cur.execute(
                "SELECT COUNT(*) n, "
                "  AVG(CASE WHEN gross_pct>0 THEN 1 ELSE 0 END) gross_wr, "
                "  AVG(CASE WHEN net_pct>0 THEN 1 ELSE 0 END) net_wr, "
                "  AVG(net_pct) avg_net "
                "FROM v7_okx_positions WHERE status='CLOSED'")
            trd = cur.fetchone() or {}
    finally:
        conn.close()

    n = int(sig.get("n") or 0)
    wr = sig.get("wr")
    out: dict[str, Any] = {
        "gate_a_signal_layer": {
            "n": n,
            "win_rate_pct": round(float(wr) * 100, 1) if wr is not None else None,
            "note": "Strong-tier tracked signals with 4h outcome backfilled",
        },
        "caveat": "signal accuracy != trading profit after costs/stops",
        "disclaimer": DISCLAIMER,
        "_source": "live",
    }
    tn = int(trd.get("n") or 0)
    if tn:
        out["trade_layer"] = {
            "n_closed": tn,
            "avg_net_bps": round(float(trd["avg_net"]) * 10000, 1)
            if trd.get("avg_net") is not None else None,
            "win_rates": {
                "gross_pct": _pct(trd.get("gross_wr")),
                "net_pct": _pct(trd.get("net_wr")),
            },
        }
    return out


# ── Tool 4: risk frame (pure computation) ──────────────────────────────

# Edge profile constants (documented in CLAUDE.md §Leverage ladder).
_MU = 0.05          # expected per-trade edge
_SIGMA = 0.30       # per-trade vol
_LEV_CAP = 2.0      # absolute hard cap


def risk_frame(entry_price: float, direction: str,
               atr: Optional[float] = None) -> dict[str, Any]:
    """Pure maths on a hypothetical entry. Computes nothing about live
    positions and places no order. atr defaults to the latest recorded
    ATR when not supplied."""
    d = 1 if direction.upper() in ("UP", "LONG", "BUY") else -1
    if atr is None:
        atr = _latest_atr()
    stop = None
    if atr:
        stop = entry_price - d * 3.0 * atr    # 3xATR trailing-stop anchor

    kelly = _MU / (_SIGMA ** 2)               # ~0.56x
    vol_drag_2x = 0.5 * (_SIGMA ** 2) * (2.0 ** 2)   # E[r]-0.5 σ²L²

    return {
        "entry_price": entry_price,
        "direction": "LONG" if d == 1 else "SHORT",
        "atr_used": atr,
        "atr_stop_price": round(stop, 2) if stop is not None else None,
        "kelly_fraction": round(kelly, 3),
        "leverage_hard_cap": _LEV_CAP,
        "vol_drag_at_2x_pct": round(vol_drag_2x * 100, 1),
        "rationale": ("Kelly-optimal ~0.56x; hard cap 2.0x from vol-drag "
                      "math (E[r]-0.5·σ²·L²), not sentiment"),
        "note": "analytical framing only — this is not an order",
        "disclaimer": DISCLAIMER,
        "_source": "seed" if _seed_mode() else "live",
    }


def _latest_atr() -> Optional[float]:
    if _seed_mode():
        return 410.0
    conn = _conn()
    try:
        with conn.cursor() as cur:
            # indicator_history stores per-bar computed fields incl. atr
            cur.execute(
                "SELECT atr FROM indicator_history "
                "WHERE atr IS NOT NULL ORDER BY dt DESC LIMIT 1")
            row = cur.fetchone()
    except Exception:
        return None
    finally:
        conn.close()
    return _f(row.get("atr")) if row else None


# ── helpers ────────────────────────────────────────────────────────────

def _f(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _pct(x) -> Optional[float]:
    return round(float(x) * 100, 1) if x is not None else None
