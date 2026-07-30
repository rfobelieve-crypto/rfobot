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

import hashlib
import hmac
import json
import os
import re
import secrets
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


# ── Tool 3b: public track record (no $ amounts, adds win-rate CI) ──────
#
# Reshapes the same two queries as track_record() for an anonymous public
# website rather than an MCP client the operator personally configured —
# different threat model, narrower payload: no avg_net_bps (the site's own
# rule is rigor over returns, see product-site README/Stats.tsx), no $
# amounts anywhere, no driver/feature names. Adds a Wilson-score win-rate
# CI, which track_record() doesn't compute in live mode.

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    """Wilson score interval for a binomial proportion, in percentage
    points. Same formula used throughout research/*.py (e.g.
    research/strong_signal_perf.py) — reimplemented standalone here so
    the agent imports nothing from research/ for this path."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, center - half) * 100, 1),
            round(min(1.0, center + half) * 100, 1))


def _mdd_pct_from_trades() -> Optional[float]:
    """MDD from a synthetic equity curve built by compounding CLOSED-trade
    net_pct in order — deliberately NOT the raw account-balance snapshot
    curve (v7_okx_balance_snapshots / report.py's _get_equity_curve_stats).
    That curve is contaminated by operator deposits/withdrawals, which are
    indistinguishable from trading losses in a raw balance series — see
    mistake.md 2026-07-13 (a fund transfer alone tripped a -30% "loss"
    kill switch). A synthetic curve built purely from trade returns is
    immune to that: it only ever reflects what the strategy did."""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT net_pct FROM v7_okx_positions "
                "WHERE status='CLOSED' AND net_pct IS NOT NULL "
                "ORDER BY exit_time")
            rets = [float(r["net_pct"]) for r in cur.fetchall()]
    except Exception:
        return None
    finally:
        conn.close()
    if len(rets) < 2:
        return None
    equity, peak, mdd = 1.0, 1.0, 0.0
    for r in rets:
        equity *= (1 + r)
        peak = max(peak, equity)
        mdd = min(mdd, (equity - peak) / peak * 100)
    return round(mdd, 1)


def public_track_record() -> dict[str, Any]:
    if _seed_mode():
        return {
            "signal_layer": {"n": 739, "win_rate_pct": 59.5, "ci95": [56.0, 63.2],
                             "note": "Strong-tier tracked signals, 4h outcome"},
            "trade_layer": {"n_closed": 34, "win_rate_pct": 52.9,
                            "ci95": [36.4, 68.7],
                            "note": "closed live OKX trades, net of costs"},
            "mdd_pct": -8.2,
            "caveat": "signal accuracy != trading profit after costs/stops",
            "disclaimer": DISCLAIMER,
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) n, SUM(correct) wins FROM tracked_signals "
                "WHERE strength='Strong' AND correct IS NOT NULL")
            sig = cur.fetchone() or {}
            cur.execute(
                "SELECT COUNT(*) n, "
                "  SUM(CASE WHEN net_pct>0 THEN 1 ELSE 0 END) wins "
                "FROM v7_okx_positions WHERE status='CLOSED'")
            trd = cur.fetchone() or {}
    finally:
        conn.close()

    sig_n = int(sig.get("n") or 0)
    sig_wins = int(sig.get("wins") or 0)
    trd_n = int(trd.get("n") or 0)
    trd_wins = int(trd.get("wins") or 0)
    sig_ci = _wilson_ci(sig_wins, sig_n) if sig_n else None
    trd_ci = _wilson_ci(trd_wins, trd_n) if trd_n else None

    return {
        "signal_layer": {
            "n": sig_n,
            "win_rate_pct": round(sig_wins / sig_n * 100, 1) if sig_n else None,
            "ci95": list(sig_ci) if sig_ci else None,
            "note": "Strong-tier tracked signals, 4h outcome",
        },
        "trade_layer": {
            "n_closed": trd_n,
            "win_rate_pct": round(trd_wins / trd_n * 100, 1) if trd_n else None,
            "ci95": list(trd_ci) if trd_ci else None,
            "note": "closed live OKX trades, net of costs",
        },
        "mdd_pct": _mdd_pct_from_trades(),
        "caveat": "signal accuracy != trading profit after costs/stops",
        "disclaimer": DISCLAIMER,
        "_source": "live",
    }


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


# ── Tool 5: cancel-flow window analysis ────────────────────────────────

def cancel_flow_analysis(minutes: int = 90, t_from: Optional[str] = None,
                         t_to: Optional[str] = None,
                         include_perp: bool = True) -> dict[str, Any]:
    """Read-only forensic window analysis of cancellation flow.

    Delegates to research.cancel_flow_analyze.analyze_window (pure SELECTs
    over depth_deltas_1m / flow_bars_1m / orderbook_snapshots_1m /
    cancel_playbook_events + pure computation — no trading-system import,
    see agent-boundary.md).
    """
    if _seed_mode():
        return {
            "meta": {"from_tpe": "07-16 20:30", "to_tpe": "07-16 20:50",
                     "n_min": 21, "lag_min": 0.0, "def_version": "v1-2026-07-16"},
            "verdict": {"gate_minutes": 3, "max_shock": 4.3, "lean": "偏多",
                        "note": "3 分鐘過鬧鐘門檻"},
            "right_edge": {"px": 63910.0, "shock": 3.1, "skew15": 0.06,
                           "net15": -0.27, "vshock": 5.6},
            "events": [{"t": "07-16 20:46", "playbook": "真破",
                        "direction": "UP", "px": 63910.0, "shock": 3.1,
                        "vshock": 5.6, "fwd_ret_60m": 0.0049,
                        "hit_60m": 1, "alerted": 0}],
            "minutes": [], "gate_minutes": [], "taker_bursts": [],
            "tilt": {}, "disclaimer": DISCLAIMER, "_source": "seed",
        }
    import time as _time

    import pandas as _pd

    # Lazy import: research/ is a namespace package at the repo root; the
    # MCP server runs with cwd=repo root (see server docstring config).
    from research.cancel_flow_analyze import analyze_window

    tpe = _pd.Timedelta(hours=8)
    if t_from and t_to:
        t0 = int((_pd.Timestamp(t_from) - tpe).timestamp() * 1000)
        t1 = int((_pd.Timestamp(t_to) - tpe).timestamp() * 1000)
    else:
        t1 = int(_time.time() * 1000)
        t0 = t1 - max(10, min(int(minutes), 24 * 60)) * 60_000
    res = analyze_window(t0, t1, perp=include_perp)
    res["disclaimer"] = DISCLAIMER + " " + str(res.get("disclaimer", ""))
    res["_source"] = "live"
    return res


# ── Cancel-flow multicoin stats (2026-07-24, KPI dashboard on product-site) ─
#
# Lightweight per-coin summary over depth_deltas_1m (whitelisted table,
# agent-boundary.md) — NOT the forensic analyze_window() used by
# cancel_flow_analysis above. This just answers "how much add/cancel
# activity has each coin seen recently, and what's the cancel share" for a
# dashboard stat card. Deliberately NOT a signal: no playbook/gate/lean
# logic, no claim of predictive value — that's still pending the 2026-08-10
# pre-registered verdict (F1/F2). Spot+perp combined per coin (a KPI card
# reads "ETH cancel pressure", not two separate market rows); the raw
# per-exchange split stays in `by_exchange` for anyone who wants it.

def public_cancel_flow_stats(window_minutes: int = 120) -> dict[str, Any]:
    window_minutes = max(5, min(int(window_minutes), 1440))
    if _seed_mode():
        return {
            "window_minutes": 120,
            "coins": [
                {"symbol": "BTC-USD", "n_minutes": 20800, "cancel_ratio_pct": 54.2,
                 "ask_bid_skew_pct": 3.1, "last_updated": _now_iso(),
                 "by_exchange": {"binance": {"n_minutes": 8900, "cancel_ratio_pct": 53.8},
                                 "binance_perp": {"n_minutes": 11900, "cancel_ratio_pct": 54.6}}},
                {"symbol": "ETH-USD", "n_minutes": 400, "cancel_ratio_pct": 51.7,
                 "ask_bid_skew_pct": -1.4, "last_updated": _now_iso(),
                 "by_exchange": {"binance": {"n_minutes": 200, "cancel_ratio_pct": 50.9},
                                 "binance_perp": {"n_minutes": 200, "cancel_ratio_pct": 52.5}}},
            ],
            "disclaimer": DISCLAIMER + " Raw order-book activity monitor, not a "
                          "trading signal — cancel-flow edge is unverified "
                          "pending the 2026-08-10 pre-registered test.",
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT exchange, canonical_symbol, "
                "       COUNT(*) AS n_minutes, "
                "       MAX(minute_start_ms) AS last_min, "
                "       SUM(bid_add_qty) AS bid_add, SUM(bid_cancel_qty) AS bid_cancel, "
                "       SUM(ask_add_qty) AS ask_add, SUM(ask_cancel_qty) AS ask_cancel "
                "FROM depth_deltas_1m "
                "WHERE minute_start_ms >= "
                "      (UNIX_TIMESTAMP(UTC_TIMESTAMP()) * 1000 - %s * 60000) "
                "GROUP BY exchange, canonical_symbol",
                (window_minutes,))
            window_rows = cur.fetchall()

            cur.execute(
                "SELECT exchange, canonical_symbol, COUNT(*) AS n_minutes_total "
                "FROM depth_deltas_1m GROUP BY exchange, canonical_symbol")
            total_rows = cur.fetchall()
    finally:
        conn.close()

    totals_by_key = {(r["exchange"], r["canonical_symbol"]): int(r["n_minutes_total"])
                     for r in total_rows}

    rows_by_symbol: dict[str, list] = {}
    for r in window_rows:
        rows_by_symbol.setdefault(r["canonical_symbol"], []).append(r)

    coins = []
    for sym, rows in rows_by_symbol.items():
        by_exchange = {}
        bid_add_c = bid_cancel_c = ask_add_c = ask_cancel_c = 0.0
        last_min_ms = 0
        for r in rows:
            bid_add = _f(r["bid_add"]) or 0.0
            bid_cancel = _f(r["bid_cancel"]) or 0.0
            ask_add = _f(r["ask_add"]) or 0.0
            ask_cancel = _f(r["ask_cancel"]) or 0.0
            bid_add_c += bid_add
            bid_cancel_c += bid_cancel
            ask_add_c += ask_add
            ask_cancel_c += ask_cancel
            row_denom = bid_add + bid_cancel + ask_add + ask_cancel
            n_total = totals_by_key.get((r["exchange"], sym), int(r["n_minutes"]))
            by_exchange[r["exchange"]] = {
                "n_minutes": n_total,
                "cancel_ratio_pct": (_pct((bid_cancel + ask_cancel) / row_denom)
                                     if row_denom else None),
            }
            last_min_ms = max(last_min_ms, int(r["last_min"] or 0))

        add_total = bid_add_c + ask_add_c
        cancel_total = bid_cancel_c + ask_cancel_c
        denom = add_total + cancel_total
        cancel_ratio = _pct(cancel_total / denom) if denom else None
        skew_denom = bid_cancel_c + ask_cancel_c
        ask_bid_skew = (_pct((ask_cancel_c - bid_cancel_c) / skew_denom)
                        if skew_denom else None)

        coins.append({
            "symbol": sym,
            "n_minutes": sum(v["n_minutes"] for v in by_exchange.values()),
            "cancel_ratio_pct": cancel_ratio,
            "ask_bid_skew_pct": ask_bid_skew,
            "last_updated": (datetime.fromtimestamp(
                last_min_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")
                             if last_min_ms else None),
            "by_exchange": by_exchange,
        })

    coins.sort(key=lambda c: (-(c["n_minutes"] or 0), c["symbol"]))

    return {
        "window_minutes": window_minutes,
        "coins": coins,
        "disclaimer": DISCLAIMER + " Raw order-book activity monitor, not a "
                      "trading signal — cancel-flow edge is unverified "
                      "pending the 2026-08-10 pre-registered test.",
        "_source": "live",
    }


# ── helpers ────────────────────────────────────────────────────────────

def _f(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _pct(x) -> Optional[float]:
    return round(float(x) * 100, 1) if x is not None else None


# ── Agent verdict cohort（三方判讀對照 LLM 端, 2026-07-18） ───────────────
# agent 只寫自己的 agent_* 命名空間（agent-boundary.md 明文允許）;
# 讀取僅 orderbook_snapshots_1m mid（允許表）。前瞻紀錄, 落表後不可改。

_VERDICT_DDL = """
CREATE TABLE IF NOT EXISTS agent_verdicts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_ms BIGINT NOT NULL,
    direction VARCHAR(8) NOT NULL,
    basis VARCHAR(500) NULL,
    event_source VARCHAR(12) NULL,
    event_id BIGINT NULL,
    base_mid DOUBLE NULL,
    fwd_ret_60m DOUBLE NULL,
    fwd_ret_120m DOUBLE NULL,
    hit_60m TINYINT NULL,
    outcome_done TINYINT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_out (outcome_done, event_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""


def _mid_at_minute(cur, minute: int) -> Optional[float]:
    cur.execute(
        "SELECT mid_price FROM orderbook_snapshots_1m "
        "WHERE canonical_symbol='BTC-USD' AND ts_ms >= %s AND ts_ms < %s "
        "ORDER BY ts_ms DESC LIMIT 1",
        (minute * 60_000, (minute + 1) * 60_000))
    r = cur.fetchone()
    return float(r["mid_price"]) if r else None


def log_agent_verdict(direction: str, basis: str = "",
                      event_source: Optional[str] = None,
                      event_id: Optional[int] = None) -> dict:
    import time as _t
    from shared.db import get_db_conn
    direction = (direction or "").strip().upper()
    if direction not in ("UP", "DOWN", "UNSURE"):
        return {"error": "direction must be UP / DOWN / UNSURE"}
    if event_source not in (None, "", "tv", "pb"):
        return {"error": "event_source must be 'tv' or 'pb' (or omitted)"}
    event_ms = (int(_t.time() * 1000) // 60_000) * 60_000
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_VERDICT_DDL)
            base = _mid_at_minute(cur, event_ms // 60_000)
            cur.execute(
                "INSERT INTO agent_verdicts (event_ms, direction, basis, "
                "event_source, event_id, base_mid) VALUES (%s,%s,%s,%s,%s,%s)",
                (event_ms, direction, (basis or "")[:500],
                 event_source or None, event_id, base))
            vid = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return {
        "verdict_id": int(vid),
        "direction": direction,
        "minute_utc": datetime.fromtimestamp(
            event_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "base_mid": base,
        "scoring": "fwd mid return at 60/120m, lazily backfilled; "
                   "UNSURE 不計入命中率",
        "immutable": "前瞻紀錄, 落表後不可改",
        "disclaimer": DISCLAIMER,
    }


def agent_verdict_stats() -> dict:
    import time as _t
    from shared.db import get_db_conn
    now_ms = int(_t.time() * 1000)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_VERDICT_DDL)
            cur.execute("SELECT id, event_ms, direction, base_mid "
                        "FROM agent_verdicts WHERE outcome_done=0 "
                        "AND event_ms <= %s", (now_ms - 121 * 60_000,))
            for r in cur.fetchall():
                m0 = int(r["event_ms"]) // 60_000
                base = (float(r["base_mid"]) if r["base_mid"]
                        else _mid_at_minute(cur, m0))
                sets, vals = ["outcome_done=1"], []
                if base:
                    for h in (60, 120):
                        mid = _mid_at_minute(cur, m0 + h)
                        if mid:
                            fwd = mid / base - 1
                            sets.append(f"fwd_ret_{h}m=%s")
                            vals.append(fwd)
                            if h == 60 and r["direction"] in ("UP", "DOWN"):
                                sets.append("hit_60m=%s")
                                vals.append(1 if (fwd > 0) ==
                                            (r["direction"] == "UP") else 0)
                vals.append(int(r["id"]))
                cur.execute("UPDATE agent_verdicts SET " + ", ".join(sets)
                            + " WHERE id=%s", vals)
            conn.commit()
            cur.execute(
                "SELECT direction, COUNT(*) n, AVG(hit_60m) hr60, "
                "AVG(fwd_ret_60m) r60, AVG(fwd_ret_120m) r120 "
                "FROM agent_verdicts WHERE outcome_done=1 "
                "GROUP BY direction")
            scored = {r["direction"]: {
                "n": int(r["n"]),
                "hit_60m": _pct(_f(r["hr60"])),
                "avg_fwd_60m_pct": _pct(_f(r["r60"])),
                "avg_fwd_120m_pct": _pct(_f(r["r120"])),
            } for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) n FROM agent_verdicts "
                        "WHERE outcome_done=0")
            pending = int(cur.fetchone()["n"])
            cur.execute(
                "SELECT id, event_ms, direction, basis, hit_60m, fwd_ret_60m "
                "FROM agent_verdicts ORDER BY id DESC LIMIT 5")
            recent = [{
                "id": int(r["id"]),
                "minute_utc": datetime.fromtimestamp(
                    int(r["event_ms"]) / 1000,
                    tz=timezone.utc).strftime("%m-%d %H:%M"),
                "direction": r["direction"],
                "basis": (r["basis"] or "")[:120],
                "hit_60m": (None if r["hit_60m"] is None
                            else int(r["hit_60m"])),
                "fwd_ret_60m_pct": _pct(_f(r["fwd_ret_60m"])),
            } for r in cur.fetchall()]
    finally:
        conn.close()
    return {
        "cohort": "LLM (this assistant) — 三方判讀對照之第三軌",
        "scored_by_direction": scored,
        "pending": pending,
        "recent": recent,
        "comparison_note": "machine=cancel_playbook_events, human="
                           "cancel_eyeball_log; 三軌對照統計待樣本累積後"
                           "以 repo research script 執行",
        "disclaimer": DISCLAIMER,
    }


# ── Public waitlist (product website, 2026-07-21) ──────────────────────
# Writes ONLY to the agent's own namespace (agent_* — explicitly allowed
# by agent-boundary.md), never to a quant table. Scope is "notify me about
# write-ups / methodology updates", not "give me access to signals" — the
# site's own stance is showcase, not a paid product (see mistake.md
# 2026-07-10 OKX ASP hackathon PARK decision on not selling edge access).

_WAITLIST_DDL = """
CREATE TABLE IF NOT EXISTS agent_waitlist_signups (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(320) NOT NULL,
    note VARCHAR(500) NULL,
    source VARCHAR(32) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def submit_waitlist(email: str, note: str = "",
                    source: Optional[str] = None) -> dict[str, Any]:
    """Insert one waitlist signup. Validates shape only (no verification
    email, no dedup beyond what a human reviewing the table would do by
    eye) — this is a notify-me list, not an auth system."""
    email = (email or "").strip().lower()
    if not email or len(email) > 320 or not _EMAIL_RE.match(email):
        return {"ok": False, "error": "invalid email"}
    if _seed_mode():
        return {"ok": True, "disclaimer": DISCLAIMER, "_source": "seed"}
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_WAITLIST_DDL)
            cur.execute(
                "INSERT INTO agent_waitlist_signups (email, note, source) "
                "VALUES (%s, %s, %s)",
                (email, (note or "")[:500], (source or "")[:32]))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "disclaimer": DISCLAIMER}


# ── Public signal history (behind Google login, product website) ───────
#
# The auth gate lives entirely in Next.js (see product-site/auth.ts) — it
# is a UX distinction ("more detail for people who sign in"), not a
# security boundary, so this function keeps the SAME agent-boundary.md
# discipline as every other public_* function: no shap_top, no
# model_version, no feature names. The only thing that's actually new
# versus latest_signal() is (a) history instead of one row and (b) the
# realized outcome (correct) once backfilled.

def public_signal_history(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    if _seed_mode():
        return {
            "signals": [
                {"signal_time": "2026-07-20T12:00:00", "direction": "UP",
                 "tier": "Strong", "confidence": 82.1, "regime": "TRENDING_BULL",
                 "correct": 1},
                {"signal_time": "2026-07-20T08:00:00", "direction": "DOWN",
                 "tier": "Moderate", "confidence": 68.4, "regime": "CHOPPY",
                 "correct": 0},
            ],
            "disclaimer": DISCLAIMER,
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, strength, confidence, "
                "       regime, correct "
                "FROM tracked_signals ORDER BY signal_time DESC LIMIT %s",
                (limit,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        "signals": [{
            "signal_time": r["signal_time"].isoformat() if r.get("signal_time") else None,
            "direction": r.get("direction"),
            "tier": r.get("strength"),
            "confidence": _f(r.get("confidence")),
            "regime": r.get("regime"),
            "correct": (None if r.get("correct") is None else int(r["correct"])),
        } for r in rows],
        "disclaimer": DISCLAIMER,
        "_source": "live",
    }


# ── Self-hosted accounts (product website, 2026-07-22) ──────────────────
#
# Replaces the earlier Google-OAuth plan — own email/password accounts,
# own table. Same boundary as agent_waitlist_signups: writes ONLY into
# the agent's own agent_* namespace, never a quant table. Passwords are
# PBKDF2-HMAC-SHA256, random salt per user, stdlib-only (no new
# dependency) — never stored or logged in plaintext.

_USER_DDL = """
CREATE TABLE IF NOT EXISTS agent_user_accounts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(320) NOT NULL UNIQUE,
    password_hash VARCHAR(200) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

_PBKDF2_ITERATIONS = 260_000


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, hash_hex = stored.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return hmac.compare_digest(dk.hex(), hash_hex)


def register_user(email: str, password: str) -> dict[str, Any]:
    email = (email or "").strip().lower()
    if not email or len(email) > 320 or not _EMAIL_RE.match(email):
        return {"ok": False, "error": "invalid email"}
    if not password or len(password) < 8:
        return {"ok": False, "error": "password must be at least 8 characters"}
    if _seed_mode():
        return {"ok": True, "_source": "seed"}
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_USER_DDL)
            cur.execute("SELECT id FROM agent_user_accounts WHERE email=%s", (email,))
            if cur.fetchone():
                return {"ok": False, "error": "email already registered"}
            cur.execute(
                "INSERT INTO agent_user_accounts (email, password_hash) "
                "VALUES (%s, %s)", (email, _hash_password(password)))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


def verify_user(email: str, password: str) -> dict[str, Any]:
    email = (email or "").strip().lower()
    if not email or not password:
        return {"ok": False}
    if _seed_mode():
        return ({"ok": True, "email": email} if email == "demo@example.com"
                else {"ok": False})
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_USER_DDL)
            cur.execute(
                "SELECT password_hash FROM agent_user_accounts WHERE email=%s",
                (email,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not _verify_password(password, row["password_hash"]):
        return {"ok": False}
    return {"ok": True, "email": email}


# ── Public: live execution status (site dashboard, 2026-07-30) ──────────
# Percentages, directions and timing ONLY — no contract sizes, no dollar
# equity. The public layer shows how the strategy behaves; account scale
# stays private. Reads only the whitelisted v7_okx_positions table.

def public_live_status() -> dict[str, Any]:
    if _seed_mode():
        return {
            "open_position": {"direction": "SHORT", "tier": "Strong",
                              "entry_price": 64100.0, "held_hours": 3.0},
            "recent": [
                {"entry_utc": "07-28 09:00", "direction": "SHORT",
                 "net_pct": 2.15, "exit_reason": "opp_signal"},
                {"entry_utc": "07-25 17:00", "direction": "LONG",
                 "net_pct": -0.68, "exit_reason": "trail_stop"},
            ],
            "totals": {"n_closed": 34, "win_rate_pct": 52.9,
                       "cum_net_pct": 4.1},
            "disclaimer": DISCLAIMER,
            "_source": "seed",
        }
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT direction, entry_tier, entry_time, entry_price "
                "FROM v7_okx_positions WHERE status='OPEN' "
                "ORDER BY id DESC LIMIT 1")
            op = cur.fetchone()
            cur.execute(
                "SELECT entry_time, direction, net_pct, exit_reason "
                "FROM v7_okx_positions WHERE status='CLOSED' "
                "ORDER BY exit_time DESC LIMIT 6")
            recent = cur.fetchall() or []
            cur.execute(
                "SELECT net_pct FROM v7_okx_positions "
                "WHERE status='CLOSED' AND net_pct IS NOT NULL "
                "ORDER BY exit_time")
            allpct = [float(r["net_pct"]) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    comp = 1.0
    for p in allpct:
        comp *= 1.0 + p / 100.0
    open_position = None
    if op:
        held_h = None
        if op.get("entry_time"):
            held_h = round(
                (datetime.utcnow() - op["entry_time"]).total_seconds() / 3600, 1)
        open_position = {
            "direction": op.get("direction"),
            "tier": op.get("entry_tier"),
            "entry_price": (float(op["entry_price"])
                            if op.get("entry_price") is not None else None),
            "held_hours": held_h,
        }
    n = len(allpct)
    wins = sum(1 for p in allpct if p > 0)
    return {
        "open_position": open_position,
        "recent": [{
            "entry_utc": (r["entry_time"].strftime("%m-%d %H:%M")
                          if r.get("entry_time") else None),
            "direction": r.get("direction"),
            "net_pct": (round(float(r["net_pct"]), 2)
                        if r.get("net_pct") is not None else None),
            "exit_reason": r.get("exit_reason"),
        } for r in recent],
        "totals": {"n_closed": n,
                   "win_rate_pct": round(100 * wins / n, 1) if n else None,
                   "cum_net_pct": round((comp - 1.0) * 100, 2)},
        "disclaimer": DISCLAIMER,
    }
