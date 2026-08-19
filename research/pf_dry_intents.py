"""M3 — SF dry-run intent flow: fresh variant-B fills -> risk engine ->
pf_intents, decisions and rejection reasons all on the ledger (2026-08-18).

Every hour, freshly filled variant-B shadow signals become Intents and go
through `risk_engine.decide()` against the REAL account state: live equity
from balance snapshots, live V7 open positions, and — honestly — the
account-layer halt (CAP-2 is firing hourly while the capital baseline
stays unresolved; those intents get rejected with `account_halted`, which
is the ledger telling the truth: resolving the baseline is on the go-live
critical path).  No orders anywhere; the execution layer does not exist
yet.  What this proves for three weeks is the full decision chain the live
path will use, plus an auditable record of what risk control would have
blocked and why.

Within a batch, approved intents occupy virtual slots so the concurrency
cap (sweep: 5) and same-symbol collision rules are exercised for real.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicator.portfolio.ledger import (  # noqa: E402
    ALL_DDL, Intent, OpenPosition, PortfolioState)
from indicator.portfolio.limits import default_limits  # noqa: E402
from indicator.portfolio.risk_engine import decide  # noqa: E402

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
FRESH_WINDOW_S = 2 * 3600       # engine runs hourly; look back 2h for safety
DIS = 3.5                        # frozen disaster-stop mult (sweep_core)


def fresh_candidates(now_s: int) -> list[dict]:
    if not LOG.exists():
        return []
    with LOG.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            fill = int(r["fill_ts"])
        except (KeyError, ValueError):
            continue
        if (now_s - fill <= FRESH_WINDOW_S
                and r.get("variant_b") == "1"
                and r.get("side") in ("LONG", "SHORT")):
            out.append(r)
    return out


def build_state(cur) -> PortfolioState:
    cur.execute("SELECT total_eq_usd e FROM v7_okx_balance_snapshots "
                "ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    equity = float(row["e"]) if row else 0.0

    open_pos: list[OpenPosition] = []
    cur.execute("SELECT direction, notional_usd FROM v7_okx_positions "
                "WHERE status = 'OPEN'")
    for p in cur.fetchall():
        open_pos.append(OpenPosition(
            strategy="v7", symbol="BTC-USD", side=p["direction"],
            risk_pct=0.25,
            notional_mult=(float(p["notional_usd"]) / equity
                           if equity > 0 else 0.0)))

    # Account layer: any unresolved CAP trigger inside the last 2h = halted.
    # CAP-2 currently fires hourly; reflecting it is the point, not a bug.
    cur.execute("SELECT COUNT(*) n FROM v7_okx_kill_log "
                "WHERE resolved_at IS NULL "
                "AND ts >= DATE_SUB(NOW(), INTERVAL 2 HOUR)")
    halted = int(cur.fetchone()["n"]) > 0

    # 2026-08-19: equity itself is a halt condition.  The operator withdrew
    # the account to ~$0 on 08-18 14:00 and the kill log went quiet (nothing
    # left to trip a cap on), so the 2h kill-log window alone reported
    # halted=False against a $0 account — a dry run must never size trades
    # off an equity that cannot fund them.  Also covers a stale-snapshot
    # feed: no fresh equity means no decision, not a decision on stale data.
    if equity < 50.0:
        halted = True
    cur.execute("SELECT MAX(ts) m FROM v7_okx_balance_snapshots")
    last_snap = cur.fetchone()["m"]
    if last_snap is not None:
        cur.execute("SELECT TIMESTAMPDIFF(HOUR, %s, UTC_TIMESTAMP()) h",
                    (last_snap,))
        if int(cur.fetchone()["h"] or 0) >= 3:
            halted = True

    # Sweep's day-R from shadow rows closed today (UTC) — the strategy-layer
    # daily cap sees what the strategy actually did today, shadow or not.
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_r = 0.0
    if LOG.exists():
        with LOG.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("status") == "CLOSED" and r.get("net_r")
                        and r.get("exit_utc", "").startswith(day)
                        and r.get("variant_b") == "1"):
                    try:
                        day_r += float(r["net_r"])
                    except ValueError:
                        pass
    return PortfolioState(
        equity_usd=equity, open_positions=open_pos,
        strategy_day_r={"sweep": day_r}, account_halted=halted)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0,
                    help="lookback window for fresh fills (default 2)")
    args = ap.parse_args()
    global FRESH_WINDOW_S
    FRESH_WINDOW_S = int(args.hours * 3600)

    from shared.db import get_db_conn
    now_s = int(datetime.now(timezone.utc).timestamp())
    cands = fresh_candidates(now_s)
    limits = default_limits()
    risk_pct = limits.get("sweep").risk_pct_per_trade

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for ddl in ALL_DDL:
                cur.execute(ddl)
            state = build_state(cur)
            n_app = n_rej = n_dup = 0
            for r in cands:
                sym = f"{r['symbol']}-USD"
                fill_dt = datetime.fromtimestamp(
                    int(r["fill_ts"]), timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S")
                cur.execute(
                    "SELECT 1 FROM pf_intents WHERE strategy='sweep' "
                    "AND symbol=%s AND ts=%s LIMIT 1", (sym, fill_dt))
                if cur.fetchone():
                    n_dup += 1
                    continue
                entry = float(r["entry_px"])
                atr = float(r["atr"])
                sgn = 1 if r["side"] == "LONG" else -1
                stop = entry - sgn * DIS * atr
                intent = Intent(strategy="sweep", symbol=sym, side=r["side"],
                                risk_pct=risk_pct, stop_px=stop,
                                entry_ref_px=entry, ttl_sec=3600)
                d = decide(intent, state, limits)
                if d.rejected:
                    n_rej += 1
                    status, reason = "rejected", d.reason
                else:
                    n_app += 1
                    status, reason = "approved", None
                    stop_frac = DIS * atr / entry
                    state.open_positions.append(OpenPosition(
                        strategy="sweep", symbol=sym, side=r["side"],
                        risk_pct=risk_pct,
                        notional_mult=(risk_pct / 100.0) / stop_frac
                        if stop_frac > 0 else 0.0))
                cur.execute(
                    "INSERT INTO pf_intents (ts, strategy, symbol, side, "
                    "risk_pct, stop_px, entry_ref_px, ttl_sec, status, "
                    "approved_risk_pct, reject_reason, decided_ts) "
                    "VALUES (%s,'sweep',%s,%s,%s,%s,%s,3600,%s,%s,%s,"
                    "UTC_TIMESTAMP())",
                    (fill_dt, sym, r["side"], risk_pct, stop, entry, status,
                     risk_pct if status == "approved" else None, reason))
        conn.commit()
        print(f"pf_dry_intents: {len(cands)} fresh, approved={n_app} "
              f"rejected={n_rej} dup={n_dup} "
              f"(halted={state.account_halted}, equity={state.equity_usd:.0f})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
