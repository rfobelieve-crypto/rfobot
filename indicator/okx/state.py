"""OkxStateStore — DB layer for v7_okx_* tables.

P2: exchange is source of truth; this store is a local mirror that we
flag for mismatch but never auto-mutate from OKX events without explicit
caller intent.

P4: dead-state preferable to dirty-state.  Every write is transactional;
read-modify-write uses the shared lock.

Tables (see migrations/012_v7_okx_tables.sql):
  - v7_okx_positions
  - v7_okx_orders
  - v7_okx_kill_log
  - v7_okx_reconciliation_log
  - v7_okx_executor_status
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Optional

from shared.db import get_db_conn

logger = logging.getLogger(__name__)


class OkxStateStore:
    """Thin DB wrapper.  No logic, just CRUD.

    Threading note: SQL execution is intrinsically protected by per-conn
    serialization, but we add an RLock so multi-op transactions are atomic
    at the application level.
    """

    def __init__(self, table_prefix: str = "v7_okx"):
        self._prefix = table_prefix
        self._lock = threading.RLock()

    # ── Positions ──────────────────────────────────────────────────────

    def insert_open_position(self, *, entry_time: datetime, direction: str,
                             entry_tier: str, entry_price: float,
                             atr_at_entry: float, stop_dist: float,
                             current_stop: float,
                             size_contracts: int, size_frac: float,
                             notional_usd: float, equity_before: float,
                             entry_cl_ord_id: str,
                             stop_algo_cl_ord_id: str,
                             model_version: Optional[str] = None,
                             paused: bool = False) -> Optional[int]:
        """Insert a new OPEN position.  Returns inserted id or None on
        duplicate (cl_ord_id conflict)."""
        sql = (
            f"INSERT INTO `{self._prefix}_positions` "
            "(entry_time, direction, entry_tier, entry_price, atr_at_entry, "
            " stop_dist, trail_extreme, current_stop, size_contracts, "
            " size_frac, notional_usd, equity_before, model_version, "
            " paused_at_signal, entry_cl_ord_id, stop_algo_cl_ord_id, "
            " status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "        %s, %s, %s, 'OPEN')"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    try:
                        cur.execute(sql, (
                            entry_time, direction, entry_tier,
                            float(entry_price), float(atr_at_entry),
                            float(stop_dist), float(entry_price),  # trail_extreme starts at entry
                            float(current_stop), int(size_contracts),
                            float(size_frac), float(notional_usd),
                            float(equity_before), model_version,
                            1 if paused else 0, entry_cl_ord_id,
                            stop_algo_cl_ord_id,
                        ))
                        new_id = cur.lastrowid
                    except Exception as e:
                        if "duplicate" in str(e).lower():
                            logger.warning("insert_open_dup cl_ord_id=%s",
                                           entry_cl_ord_id)
                            return None
                        raise
                conn.commit()
                return new_id
            finally:
                conn.close()

    def get_open_position(self) -> Optional[dict]:
        sql = (
            f"SELECT * FROM `{self._prefix}_positions` "
            "WHERE status='OPEN' ORDER BY id DESC LIMIT 1"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
                return row
            finally:
                conn.close()

    def get_all_open_positions(self) -> list[dict]:
        sql = (
            f"SELECT * FROM `{self._prefix}_positions` "
            "WHERE status='OPEN' ORDER BY id"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return list(cur.fetchall())
            finally:
                conn.close()

    def update_trail(self, *, position_id: int, trail_extreme: float,
                     current_stop: float) -> None:
        sql = (
            f"UPDATE `{self._prefix}_positions` "
            "SET trail_extreme=%s, current_stop=%s "
            "WHERE id=%s AND status='OPEN'"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (float(trail_extreme),
                                      float(current_stop),
                                      int(position_id)))
                conn.commit()
            finally:
                conn.close()

    def set_position_okx_ids(self, *, position_id: int,
                             entry_ord_id: Optional[str] = None,
                             stop_algo_id: Optional[str] = None) -> None:
        """Update OKX-side IDs once we get them back from REST/WS."""
        updates, params = [], []
        if entry_ord_id is not None:
            updates.append("entry_ord_id=%s")
            params.append(entry_ord_id)
        if stop_algo_id is not None:
            updates.append("stop_algo_id=%s")
            params.append(stop_algo_id)
        if not updates:
            return
        params.append(int(position_id))
        sql = (
            f"UPDATE `{self._prefix}_positions` "
            f"SET {', '.join(updates)} WHERE id=%s"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                conn.commit()
            finally:
                conn.close()

    def close_position(self, *, position_id: int, exit_time: datetime,
                       exit_price: float, exit_reason: str,
                       gross_pct: float, net_pct: float,
                       equity_ret_pct: float, equity_after: float,
                       exit_fees_usd: float = 0.0,
                       new_status: str = "CLOSED") -> None:
        sql = (
            f"UPDATE `{self._prefix}_positions` "
            "SET status=%s, exit_time=%s, exit_price=%s, exit_reason=%s, "
            "    exit_fees_usd=%s, gross_pct=%s, net_pct=%s, "
            "    equity_ret_pct=%s, equity_after=%s "
            "WHERE id=%s AND status='OPEN'"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        new_status, exit_time, float(exit_price),
                        exit_reason, float(exit_fees_usd),
                        float(gross_pct), float(net_pct),
                        float(equity_ret_pct), float(equity_after),
                        int(position_id),
                    ))
                conn.commit()
            finally:
                conn.close()

    # ── Executor status ────────────────────────────────────────────────

    def save_executor_status(self, *, status: str,
                             reason: Optional[str] = None,
                             trigger_id: Optional[str] = None,
                             context: Optional[dict] = None) -> None:
        """Upsert single-row status table."""
        sql = (
            f"INSERT INTO `{self._prefix}_executor_status` "
            "(id, status, last_changed_at, reason, trigger_id, context) "
            "VALUES (1, %s, NOW(), %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "  status=VALUES(status), last_changed_at=NOW(), "
            "  reason=VALUES(reason), trigger_id=VALUES(trigger_id), "
            "  context=VALUES(context)"
        )
        ctx_json = json.dumps(context) if context else None
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (status, reason, trigger_id, ctx_json))
                conn.commit()
            finally:
                conn.close()

    def load_executor_status(self) -> dict:
        sql = f"SELECT * FROM `{self._prefix}_executor_status` WHERE id=1"
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
                return row or {"status": "INIT"}
            finally:
                conn.close()

    # ── Kill log ───────────────────────────────────────────────────────

    def resolve_open_kills(self, resolution: str = "auto_recovery") -> int:
        """Mark all unresolved kill_log entries as resolved.

        Called when executor transitions from HALTED back to ACTIVE so
        the M6 (kill recovery validation) milestone + dashboard reflect
        the recovery.  Returns number of rows updated.
        """
        sql = (
            f"UPDATE `{self._prefix}_kill_log` "
            "SET resolved_at=NOW(), resolution=%s "
            "WHERE resolved_at IS NULL"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (resolution,))
                    n = cur.rowcount
                conn.commit()
                return int(n)
            finally:
                conn.close()

    def log_kill_trigger(self, *, trigger_id: str, severity: str,
                         context: dict) -> None:
        sql = (
            f"INSERT INTO `{self._prefix}_kill_log` "
            "(ts, trigger_id, severity, context) VALUES (NOW(), %s, %s, %s)"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (trigger_id, severity,
                                      json.dumps(context)))
                conn.commit()
            finally:
                conn.close()

    def get_recent_triggers(self, *, days: int = 30) -> list[dict]:
        sql = (
            f"SELECT * FROM `{self._prefix}_kill_log` "
            "WHERE ts >= DATE_SUB(NOW(), INTERVAL %s DAY) "
            "ORDER BY ts DESC"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (int(days),))
                    return list(cur.fetchall())
            finally:
                conn.close()

    # ── Reconciliation log ─────────────────────────────────────────────

    def log_reconciliation(self, *, verdict: str,
                           detail: Optional[dict] = None) -> None:
        sql = (
            f"INSERT INTO `{self._prefix}_reconciliation_log` "
            "(ts, result, detail) VALUES (NOW(), %s, %s)"
        )
        detail_json = json.dumps(detail) if detail else None
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (verdict, detail_json))
                conn.commit()
            finally:
                conn.close()

    # ── Balance snapshots ──────────────────────────────────────────────

    def insert_balance_snapshot(self, *, total_eq_usd: float,
                                 available_usd: float,
                                 source: str = "ws") -> None:
        """Persist a balance reading.  source in {'ws','rest','start'}."""
        sql = (
            f"INSERT INTO `{self._prefix}_balance_snapshots` "
            "(ts, total_eq_usd, available_usd, source) "
            "VALUES (NOW(), %s, %s, %s)"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (float(total_eq_usd),
                                      float(available_usd), source))
                conn.commit()
            finally:
                conn.close()

    def get_latest_balance(self) -> Optional[dict]:
        sql = (
            f"SELECT * FROM `{self._prefix}_balance_snapshots` "
            "ORDER BY ts DESC LIMIT 1"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return cur.fetchone()
            finally:
                conn.close()

    def get_day_start_equity(self) -> Optional[float]:
        """Latest snapshot from BEFORE today UTC midnight.

        Used as the anchor for daily_loss_cap.  Returns None if no
        pre-today snapshot exists (e.g. first day of operation).
        """
        sql = (
            f"SELECT total_eq_usd FROM `{self._prefix}_balance_snapshots` "
            "WHERE ts < UTC_DATE() ORDER BY ts DESC LIMIT 1"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
                if row and row.get("total_eq_usd") is not None:
                    return float(row["total_eq_usd"])
                return None
            finally:
                conn.close()

    # ── Approvals (manual-approval gate) ───────────────────────────────

    def insert_pending_approval(self, *, intent_json: str,
                                 expires_at: datetime) -> int:
        sql = (
            f"INSERT INTO `{self._prefix}_approvals` "
            "(created_at, expires_at, status, intent) "
            "VALUES (NOW(), %s, 'PENDING', %s)"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (expires_at, intent_json))
                    new_id = cur.lastrowid
                conn.commit()
                return int(new_id)
            finally:
                conn.close()

    def get_approval(self, approval_id: int) -> Optional[dict]:
        sql = (
            f"SELECT * FROM `{self._prefix}_approvals` "
            "WHERE id=%s LIMIT 1"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (int(approval_id),))
                    return cur.fetchone()
            finally:
                conn.close()

    def decide_approval(self, *, approval_id: int, decision: str,
                         decided_by: str) -> bool:
        """Set status to APPROVED or DENIED.  Only succeeds if currently
        PENDING and not expired.  Returns True on transition."""
        assert decision in ("APPROVED", "DENIED")
        sql = (
            f"UPDATE `{self._prefix}_approvals` "
            "SET status=%s, decided_at=NOW(), decided_by=%s "
            "WHERE id=%s AND status='PENDING' AND expires_at > NOW()"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (decision, decided_by, int(approval_id)))
                    affected = cur.rowcount
                conn.commit()
                return affected > 0
            finally:
                conn.close()

    def mark_approval_executed(self, *, approval_id: int,
                                position_id: int) -> None:
        sql = (
            f"UPDATE `{self._prefix}_approvals` "
            "SET status='EXECUTED', exec_position_id=%s "
            "WHERE id=%s AND status='APPROVED'"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (int(position_id), int(approval_id)))
                conn.commit()
            finally:
                conn.close()

    def mark_approval_stale(self, *, approval_id: int,
                             reason: str = "stale") -> None:
        sql = (
            f"UPDATE `{self._prefix}_approvals` "
            "SET status='STALE', decided_at=NOW(), decided_by=%s "
            "WHERE id=%s AND status='APPROVED'"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (reason, int(approval_id)))
                conn.commit()
            finally:
                conn.close()

    def expire_old_approvals(self) -> int:
        """Move PENDING rows past expires_at to EXPIRED.  Returns count."""
        sql = (
            f"UPDATE `{self._prefix}_approvals` "
            "SET status='EXPIRED' "
            "WHERE status='PENDING' AND expires_at <= NOW()"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    affected = cur.rowcount
                conn.commit()
                return int(affected)
            finally:
                conn.close()

    def get_executed_approval_count(self) -> int:
        """Total trades approved via manual gate so far.

        Used by ApprovalGate to flip into auto mode after threshold.
        """
        sql = (
            f"SELECT COUNT(*) AS n FROM `{self._prefix}_approvals` "
            "WHERE status='EXECUTED'"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
                return int(row["n"] or 0) if row else 0
            finally:
                conn.close()

    def get_latest_pending_approval(self) -> Optional[dict]:
        """Used by /yes /no shortcut when user doesn't supply an id."""
        sql = (
            f"SELECT * FROM `{self._prefix}_approvals` "
            "WHERE status='PENDING' AND expires_at > NOW() "
            "ORDER BY id DESC LIMIT 1"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return cur.fetchone()
            finally:
                conn.close()

    def get_consecutive_clean_days(self) -> int:
        """For Stage 3 graduation (GR-04)."""
        sql = (
            f"SELECT DATE(ts) AS d, "
            "       SUM(CASE WHEN result <> 'CONSISTENT' THEN 1 ELSE 0 END) AS bad "
            f"FROM `{self._prefix}_reconciliation_log` "
            "WHERE ts >= DATE_SUB(NOW(), INTERVAL 30 DAY) "
            "GROUP BY DATE(ts) ORDER BY d DESC"
        )
        with self._lock:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = list(cur.fetchall())
                clean = 0
                for r in rows:
                    if int(r["bad"] or 0) > 0:
                        break
                    clean += 1
                return clean
            finally:
                conn.close()


# ── Chart overlay ──────────────────────────────────────────────────────────

def fetch_okx_positions_for_chart(start_dt: datetime, end_dt: datetime) -> list:
    """LIVE OKX positions whose holding window intersects [start_dt, end_dt].

    Drop-in replacement for the old v7_paper_executor.fetch_positions_for_chart:
    returns the same dict shape (entry_time, direction, entry_price, entry_tier,
    status, exit_time, exit_price, exit_reason, win) so both charts can overlay
    the real live entries/exits. `win` is derived from gross_pct (the OKX table
    has no win column). Times are naive UTC. Returns [] if the table is absent.
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT entry_time, direction, entry_price, entry_tier, "
                    "       status, exit_time, exit_price, exit_reason, "
                    "       CASE WHEN gross_pct > 0 THEN 1 ELSE 0 END AS win "
                    "FROM `v7_okx_positions` "
                    "WHERE entry_time <= %s "
                    "  AND (exit_time IS NULL OR exit_time >= %s) "
                    "ORDER BY entry_time ASC",
                    (end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                     start_dt.strftime("%Y-%m-%d %H:%M:%S")))
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return []
                raise
    finally:
        conn.close()
    return list(rows or [])
