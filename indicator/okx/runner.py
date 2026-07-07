"""OKX executor singleton + lifecycle.

Used by indicator/app.py to:
  - Lazily construct V7OkxExecutor on first use (deferred so importing
    this module never connects to OKX)
  - Provide the same instance to update_cycle and /yes /no webhook
    handlers

Behaviour is env-gated via OKX_EXECUTOR_ENABLED:
  unset / "0" / "false"  → get_executor() returns None (no-op)
  "1" / "true"           → lazy-init on first call

This means flipping the gate is one Railway env var away.  No code
deploy needed once the binary is in place.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from indicator.okx.approval import ApprovalGate
from indicator.okx.client import OkxClient
from indicator.okx.config import (
    OkxConfig, load_okx_config_from_env, validate_okx_config,
)
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.reconciler import PositionReconciler
from indicator.okx.state import OkxStateStore

logger = logging.getLogger(__name__)


_LOCK = threading.Lock()
_INSTANCE: Optional[V7OkxExecutor] = None
_APPROVAL: Optional[ApprovalGate] = None
_INIT_FAILED: bool = False


def is_enabled() -> bool:
    """True if the env flag opts in."""
    return os.environ.get("OKX_EXECUTOR_ENABLED", "").lower() in ("1", "true")


def _stage_label() -> str:
    """Return 'live' if STAGE=live, else 'testnet' (== demo)."""
    return "live" if os.environ.get("STAGE", "") == "live" else "testnet"


def get_executor() -> Optional[V7OkxExecutor]:
    """Return the singleton (lazy-init).

    Returns None if:
      - OKX_EXECUTOR_ENABLED is not set
      - A previous init attempt failed (caller should not keep retrying
        every cycle — fail closed)
    """
    global _INSTANCE, _APPROVAL, _INIT_FAILED
    if not is_enabled():
        return None
    if _INIT_FAILED:
        return None
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        if _INIT_FAILED:
            return None
        try:
            cfg = load_okx_config_from_env(stage=_stage_label())
            validate_okx_config(cfg)
            client = OkxClient(cfg)
            store = OkxStateStore(table_prefix=cfg.table_prefix)
            reconciler = PositionReconciler(client=client, store=store,
                                              inst_id=cfg.inst_id)
            gate = ApprovalGate(
                store=store,
                alert_chat_id=cfg.telegram_critical_chat_id,
                stage_label=cfg.stage_label,
            )
            executor = V7OkxExecutor(
                client=client, store=store, reconciler=reconciler,
                cfg=cfg, approval_gate=gate,
            )
            # start() does API perms check, NTP, cold-start recon, balance
            # snapshot, then transitions to ACTIVE (or HALTED / DEMOTED
            # on failure).  Don't crash app.py if it fails.
            executor.start()
            _APPROVAL = gate
            _INSTANCE = executor
            logger.info("okx_executor_initialized status=%s",
                        executor.get_status().value)
            return _INSTANCE
        except Exception:
            logger.exception("okx_executor_init_failed_disabling")
            _INIT_FAILED = True
            return None


def get_approval_gate() -> Optional[ApprovalGate]:
    """Returns the gate associated with the singleton, or None.

    Webhook handlers (e.g. /yes_<id>) call this to decide an approval.
    """
    # Side-effect: also ensures executor is initialised
    get_executor()
    return _APPROVAL


# ── Multi-account follow-trading (Phase 1) ─────────────────────────────
#
# Friend accounts registered in okx_accounts (status=ACTIVE) each get an
# isolated executor stack: own credentials, own per-account tables
# (v7_okx_a<id>_*), own kill-switch state.  approval_gate=None → fully
# automatic execution (the operator pauses via /okx_pauseacct instead).
#
# Init failure HALTs the account in DB (no per-cycle retry storm) and the
# admin resumes it after fixing the cause.

_ACCT_EXECUTORS: dict[int, V7OkxExecutor] = {}


def get_account_executors() -> list[tuple[str, V7OkxExecutor]]:
    """(label, executor) for every ACTIVE friend account.

    Called once per update cycle.  Accounts newly flipped to ACTIVE are
    lazily initialised; accounts no longer ACTIVE are dropped from the
    cycle (their open positions stay managed only after re-activation —
    pausing mid-position means the stop order on OKX is the safety net).
    """
    if not is_enabled():
        return []
    try:
        from indicator.okx.accounts import (
            create_account_tables, load_active_accounts,
            mark_account_halted, table_prefix_for,
        )
        rows = load_active_accounts()
    except Exception:
        logger.exception("account_rows_load_failed")
        return []

    active_ids = set()
    out: list[tuple[str, V7OkxExecutor]] = []
    for row in rows:
        acct_id = row["id"]
        active_ids.add(acct_id)
        ex = _ACCT_EXECUTORS.get(acct_id)
        if ex is None:
            with _LOCK:
                ex = _ACCT_EXECUTORS.get(acct_id)
                if ex is None:
                    try:
                        cfg = load_okx_config_from_env(stage=_stage_label())
                        cfg.api_key = row["api_key"]
                        cfg.api_secret = row["api_secret"]
                        cfg.passphrase = row["passphrase"]
                        cfg.initial_capital_usd = row["initial_capital_usd"]
                        cfg.daily_loss_cap_pct = row["daily_loss_cap_pct"]
                        cfg.total_loss_cap_pct = row["total_loss_cap_pct"]
                        cfg.stage_label = f"{cfg.stage_label}:{row['label']}"
                        cfg.table_prefix = table_prefix_for(acct_id)
                        validate_okx_config(cfg)
                        create_account_tables(acct_id)
                        client = OkxClient(cfg)
                        store = OkxStateStore(table_prefix=cfg.table_prefix)
                        reconciler = PositionReconciler(
                            client=client, store=store, inst_id=cfg.inst_id)
                        ex = V7OkxExecutor(
                            client=client, store=store,
                            reconciler=reconciler, cfg=cfg,
                            approval_gate=None,  # follow accounts: full auto
                        )
                        ex.start()
                        _ACCT_EXECUTORS[acct_id] = ex
                        logger.info("acct_executor_initialized label=%s "
                                    "status=%s", row["label"],
                                    ex.get_status().value)
                    except Exception as e:
                        logger.exception("acct_executor_init_failed label=%s",
                                         row["label"])
                        try:
                            mark_account_halted(acct_id, repr(e))
                        except Exception:
                            logger.exception("acct_halt_mark_failed")
                        continue
        out.append((row["label"], ex))

    # Drop executors for accounts no longer ACTIVE (paused/deleted)
    for stale_id in list(_ACCT_EXECUTORS.keys()):
        if stale_id not in active_ids:
            _ACCT_EXECUTORS.pop(stale_id, None)
            logger.info("acct_executor_dropped id=%s", stale_id)

    return out


def reset_for_tests() -> None:
    """Clear singleton — pytest fixtures use this."""
    global _INSTANCE, _APPROVAL, _INIT_FAILED
    with _LOCK:
        _INSTANCE = None
        _APPROVAL = None
        _INIT_FAILED = False
        _ACCT_EXECUTORS.clear()
