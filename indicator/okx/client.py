"""OkxClient — composite that wraps REST + WS private.

Single entrypoint for the executor.  No business logic — just routing.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from indicator.okx.config import OkxConfig
from indicator.okx.rest import OkxRestClient
from indicator.okx.types import (
    AlgoOrderResult, AmendResult, Balance, BalanceEvent, CancelResult,
    ConnectivityStatus, OrderEvent, OrderResult, Position, PositionEvent,
    Side,
)
from indicator.okx.ws_private import OkxWsPrivateClient

logger = logging.getLogger(__name__)


class OkxClient:
    """Thin facade over OkxRestClient + OkxWsPrivateClient.

    Lifecycle:
      1. __init__       — instantiate sub-clients
      2. start_ws()     — launch private WS thread (idempotent)
      3. subscribe_*    — register callbacks
      4. <REST methods> — delegate to OkxRestClient
      5. close()        — stop WS thread
    """

    def __init__(self, cfg: OkxConfig):
        self._cfg = cfg
        self._rest = OkxRestClient(cfg)
        self._ws_private = OkxWsPrivateClient(cfg)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start_ws(self) -> None:
        self._ws_private.start()

    def close(self) -> None:
        self._ws_private.stop()

    # ── REST passthrough ───────────────────────────────────────────────

    def submit_market_order(self, *, inst_id: str, side: Side, sz: int,
                            td_mode: str,
                            cl_ord_id: Optional[str] = None,
                            pos_side: Optional[str] = None,
                            reduce_only: bool = False) -> OrderResult:
        return self._rest.submit_market_order(
            inst_id=inst_id, side=side, sz=sz, td_mode=td_mode,
            cl_ord_id=cl_ord_id, pos_side=pos_side, reduce_only=reduce_only,
        )

    def submit_algo_stop(self, *, inst_id: str, side: Side, sz: int,
                         trigger_px: float, td_mode: str,
                         algo_cl_ord_id: Optional[str] = None,
                         pos_side: Optional[str] = None,
                         reduce_only: bool = False) -> AlgoOrderResult:
        return self._rest.submit_algo_stop(
            inst_id=inst_id, side=side, sz=sz, trigger_px=trigger_px,
            td_mode=td_mode, algo_cl_ord_id=algo_cl_ord_id,
            pos_side=pos_side, reduce_only=reduce_only,
        )

    def amend_algo_stop(self, *, inst_id: str, algo_id: str,
                        new_trigger_px: float) -> AmendResult:
        # inst_id is REQUIRED: OKX amend-algos rejects missing instId with 50014
        # (2026-06-10 bug).  The caller (executor.py) and the REST layer
        # (rest.py) both pass/accept inst_id; this facade previously dropped it,
        # so every amend raised TypeError here and the trailing stop never moved
        # on the exchange.  Keep this signature in lockstep with rest.amend_algo_stop.
        return self._rest.amend_algo_stop(
            inst_id=inst_id, algo_id=algo_id, new_trigger_px=new_trigger_px,
        )

    def cancel_algo_stop(self, *, algo_id: str) -> CancelResult:
        return self._rest.cancel_algo_stop(algo_id=algo_id)

    def get_positions(self, *, inst_id: str) -> list[Position]:
        return self._rest.get_positions(inst_id=inst_id)

    def get_balance(self) -> Optional[Balance]:
        return self._rest.get_balance()

    def get_account_config(self) -> dict:
        return self._rest.get_account_config()

    def get_server_time(self) -> Optional[float]:
        return self._rest.get_server_time()

    def set_leverage(self, *, inst_id: str, lever: int, mgn_mode: str,
                     pos_side: Optional[str] = None) -> bool:
        """Proxy to OkxRestClient.set_leverage.

        Required because isolated-margin opens call this through the facade
        before each entry (executor._open_position).  Without this proxy a
        plain AttributeError surfaces as a generic 'set-leverage failed' abort
        and hides the real root cause.  Discovered 2026-06-16 when the live
        executor aborted every isolated open with a phantom error before
        even reaching OKX.
        """
        return self._rest.set_leverage(
            inst_id=inst_id, lever=lever, mgn_mode=mgn_mode, pos_side=pos_side,
        )

    def set_leverage_detail(self, *, inst_id: str, lever: int, mgn_mode: str,
                            pos_side: Optional[str] = None) -> dict:
        """Proxy to OkxRestClient.set_leverage_detail (diagnostic surface)."""
        return self._rest.set_leverage_detail(
            inst_id=inst_id, lever=lever, mgn_mode=mgn_mode, pos_side=pos_side,
        )

    # ── WS subscription passthrough ────────────────────────────────────

    def subscribe_orders(self,
                         callback: Callable[[OrderEvent], None]) -> None:
        self._ws_private.subscribe_orders(callback)

    def subscribe_positions(self,
                            callback: Callable[[PositionEvent], None]) -> None:
        self._ws_private.subscribe_positions(callback)

    def subscribe_balance(self,
                          callback: Callable[[BalanceEvent], None]) -> None:
        self._ws_private.subscribe_balance(callback)

    # ── Health ─────────────────────────────────────────────────────────

    def connectivity(self) -> ConnectivityStatus:
        status = self._ws_private.connectivity()
        status.rest_last_latency_ms = self._rest.last_latency_ms()
        status.rest_circuit_tripped = self._rest.is_circuit_tripped()
        return status

    def diag_state(self) -> dict:
        """Diagnostic snapshot — surface WS internals to /okx-status."""
        return {
            "ws_private": self._ws_private.diag_state(),
            "rest_last_latency_ms": self._rest.last_latency_ms(),
            "rest_circuit_tripped": self._rest.is_circuit_tripped(),
        }
