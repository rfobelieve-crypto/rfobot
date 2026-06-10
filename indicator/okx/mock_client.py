"""MockOkxClient — a stateful, in-memory stand-in for OkxClient.

Purpose (2026-06-07): the live OKX trade-path code was written but never run
against OKX's demo environment (testnet shakeout was skipped — see CLAUDE.md
§"跳過 testnet").  This fake lets us drive the *real* V7OkxExecutor through a
full cycle with ZERO real money, and — crucially — deliberately inject the
adverse conditions each kill switch is supposed to catch, satisfying the
CLAUDE.md hard rule that "kill switches must be verified to actually fire,
not just written into code".

Design: faithful to the OkxClient public surface (same method signatures,
same return dataclasses), but maintains real internal state:

  - a single signed net position (+long / -short) with an avg price
  - registered algo stops (id -> trigger / side / size)
  - a balance, account perms, server-clock offset, connectivity status
  - WS callbacks (orders / positions / balance)

Scenario helpers (the whole point) let a test or sim push the fake into the
exact state a given kill check guards against:

  reject_market_order(err)      -> entry/close rejection paths
  raise_on_algo()/reject_algo() -> B4 algo-stop-latency force-close
  set_server_time_offset(sec)   -> C5/C6 NTP drift
  set_connectivity(...)         -> A1/A2/A3 WS-loss kills
  set_balance(...)              -> CAP-2 / daily / total loss caps
  set_okx_position(...)         -> A4 orphan_exchange / size_diff /
                                   direction_diff  (== manual-interference!)
  fire_algo_stop(fill_price)    -> OKX trail-stop firing + WS algo fill,
                                   incl. a -10% gap stop-out
  push_balance(...)             -> WS balance event (drives daily-cap persist)

This is NOT a market simulator — there is no order book, no matching, no
latency model.  It is a controllable test double whose only job is to let us
exercise the executor's state machine and prove every guard works.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from indicator.okx.config import OkxConfig
from indicator.okx.types import (
    AlgoOrderResult, AmendResult, Balance, BalanceEvent, CancelResult,
    ConnectivityStatus, OrderEvent, OrderResult, Position, PositionEvent,
    Side,
)

logger = logging.getLogger(__name__)


class MockOkxClient:
    """In-memory fake mirroring the OkxClient facade.  See module docstring."""

    def __init__(self, cfg: OkxConfig, *, mark_price: float = 75000.0):
        self._cfg = cfg
        self._inst_id = cfg.inst_id

        # ── Market / fill model (intentionally trivial) ────────────────
        self._mark: float = mark_price
        self._slippage_frac: float = 0.0

        # ── Position state: single signed net position ─────────────────
        self._net: float = 0.0           # + long, - short, in contracts
        self._avg_price: float = 0.0

        # ── Algo stop registry: algo_id -> dict ────────────────────────
        self._algos: dict[str, dict] = {}

        # ── Account / clock / health ───────────────────────────────────
        self._balance = Balance(
            total_eq_usd=cfg.initial_capital_usd,
            available_usd=cfg.initial_capital_usd,
        )
        self._perms: list[str] = ["read", "trade"]
        self._server_time_offset_sec: float = 0.0
        self._connectivity = ConnectivityStatus(
            public_ws_ok=True, private_ws_ok=True,
            last_public_heartbeat_age_sec=0.0,
            last_private_heartbeat_age_sec=0.0,
            rest_last_latency_ms=10.0, rest_circuit_tripped=False,
            consecutive_reconnect_fails=0,
        )

        # ── Injectable failure flags ───────────────────────────────────
        self._reject_market_error: Optional[str] = None
        self._reject_market_once: bool = False
        self._reject_algo_error: Optional[str] = None
        self._algo_raises: bool = False
        self._reject_set_leverage: bool = False

        # ── WS callbacks ───────────────────────────────────────────────
        self._on_order: Optional[Callable[[OrderEvent], None]] = None
        self._on_position: Optional[Callable[[PositionEvent], None]] = None
        self._on_balance: Optional[Callable[[BalanceEvent], None]] = None

        # ── Sequence counters / audit trail for assertions ─────────────
        self._ord_seq = 0
        self._algo_seq = 0
        self._ws_started = False
        self.market_orders: list[dict] = []   # every submit_market_order call
        self.algo_orders: list[dict] = []      # every submit_algo_stop call
        self.leverage_calls: list[dict] = []   # every set_leverage call

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start_ws(self) -> None:
        self._ws_started = True

    def close(self) -> None:
        self._ws_started = False

    # ── REST: orders ───────────────────────────────────────────────────

    def submit_market_order(self, *, inst_id: str, side: Side, sz: float,
                            td_mode: str,
                            cl_ord_id: Optional[str] = None,
                            pos_side: Optional[str] = None,
                            reduce_only: bool = False) -> OrderResult:
        self.market_orders.append({
            "inst_id": inst_id, "side": side, "sz": float(sz),
            "td_mode": td_mode, "cl_ord_id": cl_ord_id,
            "pos_side": pos_side, "reduce_only": reduce_only,
        })

        if self._reject_market_error is not None:
            err = self._reject_market_error
            if self._reject_market_once:
                self._reject_market_error = None
                self._reject_market_once = False
            return OrderResult(cl_ord_id=cl_ord_id or "", status="rejected",
                               error=err)

        self._ord_seq += 1
        ord_id = f"mock-ord-{self._ord_seq}"
        fill_price = self._fill_price(side)

        # Apply to net position
        delta = float(sz) if side == Side.BUY else -float(sz)
        prev_net = self._net
        new_net = round(prev_net + delta, 8)
        if prev_net == 0.0 and new_net != 0.0:
            self._avg_price = fill_price            # fresh open
        elif new_net == 0.0:
            self._avg_price = 0.0                    # fully flat
        # (partial reduce / flip keeps the simple model's prior avg)
        self._net = new_net

        return OrderResult(cl_ord_id=cl_ord_id or "", ord_id=ord_id,
                           status="filled", fill_price=fill_price,
                           fill_size=float(sz))

    def submit_algo_stop(self, *, inst_id: str, side: Side, sz: float,
                         trigger_px: float, td_mode: str,
                         algo_cl_ord_id: Optional[str] = None,
                         pos_side: Optional[str] = None,
                         reduce_only: bool = False) -> AlgoOrderResult:
        self.algo_orders.append({
            "inst_id": inst_id, "side": side, "sz": float(sz),
            "trigger_px": trigger_px, "algo_cl_ord_id": algo_cl_ord_id,
            "pos_side": pos_side, "reduce_only": reduce_only,
        })

        if self._algo_raises:
            raise RuntimeError("mock: algo stop submit network error")
        if self._reject_algo_error is not None:
            return AlgoOrderResult(algo_cl_ord_id=algo_cl_ord_id or "",
                                   status="rejected",
                                   error=self._reject_algo_error)

        self._algo_seq += 1
        algo_id = f"mock-algo-{self._algo_seq}"
        self._algos[algo_id] = {
            "side": side, "sz": float(sz), "trigger_px": trigger_px,
            "cl_ord_id": algo_cl_ord_id or "", "pos_side": pos_side,
        }
        return AlgoOrderResult(algo_cl_ord_id=algo_cl_ord_id or "",
                               algo_id=algo_id, status="live")

    def amend_algo_stop(self, *, algo_id: str,
                        new_trigger_px: float) -> AmendResult:
        algo = self._algos.get(algo_id)
        if algo is None:
            return AmendResult(algo_id=algo_id, status="not_found")
        algo["trigger_px"] = new_trigger_px
        return AmendResult(algo_id=algo_id, status="ok")

    def cancel_algo_stop(self, *, algo_id: str) -> CancelResult:
        if algo_id in self._algos:
            del self._algos[algo_id]
            return CancelResult(algo_id=algo_id, status="ok")
        return CancelResult(algo_id=algo_id, status="not_found")

    def set_leverage(self, *, inst_id: str, lever: int, mgn_mode: str,
                     pos_side: Optional[str] = None) -> bool:
        """Mirror OkxRestClient.set_leverage.  Records the call; returns True
        unless reject_set_leverage() armed a failure (to test the abort path)."""
        self.leverage_calls.append({
            "inst_id": inst_id, "lever": lever, "mgn_mode": mgn_mode,
            "pos_side": pos_side,
        })
        return not self._reject_set_leverage

    # ── REST: account ──────────────────────────────────────────────────

    def get_positions(self, *, inst_id: str) -> list[Position]:
        if self._net == 0.0 or inst_id != self._inst_id:
            return []
        direction = "LONG" if self._net > 0 else "SHORT"
        return [Position(
            inst_id=self._inst_id, direction=direction,
            size_contracts=abs(self._net), avg_price=self._avg_price,
        )]

    def get_balance(self) -> Optional[Balance]:
        return self._balance

    def get_account_config(self) -> dict:
        return {"data": [{"perm": ",".join(self._perms)}]}

    def get_server_time(self) -> Optional[float]:
        return time.time() + self._server_time_offset_sec

    # ── WS subscriptions ───────────────────────────────────────────────

    def subscribe_orders(self,
                         callback: Callable[[OrderEvent], None]) -> None:
        self._on_order = callback

    def subscribe_positions(self,
                            callback: Callable[[PositionEvent], None]) -> None:
        self._on_position = callback

    def subscribe_balance(self,
                          callback: Callable[[BalanceEvent], None]) -> None:
        self._on_balance = callback

    # ── Health ─────────────────────────────────────────────────────────

    def connectivity(self) -> ConnectivityStatus:
        return self._connectivity

    def diag_state(self) -> dict:
        return {"mock": True, "net": self._net, "avg_price": self._avg_price,
                "algos": list(self._algos.keys()),
                "ws_started": self._ws_started}

    # ── Internal ───────────────────────────────────────────────────────

    def _fill_price(self, side: Side) -> float:
        if self._slippage_frac <= 0:
            return self._mark
        # Adverse slippage: buys fill higher, sells fill lower.
        if side == Side.BUY:
            return self._mark * (1.0 + self._slippage_frac)
        return self._mark * (1.0 - self._slippage_frac)

    # ════════════════════════════════════════════════════════════════════
    #  Scenario injection — the reason this fake exists
    # ════════════════════════════════════════════════════════════════════

    def set_mark_price(self, price: float) -> None:
        self._mark = float(price)

    def set_slippage(self, frac: float) -> None:
        """Adverse round-trip slippage fraction (e.g. 0.001 = 10 bps)."""
        self._slippage_frac = float(frac)

    def set_balance(self, total_eq_usd: float,
                    available_usd: Optional[float] = None) -> None:
        self._balance = Balance(
            total_eq_usd=float(total_eq_usd),
            available_usd=float(available_usd if available_usd is not None
                                else total_eq_usd),
        )

    def set_perms(self, perms: list[str]) -> None:
        """e.g. ['read','trade','withdraw'] to trip E4."""
        self._perms = list(perms)

    def set_server_time_offset(self, offset_sec: float) -> None:
        """Inject clock drift vs local time -> C5 (>5s) / C6 (>30s)."""
        self._server_time_offset_sec = float(offset_sec)

    def set_connectivity(self, **fields) -> None:
        """Override ConnectivityStatus fields, e.g.
        set_connectivity(last_private_heartbeat_age_sec=400) -> A1 demote.
        """
        for k, v in fields.items():
            setattr(self._connectivity, k, v)

    def reject_market_order(self, error: str = "mock_rejected",
                            *, once: bool = False) -> None:
        self._reject_market_error = error
        self._reject_market_once = once

    def clear_market_rejection(self) -> None:
        self._reject_market_error = None
        self._reject_market_once = False

    def reject_algo(self, error: str = "mock_algo_rejected") -> None:
        self._reject_algo_error = error

    def raise_on_algo(self) -> None:
        """Make submit_algo_stop raise -> tests the B4 emergency-close path."""
        self._algo_raises = True

    def reject_set_leverage(self) -> None:
        """Make set_leverage return False -> tests the isolated open-abort path."""
        self._reject_set_leverage = True

    def set_okx_position(self, direction: str, size_contracts: float,
                         avg_price: Optional[float] = None) -> None:
        """Force OKX to report a position the executor may not know about.

        This is how we simulate MANUAL INTERFERENCE (operator opened a
        trade on the same account) — reconciler should flag
        orphan_exchange / size_diff / direction_diff and the executor
        should HALT.  direction: 'LONG' / 'SHORT' / 'FLAT'.
        """
        if direction == "FLAT" or size_contracts == 0:
            self._net = 0.0
            self._avg_price = 0.0
            return
        sign = 1.0 if direction == "LONG" else -1.0
        self._net = sign * abs(float(size_contracts))
        self._avg_price = float(avg_price if avg_price is not None
                                else self._mark)

    def fire_algo_stop(self, fill_price: Optional[float] = None) -> None:
        """Simulate OKX's trail stop firing: flatten + emit WS algo fill.

        Pass a fill_price far from entry (e.g. mark * 0.90) to simulate a
        -10% gap stop-out.  Emits one OrderEvent per live algo so the
        executor's on_order callback closes its DB row and sends the exit
        alert (the path that, if broken, leaves DB OPEN forever).
        """
        from datetime import datetime
        px = float(fill_price) if fill_price is not None else self._mark
        fired = list(self._algos.items())
        # Flatten the position (the stop closed it)
        self._net = 0.0
        self._avg_price = 0.0
        for algo_id, algo in fired:
            del self._algos[algo_id]
            if self._on_order is None:
                continue
            self._on_order(OrderEvent(
                cl_ord_id="", ord_id=f"mock-stopfill-{algo_id}",
                state="filled", fill_price=px, fill_size=algo["sz"],
                ts=datetime.utcnow(),
                algo_cl_ord_id=algo.get("cl_ord_id") or "",
                algo_id=algo_id,
            ))

    def push_balance(self, total_eq_usd: float,
                     available_usd: Optional[float] = None) -> None:
        """Emit a WS balance event (executor persists it for the daily cap)."""
        from datetime import datetime
        self.set_balance(total_eq_usd, available_usd)
        if self._on_balance is not None:
            self._on_balance(BalanceEvent(
                total_eq_usd=self._balance.total_eq_usd,
                available_usd=self._balance.available_usd,
                ts=datetime.utcnow(),
            ))
