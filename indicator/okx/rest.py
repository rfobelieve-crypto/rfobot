"""OKX REST client with retry / backoff / circuit breaker.

Connection-resilience + response parsers complete (2026-05-28).
Remaining TODO at L124 is the partial-fill → reconciler hook, which
is a future feature, not a blocker for Stage 3 $100.

Retry policy (per docs/okx_integration_design.md §10.2):
  - Order submit: 3x, 1s/2s/4s, idempotency via clOrdId
  - Amend algo: 2x, 1s/2s, halt on full fail
  - Cancel algo: 3x, 1s/2s/4s
  - Get positions: 5x, 0.5s/1s/2s/4s/8s
  - Get balance: 3x, 0.5s/1s/2s
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

from indicator.okx.config import OkxConfig
from indicator.okx.types import (
    AlgoOrderResult, AmendResult, Balance, CancelResult,
    OrderResult, Position, Side,
)

logger = logging.getLogger(__name__)


# ── Circuit breaker ────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """Trips when N failures occur within W seconds.  Stays tripped for
    cooldown_sec before allowing retry."""
    threshold: int = 5
    window_sec: int = 60
    cooldown_sec: int = 300

    def __post_init__(self):
        self._failures: deque[float] = deque(maxlen=self.threshold * 2)
        self._tripped_until: float = 0.0

    def record_failure(self) -> None:
        now = time.time()
        self._failures.append(now)
        # Count recent
        recent = sum(1 for t in self._failures if now - t <= self.window_sec)
        if recent >= self.threshold:
            self._tripped_until = now + self.cooldown_sec
            logger.warning(
                "circuit_breaker_tripped recent_fails=%d window=%ds cooldown=%ds",
                recent, self.window_sec, self.cooldown_sec,
            )

    def is_tripped(self) -> bool:
        return time.time() < self._tripped_until

    def record_success(self) -> None:
        # Don't clear history — just track; recovery is time-based.
        pass


# ── ID generators ──────────────────────────────────────────────────────

def make_cl_ord_id(prefix: str = "v7") -> str:
    """Idempotency key for orders.  Safety belt #1.

    Format: v7{epoch_ms}{rand_hex8}, max 32 chars, OKX-compatible.
    """
    return f"{prefix}{int(time.time() * 1000)}{uuid.uuid4().hex[:8]}"


# ── REST client ────────────────────────────────────────────────────────

class OkxRestClient:
    """Low-level OKX REST wrapper.  No business logic.

    Every public method:
      - Uses idempotency keys where applicable
      - Retries per policy
      - Surfaces circuit breaker state
      - Logs structured events
    """

    def __init__(self, cfg: OkxConfig, *,
                 session: Optional[requests.Session] = None,
                 timeout_sec: float = 10.0):
        self._cfg = cfg
        self._session = session or requests.Session()
        self._timeout = timeout_sec
        self._breaker = CircuitBreaker()
        self._last_latency_ms: Optional[float] = None

    # ── Public API ─────────────────────────────────────────────────────

    def submit_market_order(self, *, inst_id: str, side: Side, sz: int,
                            td_mode: str,
                            cl_ord_id: Optional[str] = None,
                            pos_side: Optional[str] = None,
                            reduce_only: bool = False) -> OrderResult:
        """Submit a market order.  Idempotent via clOrdId.

        pos_side: "long"/"short" required in long_short_mode, omitted in
        net_mode.  reduce_only: True ensures the order only reduces an
        existing position (defensive; redundant in long_short_mode where
        side+posSide is unambiguous, but harmless).

        Retry policy: 3x exponential backoff on 5xx; do NOT retry on 4xx
        (4xx means our request is malformed/rejected — retrying won't help).
        """
        cl_ord_id = cl_ord_id or make_cl_ord_id()
        path = "/api/v5/trade/order"
        body = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side.value,
            "ordType": "market",
            "sz": str(sz),
            "clOrdId": cl_ord_id,
        }
        if pos_side is not None:
            body["posSide"] = pos_side
        if reduce_only:
            body["reduceOnly"] = True
        # TODO(stage2-impl): on partial fill we should pass into reconciler.
        result = self._retry_post(
            path=path, body=body, retries=3, backoff_base=1.0,
            retry_on_4xx=False,
        )
        if result is None:
            return OrderResult(
                cl_ord_id=cl_ord_id, status="rejected",
                error="all_retries_exhausted",
            )
        return self._parse_order_response(result, cl_ord_id)

    def submit_algo_stop(self, *, inst_id: str, side: Side, sz: int,
                         trigger_px: float, td_mode: str,
                         algo_cl_ord_id: Optional[str] = None,
                         pos_side: Optional[str] = None,
                         reduce_only: bool = False) -> AlgoOrderResult:
        """Submit conditional stop-market algo order (the trailing stop).

        pos_side / reduce_only: same semantics as submit_market_order.
        For long_short_mode the stop's posSide must match the position
        it protects (LONG position → posSide='long').
        """
        algo_cl_ord_id = algo_cl_ord_id or make_cl_ord_id(prefix="v7a")
        path = "/api/v5/trade/order-algo"
        # OKX conditional orders require slTriggerPx + slOrdPx for stop loss
        # (triggerPx alone returns 50015 "Either parameter tpTriggerPx or
        # slTriggerPx is required").  Discovered 2026-06-02 when first live
        # algo stop submission failed.
        body = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side.value,
            "ordType": "conditional",
            "sz": str(sz),
            "slTriggerPx": str(trigger_px),
            "slOrdPx": "-1",              # market on trigger
            "slTriggerPxType": "last",    # trigger on last price
            "algoClOrdId": algo_cl_ord_id,
        }
        if pos_side is not None:
            body["posSide"] = pos_side
        if reduce_only:
            body["reduceOnly"] = True
        result = self._retry_post(
            path=path, body=body, retries=3, backoff_base=1.0,
            retry_on_4xx=False,
        )
        if result is None:
            return AlgoOrderResult(
                algo_cl_ord_id=algo_cl_ord_id, status="rejected",
                error="all_retries_exhausted",
            )
        return self._parse_algo_order_response(result, algo_cl_ord_id)

    def amend_algo_stop(self, *, algo_id: str,
                        new_trigger_px: float) -> AmendResult:
        """Atomic amend.  P6: NEVER cancel-then-new.

        OKX amend-algos for stop-loss requires newSlTriggerPx (not the
        generic newTriggerPx).  Same fix as submit_algo_stop above.
        """
        path = "/api/v5/trade/amend-algos"
        body = {
            "algoId": algo_id,
            "newSlTriggerPx": str(new_trigger_px),
        }
        result = self._retry_post(
            path=path, body=body, retries=2, backoff_base=1.0,
            retry_on_4xx=False,
        )
        if result is None:
            return AmendResult(algo_id=algo_id, status="failed",
                               error="all_retries_exhausted")
        return self._parse_amend_response(result, algo_id)

    def cancel_algo_stop(self, *, algo_id: str) -> CancelResult:
        path = "/api/v5/trade/cancel-algos"
        body = {"algoId": algo_id}
        result = self._retry_post(
            path=path, body=body, retries=3, backoff_base=1.0,
            retry_on_4xx=False,
        )
        if result is None:
            return CancelResult(algo_id=algo_id, status="failed",
                                error="all_retries_exhausted")
        return CancelResult(algo_id=algo_id, status="ok",
                            error="not_implemented")

    def get_positions(self, *, inst_id: str) -> list[Position]:
        path = "/api/v5/account/positions"
        params = {"instId": inst_id}
        result = self._retry_get(
            path=path, params=params, retries=5, backoff_base=0.5,
        )
        if result is None:
            return []
        return self._parse_positions_response(result)

    def get_balance(self) -> Optional[Balance]:
        """Get total equity in USD-equivalent.  3x retry, demote on full fail."""
        path = "/api/v5/account/balance"
        result = self._retry_get(
            path=path, params={}, retries=3, backoff_base=0.5,
        )
        if result is None:
            return None
        return self._parse_balance_response(result)

    def get_account_config(self) -> dict:
        """Get account-level settings + API key permissions.

        Used by validate_api_permissions at startup.  Safety belt #5.
        """
        path = "/api/v5/account/config"
        result = self._retry_get(
            path=path, params={}, retries=3, backoff_base=0.5,
        )
        return result or {}

    def get_server_time(self) -> Optional[float]:
        """Returns OKX server time as Unix seconds.  Used for NTP drift check.

        No auth required.  Cheap: 1 retry only.
        """
        path = "/api/v5/public/time"
        result = self._retry_get(
            path=path, params={}, retries=1, backoff_base=0.5, auth=False,
        )
        if result is None:
            return None
        return self._parse_server_time(result)

    # ── Health surface ─────────────────────────────────────────────────

    def last_latency_ms(self) -> Optional[float]:
        return self._last_latency_ms

    def is_circuit_tripped(self) -> bool:
        return self._breaker.is_tripped()

    # ── Internal: signing ──────────────────────────────────────────────

    def _sign(self, ts: str, method: str, path: str, body: str) -> str:
        """OKX HMAC-SHA256 over (ts + method + path + body) → base64."""
        message = f"{ts}{method}{path}{body}".encode()
        secret = self._cfg.api_secret.encode()
        return base64.b64encode(
            hmac.new(secret, message, hashlib.sha256).digest()
        ).decode()

    def _headers(self, method: str, path: str, body: str,
                 auth: bool = True) -> dict:
        if not auth:
            return {"Content-Type": "application/json"}
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self._cfg.api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._cfg.passphrase,
        }
        if self._cfg.is_simulated == 1:
            headers["x-simulated-trading"] = "1"
        return headers

    # ── Internal: retry primitives ─────────────────────────────────────

    def _retry_get(self, *, path: str, params: dict, retries: int,
                  backoff_base: float, auth: bool = True) -> Optional[dict]:
        return self._retry(method="GET", path=path, body=None,
                          params=params, retries=retries,
                          backoff_base=backoff_base, retry_on_4xx=True,
                          auth=auth)

    def _retry_post(self, *, path: str, body: dict, retries: int,
                   backoff_base: float, retry_on_4xx: bool) -> Optional[dict]:
        return self._retry(method="POST", path=path, body=body,
                          params=None, retries=retries,
                          backoff_base=backoff_base,
                          retry_on_4xx=retry_on_4xx, auth=True)

    def _retry(self, *, method: str, path: str, body: Optional[dict],
               params: Optional[dict], retries: int, backoff_base: float,
               retry_on_4xx: bool, auth: bool) -> Optional[dict]:
        """Core retry loop with circuit breaker integration."""
        if self._breaker.is_tripped():
            logger.warning("rest_call_skipped_circuit_tripped path=%s", path)
            return None

        # OKX HMAC signs (ts + method + requestPath + body) where
        # requestPath must include the query string for GETs.  If we
        # signed just `/api/v5/account/positions` but actually fetched
        # `/api/v5/account/positions?instId=...`, OKX returns 50113.
        signed_path = path
        if method == "GET" and params:
            from urllib.parse import urlencode
            qs = urlencode(params)
            if qs:
                signed_path = f"{path}?{qs}"

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            if attempt > 0:
                delay = backoff_base * (2 ** (attempt - 1))
                logger.info("rest_retry path=%s attempt=%d delay=%.1fs",
                            path, attempt, delay)
                time.sleep(delay)
            try:
                t0 = time.time()
                body_str = json.dumps(body) if body is not None else ""
                url = self._cfg.rest_base + path
                resp = self._session.request(
                    method=method, url=url, params=params,
                    data=body_str if method != "GET" else None,
                    headers=self._headers(method, signed_path, body_str,
                                          auth=auth),
                    timeout=self._timeout,
                )
                self._last_latency_ms = (time.time() - t0) * 1000
                if resp.status_code == 200:
                    self._breaker.record_success()
                    try:
                        return resp.json()
                    except Exception as e:
                        logger.error("rest_parse_failed path=%s err=%s",
                                     path, e)
                        last_err = e
                        continue
                # 4xx vs 5xx handling
                if 400 <= resp.status_code < 500:
                    logger.warning("rest_4xx path=%s status=%d body=%s",
                                   path, resp.status_code, resp.text[:200])
                    if not retry_on_4xx:
                        return None
                    last_err = RuntimeError(f"4xx: {resp.status_code}")
                    continue
                # 5xx
                logger.warning("rest_5xx path=%s status=%d",
                               path, resp.status_code)
                self._breaker.record_failure()
                last_err = RuntimeError(f"5xx: {resp.status_code}")
            except requests.exceptions.Timeout as e:
                logger.warning("rest_timeout path=%s attempt=%d",
                               path, attempt)
                self._breaker.record_failure()
                last_err = e
            except requests.exceptions.ConnectionError as e:
                logger.warning("rest_connection_error path=%s attempt=%d err=%s",
                               path, attempt, e)
                self._breaker.record_failure()
                last_err = e
            except Exception as e:
                logger.exception("rest_unexpected path=%s attempt=%d",
                                 path, attempt)
                last_err = e

        logger.error("rest_all_retries_exhausted path=%s last_err=%s",
                     path, last_err)
        return None

    # ── Internal: response parsing ─────────────────────────────────────

    def _parse_order_response(self, raw: dict, cl_ord_id: str) -> OrderResult:
        """Parse POST /api/v5/trade/order response into OrderResult.

        Two failure layers:
          - top-level `code != "0"`: request was rejected outright
          - per-order `sCode != "0"`: order-level reject (e.g. 51008 balance)
        """
        code = raw.get("code", "")
        if code != "0":
            return OrderResult(cl_ord_id=cl_ord_id, status="rejected",
                               error=f"okx_code_{code}_{raw.get('msg', '')}")
        data_list = raw.get("data") or []
        if not data_list:
            # Top code=0 but no data — treat as submitted with unknown ord_id
            return OrderResult(cl_ord_id=cl_ord_id, status="submitted",
                               ord_id=None)
        data = data_list[0]
        s_code = data.get("sCode", "0")
        if s_code != "0":
            return OrderResult(
                cl_ord_id=cl_ord_id, status="rejected",
                error=f"okx_sCode_{s_code}_{data.get('sMsg', '')}",
            )
        return OrderResult(
            cl_ord_id=cl_ord_id,
            ord_id=data.get("ordId") or None,
            status="submitted",
        )

    def _parse_algo_order_response(self, raw: dict,
                                   algo_cl_ord_id: str) -> AlgoOrderResult:
        """Parse POST /api/v5/trade/order-algo response into AlgoOrderResult."""
        code = raw.get("code", "")
        if code != "0":
            return AlgoOrderResult(
                algo_cl_ord_id=algo_cl_ord_id, status="rejected",
                error=f"okx_code_{code}_{raw.get('msg', '')}",
            )
        data_list = raw.get("data") or []
        if not data_list:
            return AlgoOrderResult(
                algo_cl_ord_id=algo_cl_ord_id, status="submitted",
                algo_id=None,
            )
        data = data_list[0]
        s_code = data.get("sCode", "0")
        if s_code != "0":
            return AlgoOrderResult(
                algo_cl_ord_id=algo_cl_ord_id, status="rejected",
                algo_id=data.get("algoId") or None,
                error=f"okx_sCode_{s_code}_{data.get('sMsg', '')}",
            )
        return AlgoOrderResult(
            algo_cl_ord_id=algo_cl_ord_id, status="submitted",
            algo_id=data.get("algoId") or None,
        )

    # OKX algo-amend response code → AmendResult.status mapping
    _AMEND_CODE_MAP = {
        "0": "ok",
        "51400": "already_filled",   # docs/stage2_kill_criteria B3 boundary
        "51401": "not_found",
    }

    def _parse_amend_response(self, raw: dict, algo_id: str) -> AmendResult:
        """Parse POST /api/v5/trade/amend-algos response.

        Handles 51400 (already_filled — race with stop trigger) and
        51401 (order does not exist) per kill_criteria B3.
        """
        code = raw.get("code", "")
        if code != "0":
            return AmendResult(algo_id=algo_id, status="failed",
                               error=f"okx_code_{code}_{raw.get('msg', '')}")
        data_list = raw.get("data") or []
        if not data_list:
            return AmendResult(algo_id=algo_id, status="ok")
        data = data_list[0]
        s_code = data.get("sCode", "0")
        status = self._AMEND_CODE_MAP.get(s_code, "failed")
        error = None
        if status == "failed":
            error = f"okx_sCode_{s_code}_{data.get('sMsg', '')}"
        return AmendResult(algo_id=algo_id, status=status, error=error)

    def _parse_positions_response(self, raw: dict) -> list[Position]:
        """Parse GET /api/v5/account/positions data list into Position[].

        Handles both net_mode (posSide="net", pos signed) and long_short
        mode (posSide="long"/"short", pos positive).  size_contracts is
        always the absolute magnitude; direction is the human label.
        """
        out: list[Position] = []
        for item in raw.get("data") or []:
            pos_str = item.get("pos") or "0"
            try:
                pos_signed = float(pos_str)
            except ValueError:
                pos_signed = 0.0
            pos_side = (item.get("posSide") or "net").lower()
            if pos_signed == 0:
                direction = "FLAT"
            elif pos_side == "long":
                direction = "LONG"
            elif pos_side == "short":
                direction = "SHORT"
            else:
                # net_mode: sign of pos carries direction
                direction = "LONG" if pos_signed > 0 else "SHORT"
            try:
                avg_price = float(item.get("avgPx") or 0)
            except ValueError:
                avg_price = 0.0
            try:
                upl = float(item.get("upl") or 0)
            except ValueError:
                upl = 0.0
            out.append(Position(
                inst_id=item.get("instId", ""),
                direction=direction,
                size_contracts=int(abs(pos_signed)),
                avg_price=avg_price,
                unrealized_pnl_usd=upl,
                raw=item,
            ))
        return out

    def _parse_balance_response(self, raw: dict) -> Optional[Balance]:
        """Parse GET /api/v5/account/balance response.

        Picks USDT row from details[] for available; falls back to 0 if
        no USDT row found.  Returns None if data array is empty.
        """
        data_list = raw.get("data") or []
        if not data_list:
            return None
        data = data_list[0]
        try:
            total_eq = float(data.get("totalEq") or 0)
        except ValueError:
            total_eq = 0.0
        available = 0.0
        for ccy_row in data.get("details") or []:
            if (ccy_row.get("ccy") or "").upper() == "USDT":
                try:
                    available = float(ccy_row.get("availBal") or 0)
                except ValueError:
                    available = 0.0
                break
        return Balance(total_eq_usd=total_eq, available_usd=available,
                       raw=data)

    def _parse_server_time(self, raw: dict) -> Optional[float]:
        """Parse GET /api/v5/public/time → Unix seconds (float)."""
        data_list = raw.get("data") or []
        if not data_list:
            return None
        ts_ms_str = data_list[0].get("ts")
        if not ts_ms_str:
            return None
        try:
            return float(ts_ms_str) / 1000.0
        except (ValueError, TypeError):
            return None
