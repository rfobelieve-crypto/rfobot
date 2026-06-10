"""Unit tests for indicator/okx/rest.py.

Focus: connection-resilience (retry, backoff, circuit breaker), signing,
and idempotency-key generation.  We do NOT hit real OKX — every test
mocks requests.Session.

Per CLAUDE.md "Hard kill switches 必須先驗證能觸發": the circuit breaker
tests below are part of that gate.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.rest import CircuitBreaker, OkxRestClient, make_cl_ord_id


def _mk_resp(status_code: int, json_data: dict | None = None):
    """Build a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {"code": "0", "data": [{}]}
    resp.text = "fake-body"
    return resp


def _mk_client(*, is_simulated=1, session=None) -> OkxRestClient:
    cfg = OkxConfig(
        api_key="k", api_secret="s", passphrase="p",
        is_simulated=is_simulated,
        telegram_critical_chat_id="critical",
    )
    return OkxRestClient(cfg, session=session, timeout_sec=1.0)


# ── make_cl_ord_id ────────────────────────────────────────────────────


class TestClOrdIdGenerator:
    def test_default_prefix(self):
        cid = make_cl_ord_id()
        assert cid.startswith("v7")

    def test_custom_prefix(self):
        cid = make_cl_ord_id(prefix="v7a")
        assert cid.startswith("v7a")

    def test_two_calls_yield_unique_ids(self):
        ids = {make_cl_ord_id() for _ in range(50)}
        assert len(ids) == 50

    def test_length_under_okx_limit(self):
        # OKX clOrdId max length = 32
        cid = make_cl_ord_id()
        assert len(cid) <= 32


# ── CircuitBreaker ────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_starts_untripped(self):
        cb = CircuitBreaker(threshold=3, window_sec=60, cooldown_sec=30)
        assert not cb.is_tripped()

    def test_under_threshold_does_not_trip(self):
        cb = CircuitBreaker(threshold=3, window_sec=60, cooldown_sec=30)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_tripped()

    def test_at_threshold_trips(self):
        cb = CircuitBreaker(threshold=3, window_sec=60, cooldown_sec=30)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_tripped()

    def test_recovers_after_cooldown(self):
        cb = CircuitBreaker(threshold=2, window_sec=60, cooldown_sec=1)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_tripped()
        time.sleep(1.1)
        assert not cb.is_tripped()

    def test_old_failures_drop_out_of_window(self):
        """Failures older than window_sec should not count toward threshold."""
        cb = CircuitBreaker(threshold=3, window_sec=1, cooldown_sec=30)
        cb.record_failure()
        cb.record_failure()
        time.sleep(1.1)
        # These 2 old failures are now outside window
        cb.record_failure()  # only 1 recent
        assert not cb.is_tripped()


# ── Signing / Headers ────────────────────────────────────────────────


class TestSigningHeaders:
    def test_signature_deterministic_for_same_input(self):
        client = _mk_client()
        sig1 = client._sign("2026-05-27T12:00:00Z", "POST",
                            "/api/v5/trade/order", '{"a":1}')
        sig2 = client._sign("2026-05-27T12:00:00Z", "POST",
                            "/api/v5/trade/order", '{"a":1}')
        assert sig1 == sig2

    def test_signature_changes_with_body(self):
        client = _mk_client()
        sig1 = client._sign("2026-05-27T12:00:00Z", "POST", "/p", '{"a":1}')
        sig2 = client._sign("2026-05-27T12:00:00Z", "POST", "/p", '{"a":2}')
        assert sig1 != sig2

    def test_authed_headers_contain_required_fields(self):
        client = _mk_client()
        headers = client._headers("POST", "/p", '{"a":1}', auth=True)
        for key in ("OK-ACCESS-KEY", "OK-ACCESS-SIGN",
                    "OK-ACCESS-TIMESTAMP", "OK-ACCESS-PASSPHRASE",
                    "Content-Type"):
            assert key in headers

    def test_unauthed_headers_omit_credentials(self):
        client = _mk_client()
        headers = client._headers("GET", "/p", "", auth=False)
        assert "OK-ACCESS-KEY" not in headers
        assert "OK-ACCESS-SIGN" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_testnet_header_injected_when_simulated(self):
        client = _mk_client(is_simulated=1)
        headers = client._headers("POST", "/p", '{"a":1}', auth=True)
        assert headers.get("x-simulated-trading") == "1"

    def test_live_does_not_inject_testnet_header(self):
        client = _mk_client(is_simulated=0)
        headers = client._headers("POST", "/p", '{"a":1}', auth=True)
        assert "x-simulated-trading" not in headers


# ── Retry / backoff (no real sleep) ──────────────────────────────────


class TestSignedPathIncludesQuery:
    """Regression test for OKX 50113 Invalid Sign on GETs with params.

    OKX requires the HMAC requestPath segment to include the query
    string.  Previously _retry signed just the path; positions endpoint
    failed because it carries ?instId=BTC-USDT-SWAP.
    """

    def test_get_with_params_signs_path_with_query(self):
        captured = {}
        session = MagicMock()

        def fake_request(**kwargs):
            captured["headers"] = kwargs["headers"]
            captured["url"] = kwargs["url"]
            captured["params"] = kwargs.get("params")
            return _mk_resp(200, {"code": "0", "data": []})

        session.request.side_effect = fake_request
        client = _mk_client(session=session)

        # Patch _sign to capture what path it was called with
        signed_paths: list[str] = []
        original_sign = client._sign

        def capturing_sign(ts, method, path, body):
            signed_paths.append(path)
            return original_sign(ts, method, path, body)

        client._sign = capturing_sign
        client._retry_get(path="/api/v5/account/positions",
                          params={"instId": "BTC-USDT-SWAP"},
                          retries=0, backoff_base=0.01)
        assert signed_paths, "sign was never called"
        assert "instId=BTC-USDT-SWAP" in signed_paths[0]

    def test_get_without_params_signs_path_unchanged(self):
        session = MagicMock()
        session.request.return_value = _mk_resp(200, {"code": "0", "data": []})
        client = _mk_client(session=session)
        signed_paths: list[str] = []
        orig = client._sign

        def cap(ts, method, path, body):
            signed_paths.append(path)
            return orig(ts, method, path, body)

        client._sign = cap
        client._retry_get(path="/api/v5/account/balance", params={},
                          retries=0, backoff_base=0.01)
        assert signed_paths[0] == "/api/v5/account/balance"


class TestRetryLogic:
    def test_200_success_no_retry(self):
        session = MagicMock()
        session.request.return_value = _mk_resp(200)
        client = _mk_client(session=session)
        result = client._retry_get(path="/p", params={},
                                   retries=3, backoff_base=0.01)
        assert result is not None
        assert session.request.call_count == 1

    def test_5xx_triggers_retry(self):
        session = MagicMock()
        session.request.side_effect = [
            _mk_resp(500),
            _mk_resp(500),
            _mk_resp(200, {"code": "0", "data": []}),
        ]
        client = _mk_client(session=session)
        with patch("indicator.okx.rest.time.sleep"):
            result = client._retry_get(path="/p", params={},
                                       retries=3, backoff_base=0.01)
        assert result is not None
        assert session.request.call_count == 3

    def test_4xx_no_retry_when_disabled(self):
        # Order submission: 4xx means malformed request, never retry
        session = MagicMock()
        session.request.return_value = _mk_resp(400)
        client = _mk_client(session=session)
        result = client._retry_post(path="/p", body={"a": 1},
                                    retries=3, backoff_base=0.01,
                                    retry_on_4xx=False)
        assert result is None
        assert session.request.call_count == 1  # no retries

    def test_4xx_does_retry_when_enabled(self):
        session = MagicMock()
        session.request.side_effect = [
            _mk_resp(400), _mk_resp(400), _mk_resp(200),
        ]
        client = _mk_client(session=session)
        with patch("indicator.okx.rest.time.sleep"):
            result = client._retry_get(path="/p", params={},
                                       retries=3, backoff_base=0.01)
        # GETs default to retry_on_4xx=True
        assert result is not None

    def test_all_retries_exhausted_returns_none(self):
        session = MagicMock()
        session.request.return_value = _mk_resp(500)
        client = _mk_client(session=session)
        with patch("indicator.okx.rest.time.sleep"):
            result = client._retry_post(path="/p", body={"a": 1},
                                        retries=2, backoff_base=0.01,
                                        retry_on_4xx=False)
        assert result is None
        # 1 initial + 2 retries = 3 attempts
        assert session.request.call_count == 3

    def test_circuit_breaker_short_circuits_calls(self):
        # If breaker is tripped, no request goes out
        session = MagicMock()
        client = _mk_client(session=session)
        client._breaker._tripped_until = time.time() + 60  # forced trip
        result = client._retry_get(path="/p", params={},
                                   retries=3, backoff_base=0.01)
        assert result is None
        assert session.request.call_count == 0

    def test_timeout_exception_treated_as_failure(self):
        import requests
        session = MagicMock()
        session.request.side_effect = requests.exceptions.Timeout("slow")
        client = _mk_client(session=session)
        with patch("indicator.okx.rest.time.sleep"):
            result = client._retry_get(path="/p", params={},
                                       retries=2, backoff_base=0.01)
        assert result is None

    def test_connection_error_treated_as_failure(self):
        import requests
        session = MagicMock()
        session.request.side_effect = requests.exceptions.ConnectionError(
            "no route"
        )
        client = _mk_client(session=session)
        with patch("indicator.okx.rest.time.sleep"):
            result = client._retry_get(path="/p", params={},
                                       retries=1, backoff_base=0.01)
        assert result is None

    def test_circuit_trips_after_repeated_5xx(self):
        # Default CircuitBreaker threshold = 5.
        # 1 _retry call with retries=10 will exceed it, tripping the
        # breaker mid-loop.  After tripping, subsequent attempts skip.
        session = MagicMock()
        session.request.return_value = _mk_resp(500)
        client = _mk_client(session=session)
        with patch("indicator.okx.rest.time.sleep"):
            client._retry_post(path="/p", body={"a": 1},
                               retries=10, backoff_base=0.001,
                               retry_on_4xx=False)
        # Once breaker is tripped, follow-up calls return None immediately
        result = client._retry_get(path="/p", params={},
                                   retries=3, backoff_base=0.01)
        assert result is None
        assert client.is_circuit_tripped()


# ── posSide / reduceOnly body fields ──────────────────────────────────


class TestOrderBodyFields:
    """Verifies long_short_mode kwargs land in the OKX request body."""

    def _capture_body(self):
        """Patch session and return the captured request body dict."""
        session = MagicMock()
        session.request.return_value = _mk_resp(200,
            {"code": "0", "data": [{"ordId": "1", "sCode": "0"}]})
        client = _mk_client(session=session)
        return client, session

    def test_market_order_omits_pos_side_when_none(self):
        from indicator.okx.types import Side
        client, session = self._capture_body()
        client.submit_market_order(
            inst_id="BTC-USDT-SWAP", side=Side.BUY, sz=1,
            td_mode="cross", cl_ord_id="v7-x", pos_side=None,
        )
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert "posSide" not in body
        assert "reduceOnly" not in body

    def test_market_order_includes_pos_side_long(self):
        from indicator.okx.types import Side
        client, session = self._capture_body()
        client.submit_market_order(
            inst_id="BTC-USDT-SWAP", side=Side.BUY, sz=1,
            td_mode="cross", cl_ord_id="v7-x", pos_side="long",
        )
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert body["posSide"] == "long"

    def test_market_order_includes_reduce_only_when_true(self):
        from indicator.okx.types import Side
        client, session = self._capture_body()
        client.submit_market_order(
            inst_id="BTC-USDT-SWAP", side=Side.SELL, sz=1,
            td_mode="cross", cl_ord_id="v7-x",
            pos_side="long", reduce_only=True,
        )
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert body["reduceOnly"] is True
        assert body["posSide"] == "long"

    def test_algo_stop_includes_pos_side_and_reduce_only(self):
        from indicator.okx.types import Side
        client, session = self._capture_body()
        client.submit_algo_stop(
            inst_id="BTC-USDT-SWAP", side=Side.SELL, sz=1,
            trigger_px=74500.0, td_mode="cross",
            algo_cl_ord_id="v7a-x", pos_side="long", reduce_only=True,
        )
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert body["posSide"] == "long"
        assert body["reduceOnly"] is True


# ── set_leverage (isolated margin prerequisite) ───────────────────────


class TestSetLeverage:
    def _client_with(self, resp):
        session = MagicMock()
        session.request.return_value = resp
        return _mk_client(session=session), session

    def test_set_leverage_isolated_body_and_success(self):
        client, session = self._client_with(_mk_resp(200,
            {"code": "0", "data": [{"lever": "10", "mgnMode": "isolated"}]}))
        ok = client.set_leverage(inst_id="BTC-USDT-SWAP", lever=10,
                                 mgn_mode="isolated", pos_side="long")
        assert ok is True
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert body == {"instId": "BTC-USDT-SWAP", "lever": "10",
                        "mgnMode": "isolated", "posSide": "long"}

    def test_set_leverage_cross_omits_pos_side(self):
        client, session = self._client_with(_mk_resp(200,
            {"code": "0", "data": [{}]}))
        client.set_leverage(inst_id="BTC-USDT-SWAP", lever=5, mgn_mode="cross")
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert "posSide" not in body

    def test_set_leverage_nonzero_code_is_failure(self):
        client, _ = self._client_with(_mk_resp(200,
            {"code": "59000", "msg": "position exists", "data": []}))
        ok = client.set_leverage(inst_id="BTC-USDT-SWAP", lever=10,
                                 mgn_mode="isolated", pos_side="long")
        assert ok is False


class TestAmendAlgoBody:
    """Regression: amend-algos body MUST include instId (2026-06-10 bug —
    missing instId → 50014, trailing stop never moved on OKX)."""

    def test_amend_body_includes_instid(self):
        session = MagicMock()
        session.request.return_value = _mk_resp(200,
            {"code": "0", "data": [{"algoId": "a1", "sCode": "0"}]})
        client = _mk_client(session=session)
        res = client.amend_algo_stop(inst_id="BTC-USDT-SWAP", algo_id="a1",
                                     new_trigger_px=62560.5)
        assert res.status == "ok"
        import json
        body = json.loads(session.request.call_args.kwargs["data"])
        assert body["instId"] == "BTC-USDT-SWAP"
        assert body["algoId"] == "a1"
        assert body["newSlTriggerPx"] == "62560.5"
        assert body["newSlOrdPx"] == "-1"

    def test_amend_missing_instid_would_fail(self):
        # OKX returns sCode 50014 when instId is absent; parser → failed.
        session = MagicMock()
        session.request.return_value = _mk_resp(200,
            {"code": "1", "data": [{"algoId": "a1", "sCode": "50014",
                                    "sMsg": "Parameter instId can not be empty."}]})
        client = _mk_client(session=session)
        res = client.amend_algo_stop(inst_id="BTC-USDT-SWAP", algo_id="a1",
                                     new_trigger_px=1.0)
        assert res.status == "failed"


# ── Latency tracking ──────────────────────────────────────────────────


class TestLatencyTracking:
    def test_last_latency_recorded_after_call(self):
        session = MagicMock()
        session.request.return_value = _mk_resp(200)
        client = _mk_client(session=session)
        assert client.last_latency_ms() is None
        client._retry_get(path="/p", params={}, retries=0, backoff_base=0.01)
        assert client.last_latency_ms() is not None
        assert client.last_latency_ms() >= 0
