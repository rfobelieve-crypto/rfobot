"""Tests for REST response parsers in indicator/okx/rest.py.

Real OKX v5 API response shapes (per OKX docs); we feed them through
the parser helpers and assert the typed dataclass fields come out right.
"""
from __future__ import annotations

import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.rest import OkxRestClient
from indicator.okx.types import (
    AlgoOrderResult,
    AmendResult,
    Balance,
    OrderResult,
    Position,
    Side,
)


def _mk_client() -> OkxRestClient:
    cfg = OkxConfig(api_key="k", api_secret="s", passphrase="p",
                    telegram_critical_chat_id="critical")
    return OkxRestClient(cfg)


# ── _parse_order_response ─────────────────────────────────────────────


class TestParseOrderResponse:
    def test_success_response_populates_ord_id(self):
        raw = {
            "code": "0",
            "msg": "",
            "data": [{
                "ordId": "312269865356374016",
                "clOrdId": "v7-abc",
                "tag": "",
                "sCode": "0",
                "sMsg": "",
            }],
        }
        client = _mk_client()
        result = client._parse_order_response(raw, "v7-abc")
        assert isinstance(result, OrderResult)
        assert result.cl_ord_id == "v7-abc"
        assert result.ord_id == "312269865356374016"
        assert result.status == "submitted"
        assert result.error is None

    def test_top_level_error_code_marks_rejected(self):
        raw = {
            "code": "51008",
            "msg": "Insufficient balance",
            "data": [],
        }
        client = _mk_client()
        result = client._parse_order_response(raw, "v7-abc")
        assert result.status == "rejected"
        assert "51008" in (result.error or "")
        assert "Insufficient" in (result.error or "")

    def test_per_order_sCode_error_marks_rejected(self):
        # OKX returns top code=0 but per-order sCode!=0
        raw = {
            "code": "0",
            "msg": "",
            "data": [{
                "ordId": "",
                "clOrdId": "v7-abc",
                "sCode": "51008",
                "sMsg": "Insufficient balance",
            }],
        }
        client = _mk_client()
        result = client._parse_order_response(raw, "v7-abc")
        assert result.status == "rejected"
        assert "51008" in (result.error or "")

    def test_empty_data_array_handled(self):
        raw = {"code": "0", "msg": "", "data": []}
        client = _mk_client()
        result = client._parse_order_response(raw, "v7-abc")
        # Treat as submitted with no ord_id known yet
        assert result.cl_ord_id == "v7-abc"
        assert result.ord_id is None


# ── submit_algo_stop response parsing ─────────────────────────────────


class TestParseAlgoOrderResponse:
    def test_success_populates_algo_id(self):
        raw = {
            "code": "0",
            "msg": "",
            "data": [{
                "algoId": "555111222333",
                "algoClOrdId": "v7a-xyz",
                "sCode": "0",
                "sMsg": "",
            }],
        }
        client = _mk_client()
        result = client._parse_algo_order_response(raw, "v7a-xyz")
        assert isinstance(result, AlgoOrderResult)
        assert result.algo_cl_ord_id == "v7a-xyz"
        assert result.algo_id == "555111222333"
        assert result.status == "submitted"
        assert result.error is None

    def test_per_order_sCode_failure_marks_rejected(self):
        raw = {
            "code": "0",
            "data": [{
                "algoId": "",
                "algoClOrdId": "v7a-xyz",
                "sCode": "51400",
                "sMsg": "Already filled",
            }],
        }
        client = _mk_client()
        result = client._parse_algo_order_response(raw, "v7a-xyz")
        assert result.status == "rejected"
        assert "51400" in (result.error or "")


# ── amend_algo_stop response parsing ──────────────────────────────────


class TestParseAmendResponse:
    def test_success_returns_ok(self):
        raw = {
            "code": "0",
            "data": [{"algoId": "555", "sCode": "0", "sMsg": ""}],
        }
        client = _mk_client()
        result = client._parse_amend_response(raw, "555")
        assert isinstance(result, AmendResult)
        assert result.status == "ok"
        assert result.algo_id == "555"

    def test_51400_already_filled_recognized(self):
        # Per rest.py L181 TODO: "Handle 51400 already_filled"
        raw = {
            "code": "0",
            "data": [{
                "algoId": "555",
                "sCode": "51400",
                "sMsg": "Order has been filled",
            }],
        }
        client = _mk_client()
        result = client._parse_amend_response(raw, "555")
        assert result.status == "already_filled"

    def test_not_found_recognized(self):
        # Per OKX: 51401 = order does not exist
        raw = {
            "code": "0",
            "data": [{
                "algoId": "555",
                "sCode": "51401",
                "sMsg": "Order does not exist",
            }],
        }
        client = _mk_client()
        result = client._parse_amend_response(raw, "555")
        assert result.status == "not_found"

    def test_other_failure_marked_failed(self):
        raw = {
            "code": "0",
            "data": [{
                "algoId": "555",
                "sCode": "99999",
                "sMsg": "Unknown error",
            }],
        }
        client = _mk_client()
        result = client._parse_amend_response(raw, "555")
        assert result.status == "failed"
        assert "99999" in (result.error or "")


# ── get_positions response parsing ────────────────────────────────────


class TestParsePositions:
    def test_empty_positions(self):
        raw = {"code": "0", "data": []}
        client = _mk_client()
        positions = client._parse_positions_response(raw)
        assert positions == []

    def test_net_mode_long_position(self):
        raw = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "posSide": "net",
                "pos": "5",              # +ve = long in net mode
                "avgPx": "75000.5",
                "upl": "100.0",
                "lever": "1",
            }],
        }
        client = _mk_client()
        positions = client._parse_positions_response(raw)
        assert len(positions) == 1
        p = positions[0]
        assert p.inst_id == "BTC-USDT-SWAP"
        assert p.direction == "LONG"
        assert p.size_contracts == 5
        assert p.avg_price == pytest.approx(75000.5)
        assert p.unrealized_pnl_usd == pytest.approx(100.0)

    def test_net_mode_short_position(self):
        raw = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "posSide": "net",
                "pos": "-3",             # -ve = short in net mode
                "avgPx": "75000.0",
                "upl": "-50.0",
            }],
        }
        client = _mk_client()
        positions = client._parse_positions_response(raw)
        p = positions[0]
        assert p.direction == "SHORT"
        # size_contracts is the absolute magnitude
        assert p.size_contracts == 3

    def test_flat_position_returns_flat(self):
        raw = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "posSide": "net",
                "pos": "0",
                "avgPx": "0",
            }],
        }
        client = _mk_client()
        positions = client._parse_positions_response(raw)
        assert len(positions) == 1
        assert positions[0].direction == "FLAT"
        assert positions[0].size_contracts == 0

    def test_long_short_mode_long_side(self):
        # In long_short mode, posSide is "long" or "short" and pos is +ve
        raw = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "posSide": "long",
                "pos": "5",
                "avgPx": "75000.0",
            }],
        }
        client = _mk_client()
        positions = client._parse_positions_response(raw)
        assert positions[0].direction == "LONG"

    def test_raw_preserved(self):
        raw = {
            "code": "0",
            "data": [{
                "instId": "BTC-USDT-SWAP", "posSide": "net",
                "pos": "5", "avgPx": "75000.0",
                "extra_field": "foo",
            }],
        }
        client = _mk_client()
        positions = client._parse_positions_response(raw)
        assert positions[0].raw["extra_field"] == "foo"


# ── get_balance response parsing ──────────────────────────────────────


class TestParseBalance:
    def test_typical_balance(self):
        raw = {
            "code": "0",
            "data": [{
                "totalEq": "1054.36",
                "isoEq": "0",
                "adjEq": "1054.36",
                "details": [{
                    "ccy": "USDT",
                    "availBal": "950.0",
                    "eq": "1054.36",
                    "cashBal": "1000.0",
                }],
            }],
        }
        client = _mk_client()
        balance = client._parse_balance_response(raw)
        assert isinstance(balance, Balance)
        assert balance.total_eq_usd == pytest.approx(1054.36)
        assert balance.available_usd == pytest.approx(950.0)

    def test_empty_details_uses_zero_available(self):
        raw = {
            "code": "0",
            "data": [{"totalEq": "100.0", "details": []}],
        }
        client = _mk_client()
        balance = client._parse_balance_response(raw)
        assert balance.total_eq_usd == pytest.approx(100.0)
        assert balance.available_usd == 0.0

    def test_multi_ccy_details_picks_usdt(self):
        raw = {
            "code": "0",
            "data": [{
                "totalEq": "1054.36",
                "details": [
                    {"ccy": "BTC", "availBal": "0.01"},
                    {"ccy": "USDT", "availBal": "950.0"},
                ],
            }],
        }
        client = _mk_client()
        balance = client._parse_balance_response(raw)
        assert balance.available_usd == pytest.approx(950.0)

    def test_empty_data_returns_none(self):
        raw = {"code": "0", "data": []}
        client = _mk_client()
        balance = client._parse_balance_response(raw)
        assert balance is None


# ── get_server_time parsing ───────────────────────────────────────────


class TestParseServerTime:
    def test_typical_response(self):
        raw = {
            "code": "0",
            "data": [{"ts": "1779892800000"}],   # 2026-05-27 12:00:00 UTC in ms
        }
        client = _mk_client()
        ts_sec = client._parse_server_time(raw)
        assert ts_sec == pytest.approx(1779892800.0)

    def test_empty_data_returns_none(self):
        raw = {"code": "0", "data": []}
        client = _mk_client()
        assert client._parse_server_time(raw) is None

    def test_non_numeric_returns_none(self):
        raw = {"code": "0", "data": [{"ts": "not-a-number"}]}
        client = _mk_client()
        assert client._parse_server_time(raw) is None
