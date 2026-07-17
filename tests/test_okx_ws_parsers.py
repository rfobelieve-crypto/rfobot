"""Tests for WS private message parsers in indicator/okx/ws_private.py.

We don't open a real WS — we directly call the dispatcher with a JSON
string and assert the callback receives the right typed event.

Real OKX v5 wire format reproduced from OKX docs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.types import BalanceEvent, OrderEvent, PositionEvent
from indicator.okx.ws_private import OkxWsPrivateClient


def _mk_ws() -> OkxWsPrivateClient:
    cfg = OkxConfig(api_key="k", api_secret="s", passphrase="p",
                    telegram_critical_chat_id="critical")
    return OkxWsPrivateClient(cfg)


# ── URL selection by simulated flag ───────────────────────────────────


class TestTestnetUrl:
    def test_simulated_uses_testnet_url(self):
        cfg = OkxConfig(api_key="k", api_secret="s", passphrase="p",
                        telegram_critical_chat_id="critical",
                        is_simulated=1)
        ws = OkxWsPrivateClient(cfg)
        url = ws._ws_url()
        # OKX testnet WS uses wspap.okx.com per OKX docs
        assert "wspap" in url

    def test_live_uses_prod_url(self):
        cfg = OkxConfig(api_key="k", api_secret="s", passphrase="p",
                        telegram_critical_chat_id="critical",
                        is_simulated=0)
        ws = OkxWsPrivateClient(cfg)
        url = ws._ws_url()
        # Live = the configured ws_private URL untouched
        assert "wspap" not in url
        assert url == cfg.ws_private


# ── Heartbeat frames (2026-07-18 live log flood) ─────────────────────


class TestHeartbeatFrames:
    """Non-JSON heartbeat frames must be ignored silently — on live every
    "pong"/empty frame logged a full JSONDecodeError traceback
    (~130/15min), drowning the errors that matter."""

    @pytest.mark.parametrize("frame", ["pong", "ping", "", " ", "pong\n"])
    def test_heartbeat_frame_ignored(self, frame):
        ws = _mk_ws()
        ws._handle_message(frame)          # must not raise

    def test_genuine_garbage_still_raises_for_on_message_to_log(self):
        ws = _mk_ws()
        with pytest.raises(json.JSONDecodeError):
            ws._handle_message("not-json-at-all")


# ── Order channel dispatch ────────────────────────────────────────────


class TestOrderEventParsing:
    def test_filled_order_decoded(self):
        ws = _mk_ws()
        captured: list[OrderEvent] = []
        ws.subscribe_orders(captured.append)
        # Force subscribe to wire callback even though not connected
        # (subscribe_orders sets _on_order regardless)
        raw_msg = {
            "arg": {"channel": "orders", "instType": "SWAP"},
            "data": [{
                "ordId": "312269865356374016",
                "clOrdId": "v7-abc",
                "state": "filled",
                "fillPx": "75123.5",
                "fillSz": "5",
                "accFillSz": "5",
                "sz": "5",
                "fee": "-0.075",
                "feeCcy": "USDT",
                "instId": "BTC-USDT-SWAP",
                "uTime": "1779892800000",   # 2026-05-27 12:00:00 UTC
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert len(captured) == 1
        evt = captured[0]
        assert evt.ord_id == "312269865356374016"
        assert evt.cl_ord_id == "v7-abc"
        assert evt.state == "filled"
        assert evt.fill_price == pytest.approx(75123.5)
        assert evt.fill_size == 5
        # fee comes back signed: OKX reports negative when user pays
        assert evt.fee_usd == pytest.approx(-0.075)
        assert evt.ts is not None
        assert evt.ts.year == 2026 and evt.ts.month == 5

    def test_live_order_no_fill_yet(self):
        ws = _mk_ws()
        captured: list[OrderEvent] = []
        ws.subscribe_orders(captured.append)
        raw_msg = {
            "arg": {"channel": "orders", "instType": "SWAP"},
            "data": [{
                "ordId": "312269865356374016",
                "clOrdId": "v7-abc",
                "state": "live",
                "fillPx": "",
                "fillSz": "0",
                "fee": "0",
                "instId": "BTC-USDT-SWAP",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        evt = captured[0]
        assert evt.state == "live"
        assert evt.fill_price is None
        assert evt.fill_size == 0

    def test_partially_filled(self):
        ws = _mk_ws()
        captured: list[OrderEvent] = []
        ws.subscribe_orders(captured.append)
        raw_msg = {
            "arg": {"channel": "orders", "instType": "SWAP"},
            "data": [{
                "ordId": "999", "clOrdId": "v7-p",
                "state": "partially_filled",
                "fillPx": "75000.0",
                "fillSz": "2",
                "accFillSz": "3",
                "sz": "5",
                "fee": "-0.03",
                "instId": "BTC-USDT-SWAP",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert captured[0].state == "partially_filled"
        assert captured[0].fill_size == 2

    def test_canceled_order(self):
        ws = _mk_ws()
        captured: list[OrderEvent] = []
        ws.subscribe_orders(captured.append)
        raw_msg = {
            "arg": {"channel": "orders", "instType": "SWAP"},
            "data": [{
                "ordId": "111", "clOrdId": "v7-c",
                "state": "canceled",
                "fillPx": "", "fillSz": "0", "fee": "0",
                "instId": "BTC-USDT-SWAP",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert captured[0].state == "canceled"

    def test_missing_ts_does_not_raise(self):
        ws = _mk_ws()
        captured: list[OrderEvent] = []
        ws.subscribe_orders(captured.append)
        raw_msg = {
            "arg": {"channel": "orders", "instType": "SWAP"},
            "data": [{
                "ordId": "555", "clOrdId": "v7-x",
                "state": "live", "instId": "BTC-USDT-SWAP",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert captured[0].ts is None


# ── Position channel dispatch ─────────────────────────────────────────


class TestPositionEventParsing:
    def test_long_position_push(self):
        ws = _mk_ws()
        captured: list[PositionEvent] = []
        ws.subscribe_positions(captured.append)
        raw_msg = {
            "arg": {"channel": "positions", "instType": "SWAP"},
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "posSide": "net",
                "pos": "5",
                "avgPx": "75000.0",
                "uTime": "1779892800000",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        evt = captured[0]
        assert evt.inst_id == "BTC-USDT-SWAP"
        assert evt.pos == 5.0
        assert evt.avg_price == pytest.approx(75000.0)
        assert evt.ts is not None

    def test_short_position_negative_pos(self):
        ws = _mk_ws()
        captured: list[PositionEvent] = []
        ws.subscribe_positions(captured.append)
        raw_msg = {
            "arg": {"channel": "positions", "instType": "SWAP"},
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "posSide": "net",
                "pos": "-3",
                "avgPx": "75000.0",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        # pos field preserved with sign — direction inferred upstream
        assert captured[0].pos == -3.0

    def test_flat_position_push(self):
        ws = _mk_ws()
        captured: list[PositionEvent] = []
        ws.subscribe_positions(captured.append)
        raw_msg = {
            "arg": {"channel": "positions"},
            "data": [{
                "instId": "BTC-USDT-SWAP",
                "pos": "0", "avgPx": "0",
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert captured[0].pos == 0.0


# ── Account channel dispatch ──────────────────────────────────────────


class TestBalanceEventParsing:
    def test_typical_account_push(self):
        ws = _mk_ws()
        captured: list[BalanceEvent] = []
        ws.subscribe_balance(captured.append)
        raw_msg = {
            "arg": {"channel": "account"},
            "data": [{
                "totalEq": "1054.36",
                "adjEq": "1054.36",
                "uTime": "1779892800000",
                "details": [{
                    "ccy": "USDT",
                    "availBal": "950.0",
                    "eq": "1054.36",
                }],
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        evt = captured[0]
        assert evt.total_eq_usd == pytest.approx(1054.36)
        assert evt.available_usd == pytest.approx(950.0)
        assert evt.ts is not None

    def test_multi_ccy_picks_usdt_available(self):
        ws = _mk_ws()
        captured: list[BalanceEvent] = []
        ws.subscribe_balance(captured.append)
        raw_msg = {
            "arg": {"channel": "account"},
            "data": [{
                "totalEq": "1054.36",
                "details": [
                    {"ccy": "BTC", "availBal": "0.01"},
                    {"ccy": "USDT", "availBal": "950.0"},
                ],
            }],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert captured[0].available_usd == pytest.approx(950.0)

    def test_no_details_available_is_zero(self):
        ws = _mk_ws()
        captured: list[BalanceEvent] = []
        ws.subscribe_balance(captured.append)
        raw_msg = {
            "arg": {"channel": "account"},
            "data": [{"totalEq": "100.0", "details": []}],
        }
        ws._handle_message(json.dumps(raw_msg))
        assert captured[0].total_eq_usd == pytest.approx(100.0)
        assert captured[0].available_usd == 0.0


# ── Robustness: bad messages should not crash the dispatcher ─────────


class TestBadMessages:
    def test_malformed_json_swallowed_by_on_message(self):
        # _handle_message does json.loads internally; bad JSON raises,
        # but on_message wraps it with try/except.  We test that
        # _handle_message bubbles JSONDecodeError (so the wrapper test
        # in on_message can absorb it).
        ws = _mk_ws()
        with pytest.raises(json.JSONDecodeError):
            ws._handle_message("not-valid-json")

    def test_unknown_channel_does_not_raise(self):
        ws = _mk_ws()
        raw_msg = {
            "arg": {"channel": "nonsense"},
            "data": [{"foo": "bar"}],
        }
        # Must not raise even though no handler matches
        ws._handle_message(json.dumps(raw_msg))

    def test_subscribe_ack_ignored(self):
        ws = _mk_ws()
        ws._handle_message(json.dumps({"event": "subscribe",
                                        "arg": {"channel": "orders"}}))

    def test_pong_ignored(self):
        ws = _mk_ws()
        ws._handle_message(json.dumps({"event": "pong"}))
