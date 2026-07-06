"""Real-fee accounting + trailing-peak drawdown alert (2026-07-06 fixes).

Fee fix: net_pct was `gross - taker_cost` (flat 8 bps constant) while the
real fill fee arrived on the WS/orders channel and was dropped — Gate B was
about to be graded with a lying ruler (mistake.md 2026-06-14).  Now:
  - entry fill fee is persisted from the WS event (cumulative → overwrite)
  - manual close reads the order back via REST (real avgPx + real fee)
  - algo-stop close takes the fee straight off the fill event
  - a missing leg is estimated at cfg.taker_fee_side_est per side

Drawdown alert: M2M equity hit -21% from peak on 2026-07-02 — through the
Stage-3→4a gate line (MDD < 20%) — with zero notification, because only
daily/total caps existed.  _check_trailing_drawdown alerts (alert-only) at
warn/breach levels, dedups, and re-arms after recovery.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from indicator.okx.config import OkxConfig, validate_okx_config
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.mock_client import MOCK_TAKER_FEE_FRAC, MockOkxClient
from indicator.okx.types import (
    Balance,
    CancelResult,
    OrderDetails,
    OrderEvent,
    OrderResult,
    ReconciliationResult,
    ReconciliationVerdict,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _mk_cfg(**overrides) -> OkxConfig:
    base = OkxConfig(
        api_key="k", api_secret="s", passphrase="p",
        telegram_critical_chat_id="critical-chat",
        initial_capital_usd=100.0,
        contract_size_base=0.01,
        is_simulated=1,
        leverage=10,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _mk_executor(cfg=None):
    cfg = cfg or _mk_cfg()
    client = MagicMock()
    client.submit_market_order.return_value = OrderResult(
        cl_ord_id="v7close-x", ord_id="okx-close-1", status="filled")
    client.cancel_algo_stop.return_value = CancelResult(algo_id="x",
                                                         status="ok")
    client.get_balance.return_value = Balance(total_eq_usd=100.0,
                                              available_usd=100.0)
    client.get_order.return_value = None

    store = MagicMock()
    store.get_latest_balance.return_value = {"total_eq_usd": 100.0,
                                              "available_usd": 100.0}
    recon = MagicMock()
    recon.reconcile_cycle.return_value = ReconciliationResult(
        verdict=ReconciliationVerdict.CONSISTENT, detail={})
    exe = V7OkxExecutor(client=client, store=store, reconciler=recon,
                        cfg=cfg)
    return exe, client, store


def _pos(**overrides) -> dict:
    base = {
        "id": 7,
        "entry_time": datetime(2026, 7, 1, 0, 0, 0),
        "direction": "LONG",
        "entry_price": 60000.0,
        "atr_at_entry": 400.0,
        "stop_dist": 1200.0,
        "trail_extreme": 60000.0,
        "current_stop": 58800.0,
        "size_contracts": 0.33,
        "size_frac": 0.5,
        "notional_usd": 200.0,
        "equity_before": 100.0,
        "stop_algo_id": "okx-algo-1",
        "entry_cl_ord_id": "v7-entry-abc",
        "entry_fees_usd": 0.0,
    }
    base.update(overrides)
    return base


# ── _net_pct_with_fees ───────────────────────────────────────────────


class TestNetPctWithFees:
    def test_both_legs_real(self):
        exe, _, _ = _mk_executor()
        # 200 notional, 0.10 entry + 0.10 exit → 10 bps total
        net = exe._net_pct_with_fees(gross_pct=0.02, notional=200.0,
                                      entry_fees_usd=0.10,
                                      exit_fees_usd=0.10)
        assert net == pytest.approx(0.02 - 0.001)

    def test_entry_missing_estimated(self):
        exe, _, _ = _mk_executor()
        net = exe._net_pct_with_fees(gross_pct=0.02, notional=200.0,
                                      entry_fees_usd=0.0,
                                      exit_fees_usd=0.10)
        assert net == pytest.approx(0.02 - 0.0005 - 0.0005)

    def test_exit_missing_estimated(self):
        exe, _, _ = _mk_executor()
        net = exe._net_pct_with_fees(gross_pct=0.02, notional=200.0,
                                      entry_fees_usd=0.10,
                                      exit_fees_usd=None)
        assert net == pytest.approx(0.02 - 0.0005 - 0.0005)

    def test_no_notional_falls_back_to_legacy_constant(self):
        exe, _, _ = _mk_executor()
        net = exe._net_pct_with_fees(gross_pct=0.02, notional=0.0,
                                      entry_fees_usd=0.0,
                                      exit_fees_usd=None)
        assert net == pytest.approx(0.02 - 0.0008)


# ── manual close: REST read-back of the real fill ────────────────────


class TestCloseReadback:
    def test_readback_refines_price_and_fee(self):
        exe, client, store = _mk_executor()
        client.get_order.return_value = OrderDetails(
            ord_id="okx-close-1", state="filled",
            avg_px=61510.0, fee_usd=-0.11, acc_fill_sz=0.33)
        pos = _pos(entry_fees_usd=0.09)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._close_position(pos, exit_price=61500.0,
                                          exit_reason="opp_signal",
                                          bar_ts=datetime(2026, 7, 6))
        # exit price refined from the bar-close estimate to the real fill
        assert result.detail["exit_price"] == pytest.approx(61510.0)
        gross = 61510.0 / 60000.0 - 1.0
        net = gross - (0.09 + 0.11) / 200.0
        assert result.detail["gross_pct"] == pytest.approx(gross)
        assert result.detail["net_pct"] == pytest.approx(net)
        # real exit fee persisted on the row
        close_kwargs = store.close_position.call_args.kwargs
        assert close_kwargs["exit_fees_usd"] == pytest.approx(0.11)

    def test_readback_unavailable_falls_back_to_estimates(self):
        exe, client, store = _mk_executor()
        client.get_order.return_value = None
        pos = _pos()
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._close_position(pos, exit_price=61500.0,
                                          exit_reason="opp_signal",
                                          bar_ts=datetime(2026, 7, 6))
        gross = 61500.0 / 60000.0 - 1.0
        assert result.detail["net_pct"] == pytest.approx(gross - 0.001)
        close_kwargs = store.close_position.call_args.kwargs
        assert close_kwargs["exit_fees_usd"] == 0.0

    def test_mock_client_readback_end_to_end(self):
        """Faithful MockOkxClient: submit then read back the same order —
        the registry must return the real fill with an OKX-signed fee."""
        from indicator.okx.types import Side
        cfg = _mk_cfg()
        mock = MockOkxClient(cfg, mark_price=60000.0)
        res = mock.submit_market_order(
            inst_id=cfg.inst_id, side=Side.BUY, sz=0.33,
            td_mode="isolated", cl_ord_id="v7-e2e")
        details = mock.get_order(inst_id=cfg.inst_id, ord_id=res.ord_id)
        assert details is not None and details.state == "filled"
        assert details.avg_px == pytest.approx(res.fill_price)
        notional = res.fill_price * 0.33 * cfg.contract_size_base
        assert details.fee_usd == pytest.approx(-notional * MOCK_TAKER_FEE_FRAC)


# ── algo-stop close: fee straight off the WS fill event ──────────────


class TestAlgoFillFees:
    def _evt(self, fee=-0.12, px=58800.0) -> OrderEvent:
        return OrderEvent(cl_ord_id="", ord_id="okx-af-1", state="filled",
                          fill_price=px, fill_size=0.33, fee_usd=fee,
                          algo_cl_ord_id="v7a-stop", algo_id="okx-algo-1")

    def test_uses_event_fee_and_persisted_entry_fee(self):
        exe, _, store = _mk_executor()
        pos = _pos(entry_fees_usd=0.08)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._sync_close_from_algo_fill(pos, self._evt(fee=-0.12))
        kwargs = store.close_position.call_args.kwargs
        gross = 58800.0 / 60000.0 - 1.0
        assert kwargs["gross_pct"] == pytest.approx(gross)
        assert kwargs["net_pct"] == pytest.approx(gross - (0.08 + 0.12) / 200.0)
        assert kwargs["exit_fees_usd"] == pytest.approx(0.12)

    def test_missing_event_fee_estimates_exit_leg(self):
        exe, _, store = _mk_executor()
        pos = _pos(entry_fees_usd=0.08)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._sync_close_from_algo_fill(pos, self._evt(fee=None))
        kwargs = store.close_position.call_args.kwargs
        gross = 58800.0 / 60000.0 - 1.0
        assert kwargs["net_pct"] == pytest.approx(
            gross - 0.08 / 200.0 - 0.0005)
        assert kwargs["exit_fees_usd"] == 0.0


# ── WS wiring: entry fill persists the real entry fee ────────────────


class TestEntryFeeWiring:
    def test_entry_fill_event_persists_fee(self):
        cfg = _mk_cfg()
        mock = MockOkxClient(cfg, mark_price=60000.0)
        store = MagicMock()
        store.get_open_position.return_value = _pos()
        recon = MagicMock()
        exe = V7OkxExecutor(client=mock, store=store, reconciler=recon,
                            cfg=cfg)
        exe._wire_ws_callbacks()
        assert mock._on_order is not None
        mock._on_order(OrderEvent(
            cl_ord_id="v7-entry-abc", ord_id="okx-e-1", state="filled",
            fill_price=60000.0, fill_size=0.33, fee_usd=-0.0987))
        store.set_entry_fees.assert_called_once()
        kwargs = store.set_entry_fees.call_args.kwargs
        assert kwargs["position_id"] == 7
        assert kwargs["entry_fees_usd"] == pytest.approx(0.0987)

    def test_non_matching_event_does_not_touch_fees(self):
        cfg = _mk_cfg()
        mock = MockOkxClient(cfg, mark_price=60000.0)
        store = MagicMock()
        store.get_open_position.return_value = _pos()
        exe = V7OkxExecutor(client=mock, store=store,
                            reconciler=MagicMock(), cfg=cfg)
        exe._wire_ws_callbacks()
        mock._on_order(OrderEvent(
            cl_ord_id="someone-else", ord_id="okx-e-2", state="filled",
            fill_price=60000.0, fill_size=0.33, fee_usd=-0.05))
        store.set_entry_fees.assert_not_called()


# ── trailing-peak drawdown alert ─────────────────────────────────────


class TestTrailingDrawdownAlert:
    def _exe_with_peak(self, peak):
        exe, _, store = _mk_executor()
        store.get_peak_equity.return_value = peak
        return exe, store

    def test_no_alert_above_warn(self):
        exe, _ = self._exe_with_peak(100.0)
        with patch("indicator.okx.executor.send_critical") as tg:
            exe._check_trailing_drawdown(90.0)   # -10%
        tg.assert_not_called()

    def test_warn_fires_once_and_dedups(self):
        exe, _ = self._exe_with_peak(100.0)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            exe._check_trailing_drawdown(84.0)   # -16% → WARN
            exe._check_trailing_drawdown(83.5)   # still WARN band → no dup
        assert tg.call_count == 1
        assert "warning" in tg.call_args_list[0].args[1]

    def test_escalates_to_breach_once(self):
        exe, _ = self._exe_with_peak(100.0)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            exe._check_trailing_drawdown(84.0)   # WARN
            exe._check_trailing_drawdown(79.0)   # -21% → BREACH
            exe._check_trailing_drawdown(78.0)   # still BREACH → no dup
            exe._check_trailing_drawdown(84.0)   # back in WARN band, already
            #                                      alerted deeper → no dup
        assert tg.call_count == 2
        assert "BREACH" in tg.call_args_list[1].args[1]

    def test_rearms_after_recovery(self):
        exe, _ = self._exe_with_peak(100.0)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            exe._check_trailing_drawdown(84.0)   # WARN (1)
            exe._check_trailing_drawdown(88.0)   # -12% > warn+2pp → re-arm
            exe._check_trailing_drawdown(84.0)   # WARN again (2)
        assert tg.call_count == 2

    def test_no_peak_is_noop(self):
        exe, _ = self._exe_with_peak(None)
        with patch("indicator.okx.executor.send_critical") as tg:
            exe._check_trailing_drawdown(84.0)
        tg.assert_not_called()

    def test_peak_scoped_to_configured_era(self):
        exe, store = self._exe_with_peak(100.0)
        exe._check_trailing_drawdown(90.0)
        kwargs = store.get_peak_equity.call_args.kwargs
        assert kwargs["since_utc"] == exe._cfg.dd_peak_since_utc


# ── config validation for the new fields ─────────────────────────────


class TestConfigValidation:
    def test_valid_defaults_pass(self):
        validate_okx_config(_mk_cfg())

    def test_breach_shallower_than_warn_rejected(self):
        with pytest.raises(RuntimeError, match="dd_breach_pct"):
            validate_okx_config(_mk_cfg(dd_warn_pct=-20.0,
                                        dd_breach_pct=-15.0))

    def test_positive_dd_rejected(self):
        with pytest.raises(RuntimeError, match="negative"):
            validate_okx_config(_mk_cfg(dd_warn_pct=15.0))

    def test_fee_estimate_range(self):
        with pytest.raises(RuntimeError, match="taker_fee_side_est"):
            validate_okx_config(_mk_cfg(taker_fee_side_est=0.01))
