"""Unit tests for indicator/okx/config.py validate_okx_config.

Every fail-fast branch in validate_okx_config gets a dedicated test —
this function is the boot gate, and a missing check here means bad
config silently reaches production.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from indicator.okx.config import (
    OkxConfig,
    load_okx_config_from_env,
    validate_okx_config,
)


def _valid_cfg(**overrides) -> OkxConfig:
    """Base config that passes validation; override individual fields per test."""
    base = OkxConfig(
        api_key="test-key",
        api_secret="test-secret",
        passphrase="test-pass",
        telegram_critical_chat_id="critical-chat-123",
        is_simulated=1,
        leverage=1,   # tests target old 1x semantics; new default is 10x
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ── Happy path ────────────────────────────────────────────────────────


class TestValidConfig:
    def test_default_testnet_config_passes(self):
        validate_okx_config(_valid_cfg())  # must not raise

    def test_isolated_td_mode_passes(self):
        validate_okx_config(_valid_cfg(td_mode="isolated"))

    def test_cash_td_mode_passes(self):
        validate_okx_config(_valid_cfg(td_mode="cash"))


# ── Leverage cap (E1) — Stage 3 informed override allows 1..10 ────────


class TestLeverageCap:
    def test_leverage_1_passes(self):
        # 1x is the Kelly-respecting default for Stage 4+
        validate_okx_config(_valid_cfg(leverage=1))

    def test_leverage_10_passes(self):
        # 10x is the Stage 3 informed-override ceiling (2026-05-28)
        validate_okx_config(_valid_cfg(leverage=10))

    def test_leverage_11_fails(self):
        with pytest.raises(RuntimeError, match="E1"):
            validate_okx_config(_valid_cfg(leverage=11))

    def test_leverage_25_fails(self):
        with pytest.raises(RuntimeError, match="E1"):
            validate_okx_config(_valid_cfg(leverage=25))

    def test_leverage_0_fails(self):
        with pytest.raises(RuntimeError, match="E1"):
            validate_okx_config(_valid_cfg(leverage=0))


# ── posMode (E2) ──────────────────────────────────────────────────────


class TestPosMode:
    def test_net_mode_passes(self):
        validate_okx_config(_valid_cfg(pos_mode="net_mode"))

    def test_long_short_mode_passes(self):
        # Both modes are supported as of 2026-05-31 — executor branches
        # on cfg.pos_mode to decide whether to include posSide on orders.
        validate_okx_config(_valid_cfg(pos_mode="long_short_mode"))

    def test_typo_long_short_no_suffix_fails(self):
        # OKX returns exactly "long_short_mode"; bare "long_short" is wrong
        with pytest.raises(RuntimeError, match="E2"):
            validate_okx_config(_valid_cfg(pos_mode="long_short"))

    def test_arbitrary_string_fails(self):
        with pytest.raises(RuntimeError, match="E2"):
            validate_okx_config(_valid_cfg(pos_mode="unknown"))


class TestPosSideHelper:
    def test_net_mode_omits_pos_side(self):
        cfg = _valid_cfg(pos_mode="net_mode")
        assert cfg.pos_side_for("LONG") is None
        assert cfg.pos_side_for("SHORT") is None

    def test_long_short_mode_maps_to_lowercase_long_short(self):
        cfg = _valid_cfg(pos_mode="long_short_mode")
        assert cfg.pos_side_for("LONG") == "long"
        assert cfg.pos_side_for("SHORT") == "short"


# ── td_mode sanity ───────────────────────────────────────────────────


class TestTdModeSanity:
    def test_invalid_td_mode_fails(self):
        with pytest.raises(RuntimeError, match="td_mode"):
            validate_okx_config(_valid_cfg(td_mode="margin"))


# ── max_position_count (safety belt #10) ─────────────────────────────


class TestMaxPositionCount:
    def test_two_positions_fails(self):
        with pytest.raises(RuntimeError, match="max_position_count"):
            validate_okx_config(_valid_cfg(max_position_count=2))

    def test_zero_positions_fails(self):
        # max_count=0 is nonsensical (can't trade at all)
        with pytest.raises(RuntimeError, match="max_position_count"):
            validate_okx_config(_valid_cfg(max_position_count=0))


# ── is_simulated flag ────────────────────────────────────────────────


class TestIsSimulated:
    def test_invalid_value_fails(self):
        with pytest.raises(RuntimeError, match="is_simulated"):
            validate_okx_config(_valid_cfg(is_simulated=2))


# ── live mode guards ─────────────────────────────────────────────────


class TestLiveModeGuards:
    def test_live_without_stage_env_fails(self):
        # is_simulated=0 (live) MUST be paired with STAGE=live env var
        cfg = _valid_cfg(is_simulated=0)
        with patch.dict(os.environ, {"STAGE": ""}, clear=False):
            os.environ.pop("STAGE", None)
            with pytest.raises(RuntimeError, match="STAGE=live"):
                validate_okx_config(cfg)

    def test_live_with_stage_env_passes(self):
        cfg = _valid_cfg(is_simulated=0)
        with patch.dict(os.environ, {"STAGE": "live"}):
            validate_okx_config(cfg)  # must not raise

    def test_live_capital_above_500_fails(self):
        # Ceiling: 200 -> 1500 (2026-07-24, 6th informed override, $1218.44
        # deposit) -> 500 (2026-07-28, baseline back to $274 after the second
        # manual blow-up). It sits one documented expansion step above the
        # live baseline so capital creep still has to pass Gate A/B.
        cfg = _valid_cfg(is_simulated=0, initial_capital_usd=600.0)
        with patch.dict(os.environ, {"STAGE": "live"}):
            with pytest.raises(RuntimeError, match="\\$500"):
                validate_okx_config(cfg)

    def test_live_capital_at_500_passes(self):
        cfg = _valid_cfg(is_simulated=0, initial_capital_usd=500.0)
        with patch.dict(os.environ, {"STAGE": "live"}):
            validate_okx_config(cfg)

    def test_live_capital_at_current_baseline_passes(self):
        cfg = _valid_cfg(is_simulated=0, initial_capital_usd=311.6)
        with patch.dict(os.environ, {"STAGE": "live"}):
            validate_okx_config(cfg)  # must not raise

    def test_previous_1218_baseline_now_rejected(self):
        # The $1218.44 baseline was live until 2026-07-28. Pinning it here
        # means a stale Railway OKX_INITIAL_CAPITAL_USD cannot quietly
        # resurrect the old size — the executor refuses to start instead.
        cfg = _valid_cfg(is_simulated=0, initial_capital_usd=1218.44)
        with patch.dict(os.environ, {"STAGE": "live"}):
            with pytest.raises(RuntimeError, match="\\$500"):
                validate_okx_config(cfg)


# ── Capital sanity ───────────────────────────────────────────────────


class TestCapitalSanity:
    def test_negative_capital_fails(self):
        with pytest.raises(RuntimeError, match="capital"):
            validate_okx_config(_valid_cfg(initial_capital_usd=-100.0))

    def test_zero_capital_fails(self):
        with pytest.raises(RuntimeError, match="capital"):
            validate_okx_config(_valid_cfg(initial_capital_usd=0.0))

    def test_above_500_capital_fails(self):
        # Stage 2-3 hard ceiling: $500 (1500 -> 500 on 2026-07-28)
        with pytest.raises(RuntimeError, match="capital"):
            validate_okx_config(_valid_cfg(initial_capital_usd=600.0))


# ── Loss cap sign ────────────────────────────────────────────────────


class TestLossCapSign:
    def test_positive_daily_cap_fails(self):
        with pytest.raises(RuntimeError, match="daily_loss_cap"):
            validate_okx_config(_valid_cfg(daily_loss_cap_pct=10.0))

    def test_zero_daily_cap_fails(self):
        with pytest.raises(RuntimeError, match="daily_loss_cap"):
            validate_okx_config(_valid_cfg(daily_loss_cap_pct=0.0))

    def test_positive_total_cap_fails(self):
        with pytest.raises(RuntimeError, match="total_loss_cap"):
            validate_okx_config(_valid_cfg(total_loss_cap_pct=10.0))


# ── Credentials ──────────────────────────────────────────────────────


class TestCredentials:
    def test_missing_api_key_fails(self):
        with pytest.raises(RuntimeError, match="OKX_API_KEY"):
            validate_okx_config(_valid_cfg(api_key=""))

    def test_missing_secret_fails(self):
        with pytest.raises(RuntimeError, match="OKX_API_SECRET"):
            validate_okx_config(_valid_cfg(api_secret=""))

    def test_missing_passphrase_fails(self):
        with pytest.raises(RuntimeError, match="OKX_PASSPHRASE"):
            validate_okx_config(_valid_cfg(passphrase=""))


# ── Critical Telegram path ───────────────────────────────────────────


class TestCriticalAlertPath:
    def test_missing_critical_chat_id_fails(self):
        # No critical alert path → refuse to start
        with pytest.raises(RuntimeError, match="TG_CRITICAL_CHAT_ID"):
            validate_okx_config(_valid_cfg(telegram_critical_chat_id=""))


# ── load_okx_config_from_env ─────────────────────────────────────────


class TestLoadFromEnv:
    def test_testnet_loads_testnet_suffix(self):
        with patch.dict(os.environ, {
            "OKX_API_KEY_TESTNET": "tk",
            "OKX_API_SECRET_TESTNET": "ts",
            "OKX_PASSPHRASE_TESTNET": "tp",
            "TG_ALERT_CHAT_ID": "alert",
            "TG_CRITICAL_CHAT_ID": "critical",
        }, clear=False):
            cfg = load_okx_config_from_env(stage="testnet")
            assert cfg.api_key == "tk"
            assert cfg.api_secret == "ts"
            assert cfg.passphrase == "tp"
            assert cfg.is_simulated == 1
            assert cfg.stage_label == "testnet"

    def test_live_loads_live_suffix(self):
        with patch.dict(os.environ, {
            "OKX_API_KEY_LIVE": "lk",
            "OKX_API_SECRET_LIVE": "ls",
            "OKX_PASSPHRASE_LIVE": "lp",
            "TG_CRITICAL_CHAT_ID": "critical",
        }, clear=False):
            cfg = load_okx_config_from_env(stage="live")
            assert cfg.api_key == "lk"
            assert cfg.is_simulated == 0
            assert cfg.stage_label == "live"

    def test_missing_env_yields_empty_strings(self):
        # Don't crash if env vars absent; validation handles emptiness
        keys = ["OKX_API_KEY_TESTNET", "OKX_API_SECRET_TESTNET",
                "OKX_PASSPHRASE_TESTNET", "TG_ALERT_CHAT_ID",
                "TG_CRITICAL_CHAT_ID"]
        with patch.dict(os.environ, {}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            cfg = load_okx_config_from_env(stage="testnet")
            assert cfg.api_key == ""
            assert cfg.telegram_critical_chat_id == ""
