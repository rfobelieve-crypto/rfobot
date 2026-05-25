"""OkxConfig + env-var loading + fail-fast validation.

CLAUDE.md hard rule:
  Stage 2-3: leverage hard cap = 1.0x
  Any stage: capital cap, daily/total loss caps, MAX_POSITION_COUNT=1

Failing validation -> RuntimeError on startup.  Better to fail boot than
trade with bad config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OkxConfig:
    # ── Endpoints (default = OKX prod; testnet uses x-simulated-trading header)
    rest_base: str = "https://www.okx.com"
    ws_public: str = "wss://ws.okx.com:8443/ws/v5/public"
    ws_private: str = "wss://ws.okx.com:8443/ws/v5/private"

    # ── Mode
    is_simulated: int = 1            # 1 = demo (testnet header), 0 = live
    stage_label: str = "testnet"     # purely for logging / dashboards

    # ── Credentials (loaded from env)
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""

    # ── Instrument
    inst_id: str = "BTC-USDT-SWAP"
    td_mode: Literal["cash", "cross", "isolated"] = "cross"
    pos_mode: str = "net_mode"
    leverage: int = 1                # HARD CAP 1 for Stage 2-3

    # ── Risk caps (Stage 3 defaults; user accepted $100 + all 10 safety belts)
    initial_capital_usd: float = 100.0
    risk_frac: float = 0.02
    max_position_count: int = 1
    daily_loss_cap_pct: float = -50.0   # Safety belt #3: -50% of capital
    total_loss_cap_pct: float = -50.0   # Safety belt #4: -50% of capital

    # ── Monitoring intervals
    reconciliation_interval_sec: int = 60
    ntp_check_interval_sec: int = 300
    heartbeat_timeout_sec: int = 30

    # ── Kill-trigger thresholds (kill_criteria.md mapping)
    ntp_drift_halt_sec: float = 5.0           # C5
    ntp_drift_demote_sec: float = 30.0        # C6
    ws_disconnect_demote_sec: int = 300       # A1: > 5 min disconnect
    reconnect_fail_demote_count: int = 3      # A2
    algo_stop_max_latency_sec: float = 5.0    # Safety belt #7 / B4
    order_reject_rate_demote: float = 0.05    # B1: 5% rolling
    amend_fail_rate_demote: float = 0.10      # B2: 10% rolling

    # ── Alerts
    telegram_alert_chat_id: str = ""          # routine
    telegram_critical_chat_id: str = ""       # critical / kill triggers

    # ── DB
    table_prefix: str = "v7_okx"


def load_okx_config_from_env(stage: Literal["testnet", "live"] = "testnet") -> OkxConfig:
    """Read env vars, return OkxConfig.  Does NOT validate (call validate
    separately so test code can inject incomplete configs)."""
    suffix = "_TESTNET" if stage == "testnet" else "_LIVE"
    return OkxConfig(
        is_simulated=1 if stage == "testnet" else 0,
        stage_label=stage,
        api_key=os.environ.get(f"OKX_API_KEY{suffix}", ""),
        api_secret=os.environ.get(f"OKX_API_SECRET{suffix}", ""),
        passphrase=os.environ.get(f"OKX_PASSPHRASE{suffix}", ""),
        telegram_alert_chat_id=os.environ.get("TG_ALERT_CHAT_ID", ""),
        telegram_critical_chat_id=os.environ.get("TG_CRITICAL_CHAT_ID", ""),
    )


def validate_okx_config(cfg: OkxConfig) -> None:
    """Fail-fast validation.  Raises RuntimeError on violation.

    These checks correspond to kill_criteria.md §2E (should-never-happen):
      E1: leverage > 1
      E2: posMode != net_mode
      E4: withdraw permission (checked separately via REST query in startup)
    """
    # E1
    if cfg.leverage != 1:
        raise RuntimeError(
            f"E1: Stage 2-3 leverage hard cap = 1, got {cfg.leverage}"
        )
    # E2
    if cfg.pos_mode != "net_mode":
        raise RuntimeError(
            f"E2: Stage 2 posMode must be net_mode, got {cfg.pos_mode!r}"
        )
    # td_mode sanity
    if cfg.td_mode not in ("cash", "cross", "isolated"):
        raise RuntimeError(f"Invalid td_mode {cfg.td_mode!r}")
    # Max position count (Safety belt #10)
    if cfg.max_position_count != 1:
        raise RuntimeError(
            f"Stage 2-3: max_position_count must be 1, got {cfg.max_position_count}"
        )
    # is_simulated
    if cfg.is_simulated not in (0, 1):
        raise RuntimeError(f"is_simulated must be 0 or 1")
    # live mode extra guards
    if cfg.is_simulated == 0:
        if os.environ.get("STAGE", "") != "live":
            raise RuntimeError(
                "is_simulated=0 requires STAGE=live env var to prevent accidents"
            )
        if cfg.initial_capital_usd > 200:
            raise RuntimeError(
                f"Stage 3 live capital max $200, got ${cfg.initial_capital_usd}"
            )
    # capital sanity
    if not (0 < cfg.initial_capital_usd <= 1000):
        raise RuntimeError(
            f"Stage 2-3 capital must be in (0, 1000], got ${cfg.initial_capital_usd}"
        )
    # loss caps must be negative
    if cfg.daily_loss_cap_pct >= 0:
        raise RuntimeError("daily_loss_cap_pct must be negative")
    if cfg.total_loss_cap_pct >= 0:
        raise RuntimeError("total_loss_cap_pct must be negative")
    # Credentials present
    if not cfg.api_key:
        raise RuntimeError("OKX_API_KEY not set in env")
    if not cfg.api_secret:
        raise RuntimeError("OKX_API_SECRET not set in env")
    if not cfg.passphrase:
        raise RuntimeError("OKX_PASSPHRASE not set in env")
    # Critical alert channel present
    if not cfg.telegram_critical_chat_id:
        raise RuntimeError(
            "TG_CRITICAL_CHAT_ID not set — refusing to start without critical alert path"
        )
