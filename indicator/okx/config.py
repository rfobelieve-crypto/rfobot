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
    # OKX default for new accounts is long_short_mode (sides separate).
    # net_mode (single signed net position) is also supported; the
    # executor branches on this field to decide whether to include
    # posSide / reduceOnly on each order.
    pos_mode: Literal["net_mode", "long_short_mode"] = "long_short_mode"
    # Leverage hard cap = 10x (Stage 3 informed override 2026-05-28).
    # Required for $100 capital to fit 1 BTC-USDT-SWAP contract
    # (1 contract = 0.01 BTC ≈ $750 notional).  Trade-off documented
    # in CLAUDE.md §"10x leverage informed override".
    leverage: int = 10
    # OKX SWAP contract size in base currency (BTC for BTC-USDT-SWAP = 0.01)
    contract_size_base: float = 0.01
    # Round-trip taker cost as a fraction (mirrors v7_paper_executor)
    taker_cost: float = 0.0008

    # ── Risk caps (Stage 3 defaults; tightened for 10x leverage)
    # Default bumped 100→155 on 2026-06-01: user deposited $154.86 to
    # OKX trading account; CAP-2 (equity > 1.5×initial → HALT) was
    # tripping immediately because $154 > $150.  Configurable via
    # OKX_INITIAL_CAPITAL_USD env var if you adjust the deposit later.
    initial_capital_usd: float = 155.0
    risk_frac: float = 0.02
    max_position_count: int = 1
    # Tightened on 2026-05-28 for 10x leverage.  At 10x, a 2% BTC move
    # = 20% account move, so the old -50% cap was meaningless.
    daily_loss_cap_pct: float = -20.0   # Safety belt #3
    total_loss_cap_pct: float = -30.0   # Safety belt #4 (career-end)
    # Pre-submit STRATEGY risk-leverage ceiling (notional / equity), NOT the
    # OKX account leverage (10x, margin lockup only).  Sizing targets ~2x
    # (NOTIONAL_LEV_MULT); 3.0 leaves headroom for lot-rounding / small-account
    # min-lot inflation while still catching the int()-floor class bug that
    # forced ~7x on 2026-06-05.  See kill_checks.check_presubmit_order.
    max_effective_leverage: float = 3.0

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

    # ── Helpers ────────────────────────────────────────────────────────

    def pos_side_for(self, direction: str) -> "Optional[str]":
        """Map our LONG/SHORT direction → OKX posSide field.

        In net_mode the field is omitted (OKX infers from signed pos).
        In long_short_mode every order must carry an explicit posSide.
        """
        if self.pos_mode == "net_mode":
            return None
        return "long" if direction == "LONG" else "short"


def load_okx_config_from_env(stage: Literal["testnet", "live"] = "testnet") -> OkxConfig:
    """Read env vars, return OkxConfig.  Does NOT validate (call validate
    separately so test code can inject incomplete configs).

    OKX_INITIAL_CAPITAL_USD overrides the default 155.0 — useful when
    you change the OKX deposit without redeploying code.
    """
    suffix = "_TESTNET" if stage == "testnet" else "_LIVE"
    kwargs: dict = dict(
        is_simulated=1 if stage == "testnet" else 0,
        stage_label=stage,
        api_key=os.environ.get(f"OKX_API_KEY{suffix}", ""),
        api_secret=os.environ.get(f"OKX_API_SECRET{suffix}", ""),
        passphrase=os.environ.get(f"OKX_PASSPHRASE{suffix}", ""),
        telegram_alert_chat_id=os.environ.get("TG_ALERT_CHAT_ID", ""),
        telegram_critical_chat_id=os.environ.get("TG_CRITICAL_CHAT_ID", ""),
    )
    cap_override = os.environ.get("OKX_INITIAL_CAPITAL_USD", "").strip()
    if cap_override:
        try:
            kwargs["initial_capital_usd"] = float(cap_override)
        except ValueError:
            pass
    return OkxConfig(**kwargs)


def validate_okx_config(cfg: OkxConfig) -> None:
    """Fail-fast validation.  Raises RuntimeError on violation.

    These checks correspond to kill_criteria.md §2E (should-never-happen):
      E1: leverage outside Stage 3 informed-override range [1, 10]
      E2: posMode != net_mode
      E4: withdraw permission (checked separately via REST query in startup)
    """
    # E1 — Stage 3 informed override (2026-05-28): leverage may be 1..10
    if not (1 <= cfg.leverage <= 10):
        raise RuntimeError(
            f"E1: Stage 3 leverage must be in [1, 10], got {cfg.leverage}"
        )
    # Pre-submit strategy leverage cap must be sane: >= 1x (else nothing
    # trades) and <= 10x (CLAUDE.md Stage-3 account leverage ceiling).
    # Decoupled from cfg.leverage so low-leverage configs stay valid.
    if not (1.0 <= cfg.max_effective_leverage <= 10.0):
        raise RuntimeError(
            f"max_effective_leverage must be in [1, 10], "
            f"got {cfg.max_effective_leverage}"
        )
    # E2 — both modes supported; executor branches on cfg.pos_mode
    if cfg.pos_mode not in ("net_mode", "long_short_mode"):
        raise RuntimeError(
            f"E2: posMode must be 'net_mode' or 'long_short_mode', "
            f"got {cfg.pos_mode!r}"
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
