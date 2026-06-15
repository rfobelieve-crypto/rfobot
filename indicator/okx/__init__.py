"""OKX integration package — Stage 2/3 V7 testnet/live executor.

See docs/okx_integration_design.md for architecture.
See docs/stage2_kill_criteria.md for kill trigger IDs referenced in this code.

Status (2026-05-28): REST + WS layer complete (148 unit tests).  Only
the 9 executor.py integration TODOs remain — those wire the executor
state machine to actual OKX calls + Telegram alerts + DB persistence.
"""

from indicator.okx.config import OkxConfig, load_okx_config_from_env, validate_okx_config
from indicator.okx.types import (
    Side, ExecutorStatus, Position, OrderResult, AlgoOrderResult,
    ConnectivityStatus, KillCheckResult, ReconciliationResult,
)

__all__ = [
    "OkxConfig", "load_okx_config_from_env", "validate_okx_config",
    "Side", "ExecutorStatus", "Position",
    "OrderResult", "AlgoOrderResult", "ConnectivityStatus",
    "KillCheckResult", "ReconciliationResult",
]
