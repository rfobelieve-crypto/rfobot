"""OKX integration package — Stage 2/3 V7 testnet/live executor.

See docs/okx_integration_design.md for architecture.
See docs/stage2_kill_criteria.md for kill trigger IDs referenced in this code.

Status: skeleton (2026-05-25).  Actual OKX REST/WS API calls marked
`# TODO(stage2-impl)`; connection-resilience logic (retry / reconnect /
heartbeat) is implemented real.
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
