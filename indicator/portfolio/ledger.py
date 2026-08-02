# -*- coding: utf-8 -*-
"""統一曝險帳本的資料型別與 DDL（設計稿 §3.1）。

**這個模組不連 DB、不建表。** DDL 以字串常數留存，等 P1 才真的執行——
設計稿 §4 明訂 P1 之前不動 live。現在需要的是「型別與 DDL 已定稿且被
測試釘住」，不是「表已經建好」。

跨策略可比的關鍵是 `net_r`：以**進場當下的風險**為分母。一條策略賺
0.5R 和另一條賺 0.5R 是同一件事，不管它們的停損距離、標的、槓桿差多少。
用 % 或 USD 當共同單位都會被波動率與部位大小污染。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Side = Literal["LONG", "SHORT"]
IntentStatus = Literal["pending", "approved", "rejected", "expired"]

# 沿用 v7_okx_* 的命名慣例，組合層用 pf_ 前綴（設計稿 §3.1）
DDL_INTENTS = """
CREATE TABLE IF NOT EXISTS pf_intents (
  id            BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts            DATETIME      NOT NULL,
  strategy      VARCHAR(32)   NOT NULL,
  symbol        VARCHAR(32)   NOT NULL,
  side          ENUM('LONG','SHORT') NOT NULL,
  risk_pct      DECIMAL(8,4)  NOT NULL,
  stop_px       DECIMAL(20,8) NULL,
  entry_ref_px  DECIMAL(20,8) NULL,
  ttl_sec       INT           NOT NULL DEFAULT 60,
  status        ENUM('pending','approved','rejected','expired')
                              NOT NULL DEFAULT 'pending',
  approved_risk_pct DECIMAL(8,4) NULL,
  reject_reason VARCHAR(64)   NULL,
  decided_ts    DATETIME      NULL,
  KEY ix_strategy_ts (strategy, ts),
  KEY ix_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS pf_positions (
  id          BIGINT AUTO_INCREMENT PRIMARY KEY,
  strategy    VARCHAR(32)   NOT NULL,
  symbol      VARCHAR(32)   NOT NULL,
  side        ENUM('LONG','SHORT') NOT NULL,
  entry_ts    DATETIME      NOT NULL,
  entry_px    DECIMAL(20,8) NOT NULL,
  size        DECIMAL(20,8) NOT NULL,
  risk_usd    DECIMAL(18,6) NOT NULL,
  stop_px     DECIMAL(20,8) NULL,
  exit_ts     DATETIME      NULL,
  exit_px     DECIMAL(20,8) NULL,
  exit_reason VARCHAR(32)   NULL,
  gross_pnl   DECIMAL(18,6) NULL,
  fees        DECIMAL(18,6) NULL,
  net_pnl     DECIMAL(18,6) NULL,
  net_r       DECIMAL(12,6) NULL,
  equity_after DECIMAL(18,6) NULL,
  src_table   VARCHAR(32)   NULL,   -- dual-write 來源（P1 對帳用）
  src_id      BIGINT        NULL,
  UNIQUE KEY ux_src (src_table, src_id),
  KEY ix_strategy_entry (strategy, entry_ts),
  KEY ix_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

ALL_DDL = (DDL_INTENTS, DDL_POSITIONS)


@dataclass(frozen=True)
class Intent:
    """一條策略「想開倉」的請求。它不是訂單——訂單是風控引擎批准後，
    執行層自己的事。策略永遠不能直接下單，這是整個框架的單一守門點。"""

    strategy: str
    symbol: str
    side: Side
    risk_pct: float
    stop_px: Optional[float] = None
    entry_ref_px: Optional[float] = None
    ttl_sec: int = 60


@dataclass(frozen=True)
class OpenPosition:
    """帳本裡一筆存活中的部位（引擎唯讀）。"""

    strategy: str
    symbol: str
    side: Side
    risk_pct: float
    notional_mult: float = 0.0   # 佔 equity 的名目倍數


@dataclass
class PortfolioState:
    """引擎做決策所需的**全部**輸入。刻意做成快照而不是 DB handle：
    決策函式必須是純的，才能被測試逐條釘住，也才不會在審批路徑上
    偷偷多打一次 DB。"""

    equity_usd: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    # 各策略當日累計 net_r 與當日損益佔 equity 百分比
    strategy_day_r: dict[str, float] = field(default_factory=dict)
    strategy_day_pct: dict[str, float] = field(default_factory=dict)
    # 各策略自身權益高點回撤（負值，百分比）
    strategy_dd_pct: dict[str, float] = field(default_factory=dict)
    # 已被降級的策略（終態，需人工 + 重驗 gate 才回來）
    demoted: set[str] = field(default_factory=set)
    # 帳戶層是否已觸發（CAP-1..4）。至高無上：真 → 全部拒絕
    account_halted: bool = False
    # 30 日策略日 PnL 相關性 {(a,b): (rho, n_days)}
    correlations: dict[tuple[str, str], tuple[float, int]] = field(
        default_factory=dict)

    def positions_of(self, strategy: str) -> list[OpenPosition]:
        return [p for p in self.open_positions if p.strategy == strategy]

    def net_notional_mult(self) -> float:
        """同幣同向合併、反向淨額後的總名目（§3.4 事前規則）。

        兩條策略同時做多 BTC 不是兩份曝險——是一份。把它算成兩份，
        風險預算就會在「假分散」上被重複發放，這正是 9 幣教訓要堵的洞。
        """
        by_symbol: dict[str, float] = {}
        for p in self.open_positions:
            signed = p.notional_mult * (1 if p.side == "LONG" else -1)
            by_symbol[p.symbol] = by_symbol.get(p.symbol, 0.0) + signed
        return sum(abs(v) for v in by_symbol.values())
