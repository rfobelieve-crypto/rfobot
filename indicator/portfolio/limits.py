# -*- coding: utf-8 -*-
"""組合層風險預算與上限（docs/PORTFOLIO_RISK_FRAMEWORK.md §3.2-§3.4）。

這個模組只描述「界線在哪」，不做任何決策也不碰 DB——決策在
risk_engine.py，資料在 ledger.py。分開的理由：上限值是要被人審的，
把它埋在邏輯裡就沒人會去看。

**紀律**：本檔的每個數字都是**新增的保護層**。帳戶層既有的 CAP-1..4
（−20% 日 / −30% 總、有效槓桿 2x、Stage 3 資本 guard）不在這裡，也
不由這裡覆寫——見 §6 不變量。任何「因為有了組合框架所以可以放寬帳戶層」
的推論都是錯的。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# 提案值，尚待操作者拍板（設計稿 §5 開放問題 1）。標記在此，讓「還沒
# 簽核」這件事跟著數字走，而不是留在文件某一段。
PENDING_SIGNOFF = True


@dataclass(frozen=True)
class StrategyLimits:
    """單一策略的預算。新策略一律從最小檔開始（§3.3）。"""

    name: str
    # 單筆風險佔 equity 的百分比。變體 B 併發研究的可交易區間 0.15-0.25%
    risk_pct_per_trade: float = 0.15
    # 同時持倉上限。sweep 5-10、V7 現值 1
    max_concurrent: int = 1
    # 策略當日 net_r 加總低於此值 → 該策略當日 HALT（隔日自動重置）
    daily_loss_cap_r: float = -5.0
    # 或當日虧損佔 equity 百分比，兩者取先觸發者
    daily_loss_cap_pct: float = -2.0
    # 策略自身權益高點回撤超過此值 → DEMOTE 回 shadow（終態，需人工）
    total_dd_cap_pct: float = -6.0
    # 只當濾網用的線（例如撤單流若只有 confirm/veto 價值）不佔預算席位，
    # 也不能自己開倉。設計稿 §5 開放問題 2 未定案前，兩種掛法都支援。
    filter_only: bool = False


@dataclass(frozen=True)
class PortfolioLimits:
    """跨策略的總量與相關性規則。"""

    strategies: dict[str, StrategyLimits] = field(default_factory=dict)
    # 所有策略合計的淨名目曝險上限（× equity）。沿用 NOTIONAL_LEV_MULT
    # 的精神：組合層不會因為策略變多就自動變大。
    max_total_notional_mult: float = 2.0
    # 30 日策略日 PnL 相關性超過此值 → 該對策略共用單策略檔預算（§3.4）
    corr_squeeze_threshold: float = 0.5
    # 相關性判定所需的最少共同交易日；不足就不套用（樣本不足不是證據）
    corr_min_days: int = 20

    def get(self, name: str) -> StrategyLimits:
        """未註冊的策略拿最保守的預設值，而不是拿到無限制。"""
        return self.strategies.get(name) or StrategyLimits(name=name)


def default_limits() -> PortfolioLimits:
    """目前設計稿的提案配置。V7 維持現值（併發 1），sweep 若過 Gate F
    從最小檔起跑（cap 5 × 0.15%），撤單流先掛 filter_only——三者都對得上
    §3.3「預算解鎖：過 gate 才升級」。"""
    return PortfolioLimits(strategies={
        "v7": StrategyLimits(name="v7", risk_pct_per_trade=0.25,
                             max_concurrent=1),
        "sweep": StrategyLimits(name="sweep", risk_pct_per_trade=0.15,
                                max_concurrent=5),
        "cancel": StrategyLimits(name="cancel", filter_only=True,
                                 max_concurrent=0),
    })
