# -*- coding: utf-8 -*-
"""組合層風控框架（docs/PORTFOLIO_RISK_FRAMEWORK.md）。

P0 階段：**沒有任何 live 模組 import 這個套件**。它是一組純函式與型別，
先讓規則可測、可審；P1 才開始 dual-write，P2 才有第二條策略透過它下單。
這個順序寫在設計稿 §4，也寫在這裡，因為「先接上去再說」正是這份框架
存在的理由的反面。
"""
from indicator.portfolio.ledger import (ALL_DDL, Intent, OpenPosition,  # noqa: F401
                                        PortfolioState)
from indicator.portfolio.limits import (PortfolioLimits,  # noqa: F401
                                        StrategyLimits, default_limits)
from indicator.portfolio.risk_engine import Decision, decide  # noqa: F401
