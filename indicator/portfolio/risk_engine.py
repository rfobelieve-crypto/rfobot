# -*- coding: utf-8 -*-
"""組合層風控引擎——唯一的守門員（設計稿 §3.2-§3.4）。

`decide()` 是純函式：(Intent, PortfolioState, PortfolioLimits) → Decision。
不連 DB、不下單、不改狀態。這樣才能被測試逐條釘住，也保證審批路徑上
不會偷偷多打一次 DB。

**引擎只會拒絕或縮小，永遠不會放大。** 批准的風險 ≤ 策略要求的風險，
沒有任何一條路徑能讓 approved > requested。這是整份設計的安全底線：
框架是加一層閘門，不是給策略一個要更多額度的管道。

檢查順序刻意由「最不可協商」到「最可協商」，且**第一個拒絕就停**——
理由是拒絕原因要指向最根本的那條，而不是碰巧最先寫的那條。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from indicator.portfolio.ledger import Intent, PortfolioState
from indicator.portfolio.limits import PortfolioLimits, default_limits


@dataclass(frozen=True)
class Decision:
    approved: bool
    risk_pct: float = 0.0          # 實際核准的單筆風險（≤ 請求值）
    reason: str = ""               # 拒絕原因（機器可讀的短碼）
    detail: str = ""               # 給人看的一句話

    @property
    def rejected(self) -> bool:
        return not self.approved


def _reject(reason: str, detail: str) -> Decision:
    return Decision(approved=False, reason=reason, detail=detail)


def decide(intent: Intent, state: PortfolioState,
           limits: Optional[PortfolioLimits] = None) -> Decision:
    lim = limits or default_limits()
    s = lim.get(intent.strategy)

    # 1. 帳戶層至高無上。它一觸發，所有策略一起停——組合框架沒有任何
    #    路徑可以繞過它，這是 §6 不變量。
    if state.account_halted:
        return _reject("account_halted",
                       "帳戶層 kill 已觸發，所有策略停止開倉")

    # 2. 終態優先於當日狀態：被 DEMOTE 的策略不是「今天不能開」，是
    #    「回 shadow 直到人工重驗 gate」。
    if intent.strategy in state.demoted:
        return _reject("strategy_demoted",
                       f"{intent.strategy} 已降級回 shadow，需人工重驗")

    # 3. 濾網型的線沒有開倉權（撤單流若只有 confirm/veto 價值時的掛法）。
    if s.filter_only:
        return _reject("filter_only",
                       f"{intent.strategy} 掛在濾網席位，不得自行開倉")

    if intent.risk_pct <= 0:
        return _reject("invalid_risk", "risk_pct 必須為正")

    # 4. 策略層自身的回撤／當日虧損。三條分開判，因為後續動作不同：
    #    回撤 → DEMOTE（終態）；當日 → HALT（隔日自動恢復）。
    dd = state.strategy_dd_pct.get(intent.strategy, 0.0)
    if dd <= s.total_dd_cap_pct:
        return _reject("strategy_dd_cap",
                       f"{intent.strategy} 回撤 {dd:.1f}% 已達 "
                       f"{s.total_dd_cap_pct:.1f}% → 應 DEMOTE")
    day_r = state.strategy_day_r.get(intent.strategy, 0.0)
    if day_r <= s.daily_loss_cap_r:
        return _reject("strategy_daily_r",
                       f"{intent.strategy} 當日 {day_r:+.1f}R 已達 "
                       f"{s.daily_loss_cap_r:+.1f}R → 當日 HALT")
    day_pct = state.strategy_day_pct.get(intent.strategy, 0.0)
    if day_pct <= s.daily_loss_cap_pct:
        return _reject("strategy_daily_pct",
                       f"{intent.strategy} 當日 {day_pct:.1f}% 已達 "
                       f"{s.daily_loss_cap_pct:.1f}% → 當日 HALT")

    # 5. 併發上限（每策略）。
    open_n = len(state.positions_of(intent.strategy))
    if open_n >= s.max_concurrent:
        return _reject("concurrency_cap",
                       f"{intent.strategy} 已持有 {open_n} 筆，上限 "
                       f"{s.max_concurrent}")

    # 6. 同幣同向去重：另一條策略已在同一標的同方向持倉時，這不是新的
    #    分散，是同一份曝險再下一次。允許進場但**不額外配發預算**——
    #    核准風險降到兩者的較小檔，避免「假分散拿雙倍預算」。
    approved = intent.risk_pct
    collision = [p for p in state.open_positions
                 if p.symbol == intent.symbol and p.side == intent.side
                 and p.strategy != intent.strategy]
    if collision:
        smallest = min([lim.get(p.strategy).risk_pct_per_trade
                        for p in collision] + [s.risk_pct_per_trade])
        approved = min(approved, smallest)

    # 7. 相關性擠壓：30 日日 PnL 高度相關的兩條策略，合起來只配拿單策略
    #    檔的預算。樣本不足（< corr_min_days）不套用——樣本不足不是證據。
    for other in {p.strategy for p in state.open_positions
                  if p.strategy != intent.strategy}:
        key = tuple(sorted((intent.strategy, other)))
        rho_n = state.correlations.get(key)  # type: ignore[arg-type]
        if not rho_n:
            continue
        rho, n_days = rho_n
        if n_days >= lim.corr_min_days and abs(rho) > lim.corr_squeeze_threshold:
            squeezed = min(s.risk_pct_per_trade,
                           lim.get(other).risk_pct_per_trade)
            approved = min(approved, squeezed)

    # 8. 每策略單筆風險上限（放在擠壓之後，確保兩者都咬得住）。
    approved = min(approved, s.risk_pct_per_trade)

    # 9. 組合層總名目上限。用淨額算（同幣同向合併、反向互抵），
    #    這條是最後一關，因為它跨所有策略、最難在策略內部判斷。
    projected = state.net_notional_mult()
    if projected >= lim.max_total_notional_mult:
        return _reject("total_notional_cap",
                       f"組合淨名目 {projected:.2f}× 已達上限 "
                       f"{lim.max_total_notional_mult:.2f}×")

    if approved <= 0:
        return _reject("budget_exhausted", "核准後的風險為零")

    detail = f"{intent.strategy} {intent.side} {intent.symbol} "
    if approved < intent.risk_pct:
        detail += (f"核准 {approved:.3f}%（請求 {intent.risk_pct:.3f}%，"
                   f"因併發／相關性擠壓下調）")
    else:
        detail += f"核准 {approved:.3f}%"
    return Decision(approved=True, risk_pct=approved, detail=detail)
