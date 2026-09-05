# -*- coding: utf-8 -*-
"""宣告式範圍守衛——不允許靜默降級（2026-09-05）

**為什麼**：同一類錯誤在一天內出現兩次，兩次都差一點讓判決建立在被縮小的
樣本上。

  1. 路徑 B：三個交易所沒有 8h K 線粒度 → **5/6 的場館對算不出東西**，
     而輸出只是「表上少了幾列」，不報錯。
  2. 路徑 B2：程式裡一個未註冊的 `max_tail=25` 截斷（取 universe 排列順序）
     → **剛好排除了主張點名的 HYPE 與 XPL**。

**共同形狀：範圍被靜默縮小，輸出看起來完整。** 而 §1.18c 橫截面那種更難察覺
——少 40 個幣照樣算得出漂亮的 IC，沒有任何東西會變紅。

**規則**：每次取資料前**先宣告**預期的標的數與時間範圍，實得低於宣告就 raise。
不允許靜默降級；要降級必須明寫 `allow_shrink=` 並附理由，那個理由會被印出來
也會被寫進結果。

用法：

    scope = Scope("路徑B2 長尾組", expect_n=88, expect_days=365)
    ...
    scope.check(actual_n=len(coins), actual_days=span_days)     # 不足就 raise

    # 真的要允許縮小時，理由是強制的：
    scope.check(actual_n=80, actual_days=365,
                allow_shrink="8 個標的在 CEX 側不足 200 桶，事前規則已排除")
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ScopeShrunk(RuntimeError):
    """實得範圍小於宣告，且沒有具名理由。"""


@dataclass
class Scope:
    name: str
    expect_n: int = 0                 # 預期標的數（0 = 不檢查）
    expect_days: float = 0.0          # 預期時間跨度（天，0 = 不檢查）
    tol_n: float = 1.0                # 允許的比例，1.0 = 一個都不能少
    tol_days: float = 0.95            # 時間允許 5% 誤差（交易所補資料的常態）
    log: list = field(default_factory=list)

    def check(self, actual_n: int = None, actual_days: float = None,
              allow_shrink: str = "") -> None:
        bad = []
        if self.expect_n and actual_n is not None:
            need = self.expect_n * self.tol_n
            if actual_n < need:
                bad.append(f"標的數 {actual_n} < 宣告 {self.expect_n}")
        if self.expect_days and actual_days is not None:
            need = self.expect_days * self.tol_days
            if actual_days < need:
                bad.append(f"時間跨度 {actual_days:.0f} 天 < 宣告 {self.expect_days:.0f} 天")
        if not bad:
            return
        msg = f"[{self.name}] 範圍被縮小：" + "；".join(bad)
        if allow_shrink:
            self.log.append(f"{msg} —— 已具名允許：{allow_shrink}")
            print(f"[SCOPE] {msg}\n[SCOPE] 允許理由：{allow_shrink}", flush=True)
            return
        raise ScopeShrunk(
            msg + "。要允許必須傳 allow_shrink= 並寫明理由——"
            "靜默降級已經害過兩次（路徑 B 的 8h K 線、路徑 B2 的未註冊截斷）。")

    def as_dict(self) -> dict:
        return {"name": self.name, "expect_n": self.expect_n,
                "expect_days": self.expect_days, "shrink_notes": list(self.log)}
