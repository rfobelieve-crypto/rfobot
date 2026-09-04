# -*- coding: utf-8 -*-
"""註冊時的功效關卡 —— 一個判準有沒有能力做出判決，在開跑之前就要知道。

**為什麼有這個檔（2026-09-04，地形扳機結案之後）**

地形濾網的採用扳機凍結了一個月：「+60 筆新 Strong 且 90 天 kept vs vetoed
的 gap ≥ 8pp」。跑到 57/60、gap +5.8pp，看起來像「差一點沒過」。

把標準誤算出來才發現：兩個比例的差在 n=59/27 之下 **SE = 11.6pp**——
**比 8pp 的門檻本身還大**。不管資料怎麼走，這個設計都分辨不出過與不過。
它不是快要過了，它是從註冊那天起就不可能有答案，而我們為它累積了一個月。

`.claude/rules/mistake.md` 2026-08-26 已經寫過「寫下判準之後立刻把現有數字
代進去算一次」。那次代進去了、也算出 5.8 < 8「沒過」，但**沒有問那個 5.8
有多準**。這個檔是那條規則的下一層，而且是**由機器執行不靠記性**
（同 `tests/test_agent_boundary.py`、`tests/test_okx_client.py` 的形狀）。

**兩種判準，兩種病**

| 判準寫法 | 例子 | 會得什麼病 |
|---|---|---|
| 寫在**點估計**上 | `gap ≥ 8pp` | 雜訊可以憑運氣推過門檻——**假裝有答案** |
| 寫在 **CI** 上 | `CI 下緣 > 0` | 不可能被雜訊矇過（只會「無法下結論」），但**可能永遠下不了結論** |

所以兩種都要查，查的東西不同：
- 點估計型：門檻必須 ≥ **2.8 × SE**（80% 功效、雙尾 5% 的標準值）。
  否則「過」與「不過」都不是證據。
- CI 型：算出**最小可偵測效應 MDE = 2.8 × SE**，然後問「這條線宣稱的效應
  有比它大嗎」。CI 型判準不會說謊，但它可以徒勞。

**怎麼用**

    python research/prereg_power.py            # 印出所有已註冊時鐘的關卡
    python research/prereg_power.py --new ...  # 開新時鐘之前先跑這個

新時鐘上線前必須在 `CLOCKS` 裡加一列並讓 `tests/test_prereg_power.py` 綠。
擋下來的唯一合法出路是**改設計**（放大樣本、換指標、改寫成 CI 型），
不是把門檻調鬆——**調鬆門檻只會讓一個沒有能力的設計更沒有能力。**
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 80% power, two-sided 5%: z(0.975) + z(0.80) = 1.960 + 0.842
Z_POWER = 2.802
# the weaker "the CI merely excludes zero when the estimate lands on the
# threshold" bar, reported alongside so the gap between the two is visible
Z_CI = 1.960


@dataclass
class Clock:
    """一個預註冊時鐘的功效規格。

    每一列都是**註冊當下就該回答的問題**，不是事後補的。
    """

    name: str
    kind: str            # "prop2" | "mean" | "prop1"
    threshold: float     # 判準門檻（點估計型）或宣稱效應（CI 型）
    unit: str
    n: tuple             # 註冊樣本數（prop2 給 (n1, n2)）
    ci_based: bool       # 判準寫在 CI 上嗎
    p: float = 0.55      # prop2/prop1 的基準比例
    sd: float = 0.0      # mean 型的每筆標準差（**要用實測值**）
    note: str = ""
    exempt: Optional[str] = None   # 具名豁免＋理由，不得留空
    #: CI 型且宣稱效應 < MDE 時必填：承認它多半下不了結論，並寫出
    #: 「多少樣本才會」。留空 = 測試紅，因為這件事不可以被靜默跳過。
    underpowered_ack: Optional[str] = None

    def se(self) -> float:
        # 比例型一律回傳 **pp**（百分點),與門檻同單位——單位不一致的話
        # 這個關卡自己就會變成一個沒有測量能力的守衛。
        if self.kind == "prop2":
            n1, n2 = self.n
            return 100.0 * math.sqrt(self.p * (1 - self.p) * (1 / n1 + 1 / n2))
        if self.kind == "prop1":
            return 100.0 * math.sqrt(self.p * (1 - self.p) / self.n[0])
        if self.kind == "mean":
            return self.sd / math.sqrt(self.n[0])
        raise ValueError(self.kind)

    def mde(self) -> float:
        """最小可偵測效應（80% 功效）。"""
        return Z_POWER * self.se()

    def ratio(self) -> float:
        return self.threshold / self.se() if self.se() else float("inf")

    def n_needed(self) -> float:
        """要讓這個門檻達到 80% 功效，樣本要放大幾倍。"""
        r = self.ratio()
        return (Z_POWER / r) ** 2 if r else float("inf")

    def n_for_mde(self) -> float:
        """要讓 MDE 降到宣稱效應以下，樣本得多大（mean 型給總 n）。"""
        if not self.threshold:
            return float("inf")
        need_se = self.threshold / Z_POWER
        if self.kind == "mean":
            return (self.sd / need_se) ** 2
        if self.kind == "prop1":
            return 100.0 ** 2 * self.p * (1 - self.p) / need_se ** 2
        n2 = self.n[1]
        inv = (need_se / 100.0) ** 2 / (self.p * (1 - self.p)) - 1.0 / n2
        return float("inf") if inv <= 0 else 1.0 / inv

    def status(self) -> str:
        """BLOCKED（不准開跑）/ UNDERPOWERED（多半下不了結論）/ PASS。"""
        if self.exempt:
            return "EXEMPT"
        if self.ci_based:
            return "PASS" if self.threshold >= self.mde() else "UNDERPOWERED"
        return "PASS" if self.ratio() >= Z_POWER else "BLOCKED"

    def verdict(self) -> tuple[bool, str]:
        st = self.status()
        if st == "EXEMPT":
            return True, "EXEMPT: " + self.exempt
        if st == "UNDERPOWERED":
            msg = ("宣稱效應 %.4g%s < MDE %.4g%s —— CI 型判準不會說謊（它只會"
                   "「無法下結論」），但這個 n 多半就是下不了結論；"
                   "要下得了需要 n≈%.0f"
                   % (self.threshold, self.unit, self.mde(), self.unit,
                      self.n_for_mde()))
            if self.underpowered_ack:
                return True, "ACK: " + msg + " ／ " + self.underpowered_ack
            return False, msg + "（**尚未具名承認**）"
        if self.ci_based:
            return True, ("宣稱效應 %.4g%s ≥ MDE %.4g%s"
                          % (self.threshold, self.unit, self.mde(), self.unit))
        ok = self.ratio() >= Z_POWER
        if ok:
            return True, "門檻 = %.2f×SE（≥%.2f）" % (self.ratio(), Z_POWER)
        return False, ("門檻 %.4g%s 只有 %.2f×SE（SE=%.4g%s）——樣本要放大 "
                       "%.1f× 才有 80%% 功效"
                       % (self.threshold, self.unit, self.ratio(),
                          self.se(), self.unit, self.n_needed()))


#: 每一個開著的（或剛結案的）預註冊時鐘。**新時鐘上線前必須加進來。**
#: sd 一律填實測值並註明來源，不得用估的——用估的等於把這個關卡也變成
#: 一個沒有測量能力的守衛。
CLOCKS: list[Clock] = [
    Clock(
        name="地形扳機 Strong（2026-08-02 凍結，09-04 結案）",
        kind="prop2", threshold=8.0, unit="pp", n=(59, 27),
        ci_based=False, p=0.55,
        note="這個檔存在的原因。門檻 8pp vs SE 11.6pp。",
    ),
    Clock(
        name="§0.59 regime 濾網（meanR > 0）",
        kind="mean", threshold=0.098, unit="R", n=(150,),
        ci_based=True, sd=0.6068,
        note="sd 來自 sweep_shadow_log.csv net_r 實測 n=4305；"
             "宣稱效應用 §0.58 分解的 RANGING +0.098。",
        underpowered_ack=(
            "承認：n=150 下多半只會停在「累積中」。**不調鬆判準**（那只會讓它假裝有答案）；到 150 若 CI 仍含零，處置是延長到 n≈300 或收掉，兩者都在到期日當天決定。"),
    ),
    Clock(
        name="Gate F·A 變體 A forward（CI 下緣 > 0）",
        kind="mean", threshold=0.05, unit="R", n=(1400,),
        ci_based=True, sd=0.6068,
        note="宣稱效應 0.05R 是凍結時的 quasi 前瞻量級。",
    ),
    Clock(
        name="§0.474b 變體 E（BTC 差 > 0）",
        kind="mean", threshold=0.19, unit="R", n=(60,),
        ci_based=True, sd=0.6068,
        note="目前觀察到的差 +0.19R；n=60 的 MDE 見輸出。",
        underpowered_ack=(
            "承認：n=60 只差一點（MDE 0.220R vs 觀察 0.19R）。到 60 若 CI 含零即延長到 n≈80，這個延長現在就寫下來，不是到時候看數字再決定。"),
    ),
    Clock(
        name="M 後方磁鐵（CI 下緣 > 0）",
        kind="mean", threshold=0.085, unit="R", n=(400,),
        ci_based=True, sd=0.6068,
        underpowered_ack=(
            "承認：n=400 恰好壓在線上（MDE 0.085R = 宣稱 0.085R），等於只有約 50% 功效。到 400 若 CI 含零不得解讀為 FAIL，只能是「無法下結論」。"),
    ),
    Clock(
        name="§0.60 Q2 格內 vs 全體（CI 上緣 < 全體）",
        kind="prop2", threshold=6.0, unit="pp", n=(60, 819),
        ci_based=True, p=0.59,
        note="判準寫在 CI 上，所以不會假裝有答案；門檻欄填的是"
             "宣稱效應（該格比全體差多少）。",
        underpowered_ack=(
            "承認：格內 n=60 對上全體 819，MDE 18.4pp 遠大於宣稱的 6pp，**這一格幾乎不可能下結論**——因為小樣本那側鎖死了精度。要決定需要格內 n≈1484。列為已知徒勞，不再等它。"),
    ),
    Clock(
        name="V7 Gate B 執行驗證（net ≥ 0，30 筆）",
        kind="mean", threshold=0.0, unit="bps", n=(30,),
        ci_based=False, sd=1.0,
        exempt="判準是「不得為負」不是「要顯著為正」——它是操作驗收"
               "（滑價、trailing、kill switch 有沒有把 edge 吃掉），"
               "不是統計檢定。統計那半由 Gate A 用大樣本回答，"
               "這正是 2026-06-10 拆成兩個 Gate 的理由。",
    ),
]


def report(clocks: list[Clock]) -> int:
    bad = 0
    print("=" * 78)
    print("  預註冊功效關卡 —— 門檻 vs 註冊樣本下的標準誤")
    print("=" * 78)
    for c in clocks:
        ok, why = c.verdict()
        tag = "PASS" if ok else "BLOCKED"
        if not ok:
            bad += 1
        kindtag = "CI型" if c.ci_based else "點估計"
        print("\n  [%s] %s" % (tag, c.name))
        print("      %s | n=%s | SE %.4g%s | MDE(80%%) %.4g%s"
              % (kindtag, "/".join(str(x) for x in c.n), c.se(), c.unit,
                 c.mde(), c.unit))
        print("      → %s" % why)
        if c.note:
            print("      註：%s" % c.note)
    print("\n" + "-" * 78)
    print("  %d 個時鐘，%d 個被擋" % (len(clocks), bad))
    if bad:
        print("  擋下來的唯一合法出路是**改設計**（放大樣本、換指標、"
              "改寫成 CI 型），不是調鬆門檻。")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", nargs=5, metavar=("KIND", "THRESHOLD", "N1", "N2",
                                               "P_OR_SD"),
                    help="開新時鐘之前先試算："
                         "KIND=prop2|prop1|mean，N2 給 prop2 用（mean 填 0）")
    a = ap.parse_args()
    if a.new:
        kind, thr, n1, n2, ps = a.new
        c = Clock(name="（試算）", kind=kind, threshold=float(thr),
                  unit="", n=(int(n1), int(n2)) if kind == "prop2"
                  else (int(n1),), ci_based=False,
                  p=float(ps) if kind != "mean" else 0.55,
                  sd=float(ps) if kind == "mean" else 0.0)
        report([c])
        return 0
    return 1 if report(CLOCKS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
