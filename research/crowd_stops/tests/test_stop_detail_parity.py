# -*- coding: utf-8 -*-
"""三個群眾原型的 detail 分支不得改變任何一個部位。

2026-09-10 為了畫出「群眾的止損在哪」，`pos_breakout` / `pos_supertrend` /
`pos_psar` 各多了一個 `detail=True` 分支。這支釘住兩件事：

  1. `detail=True` 投影回部位之後，與 `detail=False` **逐元素相同**
     —— 保證止損地圖與天氣站的部位序列不可能是兩份實作
     （本 repo 光偵測層的第二份實作就咬過五次）
  2. 止損落在部位的正確一側：做多的止損必須 <= 當根收盤，
     做空的必須 >= 當根收盤。**反向證明過**（把 SuperTrend 的
     stop 改成取另一條軌，第 2 關立刻紅並指名該幣）

     **翻轉當根除外，理由不是為了讓測試變綠**：PSAR 翻轉時新的 SAR
     取的是前一段的極值點，而那根 K 的收盤可以已經回到 SAR 的另一側
     （實測 BTC i=324：翻空、SAR 65646、收盤 66169）。那是 PSAR 本身
     的性質，不是投影錯。它同時是一條**地圖規則**：已經被穿的止損不是
     待觸發的燃料，`stop_map` 只收未被穿的那些。

跑在 `research/crowd_stops/frozen/` 的**凍結切片**上，不吃 `.cache`
—— 後者是滾動 930 天窗，任何釘在它上面的東西都會無故變紅
（mistake.md 2026-09-10）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from research.crowd_battery import pos_breakout  # noqa: E402
from research.crowd_battery2 import pos_supertrend  # noqa: E402
from research.crowd_battery3 import pos_psar  # noqa: E402
from research.survival_cards import SC  # noqa: E402

FROZEN = ROOT / "research" / "crowd_stops" / "frozen"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
ARCH = {"breakout": pos_breakout, "supertrend": pos_supertrend,
        "psar": pos_psar}


def _bars(sym):
    p = FROZEN / f"{sym}USDT_1h.csv"
    if not p.exists():
        pytest.skip(f"{p.name} not frozen — run stop_map.py --freeze")
    import csv
    rows = []
    with open(p, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) < 6:
                continue
            rows.append((int(float(x[0])), float(x[1]), float(x[2]),
                         float(x[3]), float(x[4]), float(x[5])))
    return rows


@pytest.mark.parametrize("name", sorted(ARCH))
def test_detail_projects_to_the_same_positions(name):
    fn = ARCH[name]
    for s in CORE9:
        b = _bars(s)
        assert [d["pos"] for d in fn(b, detail=True)] == fn(b), \
            f"{s}/{name}: detail projection differs from the frozen series"


@pytest.mark.parametrize("name", sorted(ARCH))
def test_stop_sits_on_the_right_side_of_price(name):
    fn = ARCH[name]
    for s in CORE9:
        b = _bars(s)
        det = fn(b, detail=True)
        for i, d in enumerate(det):
            if d["pos"] == 0 or d["stop"] is None:
                continue
            if i and det[i - 1]["pos"] != d["pos"]:
                continue            # 翻轉當根，見檔頭
            c = b[i][SC.C]
            if d["pos"] == 1:
                assert d["stop"] <= c, (
                    f"{s}/{name} i={i}: long stop {d['stop']:.6f} is ABOVE "
                    f"close {c:.6f}")
            else:
                assert d["stop"] >= c, (
                    f"{s}/{name} i={i}: short stop {d['stop']:.6f} is BELOW "
                    f"close {c:.6f}")
