# -*- coding: utf-8 -*-
"""G4 look-ahead check — is the liquidation gauge usable in real time?

THE PROBLEM, which invalidates the finding if unaddressed:
`fill_ts` is the bar a fill happened in. The liquidation total FOR THAT
SAME BAR is only complete at its close. Conditioning a fill on its own
bar's liquidation reading therefore uses information that did not exist
when the trade was placed — a mild but real look-ahead. ADX does not have
this problem (ADX(14) is built from prior bars); a same-bar sum does.

An instrument that only works with same-bar information is not a gauge,
it is a description of what already happened to the trade. Same family as
mistake.md 2026-07-28 (the shadow harness replaying the hour BEFORE a
signal existed) — the number looks fine either way, so the alignment has
to be proven, not assumed.

TEST: recompute the gauge with the reading LAGGED one bar (only data
available at the fill bar's open) and compare.

Pre-committed reading:
  * lagged gap holds within ~30% of same-bar  -> real and usable
  * lagged gap collapses                       -> the effect was the trade's
                                                  own bar; NOT a gauge, drop it
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from research.positioning_gauges import (                  # noqa: E402
    _load, _z, sf_fills, Z_WIN,
)
from research.positioning_gauges_marginal import gap_ci     # noqa: E402

OUT = ROOT / "research" / "results" / "g4_lag_check.json"
random.seed(41)


def g4_map(lag_bars: int) -> dict[int, str]:
    lq = _load("cg_liq_agg_1h")
    tot = (lq["aggregated_long_liquidation_usd"].astype(float)
           + lq["aggregated_short_liquidation_usd"].astype(float))
    z = _z(np.log1p(tot))
    if lag_bars:
        z = z.shift(lag_bars)
    return {int(t.timestamp()): ("BURST 爆量" if v > 1 else "平靜")
            for t, v in z.items() if pd.notna(v)}


def main() -> int:
    sf = sf_fills()
    print("§0.65c G4 前視檢查 —— 這個儀表在真實時點用得了嗎\n")
    print("  問題：fill_ts 那根 bar 的清算總額，要到那根收盤才完整。")
    print("        拿它去條件化同一根 bar 的成交 = 用了下單當下不存在的資訊。\n")
    res = {}
    print(f"{'口徑':<26} {'平靜 n':>7} {'爆量 n':>7} {'平靜R':>9} "
          f"{'爆量R':>9} {'差':>9} {'差的日聚類CI':>22}")
    for lag, lab in ((0, "同根 bar（原始，有前視）"), (1, "落後一根（真實可得）"),
                     (2, "落後兩根（穩健性）")):
        gm = g4_map(lag)
        buck = defaultdict(list)
        for r in sf:
            s = gm.get(r["ts"])
            if s:
                buck[s].append(r)
        if len(buck) < 2:
            print(f"{lab:<26} 樣本不足")
            continue
        q, b = buck["平靜"], buck["BURST 爆量"]
        mq, mb = st.mean(x["R"] for x in q), st.mean(x["R"] for x in b)
        ci = gap_ci(q, b)
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        print(f"{lab:<26} {len(q):7d} {len(b):7d} {mq:+9.4f} {mb:+9.4f} "
              f"{mq - mb:+9.4f} {cis:>22}")
        res[f"lag{lag}"] = {"calm_n": len(q), "burst_n": len(b),
                            "calm_R": round(mq, 4), "burst_R": round(mb, 4),
                            "gap": round(mq - mb, 4),
                            "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None}

    a, c = res.get("lag0"), res.get("lag1")
    if a and c:
        keep = c["gap"] / a["gap"] if a["gap"] else 0
        print(f"\n  落後一根保留了原效果的 {100*keep:.0f}%")
        ok = keep >= 0.7 and c["ci"] and c["ci"][0] > 0
        v = ("**通過**：效果在只用下單當下可得的資訊時仍然成立，"
             "G4 是真的儀表、可即時使用"
             if ok else
             f"**不通過**：落後一根之後效果塌到 {c['gap']:+.4f}"
             f"（原 {a['gap']:+.4f}）—— 那是成交自己那根 bar 的性質，"
             "不是一個可用的環境儀表")
        print(f"\n判讀：{v}")
        res["verdict"] = v
        res["retention"] = round(keep, 3)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
