# -*- coding: utf-8 -*-
"""「勝率不高，靠盈虧比撐起來」——把這句話量出來（2026-09-03 使用者主張）。

使用者的手動判讀說法：掃單策略勝率本來就不高，獲利的關鍵是抓到反轉之後
延續的距離，也就是靠盈虧比。這是一個可以被資料否證的說法，而且它指向的
修法（讓贏家跑）跟既有證據（`exit_variants.json`：hold_12 FAIL、
trail_* FAIL、hold_4 PASS）方向相反。所以先量清楚再決定信誰。

四個量測，全部在**凍結的成交**上做，不動任何進出場規則：

  1 收益拆解  meanR = WR x 平均獲利 - (1-WR) x 平均虧損；盈虧比是多少，
              以及總 R 有多少集中在最好的 10% / 1%
  2 MFE       每筆交易在持有期內的最大有利波動（R 單位）。若 MFE 遠大於
              實現 R，時間出場正在砍掉延續的部分（支持使用者）；若兩者
              接近，就是「根本沒跑那麼遠」（反對使用者）
  3 持有曲線  若在第 1..24 根出場，平均 R 各是多少。這是**形狀診斷**不是
              參數挑選（mistake.md 2026-06-20）——看的是曲線往上還往下，
              不是選最高點。停損照舊，只換時間上限。
  4 磁鐵分解  §0.94 的近/遠磁鐵，效應是走勝率還是走盈虧比？這決定它是
              「更常對」還是「對的時候跑更遠」，兩者的用法完全不同。

Run: python research/sweep_failure/payoff_anatomy.py
Out: research/results/sweep_payoff_anatomy.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SLIP"] = "0"
import sweep_core as SC            # noqa: E402
from sweep_forward import SCEN     # noqa: E402
import room_ahead as RA            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_payoff_anatomy.json"
HOLDS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]


def walk(bars, sym):
    """Re-walk the frozen fills, recording MFE/MAE and R at many time caps.

    Entry, stop and slippage are exactly sweep_core's; only the moment we
    stop looking changes. A stop-out closes the trade for every longer cap
    too -- otherwise a 'longer hold' would silently be allowed to survive a
    stop it actually hit.
    """
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    idx = {b[0]: i for i, b in enumerate(bars)}
    pools = RA.all_pools(bars)
    s = SCEN["A"]
    out = []
    for fill_ts, _e, r_ref, lvl, atr, stopped, pierce, side in SC.backtest_symbol(bars):
        f = idx.get(fill_ts)
        if f is None:
            continue
        d = 1 if side == "LONG" else -1
        entry = lvl + d * SC.SLIP * atr
        risk = SC.DIS * atr
        cost = (s["entry"] + (s["sexit"] if stopped else s["texit"])) \
            / 1e4 * lvl / (SC.DIS * atr)
        stop = entry - d * risk
        mfe = mae = 0.0
        hit = None
        by_hold = {}
        for k in range(f + 1, min(f + max(HOLDS) + 1, n)):
            fav = d * (h[k] - entry) / risk if d == 1 else d * (l[k] - entry) / risk
            adv = d * (l[k] - entry) / risk if d == 1 else d * (h[k] - entry) / risk
            mfe = max(mfe, fav)
            mae = min(mae, adv)
            if hit is None and ((d == 1 and l[k] <= stop) or (d == -1 and h[k] >= stop)):
                hit = k
            step = k - f
            if step in HOLDS:
                if hit is not None:
                    by_hold[step] = -1.0 - SC.SLIP / SC.DIS
                else:
                    ex = c[k] - d * SC.SLIP * atr
                    by_hold[step] = d * (ex - entry) / risk
        if 8 not in by_hold:
            continue
        _room, magnet = RA.distances(pools, f, lvl, d, atr)
        out.append(dict(sym=sym, ts=fill_ts, net=by_hold[8] - cost, cost=cost,
                        mfe=mfe, mae=mae, magnet=magnet, pierce=pierce,
                        stopped=hit is not None, side=side,
                        holds={k: v - cost for k, v in by_hold.items()}))
    return out


def decomp(rows, label):
    rs = [x["net"] for x in rows]
    n = len(rs)
    w = [x for x in rs if x > 0]
    lo = [x for x in rs if x <= 0]
    wr = len(w) / n
    aw = sum(w) / len(w) if w else 0.0
    al = sum(lo) / len(lo) if lo else 0.0
    srt = sorted(rs, reverse=True)
    tot = sum(srt)
    top10 = sum(srt[:max(1, n // 10)]) / tot if tot else float("nan")
    top1 = sum(srt[:max(1, n // 100)]) / tot if tot else float("nan")
    print(f"  {label:<16}n={n:<6} meanR {sum(rs)/n:+.4f}  勝率 {100*wr:.1f}%  "
          f"平均獲利 {aw:+.3f}  平均虧損 {al:+.3f}  盈虧比 "
          f"{abs(aw/al) if al else float('nan'):.2f}")
    print(f"  {'':<16}最好 10% 貢獻總 R 的 {100*top10:.0f}%，"
          f"最好 1% 貢獻 {100*top1:.0f}%")
    return dict(n=n, mean=sum(rs) / n, wr=100 * wr, avg_win=aw, avg_loss=al,
                payoff=abs(aw / al) if al else None,
                top10_share=top10, top1_share=top1)


def main() -> int:
    print("=" * 88)
    print("  「勝率不高、靠盈虧比」——收益拆解 / MFE / 持有曲線 / 磁鐵分解")
    print("=" * 88)
    rows = []
    for sym in RA.SYMS:
        p = RA.CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            continue
        rows += walk(SC.load_csv(str(p)), sym)
    res = {}

    print("\n  1 收益拆解（core9 全歷史回測，凍結出場＝停損 3.5ATR + 8 根時間上限）")
    res["all"] = decomp(rows, "全體")

    print("\n  2 MFE：贏家到底有沒有跑遠，時間出場砍掉多少")
    mfe = sorted(x["mfe"] for x in rows)
    real = [x["net"] for x in rows]
    q = lambda a, p: a[int(p * (len(a) - 1))]  # noqa: E731
    print(f"     MFE 分位  p25 {q(mfe,.25):.2f}R  中位 {q(mfe,.5):.2f}R  "
          f"p75 {q(mfe,.75):.2f}R  p90 {q(mfe,.9):.2f}R  p99 {q(mfe,.99):.2f}R")
    print(f"     實現 R 中位 {sorted(real)[len(real)//2]:+.3f}R，"
          f"平均 MFE {sum(mfe)/len(mfe):.3f}R")
    cap = sum(1 for x in rows if x["mfe"] >= 1.0)
    got = sum(1 for x in rows if x["net"] >= 1.0)
    print(f"     MFE 曾達 +1R 的：{cap} 筆（{100*cap/len(rows):.1f}%）；"
          f"真的收在 +1R 以上的：{got} 筆（{100*got/len(rows):.1f}%）")
    print(f"     -> 每 1 筆摸到 +1R 的交易，只有 {got/max(cap,1):.2f} 筆守得住")
    res["mfe"] = dict(median=q(mfe, .5), p90=q(mfe, .9), mean=sum(mfe) / len(mfe),
                      reach_1r=cap, keep_1r=got, n=len(rows))

    print("\n  3 持有曲線（形狀診斷，不是挑參數；停損不變，只換時間上限）")
    print(f"     {'上限(根)':<10}{'meanR':>10}{'勝率':>9}{'停損率':>9}")
    curve = {}
    for k in HOLDS:
        sub = [x["holds"][k] for x in rows if k in x["holds"]]
        if not sub:
            continue
        wr = 100 * sum(1 for v in sub if v > 0) / len(sub)
        sr = 100 * sum(1 for v in sub if v < -0.9) / len(sub)
        curve[k] = dict(mean=sum(sub) / len(sub), wr=wr, stop=sr, n=len(sub))
        mark = "  <- 現行" if k == 8 else ""
        print(f"     {k:<10}{curve[k]['mean']:>+10.4f}{wr:>8.1f}%{sr:>8.1f}%{mark}")
    res["hold_curve"] = curve

    print("\n  4 §0.94 磁鐵效應：走勝率還是走盈虧比？")
    near = [x for x in rows if x["magnet"] <= 1.00]
    far = [x for x in rows if 1.00 < x["magnet"] < RA.NOPOOL and x["magnet"] > 2.23]
    res["near"] = decomp(near, "近磁鐵 ≤1.0")
    res["far"] = decomp(far, "遠磁鐵 >2.23")
    mn = sum(x["mfe"] for x in near) / len(near)
    mf = sum(x["mfe"] for x in far) / len(far)
    print(f"  {'':<16}平均 MFE：近 {mn:.3f}R vs 遠 {mf:.3f}R")
    res["near"]["mfe_mean"] = mn
    res["far"]["mfe_mean"] = mf

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
