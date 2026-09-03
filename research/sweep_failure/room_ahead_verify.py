# -*- coding: utf-8 -*-
"""room_ahead 的反向結果先驗儀器,再談解讀(mistake.md 2026-08-02)。

room_ahead.py 的兩個預註冊預測都「陣亡」,但後方磁鐵那一維出現了一個
**與宣稱方向相反、卻極度一致**的效應(近 +0.104 / 遠 −0.035,9/9 幣同號,
日聚類 CI [−0.186,−0.094])。這種結果的第一動作是查產生它的程式碼,不是
解讀它——尤其這支 pool 生命週期是我剛寫的第二份實作。

三道查證:
  A 池死亡時點:堆積法 vs 逐池暴力掃描,必須逐池完全相同
  B 哨兵值:把「後方沒有任何池」的 99 ATR 那批(2.2%)整批剔除後效應還在嗎
  C 這是不是 regime 的替身:同一效應在 ADX 的 RANGING / TRENDING 各自
    格內還在嗎?若只在跨格出現,它量的是 regime(§0.59 已有),不是新資訊

Run: python research/sweep_failure/room_ahead_verify.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SLIP"] = "0"
import sweep_core as SC            # noqa: E402
from sweep_forward import SCEN     # noqa: E402
import room_ahead as RA            # noqa: E402
from research.crowd_battery2 import adx_state  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def naive_pools(bars):
    """The O(pools x bars) reference implementation — slow but obvious."""
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    raw = []
    for i in range(SC.PIVOT, n - SC.PIVOT):
        seg = range(i - SC.PIVOT, i + SC.PIVOT + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            raw.append((i + SC.PIVOT, h[i], 1))
        if all(l[i] <= l[k] for k in seg) and any(l[i] < l[k] for k in seg if k != i):
            raw.append((i + SC.PIVOT, l[i], -1))
    import level_types as LT
    for _kind, items in LT.build_levels(bars).items():
        raw.extend(items)
    out = []
    for est, price, side in raw:
        death = n
        for j in range(est + 1, n):
            if (side == 1 and h[j] > price) or (side == -1 and l[j] < price):
                death = j
                break
        out.append((est, price, death))
    return out


def cells(rows, lo_q, hi_q):
    lo = [r for r in rows if r[3] <= lo_q]
    hi = [r for r in rows if r[3] > hi_q]
    if not lo or not hi:
        return None
    m = lambda g: sum(x[1] for x in g) / len(g)  # noqa: E731
    return dict(n_lo=len(lo), n_hi=len(hi), lo=m(lo), hi=m(hi), gap=m(hi) - m(lo))


def main() -> int:
    print("=" * 80)
    print("  room_ahead 儀器查證")
    print("=" * 80)

    # ---- A 池死亡時點對帳（BTC，全量）
    bars = SC.load_csv(str(RA.CACHE / "BTCUSDT_1h.csv"))
    fast = sorted(RA.all_pools(bars))
    slow = sorted(naive_pools(bars))
    same = fast == slow
    print(f"\n  A 池生命週期對帳（BTC，{len(fast)} 個池）："
          f"{'完全相同 ✅' if same else '不一致 ❌'}")
    if not same:
        diff = [(a, b) for a, b in zip(fast, slow) if a != b][:5]
        for a, b in diff:
            print(f"      堆積 {a}  vs  暴力 {b}")
        print("      -> 停手,先修對帳")
        return 1

    # ---- rebuild the labelled rows, this time also carrying the regime cell
    rows = []
    for sym in RA.SYMS:
        p = RA.CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            continue
        b = SC.load_csv(str(p))
        idx = {x[0]: i for i, x in enumerate(b)}
        st = adx_state(b)                      # 凍結儀器,不是第二份實作
        pools = RA.all_pools(b)
        s = SCEN["A"]
        for fill_ts, _e, r, lvl, atr, stopped, _pi, side in SC.backtest_symbol(b):
            f = idx.get(fill_ts)
            if f is None:
                continue
            d = 1 if side == "LONG" else -1
            _room, magnet = RA.distances(pools, f, lvl, d, atr)
            cost = (s["entry"] + (s["sexit"] if stopped else s["texit"])) \
                / 1e4 * lvl / (SC.DIS * atr)
            cell = st.get(b[f][0] // 3600 * 3600, "")
            rows.append((fill_ts, r - cost, sym, magnet, cell))

    lo_q, hi_q = 1.00, 2.23                     # room_ahead 印出的切點,不重算
    base = cells(rows, lo_q, hi_q)
    print(f"\n  複算全體：低 {base['lo']:+.4f} (n={base['n_lo']}) / "
          f"高 {base['hi']:+.4f} (n={base['n_hi']}) / 差 {base['gap']:+.4f}")

    # ---- B 哨兵值
    keep = [r for r in rows if r[3] < RA.NOPOOL]
    b2 = cells(keep, lo_q, hi_q)
    drop = len(rows) - len(keep)
    print(f"\n  B 剔除「後方無池」{drop} 筆（{100*drop/len(rows):.1f}%）後：")
    print(f"      低 {b2['lo']:+.4f} / 高 {b2['hi']:+.4f} / 差 {b2['gap']:+.4f}"
          f"  → {'效應仍在 ✅' if b2['gap'] < -0.05 else '效應消失 ❌'}")

    # ---- C regime 替身檢定
    print(f"\n  C 分 regime 格內（ADX §0.49d 凍結儀器）——"
          f"若只在跨格出現就是 regime 的替身：")
    byc = defaultdict(list)
    for r in keep:
        byc[r[4] or "(暖機)"].append(r)
    print(f"      {'格':<10}{'n':>7}{'近':>10}{'遠':>10}{'差':>10}")
    for c in ("RANGING", "NEUTRAL", "TRENDING", "(暖機)"):
        sub = byc.get(c, [])
        st_ = cells(sub, lo_q, hi_q)
        if st_ is None:
            print(f"      {c:<10}{len(sub):>7}   樣本不足")
            continue
        print(f"      {c:<10}{len(sub):>7}{st_['lo']:>+10.4f}"
              f"{st_['hi']:>+10.4f}{st_['gap']:>+10.4f}")

    # ---- D 前視洩漏：只用「時間定義」的池重算
    # 2026-07-29 的 equal-levels 主張因為 pivot 鄰域會洩漏而撤回（價格一直
    # 尊重的價位會累積 pivot，而「價格回來了」正是標籤）。本維度量的是
    # level「之外」而不是鄰域，但最便宜的了結方式是整批換成當時判定為
    # 免疫的時間定義池（session / PDH-PDL / PWH-PWL）再看效應在不在。
    print("\n  D 只用時間定義池（session/PDH-PDL/PWH-PWL，已知免疫前視）：")
    rows_t = []
    for sym in RA.SYMS:
        p = RA.CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            continue
        b = SC.load_csv(str(p))
        idx = {x[0]: i for i, x in enumerate(b)}
        pools = time_pools(b)
        s = SCEN["A"]
        for fill_ts, _e, r, lvl, atr, stopped, pierce, side in SC.backtest_symbol(b):
            f = idx.get(fill_ts)
            if f is None:
                continue
            d = 1 if side == "LONG" else -1
            _room, magnet = RA.distances(pools, f, lvl, d, atr)
            cost = (s["entry"] + (s["sexit"] if stopped else s["texit"])) \
                / 1e4 * lvl / (SC.DIS * atr)
            rows_t.append((fill_ts, r - cost, sym, magnet, pierce))
    v = sorted(x[3] for x in rows_t if x[3] < RA.NOPOOL)
    tl, th = v[len(v) // 3], v[2 * len(v) // 3]
    kt = [r for r in rows_t if r[3] < RA.NOPOOL]
    st_ = cells(kt, tl, th)
    per = {}
    for sym in RA.SYMS:
        sub = cells([r for r in kt if r[2] == sym], tl, th)
        per[sym] = sub["gap"] if sub else None
    pos = sum(1 for x in per.values() if x is not None and x < 0)
    print(f"      切點 {tl:.2f}/{th:.2f} ATR  低 {st_['lo']:+.4f} / "
          f"高 {st_['hi']:+.4f} / 差 {st_['gap']:+.4f}")
    print(f"      逐幣同號 {pos}/9  → "
          f"{'效應仍在,非 pivot 鄰域洩漏 ✅' if st_['gap'] < -0.05 and pos >= 6 else '存疑 ❌'}")

    # ---- E 是不是 pierce 的替身（變體 B 那個前瞻失敗的特徵）
    print("\n  E 在 pierce 三等分格內（若只在跨格出現＝pierce 的替身）：")
    pv = sorted(x[4] for x in kt)
    pl, ph = pv[len(pv) // 3], pv[2 * len(pv) // 3]
    print(f"      {'pierce 格':<14}{'n':>7}{'近':>10}{'遠':>10}{'差':>10}")
    for name, sel in (("淺 ≤%.2f" % pl, lambda x: x[4] <= pl),
                      ("中", lambda x: pl < x[4] <= ph),
                      ("深 >%.2f" % ph, lambda x: x[4] > ph)):
        sub = [r for r in kt if sel(r)]
        s2 = cells(sub, tl, th)
        if s2 is None:
            print(f"      {name:<14}{len(sub):>7}   樣本不足")
            continue
        print(f"      {name:<14}{len(sub):>7}{s2['lo']:>+10.4f}"
              f"{s2['hi']:>+10.4f}{s2['gap']:>+10.4f}")
    return 0


def time_pools(bars):
    """Only the three time-defined pool families, with death bars."""
    import heapq
    import level_types as LT
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    raw = []
    for _kind, items in LT.build_levels(bars).items():
        raw.extend(items)
    born = defaultdict(list)
    for est, price, side in raw:
        born[est].append((price, side))
    up, dn, death = [], [], {}
    for j in range(n):
        while up and up[0][0] < h[j]:
            _p, key = heapq.heappop(up)
            death[key] = j
        while dn and -dn[0][0] > l[j]:
            _p, key = heapq.heappop(dn)
            death[key] = j
        for price, side in born.get(j, ()):
            key = (j, price, side)
            (heapq.heappush(up, (price, key)) if side == 1
             else heapq.heappush(dn, (-price, key)))
    return [(est, price, death.get((est, price, side), n))
            for est, price, side in raw]


if __name__ == "__main__":
    raise SystemExit(main())
