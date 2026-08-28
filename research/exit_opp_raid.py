# -*- coding: utf-8 -*-
"""Exit at the OPPOSITE raid — liquidity-to-liquidity holding. §0.77.

The operator's proposal, in their words: 「如果我不按時間出場,才有機會吃到
一大段 … 比如說我現在做空,我就等到下一次做多獵取 … 當作止盈」.

Concretely: a SHORT entered off a swept HIGH stays open until the first
sweep of a LOW (the moment a long-raid signal would be born) — the down
leg has just consumed the opposite pool, so that is where the segment it
was riding ends. The 3.5 ATR disaster stop stays; the 8-bar time cap goes.

WHY THIS IS NOT A REPEAT of the nine tested variants: pool_target is the
closest and its docstring is explicit — "it cannot lengthen holds — the
time cap still binds". Nothing tested ever HELD PAST the cap, and nothing
used the opposite entry signal as information. Precedent for the idea:
V7's opp_signal exit was historically its best (85.7% WR backtest); the
mechanism — a reversal of the entry signal is itself strong information —
transfers naturally.

PRIOR EVIDENCE AGAINST, stated before running: the exit campaign's verdict
was "the take-profit family all died and the two survivors both leave
EARLIER" (fail_fast, hold_4). The mean-reversion mechanism pays fast; the
long hold exposes the position to the trend that raids the other side. So
the in-house prior says this loses. It gets tested because the operator's
hypothesis is precise, untested, and mechanically coherent — not because
the prior favours it.

TWO variants frozen together, so the answer can be attributed:
  opp_raid   stop 3.5 ATR ∨ first opposite-side sweep (exit at that bar's
             close — the sweep fact is knowable by the close) ∨ 720-bar
             safety cap
  no_cap     stop 3.5 ATR ∨ 720-bar cap only (no signal). The control:
             if opp_raid wins and no_cap loses, the SIGNAL carries the
             information; if both win, it is just duration.

Same paired harness as the nine (exit_variants.entries / cost model:
0.05 ATR per side, stop pays slip beyond, no intrabar look-ahead).
Variant-B population (pierce <= 0.25), all cached symbols, core9 breadth.

HONEST LIMIT, before the numbers: entries() has no non-overlap constraint,
and long holds overlap heavily — a real book cannot take every entry while
one is open. Average hold and the implied overlap are reported so the
per-trade R is read as per-OPPORTUNITY R, not portfolio return.

Pre-committed reading (established-arm gates as everywhere):
  * opp_raid beats baseline with CI off zero, breadth >= 6/9, halves agree,
    AND beats no_cap  -> the operator's mechanism is real; candidate for
    pre-registration on fresh fills
  * both long-hold variants lose -> the campaign verdict extends: this
    edge pays fast and holding is where it dies; record and close
  * opp_raid ~ no_cap -> duration, not signal; the raid carries no exit
    information beyond "later"
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from exit_variants import entries, run_exit                # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "exit_opp_raid.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
PIERCE_B = 0.25
SAFETY = 720                       # 30 days — a bound, not a tuned knob
random.seed(233)


def run_long_exit(bars, e, opp_sweeps, use_signal):
    """opp_raid / no_cap under the harness's exact cost conventions."""
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    op = [b[SC.O] for b in bars]
    n = len(bars)
    d, A, lvl, fill = e["d"], e["A"], e["lvl"], e["fill"]
    entry = lvl + d * SC.SLIP * A
    risk = SC.DIS * A
    stop = entry - d * risk
    last = min(fill + SAFETY, n - 1)
    # first opposite sweep strictly after the fill (binary search-free: the
    # list is sorted by bar)
    opp_j = None
    if use_signal:
        for j in opp_sweeps:
            if j > fill:
                opp_j = j
                break
    for k in range(fill + 1, last + 1):
        if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
            o = op[k]
            px = (min(stop, o) if d == 1 else max(stop, o)) - d * SC.SLIP * A
            return d * (px - entry) / risk, k
        if opp_j is not None and k >= opp_j:
            # the sweep fact is knowable by that bar's close; exit at close
            px = c[k] - d * SC.SLIP * A
            return d * (px - entry) / risk, k
    px = c[last] - d * SC.SLIP * A
    return d * (px - entry) / risk, last


def clustered_ci(pairs, n_boot=2500):
    by = defaultdict(list)
    for dd, v in pairs:
        by[dd].append(v)
    days = list(by)
    if len(days) < 4:
        return None
    m = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        vals = [x for dd in pick for x in by[dd]]
        if vals:
            m.append(st.mean(vals))
    m.sort()
    return m[int(.025 * len(m))], m[int(.975 * len(m))]


def main() -> int:
    rows = []          # per fill: ts, sym, base R, opp R, nocap R, holds
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        bars = SC.load_csv(str(fp))
        es = entries(bars)
        sweeps = SC.detect_sweeps(bars)
        by_side = {1: sorted(e2["j"] for e2 in sweeps if e2["kind"] == "buy"),
                   -1: sorted(e2["j"] for e2 in sweeps if e2["kind"] == "sell")}
        for e in es:
            pierce = ((bars[e["j"]][SC.H] - e["lvl"]) if e["d"] == -1
                      else (e["lvl"] - bars[e["j"]][SC.L])) / e["A"]
            if pierce > PIERCE_B:
                continue
            got = run_exit(bars, e, "baseline")
            if got is None:
                continue
            base_R, base_x = got
            # opposite raid for a SHORT (d=-1, swept high) = a swept LOW
            # = kind 'sell' (kd=-1); for a LONG the mirror.
            opp = by_side[1] if e["d"] == 1 else by_side[-1]
            # NOTE side mapping: d=+1 LONG came from a swept LOW (kd=-1);
            # its opposite signal is a swept HIGH (kd=+1) -> by_side key
            # matches e["d"] deliberately: LONG exits on 'buy'-side sweep.
            o_R, o_x = run_long_exit(bars, e, opp, use_signal=True)
            n_R, n_x = run_long_exit(bars, e, opp, use_signal=False)
            rows.append({"ts": bars[e["fill"]][0], "sym": sym,
                         "base": base_R, "opp": o_R, "nocap": n_R,
                         "h_opp": o_x - e["fill"], "h_nocap": n_x - e["fill"]})

    print("§0.77 對面獵取當止盈 —— 流動性騎到流動性\n")
    print(f"  母體：變體 B、{len({r['sym'] for r in rows})} 幣、n={len(rows)} 筆（配對）")
    print(f"  災難停損 3.5 ATR 保留；時間上限拿掉（安全上限 {SAFETY} 根）\n")

    mid = sorted(r["ts"] for r in rows)[len(rows) // 2]
    res = {}
    print(f"  {'出場':<26} {'meanR':>9} {'Δ vs 基準':>10} "
          f"{'Δ 日聚類CI':>20} {'廣度':>7} {'前半Δ':>8} {'後半Δ':>8} {'平均持有':>8}")
    for key, lab in (("base", "baseline（8根時間+停損）"),
                     ("opp", "opp_raid（對面獵取止盈）"),
                     ("nocap", "no_cap（純停損對照）")):
        vals = [r[key] for r in rows]
        m = st.mean(vals)
        if key == "base":
            print(f"  {lab:<26} {m:+9.4f} {'—':>10} {'—':>20} {'—':>7} "
                  f"{'—':>8} {'—':>8} {'8.0':>8}")
            res[key] = {"meanR": round(m, 4)}
            continue
        dif = [r[key] - r["base"] for r in rows]
        ci = clustered_ci([(r["ts"] // 86400, r[key] - r["base"]) for r in rows])
        per = defaultdict(list)
        for r in rows:
            if r["sym"] in CORE9:
                per[r["sym"]].append(r[key] - r["base"])
        br = sum(1 for s2 in per if st.mean(per[s2]) > 0)
        h1 = [r[key] - r["base"] for r in rows if r["ts"] < mid]
        h2 = [r[key] - r["base"] for r in rows if r["ts"] >= mid]
        hold = st.mean(r["h_opp" if key == "opp" else "h_nocap"] for r in rows)
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        print(f"  {lab:<26} {m:+9.4f} {st.mean(dif):+10.4f} {cis:>20} "
              f"{br:4d}/9  {st.mean(h1):+8.4f} {st.mean(h2):+8.4f} "
              f"{hold:7.1f}根")
        res[key] = {"meanR": round(m, 4), "delta": round(st.mean(dif), 4),
                    "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                    "breadth": f"{br}/9",
                    "h1": round(st.mean(h1), 4), "h2": round(st.mean(h2), 4),
                    "avg_hold": round(hold, 1)}

    o, nc = res["opp"], res["nocap"]
    print(f"\n  平均持有：baseline 8 根 → opp_raid {o['avg_hold']} 根 → "
          f"no_cap {nc['avg_hold']} 根")
    print("  （持有越長重疊越多——這裡的 R 是每『機會』的 R，不是組合報酬）")

    def established(x):
        return (x["ci"] is not None
                and (x["ci"][0] > 0 or x["ci"][1] < 0)
                and int(x["breadth"].split("/")[0]) >= 6
                and (x["h1"] > 0) == (x["h2"] > 0))

    print()
    if o["delta"] > 0 and established(o) and o["delta"] > nc["delta"] + 0.02:
        v = ("**機制成立**：對面獵取止盈贏過基準且贏過純停損對照 —— "
             "反向訊號帶資訊，不只是抱久。成為預註冊候選（新成交驗證）。")
    elif o["delta"] > 0 and established(o):
        v = ("**是持有時間不是訊號**：opp_raid 與 no_cap 差不多 —— "
             "拉長持有本身有利，反向訊號沒有額外資訊。那是另一個主張，"
             "要自己的機制解釋。")
    elif o["delta"] < 0 and established(o):
        v = ("**出場戰役的判決延伸**：這個 edge 付錢付得快，抱久是它死的"
             "地方 —— 連「騎到對面流動性」也救不了長持有。記錄並關閉。")
    else:
        v = f"未達成立門檻（Δ {o['delta']:+.4f}、CI {o['ci']}、廣度 {o['breadth']}、兩半 {o['h1']:+.3f}/{o['h2']:+.3f}）—— 列觀察"
    print(f"判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
