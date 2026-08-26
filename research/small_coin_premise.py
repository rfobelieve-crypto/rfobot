# -*- coding: utf-8 -*-
"""Do less-liquid coins actually carry more edge? — TODO §0.67.

The operator's plan is to port V7's concept to small caps, reasoning that
BTC is too efficient and inefficiency lives in smaller markets. That is a
reasonable prior and it is worth a lot of engineering IF TRUE — so it gets
tested first, on data already in hand, before the harness is built.

Two pieces of in-house counter-evidence make the test necessary:

  1. ETH was WORSE than BTC, not better. V7's clean AUC ported to ETH gave
     0.5057 against BTC's 0.5412 (2026-07-23), and three follow-ups
     (hyperparameter retune, feature elimination, suspect verification)
     all confirmed NO-GO. ETH is less efficient than BTC on any ranking,
     so the premise predicted the opposite of what happened.

  2. The raid line's WIDER universe scores worse: 29 coins n=1127
     meanR -0.0206 against core9 n=346 -0.0069. The nine most liquid names
     beat the full basket.

Neither is decisive — ETH is not a small cap, and the universe comparison
mixes many things. But the raid ledger spans 29 coins with a frozen rule
applied identically to all of them, which is close to a controlled test of
exactly this question: same strategy, same parameters, different liquidity.

METHOD: per-coin meanR from the frozen shadow ledger, ranked against a
liquidity proxy computed from the coin's own bars (median hourly quote
volume). Spearman correlation between liquidity rank and edge, plus the
coarse core9-vs-rest split. Day-clustered CI on the group difference.

Pre-committed reading:
  * edge rises as liquidity falls  -> premise supported, build the harness
  * no relation                    -> premise unsupported; the harness is
                                      still worth building (cheap testing
                                      has value) but expectations must drop
  * edge falls as liquidity falls  -> premise contradicted on this line;
                                      say so plainly before any engineering

LIMIT stated up front: this tests the premise for the SWEEP-FAILURE rule,
not for V7's direction model. A liquidity effect can differ between a
microstructure rule and a 4h prediction model. It is evidence, not proof.
"""
from __future__ import annotations

import csv
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

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
OUT = ROOT / "research" / "results" / "small_coin_premise.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
random.seed(53)


def spearman(a, b):
    def rank(x):
        s = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(s):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math_sqrt(sum((x - ma) ** 2 for x in ra))
    db = math_sqrt(sum((x - mb) ** 2 for x in rb))
    return num / (da * db) if da and db else 0.0


def math_sqrt(x):
    return x ** 0.5


def gap_ci(a, b, n_boot=4000):
    by = defaultdict(lambda: ([], []))
    for d, v in a:
        by[d][0].append(v)
    for d, v in b:
        by[d][1].append(v)
    days = list(by)
    if len(days) < 5:
        return None
    out = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        x = [v for k in pick for v in by[k][0]]
        y = [v for k in pick for v in by[k][1]]
        if x and y:
            out.append(st.mean(x) - st.mean(y))
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def main() -> int:
    per = defaultdict(list)
    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("variant_b") != "1" or r.get("status") != "CLOSED":
                continue
            try:
                per[r["symbol"]].append(
                    (int(float(r["fill_ts"])), float(r["net_r"])))
            except (ValueError, TypeError, KeyError):
                continue

    liq = {}
    for sym in list(per):
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        # median hourly quote volume = close * volume, robust to spikes
        v = sorted(b[SC.C] * b[SC.V] for b in bars[-2000:]
                   if b[SC.C] and b[SC.V])
        if v:
            liq[sym] = v[len(v) // 2]

    syms = [s for s in per if s in liq and len(per[s]) >= 15]
    syms.sort(key=lambda s: -liq[s])
    print("§0.67 小幣真的比較沒效率嗎 —— 用凍結規則、29 幣、同一套參數\n")
    print(f"{'幣':<8} {'n':>5} {'meanR':>9} {'中位小時成交額':>16} {'流動性排名':>9}")
    rows = []
    for i, s in enumerate(syms, 1):
        m = st.mean(x[1] for x in per[s])
        tag = "  core9" if s in CORE9 else ""
        print(f"{s:<8} {len(per[s]):5d} {m:+9.4f} {liq[s]:16,.0f} {i:8d}{tag}")
        rows.append({"sym": s, "n": len(per[s]), "meanR": round(m, 4),
                     "liq": liq[s], "rank": i})

    rho = spearman([r["rank"] for r in rows], [r["meanR"] for r in rows])
    print(f"\n  流動性排名 vs meanR 的 Spearman：{rho:+.3f}")
    print("  （排名 1 = 最有流動性。正相關 = 越不流動越賺 = 前提成立）")

    a = [(t // 86400, v) for s in syms if s not in CORE9 for t, v in per[s]]
    b = [(t // 86400, v) for s in syms if s in CORE9 for t, v in per[s]]
    ma, mb = st.mean(v for _, v in a), st.mean(v for _, v in b)
    ci = gap_ci(a, b)
    cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
    print(f"\n  非 core9（較小）n={len(a)}  meanR {ma:+.4f}")
    print(f"  core9（較大）   n={len(b)}  meanR {mb:+.4f}")
    print(f"  差（小 − 大）{ma - mb:+.4f}   日聚類 CI {cis}")

    if rho > 0.3 and ma > mb:
        v = "**前提成立**：越不流動越賺，值得投入多幣工程"
    elif abs(rho) <= 0.3 and ci and ci[0] <= 0 <= ci[1]:
        v = ("**前提未獲支持**：流動性與 edge 無關係，兩組差的 CI 含零。"
             "框架仍值得建（便宜的測試本身有價值），但期望值要調低——"
             "不要預設換個幣就會有 edge。")
    else:
        v = (f"**前提被否定**：越不流動反而越差（rho {rho:+.3f}，"
             f"小幣 {ma:+.4f} vs 大幣 {mb:+.4f}）。"
             "工程可以做，但論證要換一個，不能是「小幣比較沒效率」。")
    print(f"\n判讀：{v}")
    print("\n  界定：這測的是**掃單失敗規則**的流動性效應，不是 V7 方向模型的。"
          "微結構規則與 4h 預測模型的流動性關係可以不同——這是證據不是證明。")
    OUT.write_text(json.dumps(
        {"per_coin": rows, "spearman_rank_vs_edge": round(rho, 4),
         "small_meanR": round(ma, 4), "large_meanR": round(mb, 4),
         "gap_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
         "verdict": v}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
