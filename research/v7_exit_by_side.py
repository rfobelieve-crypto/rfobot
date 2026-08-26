# -*- coding: utf-8 -*-
"""Do LONG and SHORT need different exits? — TODO §0.64.

Operator's proposal: split V7 into two strategies, long and short, each
with its own entry and exit.

THE ENTRY HALF IS ALREADY ANSWERED — twice, NO-GO:
  * asymmetric cutoffs, 2026-07-06 (project_exit_asym_sweep_nogo)
  * separate LONG/SHORT direction models, CLOSED 2026-07-06: on fresh OOS
    the two sides came out symmetric (70.8% / 69.5%), so the premise died
  * and measured again today on n=815 signals: UP 58.1% / +18.8 bps vs
    DOWN 60.3% / +18.7 bps. The SIDE axis is flat. The variance lives on
    side x regime (48.1% .. 67.2%), which is what §0.60 Q2 pre-registers.

THE EXIT HALF IS GENUINELY UNTESTED, and today's measurement gave it a
mechanism: post-DECODE_EPOCH the Strong DOWN firing rate in TREND_UP is
1.54%/bar against 13-18% in the other cells (§0.63). If that holds, then
`opp_signal` — which needs a reverse signal to fire — is structurally
weaker for LONGs than for SHORTs. A structural asymmetry in the exit
machinery is a different claim from a statistical asymmetry in edge, and
it does not depend on the side axis carrying any edge at all.

So this file asks ONE question: run the frozen exit variants separately
per side — does the ranking differ?

METHOD: reuse `variants_catalog` / `simulate_with_policy` from the frozen
harness unchanged, then slice its trades by side. Baseline is production
(3xATR trail, 72h cap, opp_signal=any).

HONEST LIMIT, stated before the numbers: the simulator holds ONE position
at a time, so an exit that fires earlier frees the book for entries a
later exit would have missed. The per-side trade populations are
therefore NOT strictly paired across variants — n is reported per arm and
a material n gap is itself a finding, not something to average over.

Pre-committed reading:
  * baseline wins on both sides            -> no case for splitting exits
  * the same variant wins on both sides    -> change it for both; still no
                                              case for splitting
  * a variant beats baseline on ONE side, survives the two-half check, and
    does NOT beat on the other                -> a case for a SIDE-CONDITIONAL
                                              exit rule. Note that is still
                                              one strategy with one extra
                                              condition, NOT two strategies:
                                              splitting doubles the parameter
                                              surface on a thin sample, which
                                              is the mistake.md 2026-04-13
                                              failure (BEAR submodel AUC 0.378)

MULTIPLE COMPARISONS: 6 non-baseline variants x 2 sides = 12 tests. At
p<0.05 roughly 0.6 pass by chance. Nothing here may be wired to anything;
a survivor becomes a PRE-REGISTRATION candidate, not a change.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from research.exit_variants_backtest import (                # noqa: E402
    variants_catalog, simulate_with_policy,
)
from research.v71_v7_sizing_1x import (                       # noqa: E402
    ATR_PERIOD, load_data, decode_signals, _atr_wilder,
)

OUT = ROOT / "research" / "results" / "v7_exit_by_side.json"


def arm(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0}
    net = tr["net_pct"].to_numpy(float) * 100.0      # -> bps
    return {"n": int(len(tr)), "wr": float(tr["win"].mean() * 100),
            "bps": float(net.mean()),
            "hold": float(tr["bars_held"].mean()),
            "opp_share": float((tr["exit_reason"] == "opp_signal").mean() * 100)}


def main() -> int:
    df = load_data()
    df["atr"] = _atr_wilder(df, ATR_PERIOD)
    direction, tier, warm = decode_signals(df)
    df = df.iloc[warm:].copy()
    direction, tier = direction[warm:], tier[warm:]
    mid = df.index[len(df) // 2]

    cat = variants_catalog()
    runs = {}
    for name, pol in cat.items():
        runs[name] = simulate_with_policy(df, direction, tier, pol)

    base = runs["baseline"]
    print("§0.64 出場變體按方向切 —— 多單與空單需要不同的出場嗎")
    print(f"  OOS 窗 {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}"
          f"  基準交易數 {len(base)}\n")

    # ── the operator's underlying question, answered first ──────────────
    print("── 先答根本問題：現行出場在多單上是不是比較差 ──")
    res = {"baseline_by_side": {}}
    for side in ("LONG", "SHORT"):
        a = arm(base[base["side"] == side])
        res["baseline_by_side"][side] = a
        if a["n"]:
            print(f"  {side:6} n={a['n']:<4} WR {a['wr']:5.1f}%  "
                  f"{a['bps']:+7.1f} bps  持有 {a['hold']:5.1f} bar  "
                  f"opp_signal 收尾 {a['opp_share']:4.1f}%")
    lo, sh = res["baseline_by_side"]["LONG"], res["baseline_by_side"]["SHORT"]
    if lo["n"] and sh["n"]:
        print(f"\n  opp_signal 收尾佔比：多單 {lo['opp_share']:.1f}% vs "
              f"空單 {sh['opp_share']:.1f}%"
              f"  → {'多單確實較少靠反向訊號出場' if lo['opp_share'] < sh['opp_share'] - 5 else '兩側差異不大'}")

    # ── the variant ranking, per side ───────────────────────────────────
    print("\n── 每個變體 vs 基準，分方向（Δbps；括號內為該臂 n）──")
    print(f"{'variant':16} {'多單 Δ':>16} {'空單 Δ':>16} {'兩半同號':>18}")
    res["by_variant"] = {}
    for name in cat:
        if name == "baseline":
            continue
        tr = runs[name]
        row, halves = {}, {}
        for side in ("LONG", "SHORT"):
            b = base[base["side"] == side]
            v = tr[tr["side"] == side]
            if b.empty or v.empty:
                row[side] = None
                continue
            d = (v["net_pct"].mean() - b["net_pct"].mean()) * 100 * 100
            h1 = ((v[v.index < 0].shape[0]), )      # placeholder, see below
            row[side] = {"delta_bps": float(d), "n": int(len(v)),
                         "n_base": int(len(b))}
            # two-half check on the entry timestamp
            d1 = d2 = float("nan")
            bm, vm = b[b["entry_ts"] < mid], v[v["entry_ts"] < mid]
            bn, vn = b[b["entry_ts"] >= mid], v[v["entry_ts"] >= mid]
            if len(bm) > 10 and len(vm) > 10:
                d1 = (vm["net_pct"].mean() - bm["net_pct"].mean()) * 10000
            if len(bn) > 10 and len(vn) > 10:
                d2 = (vn["net_pct"].mean() - bn["net_pct"].mean()) * 10000
            halves[side] = (d1, d2)
            row[side]["h1"] = None if d1 != d1 else round(d1, 1)
            row[side]["h2"] = None if d2 != d2 else round(d2, 1)
            row[side]["both_halves_same_sign"] = bool(
                d1 == d1 and d2 == d2 and (d1 > 0) == (d2 > 0))
        res["by_variant"][name] = row

        def cell(x):
            if not x:
                return "        —"
            return f"{x['delta_bps']:+9.1f} (n={x['n']})"
        same = []
        for side in ("LONG", "SHORT"):
            x = row.get(side)
            same.append("✓" if x and x.get("both_halves_same_sign") else "✗")
        print(f"{name:16} {cell(row.get('LONG')):>16} "
              f"{cell(row.get('SHORT')):>16}   多{same[0]} 空{same[1]}")

    # ── verdict against the pre-committed reading ───────────────────────
    def winner(side):
        best, bn = None, 0.0
        for name, row in res["by_variant"].items():
            x = row.get(side)
            if x and x["delta_bps"] > bn and x.get("both_halves_same_sign"):
                best, bn = name, x["delta_bps"]
        return best, bn

    wl, dl = winner("LONG")
    ws, ds = winner("SHORT")
    print(f"\n  多單：{wl or '無變體通過（基準勝）'}"
          + (f"  {dl:+.1f} bps" if wl else ""))
    print(f"  空單：{ws or '無變體通過（基準勝）'}"
          + (f"  {ds:+.1f} bps" if ws else ""))

    # POWER GATE — before any reading. Added after the first run, whose
    # verdict ("baseline wins both sides") was stronger than n=18/17 can
    # support: with <10 trades per half, the two-half check fails by
    # construction, so EVERY variant is rejected whether or not it works.
    # A test that cannot pass anything is not evidence of absence.
    thin = min(lo["n"], sh["n"]) < 40
    if thin:
        v = (f"**統計力不足，不出判讀**（多 n={lo['n']}／空 n={sh['n']}，"
             "門檻 40）。每半不足 10 筆時兩半同號檢查依建構必然失敗——"
             "一個什麼都通不過的檢定，不是「沒有效果」的證據。"
             "另注：本 harness 的 OOS 窗結束於解碼修法之前，"
             "**無法測到 §0.63 提出的那個機制**。")
    elif wl is None and ws is None:
        v = ("基準兩側皆勝 —— **沒有理由把出場按方向切**。"
             "現行 3×ATR trailing 對多空一樣是最好的那個。")
    elif wl == ws:
        v = (f"同一個變體（{wl}）在兩側都贏 —— 那是「該不該換出場」的問題，"
             "**不是該不該按方向切**。仍然一套規則。")
    else:
        v = (f"多空的最佳變體不同（多={wl}／空={ws}）—— **側別條件式出場**"
             "成為預註冊候選。注意這是一套策略加一個條件，"
             "不是兩套策略（切開會讓參數面翻倍、樣本減半）。")
    print(f"\n判讀：{v}")
    print("  探索性：6 變體 × 2 側 = 12 次比較，p<0.05 下約 0.6 個僥倖過關。"
          "任何存活者只成為預註冊候選，不得直接接線。")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
