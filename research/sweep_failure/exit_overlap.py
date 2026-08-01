# -*- coding: utf-8 -*-
"""Are hold_4 and fail_fast the same effect wearing two hats?

Both survivors of exit_variants.py shorten holding time and both are
exit-on-close rules. Before either could ever be registered as a
replacement exit, we have to know whether they are one finding or two —
otherwise a "combined" variant would double-count the same mechanism and
look better than it is.

Three questions, in the order that can kill the idea fastest:
  1 how often do they exit on the SAME bar (mechanical overlap)
  2 are their per-trade improvements correlated (statistical overlap)
  3 does each survive INSIDE the other — i.e. does fail_fast still add
    anything once holding is already capped at 4, and vice versa. This
    is the anti-repackaging test the terrain campaign used as G2, and it
    is the one that decides whether a combination is worth registering.

A combined variant is also scored, but only for the record: if the
residual tests fail, the combination is one effect and its number must
not be presented as if two independent edges stacked.

Run: python research/sweep_failure/exit_overlap.py
Out: research/results/exit_overlap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from exit_variants import SYMS, entries, run_exit, _cols  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/exit_overlap.json"


def run_combined(bars, e):
    """hold cap 4 AND leave on a close back through the level — whichever
    comes first. Implemented here rather than in exit_variants so the
    frozen variant list stays exactly as pre-registered."""
    h, lo, c, op = _cols(bars)
    n = len(bars)
    d, A, lvl, fill = e["d"], e["A"], e["lvl"], e["fill"]
    entry = lvl + d * SC.SLIP * A
    risk = SC.DIS * A
    stop = entry - d * risk
    last = min(fill + 4, n - 1)
    for k in range(fill + 1, last + 1):
        if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
            px = (min(stop, op[k]) if d == 1 else max(stop, op[k])) \
                - d * SC.SLIP * A
            return d * (px - entry) / risk, k
        through = (c[k] < lvl) if d == 1 else (c[k] > lvl)
        if through:
            px = c[k] - d * SC.SLIP * A
            return d * (px - entry) / risk, k
    px = c[last] - d * SC.SLIP * A
    return d * (px - entry) / risk, last


def paired_stats(diffs, rng):
    a = np.concatenate(diffs)
    m = float(a.mean())
    boots = [float(rng.choice(a, len(a), True).mean()) for _ in range(2000)]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    null = [float((a * rng.choice([-1, 1], len(a))).mean()) for _ in range(2000)]
    p = float((np.abs(null) >= abs(m)).mean())
    return m, float(lo_ci), float(hi_ci), p


def main() -> int:
    print("=" * 78)
    print("  hold_4 vs fail_fast — 同一個效果，還是兩個？")
    print("=" * 78)
    rng = np.random.default_rng(23)
    same_bar = tot = 0
    corrs = []
    d_h, d_f, d_c = [], [], []
    d_f_in_h, d_h_in_f = [], []
    for sym in SYMS:
        try:
            bars = SC.load_csv(str(LT.CACHE / f"{sym}USDT_1h.csv"))
        except Exception:
            continue
        es = entries(bars)
        base = [run_exit(bars, e, "baseline") for e in es]
        h4 = [run_exit(bars, e, "hold_4") for e in es]
        ff = [run_exit(bars, e, "fail_fast") for e in es]
        cb = [run_combined(bars, e) for e in es]
        same_bar += sum(1 for a, b in zip(h4, ff) if a[1] == b[1])
        tot += len(es)
        dh = np.array([a[0] - b[0] for a, b in zip(h4, base)])
        df = np.array([a[0] - b[0] for a, b in zip(ff, base)])
        dc = np.array([a[0] - b[0] for a, b in zip(cb, base)])
        d_h.append(dh)
        d_f.append(df)
        d_c.append(dc)
        # residual: what each adds ON TOP of the other
        d_f_in_h.append(np.array([a[0] - b[0] for a, b in zip(cb, h4)]))
        d_h_in_f.append(np.array([a[0] - b[0] for a, b in zip(cb, ff)]))
        if dh.std() > 0 and df.std() > 0:
            corrs.append((sym, float(np.corrcoef(dh, df)[0, 1])))

    print(f"\n  [1] 機械重疊：{100*same_bar/tot:.0f}% 的交易兩者在同一根出場"
          f"（n={tot}）")
    cs = [c for _s, c in corrs]
    print(f"  [2] 統計重疊：逐幣 dR 相關係數 中位 {np.median(cs):+.2f} "
          f"（範圍 {min(cs):+.2f}~{max(cs):+.2f}）")

    res = {"same_bar_pct": round(100 * same_bar / tot, 1),
           "corr_median": round(float(np.median(cs)), 3),
           "corr_by_symbol": {s: round(c, 3) for s, c in corrs}}
    print(f"\n  {'比較':<26}{'dR':>9}{'CI':>18}{'p':>8}")
    for lab, diffs, key in (
            ("hold_4 vs baseline", d_h, "hold_4"),
            ("fail_fast vs baseline", d_f, "fail_fast"),
            ("兩者合併 vs baseline", d_c, "combined"),
            ("fail_fast 加在 hold_4 上", d_f_in_h, "ff_residual"),
            ("hold_4 加在 fail_fast 上", d_h_in_f, "h4_residual")):
        m, lo_ci, hi_ci, p = paired_stats(diffs, rng)
        mark = ""
        if key.endswith("residual"):
            mark = "  ← 殘餘增量" + (" 顯著" if (lo_ci > 0 and p < 0.05)
                                     else " 不顯著")
        print(f"  {lab:<26}{m:>+9.4f}{f'[{lo_ci:+.4f},{hi_ci:+.4f}]':>18}"
              f"{p:>8.3f}{mark}")
        res[key] = {"dR": round(m, 4), "ci": [round(lo_ci, 4), round(hi_ci, 4)],
                    "p": p}

    ff_sig = res["ff_residual"]["ci"][0] > 0 and res["ff_residual"]["p"] < 0.05
    h4_sig = res["h4_residual"]["ci"][0] > 0 and res["h4_residual"]["p"] < 0.05
    res["two_effects"] = bool(ff_sig and h4_sig)
    print()
    if ff_sig and h4_sig:
        print("  → 兩個獨立效果：合併值可視為疊加，值得註冊合併變體")
    elif ff_sig or h4_sig:
        print("  → 一個效果為主、另一個只在特定情況補刀：只註冊較強的那個，"
              "合併不算兩份證據")
    else:
        print("  → 同一個效果的兩種寫法：**不可**把合併數字當成兩個 edge 疊加")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
