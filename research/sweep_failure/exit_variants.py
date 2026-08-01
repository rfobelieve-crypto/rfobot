# -*- coding: utf-8 -*-
"""Strategy #2 exit research — PRE-REGISTERED variants vs the frozen exit.

The frozen exit (sweep_core.backtest_symbol): stop at 3.5 ATR from entry,
time exit at the close of bar +8, 0.05 ATR slippage per side. In practice
87% of trades leave by the time cap and 13% by the stop; the time-exit
group averages +0.24R at 64% WR while every stop-out books -1.014R.

Why this study is shaped the way it is:
  * The ENTRY rules are frozen and untouched. Every variant re-exits the
    exact same fills, so comparisons are PAIRED — far more powerful than
    comparing two independent samples, and it removes entry luck entirely.
  * Variants are CATEGORICAL, not a parameter sweep. Sweeping stop/hold
    grids on 9 symbols is the mistake.md 2026-06-20 trap; each variant
    below encodes a stated mechanism instead.
  * V7's own exit study is the prior: 11 variants ALL lost to plain
    3xATR because tightening clips winners early. Expect the same here;
    the point is to find out, not to find a winner.

Pre-registered variants (frozen before the first run):
  V1 trail_2atr   trailing stop 2.0 ATR below the running peak, armed
                  once price is +1R ahead. Tests "protect the winner".
  V2 trail_1atr   same, tighter (1.0 ATR). Tests whether V1 failing is
                  about the distance or about trailing at all.
  V3 half_at_1r   scale out half at +1R, remainder runs to the time cap.
                  Tests "bank some, keep optionality".
  V4 hold_12      time cap 12 bars instead of 8 (docstring says HOLD in
                  {8,12} is param-robust; 20 kills the effect).
  V5 hold_4       time cap 4 bars. Tests whether the edge front-loads.
  V6 fail_fast    exit early if price closes back THROUGH the swept level
                  against us (the retest thesis is dead). Mechanism, not
                  a number.
  V7 giveback     exit if the trade gives back 50% of its max favourable
                  excursion, once MFE >= 1R. Tests MFE-based exits.

Gates (declared before running):
  G1 pooled paired mean dR > 0 AND both halves agree in sign
  G2 >= 6/9 symbols show the same sign (basket consistency)
  G3 paired bootstrap CI on dR excludes 0 AND a paired permutation
     (sign-flip) test p < 0.05
A variant must pass all three to be a candidate; a candidate still needs
its own forward registration before it could replace the frozen exit —
Gate F is running on the current one and must not be contaminated.

Run: python research/sweep_failure/exit_variants.py
Out: research/results/exit_variants.json
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/exit_variants.json"
SYMS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK")


def entries(bars):
    """Re-derive the frozen entries WITHOUT their exit, so every variant
    re-exits identical fills. Mirrors sweep_core.backtest_symbol up to the
    fill; any drift here would invalidate the whole comparison, so the
    baseline is recomputed through this same path and cross-checked
    against sweep_core's own output before anything is reported."""
    n = len(bars)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    a = SC.atr14(bars)
    out = []
    for e in SC.detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        if a[j] is None or a[j] == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd
        fill = None
        for f in range(j + 1, min(j + 1 + SC.W, n)):
            if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                fill = f
                break
        if fill is None or fill + 1 >= n:
            continue
        out.append({"j": j, "fill": fill, "lvl": lvl, "d": d, "A": a[j]})
    return out


_COLS: dict[int, tuple] = {}


def _cols(bars):
    """OHLC columns, built once per bars object. run_exit is called ~800k
    times across the basket; rebuilding four 13k-element lists inside it
    was the whole runtime."""
    key = id(bars)
    got = _COLS.get(key)
    if got is None:
        got = ([b[SC.H] for b in bars], [b[SC.L] for b in bars],
               [b[SC.C] for b in bars], [b[SC.O] for b in bars])
        _COLS[key] = got
    return got


def run_exit(bars, e, kind):
    """Return (R, exit_bar) for one entry under one exit rule. Costs are
    charged identically in every variant: 0.05 ATR entering, 0.05 ATR
    leaving (and stop-outs pay the extra slip beyond the stop)."""
    h, lo, c, op = _cols(bars)
    n = len(bars)

    def touch_fill(level, k, d_):
        """Realistic fill for a level-touch exit: if the bar OPENED
        already through the level, the fill is the open, not the level.
        Applied to every price-level exit including the baseline's
        disaster stop, so the paired comparison stays fair."""
        o = op[k]
        return min(level, o) if d_ == 1 else max(level, o)

    d, A, lvl, fill = e["d"], e["A"], e["lvl"], e["fill"]
    entry = lvl + d * SC.SLIP * A
    risk = SC.DIS * A
    stop = entry - d * risk
    hold = {"hold_12": 12, "hold_4": 4}.get(kind, SC.HOLD)
    trail_atr = {"trail_2atr": 2.0, "trail_1atr": 1.0}.get(kind)
    last = min(fill + hold, n - 1)
    peak = entry
    booked = 0.0            # realised part for the scale-out variant
    frac = 1.0              # position still open
    for k in range(fill + 1, last + 1):
        hi_k, lo_k, c_k = h[k], lo[k], c[k]
        # NO INTRABAR LOOK-AHEAD: trail/giveback levels are derived from
        # the peak as of the PREVIOUS bar, tested against this bar, and
        # only then is the peak updated. Deriving the level from this
        # bar's own extreme would assume the peak happened before the
        # retrace — unknowable from OHLC, and it inflates exactly the
        # variants that exit on a retrace (first run: giveback +0.042R,
        # 9/9 symbols, p=0.000 — too pretty, and this was why).
        mfe_prev = d * (peak - entry) / risk
        if (d == 1 and lo_k <= stop) or (d == -1 and h[k] >= stop):
            px = touch_fill(stop, k, d) - d * SC.SLIP * A
            return booked + frac * (d * (px - entry) / risk), k
        if trail_atr is not None and mfe_prev >= 1.0:
            tr = peak - d * trail_atr * A
            if (d == 1 and lo_k <= tr) or (d == -1 and h[k] >= tr):
                px = touch_fill(tr, k, d) - d * SC.SLIP * A
                return booked + frac * (d * (px - entry) / risk), k
        if kind == "giveback" and mfe_prev >= 1.0:
            gb = peak - d * 0.5 * (peak - entry)
            if (d == 1 and lo_k <= gb) or (d == -1 and h[k] >= gb):
                px = touch_fill(gb, k, d) - d * SC.SLIP * A
                return booked + frac * (d * (px - entry) / risk), k
        # a resting limit at +1R fills when touched — order within the
        # bar does not matter for a level ABOVE the market, so this one
        # may legitimately use the current bar
        fav = d * ((hi_k if d == 1 else lo_k) - entry) / risk
        if kind == "half_at_1r" and frac == 1.0 and fav >= 1.0:
            px = entry + d * risk            # fill at +1R
            booked += 0.5 * ((d * (px - entry) / risk) - SC.SLIP * A / risk)
            frac = 0.5
        peak = max(peak, hi_k) if d == 1 else min(peak, lo_k)
        if kind == "fail_fast":
            # closed back through the swept level against us => the
            # retest thesis is dead; leave at this close
            through = (c_k < lvl) if d == 1 else (c_k > lvl)
            if through:
                px = c_k - d * SC.SLIP * A
                return booked + frac * (d * (px - entry) / risk), k
    px = c[last] - d * SC.SLIP * A
    return booked + frac * (d * (px - entry) / risk), last


VARIANTS = ("baseline", "trail_2atr", "trail_1atr", "half_at_1r",
            "hold_12", "hold_4", "fail_fast", "giveback")


def main() -> int:
    print("=" * 78)
    print("  策略 #2 出場變體 — 配對比較（同一批進場，只換出場）")
    print("=" * 78)
    per_sym = {}
    pooled = {v: [] for v in VARIANTS}
    for sym in SYMS:
        try:
            bars = SC.load_csv(str(LT.CACHE / f"{sym}USDT_1h.csv"))
        except Exception:
            continue
        es = entries(bars)
        # instrument check: baseline through THIS path must match
        # sweep_core's own backtest, else the comparison is meaningless
        ref = [t[2] for t in SC.backtest_symbol(bars)]
        mine = [run_exit(bars, e, "baseline")[0] for e in es]
        rows = {v: [run_exit(bars, e, v)[0] for e in es] for v in VARIANTS}
        per_sym[sym] = {"n": len(es), "ref_n": len(ref),
                        "ref_mean": float(np.mean(ref)) if ref else None,
                        "mine_mean": float(np.mean(mine)) if mine else None}
        for v in VARIANTS:
            pooled[v].append((sym, rows[v], rows["baseline"]))
        print(f"  {sym}: 進場 {len(es)} 筆（sweep_core 的可交易子集 {len(ref)}"
              f"；本管線含重疊倉，均 R {np.mean(mine):+.3f} vs 參考"
              f" {np.mean(ref):+.3f}）")

    res = {"per_symbol": per_sym, "variants": {}}
    print(f"\n  {'變體':<12}{'配對 dR':>10}{'兩半':>16}{'幣別同向':>10}"
          f"{'CI':>18}{'p':>8}")
    rng = np.random.default_rng(11)
    for v in VARIANTS:
        if v == "baseline":
            continue
        diffs = []
        sym_signs = []
        for sym, rs, base in pooled[v]:
            d_ = np.array(rs) - np.array(base)
            diffs.append(d_)
            sym_signs.append((sym, float(d_.mean())))
        allд = np.concatenate(diffs)
        m = float(allд.mean())
        half = len(allд) // 2
        h1, h2 = float(allд[:half].mean()), float(allд[half:].mean())
        pos = sum(1 for _s, x in sym_signs if x > 0)
        boots = [float(rng.choice(allд, len(allд), True).mean())
                 for _ in range(2000)]
        lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
        null = [float((allд * rng.choice([-1, 1], len(allд))).mean())
                for _ in range(2000)]
        p = float((np.abs(null) >= abs(m)).mean())
        g1 = m > 0 and h1 * h2 > 0
        g2 = pos >= 6
        g3 = lo_ci > 0 and p < 0.05
        verdict = "PASS" if (g1 and g2 and g3) else "FAIL"
        print(f"  {v:<12}{m:>+10.4f}{f'{h1:+.3f}/{h2:+.3f}':>16}"
              f"{f'{pos}/9':>10}{f'[{lo_ci:+.3f},{hi_ci:+.3f}]':>18}"
              f"{p:>8.3f}  {verdict}")
        res["variants"][v] = {"mean_dR": round(m, 4), "h1": round(h1, 4),
                              "h2": round(h2, 4), "sym_positive": pos,
                              "ci": [round(float(lo_ci), 4),
                                     round(float(hi_ci), 4)],
                              "p": p, "verdict": verdict,
                              "per_symbol": {s: round(x, 4)
                                             for s, x in sym_signs}}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
