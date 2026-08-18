# -*- coding: utf-8 -*-
"""Does sweep-failure work on TIME-defined liquidity, not just swing pivots?

The user's critique (2026-07-29): #3 defines liquidity purely as a swing
pivot, which is one arbitrary definition among several. The LMSR Pine map
they shared draws four more kinds of pool — session extremes, PDH/PDL,
PWH/PWL — and its own header admits: "Session, daily and weekly levels are
drawn but never fire. Nobody has tested whether raiding those behaves the
same way."

This tests them. Time-defined levels are also immune to the leak that killed
the equal-levels idea earlier today: a session high is fixed the moment the
session closes, so counting or locating it cannot smuggle in information
about what price did afterwards. A swing pivot's NEIGHBOURHOOD could (levels
that price kept respecting accumulate pivots, and "price came back" is the
label) — a previous-day high cannot.

Rules held identical to the frozen engine so the comparison is clean:
  level raided (price trades through) -> wait up to W=8 bars for a return to
  the level -> enter there -> 3.5 x ATR(sweep bar) stop -> 8-bar time exit ->
  one position at a time per (symbol, level type).
Only the SOURCE of the level changes. pierce_atr is computed for each event
so the shallow-pierce filter can be checked on every level type as well.

Level definitions (all causal — established strictly before they can be hit):
  session   Asia 00-08, London 07-16, New York 12-21 UTC; the high and low of
            a COMPLETED session become two pools at its close
  pdh/pdl   previous UTC day's high/low, live from the new day's first bar
  pwh/pwl   previous ISO week's high/low, live from the new week's first bar
A pool rests until price trades through it, exactly as the indicator draws it.

Run: python research/sweep_failure/level_types.py
Out: research/results/sweep_level_types.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_level_types.json"
CACHE = HERE / ".cache"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX",
         "TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
         "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
         "SAND", "AXS"]
TAKER = 5.0
SESSIONS = {"asia": (0, 8), "london": (7, 16), "ny": (12, 21)}


def net(R, lvl, atr):
    return R - 2 * TAKER / 1e4 * lvl / (SC.DIS * atr)


def build_levels(bars) -> dict[str, list[tuple[int, float, int]]]:
    """-> {kind: [(established_bar, price, side)]}, all causal."""
    H, L = SC.H, SC.L
    out = defaultdict(list)
    dts = [datetime.fromtimestamp(b[0], tz=timezone.utc) for b in bars]

    # session extremes: emitted at the bar where the session has just closed
    for name, (h0, h1) in SESSIONS.items():
        cur_hi = cur_lo = None
        inside_prev = False
        for i, dt in enumerate(dts):
            inside = h0 <= dt.hour < h1
            if inside:
                cur_hi = bars[i][H] if not inside_prev else max(cur_hi, bars[i][H])
                cur_lo = bars[i][L] if not inside_prev else min(cur_lo, bars[i][L])
            elif inside_prev and cur_hi is not None:
                out["session"].append((i, cur_hi, 1))
                out["session"].append((i, cur_lo, -1))
                cur_hi = cur_lo = None
            inside_prev = inside

    # previous day / previous ISO week, live from the first bar of the new period
    for kind, keyfn in (("pdh_pdl", lambda d: d.date()),
                        ("pwh_pwl", lambda d: d.isocalendar()[:2])):
        cur_key = None
        cur_hi = cur_lo = None
        prev_hi = prev_lo = None
        for i, dt in enumerate(dts):
            k = keyfn(dt)
            if cur_key is None:
                cur_key, cur_hi, cur_lo = k, bars[i][H], bars[i][L]
                continue
            if k != cur_key:
                prev_hi, prev_lo = cur_hi, cur_lo
                cur_key, cur_hi, cur_lo = k, bars[i][H], bars[i][L]
                out[kind].append((i, prev_hi, 1))
                out[kind].append((i, prev_lo, -1))
            else:
                cur_hi = max(cur_hi, bars[i][H])
                cur_lo = min(cur_lo, bars[i][L])
    return out


def trade_levels(bars, levels) -> list[tuple]:
    """(fill_ts, exit_ts, netR, pierce_atr, lvl, atr, stopped, side) under the frozen
    entry/exit rules. Single source of truth: the shadow recorder and the gate
    scorer both call THIS, so the forward log cannot drift from what is scored."""
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)
    # a pool rests from its established bar until price trades through it
    pending = sorted(levels)
    out, last_exit, idx = [], -1, 0
    live: list[tuple[float, int]] = []
    for j in range(n):
        while idx < len(pending) and pending[idx][0] <= j:
            live.append((pending[idx][1], pending[idx][2]))
            idx += 1
        if a[j] is None or a[j] == 0:
            continue
        hit = [(p, s) for p, s in live
               if (h[j] > p if s == 1 else lo[j] < p)]
        if not hit:
            continue
        live = [x for x in live if x not in hit]
        for lvl, s in hit:
            kd = s
            d = -kd
            fill = None
            for f in range(j + 1, min(j + 1 + SC.W, n)):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    fill = f
                    break
            if fill is None or fill <= last_exit or fill + 1 >= n:
                continue
            A = a[j]
            risk = SC.DIS * A
            stop = lvl - d * risk
            R, xb = None, min(fill + SC.HOLD, n - 1)
            for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
                if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                    R, xb = -1.0, k
                    break
            if R is None:
                R = d * (cl[xb] - lvl) / risk
            last_exit = xb
            pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / A
            out.append((bars[fill][0], bars[xb][0], net(R, lvl, A), pierce,
                        lvl, A, R <= -1.0,
                        "LONG" if d == 1 else "SHORT"))   # M2 additive
    return out


def stat(rs):
    n = len(rs)
    if n < 60:
        return None
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    return {"n": n, "mean": m, "t": m / (sd / math.sqrt(n))}


def fmt(s):
    return f"{s['mean']:+.4f} (t{s['t']:+5.1f}, n={s['n']:>5})" if s else "thin"


def main() -> int:
    print("=" * 78)
    print("  LEVEL TYPES — does sweep-failure work on time-defined liquidity?")
    print("=" * 78)
    agg = defaultdict(list)
    for sym in COINS:
        p = CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            continue
        bars = SC.load_csv(str(p))
        lv = build_levels(bars)
        for kind, items in lv.items():
            agg[kind] += [(r[2], r[3]) for r in trade_levels(bars, items)]
        # swing baseline from the frozen engine, same accounting
        agg["swing (baseline)"] += [
            (net(R, l_, at), pc)
            for (_t, _x, R, l_, at, _s, pc) in SC.backtest_symbol(bars)]

    res = {}
    print(f"  {'level type':<20}{'all':<32}{'pierce<=0.25 ATR':<30}")
    for kind in ("swing (baseline)", "session", "pdh_pdl", "pwh_pwl"):
        rows = agg.get(kind, [])
        if not rows:
            continue
        all_s = stat([r for r, _ in rows])
        sh_s = stat([r for r, pc in rows if pc <= 0.25])
        res[kind] = {"all": all_s, "shallow": sh_s}
        print(f"  {kind:<20}{fmt(all_s):<32}{fmt(sh_s):<30}")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: these levels are fixed by the clock, so they cannot leak the "
          "outcome the way pivot NEIGHBOURHOODS did. A null here is a real null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
