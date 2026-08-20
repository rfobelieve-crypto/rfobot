"""Vol-targeting shadow replay scorer for TODO §0.53.

Replays the frozen sweep-failure shadow ledger under the frozen rule

    sigma_hat = (sigma_24h + sigma_168h) / 2         (trailing only)
    sigma_ref = trailing 720h median of sigma_hat    (trailing only)
    w         = clamp(sigma_ref / sigma_hat, 0.25, 1.0)

and scores it against the §0.53 predictions.  Three comparators are always
printed together:

    base        the ledger as recorded (weight 1)
    vol-target  the rule above
    blind null  every trade scaled by the rule's AVERAGE weight -- the
                exposure-matched control that separates "the gauge knows
                something" from "we just bet less" (§0.52 first-run lesson,
                in the criteria from day one here)
    ADX x0.5    the §0.52 binary rule, for the V-P3 head-to-head

Judgement discipline (all frozen in §0.53 before this file ran):
    - fills before 2026-08-20 are in-sample REFERENCE ONLY
    - the POST window refuses to render a verdict under 30 days of span
      (sweep fills cluster in time; a 3-day window is one episode)
    - a run that printed only a pooled number would be an instrument
      failure, so pre/post are always separated

Read-only research code.

    python research/vol_target_shadow.py
"""
from __future__ import annotations

import bisect
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC                                    # noqa: E402
from research.crowd_battery2 import adx_state              # noqa: E402

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
CACHE = ROOT / "research" / "sweep_failure" / ".cache"
FREEZE = datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp()
FAST, SLOW, REF = 24, 168, 720
W_LO, W_HI = 0.25, 1.0
MIN_SPAN_DAYS = 30


# ── weight machinery ─────────────────────────────────────────────────────

def rolling_std(vals: list[float], win: int) -> list[float | None]:
    """Population std of the trailing `win` values, O(n) via prefix sums."""
    n = len(vals)
    ps = [0.0] * (n + 1)
    ps2 = [0.0] * (n + 1)
    for i, v in enumerate(vals):
        ps[i + 1] = ps[i] + v
        ps2[i + 1] = ps2[i] + v * v
    out: list[float | None] = [None] * n
    for i in range(win, n + 1):
        s = ps[i] - ps[i - win]
        s2 = ps2[i] - ps2[i - win]
        var = s2 / win - (s / win) ** 2
        out[i - 1] = math.sqrt(var) if var > 0 else None
    return out


def weight_series(bars) -> dict[int, float]:
    """hour_ts -> frozen §0.53 weight, computed from PAST bars only.

    sigma_hat at bar i uses returns up to and including i; the weight is
    keyed to bar i's hour and applied to fills stamped in that hour, so a
    fill never sees information after its own bar.
    """
    c = [b[SC.C] for b in bars]
    rets = [0.0] + [c[i] / c[i - 1] - 1 for i in range(1, len(c))]
    f = rolling_std(rets, FAST)
    s = rolling_std(rets, SLOW)
    hat: list[float | None] = [
        (f[i] + s[i]) / 2 if f[i] is not None and s[i] is not None else None
        for i in range(len(c))]

    out: dict[int, float] = {}
    window: list[float] = []          # sorted trailing sigma_hat values
    queue: list[float] = []           # same values in arrival order
    for i in range(len(c)):
        h = hat[i]
        if h is None:
            continue
        if len(queue) >= REF:
            ref = window[len(window) // 2]
            w = max(W_LO, min(W_HI, ref / h)) if h > 0 else 1.0
            out[bars[i][0] // 3600 * 3600] = w
        bisect.insort(window, h)
        queue.append(h)
        if len(queue) > REF:
            old = queue.pop(0)
            window.pop(bisect.bisect_left(window, old))
    return out


# ── ledger replay ────────────────────────────────────────────────────────

def max_drawdown(seq: list[float]) -> float:
    peak = cum = mdd = 0.0
    for x in seq:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


def load_rows() -> list[dict]:
    out = []
    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "CLOSED" or r.get("variant_b") != "1":
                continue
            try:
                out.append({"sym": r["symbol"],
                            "ts": int(float(r["fill_ts"])),
                            "r": float(r["net_r"])})
            except (ValueError, KeyError, TypeError):
                continue
    out.sort(key=lambda x: x["ts"])
    return out


def score(rows, wmap, adxmap):
    base, vt, adx, ws = [], [], [], []
    missing = 0
    for x in rows:
        h = x["ts"] // 3600 * 3600
        w = wmap.get(x["sym"], {}).get(h)
        if w is None:
            missing += 1
            w = 1.0
        lab = adxmap.get(x["sym"], {}).get(h)
        base.append(x["r"])
        vt.append(x["r"] * w)
        adx.append(x["r"] * (0.5 if lab == "TRENDING" else 1.0))
        ws.append(w)
    w_avg = sum(ws) / len(ws) if ws else 1.0
    null = [x * w_avg for x in base]
    return {"n": len(rows), "missing": missing, "w_avg": w_avg,
            "sums": {k: sum(v) for k, v in
                     (("base", base), ("vt", vt), ("null", null), ("adx", adx))},
            "mdds": {k: max_drawdown(v) for k, v in
                     (("base", base), ("vt", vt), ("null", null), ("adx", adx))}}


def report(tag, s, judge, span_days):
    if not s["n"]:
        print(f"\n[{tag}] no rows")
        return
    print(f"\n[{tag}]  n={s['n']}  span {span_days:.1f}d  "
          f"(weight missing on {s['missing']}, avg weight {s['w_avg']:.3f})")
    print(f"  {'':12}{'base':>9}{'vol-tgt':>9}{'null':>9}{'ADX x0.5':>10}")
    print("  total netR " + "".join(f"{s['sums'][k]:+9.3f}"
          for k in ("base", "vt", "null")) + f"{s['sums']['adx']:+10.3f}")
    print("  max DD     " + "".join(f"{s['mdds'][k]:+9.3f}"
          for k in ("base", "vt", "null")) + f"{s['mdds']['adx']:+10.3f}")

    if not judge:
        print("  -> REFERENCE ONLY (pre-freeze, in-sample); no verdict drawn")
        return
    if span_days < MIN_SPAN_DAYS:
        print(f"  -> INSUFFICIENT SPAN ({span_days:.1f}d < {MIN_SPAN_DAYS}d): "
              "no verdict (fills cluster in time)")
        return

    b_mdd, v_mdd = s["mdds"]["base"], s["mdds"]["vt"]
    if b_mdd >= 0:
        print("  -> no drawdown in base; tier-1 not evaluable")
        return
    mdd_gain = (v_mdd - b_mdd) / abs(b_mdd)
    give = s["sums"]["base"] - s["sums"]["vt"]
    beats_null = v_mdd > s["mdds"]["null"] and s["sums"]["vt"] >= s["sums"]["null"]
    print(f"  V-P1/P2: MDD {mdd_gain*100:+.1f}% rel, give-back {give:+.3f}R, "
          f"vs null -> {'beats' if beats_null else 'DOES NOT beat'}")

    # V-P3 head-to-head: MDD improvement bought per unit of return given up.
    a_gain = (s["mdds"]["adx"] - b_mdd) / abs(b_mdd)
    a_give = s["sums"]["base"] - s["sums"]["adx"]
    vt_rate = mdd_gain / give if give > 0 else float("inf")
    adx_rate = a_gain / a_give if a_give > 0 else float("inf")
    print(f"  V-P3: MDD-per-R  vol-tgt {vt_rate:.3f} vs ADX {adx_rate:.3f}"
          f"  -> {'vol-tgt wins' if vt_rate > adx_rate else 'ADX wins'}")

    ok = mdd_gain >= 0.10 and give <= abs(b_mdd) * mdd_gain * 0.5 and beats_null
    print(f"  VERDICT tier-1: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else "  (withdrawn, not re-tuned -- §0.53)"))


def main() -> int:
    rows = load_rows()
    if not rows:
        print(f"missing/empty {LOG}")
        return 1
    syms = {x["sym"] for x in rows}
    wmap, adxmap = {}, {}
    for s in sorted(syms):
        fp = CACHE / f"{s}USDT_1h.csv"
        if fp.exists():
            bars = SC.load_csv(str(fp))
            wmap[s] = weight_series(bars)
            adxmap[s] = adx_state(bars)

    span0 = (datetime.fromtimestamp(rows[0]["ts"], timezone.utc).date(),
             datetime.fromtimestamp(rows[-1]["ts"], timezone.utc).date())
    print(f"variant B closed fills: n={len(rows)}  {span0[0]} -> {span0[1]}")
    print(f"rule: w = clamp(median720(sigma_hat)/sigma_hat, {W_LO}, {W_HI}), "
          f"sigma_hat = (std{FAST}h + std{SLOW}h)/2   [frozen §0.53]")

    pre = [x for x in rows if x["ts"] < FREEZE]
    post = [x for x in rows if x["ts"] >= FREEZE]

    def span(v):
        return (v[-1]["ts"] - v[0]["ts"]) / 86400.0 if len(v) > 1 else 0.0

    report("PRE-FREEZE  in-sample", score(pre, wmap, adxmap), False, span(pre))
    report("POST-FREEZE forward", score(post, wmap, adxmap), True, span(post))
    print("\n(pooled figure deliberately not printed -- §0.53)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
