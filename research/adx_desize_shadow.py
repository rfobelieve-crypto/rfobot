"""ADX de-size rule -- shadow replay scorer for TODO §0.52.

Replays the frozen sweep-failure shadow ledger under the de-size rule
(TRENDING -> notional x0.5, else x1.0) and scores it against the frozen
predictions A-P1/A-P2/A-P3.  Writes nothing to any live path.

The rule is a WEIGHT TRANSFORM on signals that already exist, so it can be
replayed instead of waited for.  That is also how it could cheat, so the
registration froze the split: the ADX thresholds were frozen 2026-08-17,
therefore

    fills BEFORE 2026-08-17  = in-sample, REFERENCE ONLY
    fills ON/AFTER           = clean forward, the only judgement basis

and both halves are always printed.  A run that reports only the pooled
number is an instrument failure, not a result (§0.52).

Mechanism being leaned on is NOT trend forecasting -- ADX has none
(2026-08-20: forward efficiency ratio gap -0.007/-0.010/-0.011 across
12/24/48h with CIs spanning zero, breadth decaying 7/29 -> 2/29; variance
ratio gap -0.028, breadth 1/9).  What survives is concurrent state plus
inertia: forward-8h realised vol is +22% in TRENDING with breadth 9/9, and
labels persist (P(TRENDING->TRENDING, 8h) = 77.8% vs 48.2% base).  So the
claim is "fixed notional in a 22%-more-volatile state is 22% more risk",
not "a trend is coming".

Read-only research code.

    python research/adx_desize_shadow.py
"""
from __future__ import annotations

import csv
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
FREEZE = datetime(2026, 8, 17, tzinfo=timezone.utc).timestamp()
DESIZE = 0.5                     # frozen; not swept (§0.52)


def max_drawdown(seq: list[float]) -> float:
    """Deepest peak-to-trough of the cumulative curve, in R."""
    peak = cum = 0.0
    mdd = 0.0
    for x in seq:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


def load_rows() -> list[dict]:
    if not LOG.exists():
        print(f"missing {LOG}")
        return []
    out = []
    with open(LOG, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "CLOSED":
                continue
            if r.get("variant_b") != "1":
                continue              # judgement variant is B (Gate F)
            try:
                out.append({"sym": r["symbol"],
                            "ts": int(float(r["fill_ts"])),
                            "r": float(r["net_r"])})
            except (ValueError, KeyError, TypeError):
                continue
    out.sort(key=lambda x: x["ts"])
    return out


def adx_for(symbols: set[str]) -> dict[str, dict[int, str]]:
    st = {}
    for s in sorted(symbols):
        fp = CACHE / f"{s}USDT_1h.csv"
        if fp.exists():
            st[s] = adx_state(SC.load_csv(str(fp)))
    return st


def score(rows: list[dict], st: dict) -> dict:
    base, desized, weights, by_state = [], [], [], defaultdict(list)
    unlabelled = 0
    for x in rows:
        lab = st.get(x["sym"], {}).get(x["ts"] // 3600 * 3600)
        if lab is None:
            unlabelled += 1
        w = DESIZE if lab == "TRENDING" else 1.0
        base.append(x["r"])
        desized.append(x["r"] * w)
        weights.append(w)
        if lab:
            by_state[lab].append(x["r"])
    # Exposure-matched null. Without it a "PASS" is unreadable: over any
    # window where the strategy loses money, EVERY de-size improves both
    # return and drawdown, and the rule gets credit for simply betting
    # less. The honest question is whether ADX beats cutting the same
    # total exposure BLINDLY, so scale every trade by the same factor the
    # ADX rule averaged.
    w_avg = sum(weights) / len(weights) if weights else 1.0
    flat = [x * w_avg for x in base]
    return {"n": len(rows), "unlabelled": unlabelled,
            "base_sum": sum(base), "des_sum": sum(desized),
            "base_mdd": max_drawdown(base), "des_mdd": max_drawdown(desized),
            "flat_w": w_avg, "flat_sum": sum(flat),
            "flat_mdd": max_drawdown(flat), "by_state": by_state}


MIN_SPAN_DAYS = 30


def report(tag: str, s: dict, judge: bool, span_days: float = 0.0) -> None:
    if not s["n"]:
        print(f"\n[{tag}] no rows")
        return
    print(f"\n[{tag}]  n={s['n']}  span {span_days:.1f}d  "
          f"(unlabelled {s['unlabelled']})")
    print(f"  avg weight kept {s['flat_w']:.3f}  "
          f"(null = every trade scaled by this, blindly)")
    print(f"  total netR   base {s['base_sum']:+8.3f}  ->  "
          f"ADX {s['des_sum']:+8.3f}   |  blind null {s['flat_sum']:+8.3f}")
    print(f"  max drawdown base {s['base_mdd']:+8.3f}  ->  "
          f"ADX {s['des_mdd']:+8.3f}   |  blind null {s['flat_mdd']:+8.3f}")

    # A-P1: is TRENDING actually the painful state?
    for lab in ("TRENDING", "RANGING", "NEUTRAL"):
        v = s["by_state"].get(lab, [])
        if v:
            print(f"    {lab:9} n={len(v):4d}  meanR {sum(v)/len(v):+.4f}  "
                  f"MDD {max_drawdown(v):+.3f}")
    if not judge:
        print("  -> REFERENCE ONLY (pre-freeze, in-sample); no verdict drawn")
        return

    # Span gate, added 2026-08-20 on the first run (§0.52 registered a
    # tier-1 threshold but no minimum window -- an omission, and this
    # closes it rather than relaxing anything). Trades from a handful of
    # days are not n independent observations: sweep fills cluster hard in
    # time, so 131 fills across 2 days share one market episode. Judging
    # that is the concentration version of the small-sample error this
    # project has made before.
    if span_days < MIN_SPAN_DAYS:
        print(f"  -> INSUFFICIENT SPAN ({span_days:.1f}d < {MIN_SPAN_DAYS}d): "
              f"no verdict. Fills cluster in time; these are not "
              f"{s['n']} independent observations.")
        return

    # A-P2 tier-1: MDD improves >=10% relative AND return given back is at
    # most half of that improvement.
    if s["base_mdd"] >= 0:
        print("  -> no drawdown in base; A-P2 not evaluable")
        return
    mdd_gain = (s["des_mdd"] - s["base_mdd"]) / abs(s["base_mdd"])
    give = s["base_sum"] - s["des_sum"]
    print(f"  A-P2 as registered: MDD improved {mdd_gain*100:+.1f}% relative, "
          f"return given back {give:+.3f}R")
    as_written = mdd_gain >= 0.10 and give <= abs(s["base_mdd"]) * mdd_gain * 0.5

    # Added 2026-08-20 after the first run, and this TIGHTENS the bar --
    # the registered A-P2 compares against doing nothing, which any
    # de-size passes on a losing stretch. Beating the exposure-matched
    # blind null is what separates "ADX knows something" from "we bet
    # less". Loosening a criterion after seeing numbers is forbidden;
    # closing a false-positive hole is the opposite and is logged in §0.52.
    beats_null = (s["des_mdd"] > s["flat_mdd"]) and (s["des_sum"] >= s["flat_sum"])
    print(f"  vs blind null: MDD {s['des_mdd']:+.3f} vs {s['flat_mdd']:+.3f}"
          f"  netR {s['des_sum']:+.3f} vs {s['flat_sum']:+.3f}"
          f"  -> {'ADX adds' if beats_null else 'ADX adds NOTHING'}")
    ok = as_written and beats_null
    print(f"  VERDICT tier-1: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else "  (rule is withdrawn, not re-tuned -- §0.52)"))


def main() -> int:
    rows = load_rows()
    if not rows:
        return 1
    st = adx_for({x["sym"] for x in rows})
    span = (datetime.fromtimestamp(rows[0]["ts"], timezone.utc).date(),
            datetime.fromtimestamp(rows[-1]["ts"], timezone.utc).date())
    print(f"variant B closed fills: n={len(rows)}  {span[0]} -> {span[1]}")
    print(f"de-size x{DESIZE} on TRENDING; ADX(14) 25/20 frozen 2026-08-17")

    pre = [x for x in rows if x["ts"] < FREEZE]
    post = [x for x in rows if x["ts"] >= FREEZE]

    def span(v):
        return (v[-1]["ts"] - v[0]["ts"]) / 86400.0 if len(v) > 1 else 0.0

    report("PRE-FREEZE  in-sample", score(pre, st), False, span(pre))
    report("POST-FREEZE forward", score(post, st), True, span(post))
    print("\n(pooled figure deliberately not printed -- see §0.52)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
