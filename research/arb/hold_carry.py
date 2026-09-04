# -*- coding: utf-8 -*-
"""1.07 -- hold-and-carry instead of close-at-half-band (pre-registered, exploration)

Operator decision 2026-09-04 (COST_INVENTORY solution #8, "可以試試看").

The frozen family scorer books a round trip as: enter when the premium
deviates >= band from its 6h midline, exit when it comes back within half
the band. Four crossings, capture band/2. This asks whether, WHEN the
funding differential pays us for holding, it is better to keep the position:
  * capture the WHOLE deviation instead of half (exit when the premium
    crosses the midline, not half-band),
  * collect funding on the way,
  * pay nothing extra in crossings (still one round trip).
The price is time: capital, carry sign flipping, and every tail in bucket 7
grows with the hold. So the rule caps the hold and exits the moment carry
turns against us.

FROZEN RULE (hold path)
  enter   as the scorer does (deviation >= band, same sign convention)
  hold    while ALL: carry sign favourable (we receive), premium has not
          crossed the midline, hold < MAX_HOLD_H
  exit    at the first violation; also exit at the scorer's half-band point
          if carry was never favourable (then it IS the baseline)
  pnl     captured deviation (entry dev - exit dev, same sign) + carry
          received (fund_diff x hold / 8h) - round-trip fees (fees.py,
          taker_taker, operator rebates)
  baseline pnl = band/2 - same fees   (what the scorer implies)
Carry sign: recorder fund_diff = hedge - entropy (bps/8h). A high-premium
episode is short entropy / long hedge and RECEIVES -fund_diff; a low-premium
episode receives +fund_diff.

PRE-REGISTERED PREDICTIONS (written before the run)
  P1  mean net per episode, hold path > baseline, on >= 4 of the pairs that
      have funding columns (7 pairs today)
  P2  the gain is not one pair: removing the best pair still leaves P1 true
  P3  median extra hold < 8 h -- longer means this is a basis trade with a
      different risk book, not an execution tweak; then it needs its own
      registration, not this one
SURVIVAL: all three -> forward clock on rows from today; else the idea stays
a note. Exploration on the selection window only; nothing here is a verdict.

Run: python research/arb/hold_carry.py
Out: research/results/arb_hold_carry.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import premium_verdict as PV      # noqa: E402  frozen loader / band / episodes
import fees as FEES               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = HERE.parents[1] / "research" / "results" / "arb_hold_carry.json"
MAX_HOLD_H = 24.0
MIN_EPISODES = 5


def p90(xs):
    xs = sorted(xs)
    return xs[int(0.9 * len(xs))] if xs else float("nan")


def episodes_with_paths(rows, band):
    """Re-detect the scorer's episodes (same midline, same band) but keep
    the row index so the hold path can be walked. Same arithmetic as
    PV.convergence; verified below by matching its episode count."""
    import statistics as _st
    prems = [x["prem"] for x in rows]
    eps, i, n = [], PV.MIDLINE_WIN, len(rows)
    while i < n:
        mid = _st.median(prems[i - PV.MIDLINE_WIN:i])
        dev = prems[i] - mid
        if abs(dev) >= band:
            sign = 1 if dev > 0 else -1
            j = i + 1
            while j < n and sign * (prems[j] - mid) > band * PV.CONV_RETURN_FRAC:
                j += 1
            eps.append((i, j if j < n else None, mid, sign))
            i = j + 1
        else:
            i += 1
    return eps


NO_CARRY = "--no-carry" in sys.argv     # ablation: same exit rule, carry income zeroed


def main() -> int:
    print("=" * 96)
    print("  §1.07 持有收租 vs 回半帶就平（預註冊，探索；判準見檔頭）")
    print("=" * 96)
    res = {"pairs": {}}
    per_pair_gain = {}
    extra_hold = []
    for pid, sub, a, b, note in PV.PAIRS:
        rows = PV.load(PV.LOGS / sub)
        if not rows or not any(r["f_d"] is not None for r in rows):
            continue
        va, vb = PV.VENUE_KEYS[pid]
        fee_rt = FEES.round_trip_bps(va, vb, "taker_taker", True)
        out_pair = {}
        for key, lab in (("sell_max", "sell"), ("buy_max", "buy")):
            band = max(p90([x[key] for x in rows]), PV.NET_BPS_MIN)
            eps = episodes_with_paths(rows, band)
            chk = PV.convergence(rows, band)
            assert chk.get("episodes", 0) == len(eps), (pid, lab, chk, len(eps))
            base, hold, holds_h, carried = [], [], [], 0
            for i, j, mid, sign in eps:
                if j is None:
                    continue                      # never converged in window
                # baseline: capture half band at j
                base.append(band / 2 - fee_rt)
                # hold path
                fd = rows[i]["f_d"]
                recv = (-fd if sign > 0 else fd) if fd is not None else 0.0
                if fd is None or recv <= 0:
                    hold.append(band / 2 - fee_rt)
                    holds_h.append((rows[j]["ts"] - rows[i]["ts"]) / 3600)
                    continue
                carried += 1
                k, carry = i + 1, 0.0
                t0 = rows[i]["ts"]
                exit_k = None
                while k < len(rows):
                    dt_h = (rows[k]["ts"] - rows[k - 1]["ts"]) / 3600
                    fdk = rows[k]["f_d"]
                    rk = (-fdk if sign > 0 else fdk) if fdk is not None else recv
                    if rk <= 0:                       # carry flipped -> exit
                        exit_k = k
                        break
                    carry += 0.0 if NO_CARRY else rk * dt_h / 8.0
                    if sign * (rows[k]["prem"] - mid) <= 0:     # crossed midline
                        exit_k = k
                        break
                    if (rows[k]["ts"] - t0) / 3600 >= MAX_HOLD_H:
                        exit_k = k
                        break
                    k += 1
                if exit_k is None:
                    exit_k = len(rows) - 1
                dev_in = sign * (rows[i]["prem"] - mid)
                dev_out = sign * (rows[exit_k]["prem"] - mid)
                captured = max(min(dev_in - dev_out, dev_in), -band)   # cap: cannot
                # capture more than the entry deviation; loss capped at one band
                hold.append(captured + carry - fee_rt)
                holds_h.append((rows[exit_k]["ts"] - t0) / 3600)
            if len(base) < MIN_EPISODES:
                out_pair[lab] = {"n": len(base), "skip": "episodes < 5"}
                continue
            mb, mh = st.mean(base), st.mean(hold)
            h = len(hold) // 2
            halves = [round(st.mean(hold[:h]) - st.mean(base[:h]), 3),
                      round(st.mean(hold[h:]) - st.mean(base[h:]), 3)] if h else None
            out_pair[lab] = {"n": len(base), "carried": carried, "band": round(band, 2),
                             "base_mean": round(mb, 3), "hold_mean": round(mh, 3),
                             "gain": round(mh - mb, 3), "halves": halves,
                             "median_hold_h": round(st.median(holds_h), 2)}
            per_pair_gain[(pid, lab)] = mh - mb
            extra_hold.extend(holds_h)
            print(f"  {pid:<8}{lab:<5} n={len(base):>4} 持有路徑觸發 {carried:>3}  帶 {band:>6.2f}"
                  f"  基準 {mb:>+7.2f}  持有 {mh:>+7.2f}  差 {mh-mb:>+6.2f}"
                  f"  兩半 {halves}  持有中位 {st.median(holds_h):.2f}h")
        res["pairs"][pid] = out_pair

    # bars: per PAIR (best of its two sides) so one pair counts once
    by_pair = {}
    for (pid, lab), g in per_pair_gain.items():
        by_pair[pid] = max(g, by_pair.get(pid, -1e9))
    n_pos = sum(1 for g in by_pair.values() if g > 0)
    best = max(by_pair, key=by_pair.get) if by_pair else None
    n_pos_wo_best = sum(1 for p, g in by_pair.items() if g > 0 and p != best)
    med_hold = st.median(extra_hold) if extra_hold else float("nan")
    bars = {
        "P1 持有路徑優於基準的配對 ≥ 4": n_pos >= 4,
        "P2 拿掉最好的那個配對後仍 ≥ 4": n_pos_wo_best >= 4,
        "P3 持有時間中位 < 8h": med_hold < 8.0,
    }
    print(f"\n  配對層：{n_pos}/{len(by_pair)} 持有優於基準；最好的是 {best}；持有中位 {med_hold:.2f}h")
    for k, v in bars.items():
        print(f"    {'✅' if v else '❌'} {k}")
    res.update({"by_pair_gain": {k: round(v, 3) for k, v in by_pair.items()},
                "bars": bars, "median_hold_h": round(med_hold, 2) if extra_hold else None,
                "verdict": "存活：開前瞻時鐘" if all(bars.values()) else "未通過——留作筆記，不開時鐘"})
    print(f"  → {res['verdict']}")
    if NO_CARRY:
        print("  (ablation: carry income zeroed -- if the gains match the full run, "
              "the driver is the exit target, not carry)")
        return 0
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
