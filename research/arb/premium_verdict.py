# -*- coding: utf-8 -*-
"""§0.75 arbitrage clock — the verdict scorer, frozen 2026-08-28.

This file IS the registered criteria in executable form, and its main job
before day 7 is to REFUSE to render a verdict (the v7_regime_q2_clock
pattern: the failure mode to guard against is peeking until a number looks
right, and an EV-first line with zero fees is exactly where a good-looking
half-day of data would tempt an early call).

FROZEN GATE (written before more than two minutes of data existed):
  after >= 7 FULL days of recording, the line proceeds to the engineering
  gate iff there EXISTS a threshold band with
    * fee-net round-trip >= 1.0 bps        (fees are 0+0 on this pair, so
                                            net == raw executable room)
    * fired on average >= 10 times/day
    * BOTH halves of the recording satisfy the above independently
  else the line closes. The band is NOT swept for the best number: the
  candidate band is analyze.py's suggestion methodology (p90 of executable
  room), applied identically to both halves.

Progress source: ../entropy-arb/logs/<pair>/minutes.csv (outside this repo — the
third-party clone carries the recorder; THIS repo carries the judgment).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fees as FEES               # noqa: E402  single source of truth

LOGS = ROOT.parent / "entropy-arb" / "logs"
OUT = ROOT / "research" / "results" / "arb_premium_verdict.json"

# ── recording family (registered 2026-08-30, TODO §0.75) ───────────────────
# One frozen gate, applied to EVERY pair, ALL pairs reported — the family is
# not a menu to pick the best-looking member from (variant C/D lesson).
# START = first recorded minute (SNDK fixed at registration; the others take
# their own first row, so each clock runs 7 days from its own start).
# BTC is the CONTROL pair: two deep zero-fee books, band expected ~0. If BTC
# ever shows a fat band the instrument is broken, not the market (mistake.md
# 2026-07-29: run a new instrument on data whose answer is known first).
# io:OAI and io:EWY were registered and found DELISTED on Entropy at first
# connect (2026-08-30 22:30Z; meta still lists them, isDelisted=true) —
# dropped before any row. Delisting risk on these venues is real.
PAIRS = [
    # id,    csv subpath,        leg A,             leg B,                  note
    ("SNDK", "minutes.csv",      "Entropy io:SNDK", "lighter-rh SNDK",      "stock perp (original)"),
    ("NBIS", "NBIS/minutes.csv", "Entropy io:NBIS", "lighter NBIS",         "stock perp"),
    ("ANTH", "ANTH/minutes.csv", "Entropy io:ANTH", "lighter-rh ANTHROPIC", "private-co perp, no spot anchor"),
    ("BTC",  "BTC/minutes.csv",  "HL BTC",          "lighter-rh BTC",       "CONTROL - band expected ~0"),
    ("HYPE", "HYPE/minutes.csv", "HL HYPE",         "lighter-rh HYPE",      "thin crypto; largest funding gap on 09-01"),
    ("ZEC",  "ZEC/minutes.csv",  "HL ZEC",          "lighter-rh ZEC",       "thin crypto"),
    ("NEAR", "NEAR/minutes.csv", "HL NEAR",         "lighter-rh NEAR",      "thin crypto"),
]
# Venue keys for the fee table (2026-09-03). Written explicitly rather than
# parsed out of the prose above: the leg descriptions are for humans, and a
# fee must never depend on string-matching a comment.
VENUE_KEYS = {
    "SNDK": ("IO", "lighter-rh"), "NBIS": ("IO", "lighter"),
    "ANTH": ("IO", "lighter-rh"), "BTC": ("HL", "lighter-rh"),
    "HYPE": ("HL", "lighter-rh"), "ZEC": ("HL", "lighter-rh"),
    "NEAR": ("HL", "lighter-rh"),
}

FIXED_START = {"SNDK": datetime(2026, 8, 28, 10, 28, tzinfo=timezone.utc)}
GATE_DAYS = 7
NET_BPS_MIN = 1.0
FIRES_PER_DAY_MIN = 10.0


def _f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return None


def load(CSV):
    """Read the rotated pre-instrument file plus the current one.

    2026-08-28 the recorder gained depth/staleness columns; its own header
    guard rotated the first 230+ minutes to minutes.csv.old. Both belong to
    the same 7-day window. Pre-instrument rows have None for the new fields
    and are excluded from size/staleness statistics only.
    """
    rows = []
    # 2026-09-01: rotations are now TIMESTAMPED (minutes.csv.<ts>.old) so a
    # second schema change cannot overwrite the first one's file. Glob every
    # rotation, oldest first, then the live file — the clock is the union.
    import glob as _glob
    _olds = sorted(_glob.glob(str(CSV) + "*.old"))
    for fp in [Path(x) for x in _olds] + [CSV]:
        if not fp.exists():
            continue
        with open(fp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append({
                        "ts": int(r["minute_ts"]),
                        "prem": float(r["premium_mean_bps"]),
                        "sell_max": float(r["sell_edge_max_bps"]),
                        "buy_max": float(r["buy_edge_max_bps"]),
                        "n": int(r["samples"]),
                        "sell_ntl": _f(r, "sell_max_notional_usd"),
                        "sell_age": _f(r, "sell_max_age_s"),
                        "buy_ntl": _f(r, "buy_max_notional_usd"),
                        "buy_age": _f(r, "buy_max_age_s"),
                        # 2026-09-01 funding columns (blank on pre-patch rows)
                        "f_e": _f(r, "fund_entropy_bps8h"),
                        "f_h": _f(r, "fund_hedge_bps8h"),
                        "f_d": _f(r, "fund_diff_bps8h"),
                    })
                except (ValueError, KeyError):
                    continue
    rows.sort(key=lambda x: x["ts"])
    return rows


# ── convergence (registered 2026-08-28, after 230 min ≈ 3% of the window;
#    a TIGHTENING amendment, disclosed in TODO §0.75) ─────────────────────
# A persistent one-sided gap is not capturable: entering the position only
# pays when the gap CLOSES. So the verdict additionally requires that the
# premium, once it deviates from its rolling midline beyond the band,
# actually comes back: >=70% of deviation episodes must return to within
# half the band inside 240 minutes. Fail => the "edge" is structural
# offset / drift, and the line closes regardless of how fat it looks.
MIDLINE_WIN = 360          # minutes of trailing median for the midline
CONV_RETURN_FRAC = 0.5     # "converged" = back within band*this
CONV_MAX_MIN = 240
CONV_PASS_FRAC = 0.70


def convergence(rows, band_bps, with_starts=False):
    """with_starts=True additionally returns the start ts of every episode
    (2026-09-04, session-window exploration). Default output is unchanged
    -- the verdict path never passes it -- so the frozen scorer's numbers
    cannot move; the flag only exposes what the loop already knew."""
    import statistics as st
    prems = [x["prem"] for x in rows]
    starts = []
    episodes, i, n = [], MIDLINE_WIN, len(rows)
    while i < n:
        mid = st.median(prems[i - MIDLINE_WIN:i])
        dev = prems[i] - mid
        if abs(dev) >= band_bps:
            sign = 1 if dev > 0 else -1
            j = i + 1
            while j < n:
                if sign * (prems[j] - mid) <= band_bps * CONV_RETURN_FRAC:
                    break
                j += 1
            mins = (rows[j]["ts"] - rows[i]["ts"]) / 60 if j < n else None
            episodes.append(mins)
            starts.append(rows[i]["ts"])
            i = j + 1
        else:
            i += 1
    if not episodes:
        return {"episodes": 0}
    ok = sum(1 for m in episodes if m is not None and m <= CONV_MAX_MIN)
    med = st.median([m for m in episodes if m is not None] or [float("inf")])
    out = {"episodes": len(episodes), "converged_4h": ok,
           "frac": round(ok / len(episodes), 2),
           "median_minutes": round(med, 1) if med != float("inf") else None,
           "passed": ok / len(episodes) >= CONV_PASS_FRAC}
    if with_starts:
        out["starts"] = list(zip(starts, episodes))
    return out


def instrument_stats(rows, side):
    """Depth + staleness at the fat prints — instrumented rows only."""
    ins = [x for x in rows if x[f"{side}_ntl"] is not None]
    if not ins:
        return None
    fat = sorted(ins, key=lambda x: -x[f"{side}_max"])[:max(1, len(ins) // 10)]
    ntl = sorted(x[f"{side}_ntl"] for x in fat)
    stale = sum(1 for x in fat if (x[f"{side}_age"] or 0) > 5)
    return {"instrumented_min": len(ins),
            "fat_prints": len(fat),
            "fat_median_notional_usd": round(ntl[len(ntl) // 2], 0),
            "fat_stale_gt5s": stale}


def funding_stats(rows):
    """Carry side of the ledger (2026-09-01).

    The price spread pays only if it CONVERGES; the funding differential
    pays for HOLDING. MOB's own Delta-Neutral card says as much ("the
    funding spread is the return"). A snapshot cannot tell a stable carry
    from a number that flips daily, so this reports the distribution and
    the sign stability of diff = hedge - entropy (bps per 8h), plus the
    annualised median. REPORT ONLY — no gate, no verdict: the frozen
    2026-08-28 criteria are about the price spread and do not get amended
    after the fact to cover a second payoff (that would be exactly the
    post-hoc gate change the pre-registration exists to forbid).
    """
    vals = [r["f_d"] for r in rows if r.get("f_d") is not None]
    if len(vals) < 30:
        return None
    vals_sorted = sorted(vals)
    med = vals_sorted[len(vals_sorted) // 2]
    pos = sum(1 for v in vals if v > 0) / len(vals)
    return {"n": len(vals),
            "median_bps_8h": round(med, 4),
            "p10_bps_8h": round(vals_sorted[int(0.1 * len(vals_sorted))], 4),
            "p90_bps_8h": round(vals_sorted[int(0.9 * len(vals_sorted))], 4),
            "frac_positive": round(pos, 3),
            "annualised_pct_at_median": round(med / 1e4 * 3 * 365 * 100, 2)}


def side_stats(rows, key):
    """p90 candidate band per analyze.py's methodology, then fire counts."""
    vals = sorted(x[key] for x in rows)
    if not vals:
        return None
    p90 = vals[int(0.9 * len(vals))]
    band = max(p90, NET_BPS_MIN)
    days = max((rows[-1]["ts"] - rows[0]["ts"]) / 86400, 1e-9)
    fires = sum(1 for x in rows if x[key] >= band)
    return {"p90_bps": round(p90, 3), "band_bps": round(band, 3),
            "fires": fires, "fires_per_day": round(fires / days, 1)}


def capturable_usd_per_day(side, ins, conv=None):
    """REPORT-ONLY (not in the gate): fires/day x band x depth at the fat
    prints. Registered 2026-08-30 after SNDK's interim showed $200-400 books:
    a pair can pass the statistical gate and still be worth cents. This
    number decides whether a PASS is worth engineering, never whether it
    passes."""
    if not side or not ins or ins.get("fat_median_notional_usd") is None:
        return None
    # 2026-09-01 — THE CONTROL PAIR EARNED ITS KEEP. BTC (two deep zero-fee
    # books, band expected ~0) reported the family's LARGEST capturable
    # number, $97.8/day.  The data was real: HL quotes BTC a persistent
    # 3-5 bps above the Robinhood chain, positive in 100% of minutes.  But
    # its convergence column read `episodes: 0` — the premium never
    # deviates from its own rolling midline because it IS the midline: a
    # structural offset, not a spread that comes back.  You can lock it in
    # and never get paid.  The GATE was right (no convergence, no pass);
    # this REPORT metric was wrong, because band x fires x depth counts a
    # permanent offset as money.  Gated on convergence now: "fat but never
    # closes" is the most seductive way for this line to lose money.
    if conv is not None and not conv.get("passed"):
        return 0.0
    return round(side["fires_per_day"] * side["band_bps"] / 1e4
                 * ins["fat_median_notional_usd"], 2)


def capturable_usd_per_day_tradeable(side, ins, conv, days):
    """REPORT-ONLY, and the one to quote.  2026-09-03: the operator asked
    where the ceiling number above comes from, and re-deriving it found it
    overstates by one to two ORDERS OF MAGNITUDE.  Two independent errors,
    both in the same direction:

      1. `fires` counts MINUTES, not trades.  The band is the p90 of the
         executable room, so by construction 10% of minutes fire — 144/day
         out of 1440, for every pair, always.  But a deviation lasting an
         hour is ONE trade with sixty fire-minutes, not sixty trades.  The
         count of actual opportunities is the count of deviation EPISODES,
         which the convergence check already computes: SNDK's buy side has
         846 fire-minutes and **4 episodes**.
      2. It pays the FULL band per trade.  "Converged" is defined as coming
         back within HALF the band, so a trade entered at the band and
         closed on that definition captures half of it, not all of it.

    So: episodes/day x half the band x depth.  Still a ceiling (it assumes
    every episode is caught, filled at top-of-book on both legs, at zero
    fees), but a ceiling of the right order.  The old number is kept beside
    it as `_ceiling` rather than deleted, because a number that was quoted
    should stay auditable.

    The GATE is untouched by this: it was always about band width, fire
    frequency, both halves and convergence — never about this figure.
    """
    if conv is None or not conv.get("passed") or not days:
        return 0.0
    ep = conv.get("episodes") or 0
    if not ep or not side or not ins or ins.get("fat_median_notional_usd") is None:
        return 0.0
    return round(ep / days * (side["band_bps"] / 2) / 1e4
                 * ins["fat_median_notional_usd"], 2)


def score_pair(pid, csv_sub, leg_a, leg_b, note, now):
    CSV = LOGS / csv_sub
    res = {"pair": pid, "legs": f"{leg_a} vs {leg_b}", "note": note}
    if not CSV.exists():
        print(f"\n[{pid}] {leg_a} vs {leg_b} —— 錄製檔不存在（freshness board 應該在響）")
        res["status"] = "missing"
        return res
    rows = load(CSV)
    if not rows:
        print(f"\n[{pid}] {leg_a} vs {leg_b} —— 尚無資料列")
        res["status"] = "empty"
        return res
    start = FIXED_START.get(pid) or datetime.fromtimestamp(rows[0]["ts"], tz=timezone.utc)
    days = (now - start).total_seconds() / 86400
    print(f"\n[{pid}] {leg_a} vs {leg_b}（{note}）")
    print(f"  已錄 {len(rows)} 分鐘｜起 {start:%m-%d %H:%M}Z｜經過 {days:.1f}／{GATE_DAYS} 天")
    res.update({"minutes": len(rows), "start_utc": start.strftime("%Y-%m-%d %H:%M"),
                "days": round(days, 2), "gate_days": GATE_DAYS,
                "gate_met": days >= GATE_DAYS})
    if days < GATE_DAYS:
        if len(rows) >= 60:
            s = side_stats(rows, "sell_max")
            b = side_stats(rows, "buy_max")
            res["interim"] = {"sell": s, "buy": b}
            print(f"  期中（**不是判決**）：sell p90 {s['p90_bps']:+.2f} bps｜buy p90 {b['p90_bps']:+.2f} bps")
            if len(rows) > MIDLINE_WIN + 30:
                for lab, st_ in (("sell", s), ("buy", b)):
                    c = convergence(rows, st_["band_bps"])
                    res["interim"][f"conv_{lab}"] = c
                    if c.get("episodes"):
                        print(f"    {lab} 帶偏離 {c['episodes']} 次、4h 內收斂 {c['converged_4h']} 次（中位 {c['median_minutes']} 分）")
            for lab, st_ in (("sell", s), ("buy", b)):
                ins = instrument_stats(rows, lab)
                if ins:
                    res["interim"][f"depth_{lab}"] = ins
                    conv_ = res["interim"].get(f"conv_{lab}")
                    ceil = capturable_usd_per_day(st_, ins, conv_)
                    cap = capturable_usd_per_day_tradeable(st_, ins, conv_, days)
                    res["interim"][f"capturable_usd_per_day_{lab}"] = cap
                    res["interim"][f"capturable_usd_per_day_{lab}_ceiling"] = ceil
                    if ceil == 0.0:
                        print(f"    {lab} 帶存在但**不收斂**（結構性偏移，"
                              f"不是會回來的價差）→ 可捕獲記 0")
                    ep = (conv_ or {}).get("episodes") or 0
                    print(f"    {lab} 最肥時刻深度中位 ${ins['fat_median_notional_usd']:,.0f}"
                          f"｜卡價(>5s) {ins['fat_stale_gt5s']}/{ins['fat_prints']}"
                          f"｜可捕獲 ≈ ${cap}/天"
                          f"（{ep} 次偏離 × 半個帶；分鐘計數的上限是 ${ceil}"
                          f"，兩個都是報告用不進判準）")
            f = funding_stats(rows)
            if f:
                res["interim"]["funding"] = f
                print(f"    carry：資金費率差中位 {f['median_bps_8h']:+.3f} bps/8h"
                      f"（年化 {f['annualised_pct_at_median']:+.1f}%）"
                      f"｜同號比例 {f['frac_positive']*100:.0f}%"
                      f"｜p10/p90 {f['p10_bps_8h']:+.2f}/{f['p90_bps_8h']:+.2f}"
                      f"（n={f['n']}，報告用，不進判準）")
        res["status"] = "accumulating"
        print("  → 閘門未達，不出判決。")
        return res
    # ── verdict path ────────────────────────────────────────────────────
    mid = rows[len(rows) // 2]["ts"]
    halves = ([r for r in rows if r["ts"] < mid],
              [r for r in rows if r["ts"] >= mid])
    verdict_sides = {}
    ok_any = False
    ok_any_at_fee = False
    # THE FROZEN GATE'S OWN PARENTHETICAL IS NOW KNOWN TO BE CONDITIONAL.
    # It reads "fees are 0+0 on this pair, so net == raw executable room",
    # written 2026-08-28. On 2026-09-03 the fee check found that the zero on
    # the Entropy/HL leg is Entropy's referral rebate (a promotion, and
    # UNCONFIRMED on a real fill), not a schedule -- and for the HL-core
    # pairs (BTC/HYPE/ZEC/NEAR) there is no rebate claimed at all, so their
    # true requirement is 18 bps, not 1.
    #
    # The registered threshold is NOT touched: `passed` below is still the
    # frozen criterion, because moving a bar the day before the verdict is
    # the thing pre-registration exists to stop. What IS added is the fee
    # truth beside it -- a second flag, named, so the verdict cannot be
    # read without seeing which assumption it rests on.
    va, vb = VENUE_KEYS.get(pid, ("HL", "lighter"))
    req_rebate = FEES.required_band_bps(va, vb)
    req_sched = FEES.required_band_bps(va, vb, rebate=False)
    unver = FEES.unverified(va, vb)
    print(f"  費率：{va}+{vb} → 需要帶 {req_rebate:.1f} bps（含返佣）／"
          f"{req_sched:.1f} bps（只看費率表）"
          + (f"｜未查證的腿：{','.join(unver)}" if unver else ""))
    for key, lab in (("sell_max", "sell"), ("buy_max", "buy")):
        full = side_stats(rows, key)
        h = [side_stats(hh, key) for hh in halves]
        conv = convergence(rows, full["band_bps"])
        ins = instrument_stats(rows, lab)
        passed = (full["band_bps"] >= NET_BPS_MIN
                  and full["fires_per_day"] >= FIRES_PER_DAY_MIN
                  and all(x["fires_per_day"] >= FIRES_PER_DAY_MIN
                          and x["band_bps"] >= NET_BPS_MIN for x in h)
                  and bool(conv.get("passed")))
        ceil = capturable_usd_per_day(full, ins, conv)
        cap = capturable_usd_per_day_tradeable(full, ins, conv, days)
        net_rebate = FEES.net_per_trade_bps(full["band_bps"], va, vb)
        net_sched = FEES.net_per_trade_bps(full["band_bps"], va, vb,
                                           rebate=False)
        passed_at_fee = passed and net_rebate > 0
        verdict_sides[lab] = {"full": full, "halves": h,
                              "convergence": conv, "depth": ins,
                              "capturable_usd_per_day": cap,
                              "capturable_usd_per_day_ceiling": ceil,
                              "passed": passed,
                              "fee_venues": [va, vb],
                              "required_band_bps_with_rebate": round(req_rebate, 2),
                              "required_band_bps_schedule": round(req_sched, 2),
                              "net_bps_per_trade_with_rebate": round(net_rebate, 2),
                              "net_bps_per_trade_schedule": round(net_sched, 2),
                              "fee_unverified_legs": unver,
                              "passed_at_fee": passed_at_fee}
        ok_any = ok_any or passed
        ok_any_at_fee = ok_any_at_fee or passed_at_fee
        cs = (f"收斂 {conv['frac']*100:.0f}%（門檻 {CONV_PASS_FRAC*100:.0f}%）"
              if conv.get("episodes") else "收斂事件 0（不可判）")
        mark = "✓" if passed else "✗"
        mark += (" 扣費後仍正" if passed_at_fee
                 else (" 扣費後轉負" if passed else ""))
        print(f"  {lab}: 帶 {full['band_bps']:.2f} bps、{full['fires_per_day']:.1f} 次/天、"
              f"兩半 {h[0]['fires_per_day']:.1f}/{h[1]['fires_per_day']:.1f}、{cs} → {mark}")
        if ins:
            print(f"        最肥時刻深度中位 ${ins['fat_median_notional_usd']:,.0f}"
                  f"｜卡價(>5s) {ins['fat_stale_gt5s']}/{ins['fat_prints']}"
                  f"｜可捕獲 ≈ ${cap}/天"
                  f"（{conv.get('episodes') or 0} 次偏離 × 半個帶；"
                  f"分鐘計數的上限是 ${ceil}）——供工程閘門評估，不進本判準)")
    if pid == "BTC" and ok_any:
        v = ("**對照組亮了** —— BTC 兩個深簿之間不該有帶；先查儀器，不是查市場。"
             f"（本句原文寫「兩邊零費率」，2026-09-03 費率查證後改正："
             f"Lighter 是 0，HL 不是，這個配對需要帶 {req_rebate:.1f} bps。）")
    elif ok_any and ok_any_at_fee:
        v = ("**過閘** —— 進工程閘門討論（審計下單路徑、統一風控、資金拆分）。"
             "注意這只證明溢價存在，不證明抓得到它；看可捕獲美元再決定值不值得。"
             f"帶也撐得過實際費率（需要 {req_rebate:.1f} bps）"
             + (f"，但 {','.join(unver)} 的費率／返佣尚未在真實成交上查證——"
                "這一條沒查證前不得動錢。" if unver else "。"))
    elif ok_any:
        v = ("**過閘（僅在零費率假設下）** —— 帶過了 2026-08-28 凍結的 "
             f"{NET_BPS_MIN:.1f} bps 門檻，但按費率表這個配對需要 "
             f"{req_rebate:.1f} bps（只看費率表 {req_sched:.1f} bps），"
             "所以扣掉實際費率之後每筆是負的。凍結門檻寫的是「扣費後」，"
             "而它當時把費用當成 0+0——**判準沒被改，是它的前提被推翻了**。"
             "要救這條線只有兩條路：查證返佣，或改掛單執行。")
    else:
        v = "**關線** —— 扣費後的可成交空間撐不起門檻；一週結案，成本≈零。"
    print(f"  判決：{v}")
    res.update({"status": "verdict", "sides": verdict_sides, "verdict": v,
                "fee": {"venues": [va, vb],
                        "required_band_bps_with_rebate": round(req_rebate, 2),
                        "required_band_bps_schedule": round(req_sched, 2),
                        "unverified_legs": unver},
                "passed_at_fee": ok_any_at_fee})
    return res


def main() -> int:
    now = datetime.now(timezone.utc)
    print("§0.75 兩場館套利時鐘（判準 2026-08-28 凍結；家族 2026-08-30 登記，同一判準、全報不挑）")
    out = {"asof_utc": now.strftime("%Y-%m-%d %H:%M"), "gate_days": GATE_DAYS,
           "pairs": {}}
    for pid, sub, a, b, note in PAIRS:
        out["pairs"][pid] = score_pair(pid, sub, a, b, note, now)
    # backward-compatible top level = the original SNDK clock
    sn = out["pairs"].get("SNDK", {})
    for k in ("minutes", "days", "gate_met", "interim", "sides", "verdict"):
        if k in sn:
            out[k] = sn[k]
    print("\n  零費率＋看起來很肥的半天資料正是最誘人提早開獎的組合——判準凍結的意義就在此刻。")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
