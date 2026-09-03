# -*- coding: utf-8 -*-
"""Variant E, taken apart: which of the three panels is doing the work?

E (registered 2026-08-02, TODO 0.474) is the operator's own manual read of a
BTC raid, encoded from columns already recorded prospectively:

    E = OI down at the raid hour          (positions being flushed, not added)
      AND taker flow WITH the break       (the push is aggressive, not passive)
      AND liquidation burst >= the causal median of BTC's earlier raids

The forward clock (shadow log, 22 closed rows since registration) is the only
thing allowed to judge whether E works. This script does something different
and strictly descriptive: it takes the SAME three conditions to the full
historical BTC raid set (research/sweep_raid_anatomy + _derivs, ~2.7k raids)
and asks what the rule is made of.

Pre-stated questions, fixed before running (no search, no thresholds tuned):
  Q1  the full 2x2x2 panel table: every combination reported, not just E's cell
  Q2  leave-one-out: drop each of the three conditions from E in turn
  Q3  stability: first half vs second half, and per calendar year
  Q4  confounds: does E just select one side / deeper pierces / higher vol?
  Q5  bootstrap CI on E's netR and on (E minus everything else)

READ THIS BEFORE USING ANY NUMBER BELOW: this is the sample E was born from.
It cannot confirm E. It can only say what E is -- and if a single panel turns
out to carry everything, that is a NEW hypothesis needing its own frozen
clock, not a licence to edit E. Historical netR here is the anatomy engine's
gross retest outcome (no bps costs), so levels are not comparable with the
shadow ledger's net R; comparisons WITHIN this table are apples to apples.

Run: python research/sweep_raid_variant_e.py
Out: research/results/sweep_raid_variant_e.json
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_variant_e.json"
MIN_PRIOR = 5          # same as shadow_engine.variant_e_pred
BOOT = 4000


def causal_liq_high(rows):
    """liq_burst >= median of STRICTLY EARLIER raids (>= MIN_PRIOR of them).

    Same convention as shadow_engine.variant_e_pred: no global median, no
    tuned cut -- the bar at each raid is what BTC's own past raids looked
    like up to that moment.
    """
    rows = sorted(rows, key=lambda r: r["ts"])
    prior = []
    for r in rows:
        lb = r.get("liq_burst")
        r["liq_high"] = (lb is not None and len(prior) >= MIN_PRIOR
                         and lb >= st.median(prior))
        if lb is not None:
            prior.append(lb)
    return rows


def cell(rows):
    n = len(rows)
    if not n:
        return None
    nets = [r["netR"] for r in rows if r.get("netR") is not None]
    rv = 100 * sum(1 for r in rows if r["cls"] == "REVERSAL") / n
    out = {"n": n, "reversal_pct": round(rv, 1), "n_retest": len(nets)}
    if nets:
        out["netR"] = round(sum(nets) / len(nets), 4)
        out["wr_pct"] = round(100 * sum(1 for x in nets if x > 0) / len(nets), 1)
    return out


def boot_ci(vals, nb=BOOT, seed=7):
    if len(vals) < 5:
        return None, None
    rnd = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(vals[rnd.randrange(n)] for _ in range(n)) / n
                for _ in range(nb))
    return round(ms[int(0.025 * nb)], 4), round(ms[int(0.975 * nb)], 4)


def boot_diff(a, b, nb=BOOT, seed=11):
    if len(a) < 5 or len(b) < 5:
        return None, None
    rnd = random.Random(seed)
    na, nb_ = len(a), len(b)
    ds = sorted((sum(a[rnd.randrange(na)] for _ in range(na)) / na)
                - (sum(b[rnd.randrange(nb_)] for _ in range(nb_)) / nb_)
                for _ in range(nb))
    return round(ds[int(0.025 * nb)], 4), round(ds[int(0.975 * nb)], 4)


def nets(rows):
    return [r["netR"] for r in rows if r.get("netR") is not None]


def main() -> int:
    raw = A.raids("BTC")
    rows = D.attach(raw, D.load_state())
    rows = [r for r in rows
            if r.get("oi_chg_raid") is not None
            and r.get("fut_taker_signed") is not None
            and r.get("liq_burst") is not None]
    rows = causal_liq_high(rows)
    for r in rows:
        r["oi_dn"] = r["oi_chg_raid"] < 0
        r["with_break"] = r["fut_taker_signed"] > 0
        r["Q"] = r["oi_dn"] and r["with_break"]
        r["E"] = r["Q"] and r["liq_high"]

    res = {"n_raids": len(rows),
           "span": [datetime.fromtimestamp(min(r["ts"] for r in rows), timezone.utc)
                    .strftime("%Y-%m-%d"),
                    datetime.fromtimestamp(max(r["ts"] for r in rows), timezone.utc)
                    .strftime("%Y-%m-%d")]}
    print("=" * 78)
    print(f"  VARIANT E ANATOMY — BTC raids {res['span'][0]} .. {res['span'][1]}"
          f"  (n={len(rows)})")
    print("=" * 78)
    print("  descriptive only: this is the sample E was discovered in.\n")

    # ── Q1: the whole 2x2x2 ──────────────────────────────────────────
    print("  [Q1] 三個盤的全部八格（不挑格）")
    print(f"  {'OI':<5}{'taker':<7}{'liq':<6}{'n':>6}{'反轉%':>8}"
          f"{'netR':>9}{'WR%':>7}")
    tbl = {}
    for oi in (True, False):
        for tk in (True, False):
            for lq in (True, False):
                sub = [r for r in rows if r["oi_dn"] == oi
                       and r["with_break"] == tk and r["liq_high"] == lq]
                c = cell(sub)
                key = f"{'OI↓' if oi else 'OI↑'}/{'順破' if tk else '逆破'}/" \
                      f"{'爆量' if lq else '平量'}"
                tbl[key] = c
                if c:
                    print(f"  {'dn' if oi else 'up':<5}"
                          f"{'with' if tk else 'anti':<7}"
                          f"{'high' if lq else 'low':<6}{c['n']:>6}"
                          f"{c['reversal_pct']:>8.1f}"
                          f"{c.get('netR', float('nan')):>+9.4f}"
                          f"{c.get('wr_pct', float('nan')):>7.1f}")
    res["cells"] = tbl

    # ── Q2: leave-one-out ────────────────────────────────────────────
    print("\n  [Q2] 從 E 拿掉一個條件（其餘不變）")
    defs = {
        "E = OI↓ ∧ 順破 ∧ 爆量": lambda r: r["oi_dn"] and r["with_break"] and r["liq_high"],
        "  −OI（順破∧爆量）": lambda r: r["with_break"] and r["liq_high"],
        "  −順破（OI↓∧爆量）": lambda r: r["oi_dn"] and r["liq_high"],
        "  −爆量（Q = OI↓∧順破）": lambda r: r["oi_dn"] and r["with_break"],
        "  只有爆量": lambda r: r["liq_high"],
        "  只有 OI↓": lambda r: r["oi_dn"],
        "  只有順破": lambda r: r["with_break"],
        "  全部 raid（基準）": lambda r: True,
    }
    base_nets = nets(rows)
    res["leave_one_out"] = {}
    print(f"  {'定義':<26}{'n':>6}{'反轉%':>8}{'netR':>9}{'WR%':>7}"
          f"   {'CI95(netR)':>20}   {'vs 基準 CI':>20}")
    for name, pred in defs.items():
        sub = [r for r in rows if pred(r)]
        c = cell(sub)
        if not c:
            continue
        ns = nets(sub)
        lo, hi = boot_ci(ns)
        rest = [r for r in rows if not pred(r)]
        dlo, dhi = boot_diff(ns, nets(rest))
        c.update({"ci": [lo, hi], "vs_rest_ci": [dlo, dhi]})
        res["leave_one_out"][name.strip()] = c
        print(f"  {name:<26}{c['n']:>6}{c['reversal_pct']:>8.1f}"
              f"{c.get('netR', 0):>+9.4f}{c.get('wr_pct', 0):>7.1f}"
              f"   [{lo:+.4f},{hi:+.4f}]"
              + (f"   [{dlo:+.4f},{dhi:+.4f}]" if dlo is not None else ""))

    # ── Q3: stability ────────────────────────────────────────────────
    print("\n  [Q3] 穩定性")
    ordered = sorted(rows, key=lambda r: r["ts"])
    half = len(ordered) // 2
    for tag, sub in (("前半", ordered[:half]), ("後半", ordered[half:])):
        e = [r for r in sub if r["E"]]
        o = [r for r in sub if not r["E"]]
        ce, co = cell(e), cell(o)
        if ce and co:
            print(f"  {tag}: E n={ce['n']:>4} netR{ce.get('netR', 0):+.4f} "
                  f"反轉{ce['reversal_pct']:.0f}%  |  非E n={co['n']:>4} "
                  f"netR{co.get('netR', 0):+.4f} 反轉{co['reversal_pct']:.0f}%")
    byyear = defaultdict(list)
    for r in ordered:
        byyear[datetime.fromtimestamp(r["ts"], timezone.utc).year].append(r)
    res["by_year"] = {}
    for y in sorted(byyear):
        e = [r for r in byyear[y] if r["E"]]
        o = [r for r in byyear[y] if not r["E"]]
        ce, co = cell(e), cell(o)
        if ce:
            res["by_year"][y] = {"E": ce, "nonE": co}
            print(f"  {y}: E n={ce['n']:>4} netR{ce.get('netR', 0):+.4f} "
                  f"反轉{ce['reversal_pct']:.0f}%  |  非E netR"
                  f"{(co or {}).get('netR', 0):+.4f} 反轉"
                  f"{(co or {}).get('reversal_pct', 0):.0f}%")

    # ── Q4: confounds ────────────────────────────────────────────────
    print("\n  [Q4] E 是不是只是選到了別的東西")
    e = [r for r in rows if r["E"]]
    o = [r for r in rows if not r["E"]]
    res["confounds"] = {}
    for label, key in (("做多側佔比(side=-1 掃低點)", "side"),
                       ("liq_burst 中位數", "liq_burst"),
                       ("|OI 變化|中位數", "oi_chg_raid"),
                       ("taker 強度中位數", "fut_taker_signed"),
                       ("stop_fuel 中位數", "stop_fuel")):
        if key == "side":
            ve = 100 * sum(1 for r in e if r["side"] == -1) / len(e)
            vo = 100 * sum(1 for r in o if r["side"] == -1) / len(o)
        else:
            ee = [abs(r[key]) if key == "oi_chg_raid" else r[key]
                  for r in e if r.get(key) is not None]
            oo = [abs(r[key]) if key == "oi_chg_raid" else r[key]
                  for r in o if r.get(key) is not None]
            ve, vo = (st.median(ee) if ee else 0), (st.median(oo) if oo else 0)
        res["confounds"][label] = [round(ve, 4), round(vo, 4)]
        print(f"  {label:<26} E {ve:>9.4f}   非E {vo:>9.4f}")

    # ── Q5: is it just a side bet? ───────────────────────────────────
    # E fires 85% on low sweeps (the fade is LONG) versus 40% at base --
    # mechanically, because long liquidations are what makes a burst. So the
    # only honest test is WITHIN each side: if E beats base on both sides,
    # the panels are doing something; if only on one, E is a direction bet
    # wearing a signal's clothes.
    print("\n  [Q5] 分邊看（E 有 85% 落在掃低點那一側）")
    print(f"  {'側':<16}{'定義':<22}{'n':>6}{'netR':>10}{'WR%':>7}"
          f"   {'vs 同側基準 CI':>22}")
    res["by_side"] = {}
    for sd, zh in ((-1, "掃低點→做多"), (1, "掃高點→做空")):
        side_rows = [r for r in rows if r["side"] == sd]
        base_ns = nets(side_rows)
        for name, pred in (("基準（全部）", lambda r: True),
                           ("E（三盤）", lambda r: r["E"]),
                           ("E′（OI↓∧爆量）", lambda r: r["oi_dn"] and r["liq_high"]),
                           ("只有 OI↓", lambda r: r["oi_dn"]),
                           ("只有爆量", lambda r: r["liq_high"])):
            sub = [r for r in side_rows if pred(r)]
            c = cell(sub)
            if not c or c["n"] < 20:
                continue
            ns = nets(sub)
            dlo, dhi = boot_diff(ns, base_ns)
            res["by_side"][f"{zh}/{name}"] = dict(c, vs_side_base_ci=[dlo, dhi])
            tail = f"   [{dlo:+.4f},{dhi:+.4f}]" if dlo is not None else ""
            print(f"  {zh:<16}{name:<22}{c['n']:>6}{c.get('netR', 0):>+10.4f}"
                  f"{c.get('wr_pct', 0):>7.1f}{tail}")

    # ── Q6: what the CVD panel actually costs ────────────────────────
    # E' minus E is exactly the "OI down AND burst BUT taker against the
    # break" pocket. Small by construction; reported so the CVD panel's
    # marginal value is a number, not an impression.
    pocket = [r for r in rows if r["oi_dn"] and r["liq_high"] and not r["with_break"]]
    c = cell(pocket)
    if c:
        lo, hi = boot_ci(nets(pocket))
        res["cvd_pocket"] = dict(c, ci=[lo, hi])
        print(f"\n  [Q6] E′ 比 E 多出來的那一格（OI↓∧爆量∧**逆**破）："
              f"n={c['n']} netR{c.get('netR', 0):+.4f} "
              f"WR{c.get('wr_pct', 0):.0f}% CI[{lo:+.4f},{hi:+.4f}]")
        print("       CVD 那一盤擋掉的就是這一格——擋得對不對，看它是不是比 E 差。")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
