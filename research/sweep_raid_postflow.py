# -*- coding: utf-8 -*-
"""POST-raid order flow — the window the four anatomy rounds never cut.

User request (2026-08-01): 每一次流動性獵取「後」的訂單流特徵，分辨接下來
會反轉還是延續. Rounds 2-4 measured the raid HOUR (attack window) and the
pre-raid buildup (FAIL). The window between raid-hour close and the retest
is the last information available BEFORE the entry decision, and it is
causally clean for formulation 1: when price touches the level (fill), the
post-raid flow up to that minute is fully known.

Two pre-stated formulations, no dredging beyond these:

  F1 (entry quality, fills only, gap >= 30 min so a window exists):
     features over [sweep_hour_end, fill_bar_open), target = netR.
       pf_taker    signed taker share IN BREAK DIRECTION over the window
                   (+ = flow still pushing the break -> continuation fuel;
                    - = flow flipped against the break -> reversal fuel)
       pf_volrate  window vol/min divided by attack vol/min (exhaustion:
                   low = the push died; high = still fighting)
       pf_hold     fraction of window minutes with price still beyond the
                   level (price-side control; the "¬R holds outside"
                   continuation read extended past the raid hour)
     Prediction (stated before running): reversal quality is best when
     pf_taker < 0 (flow flipped) and pf_volrate low (exhausted).

  F2 (resolution forecast at a FIXED clock): among raids NOT yet retested
     by sweep+2h, features over [sweep+1h, sweep+2h) -> eventual class
     (BREAKOUT vs retested). Survivors only — base rates shift and are
     reported; fills inside that hour are excluded to avoid label leakage.

Discipline: 3 features x 2 formulations ~ 6 looks; bar = monotone + BOTH
symbols + halves-consistent + material size. BTC+ETH, ~100d of 1m flow.
Selection caveat printed: F1's gap>=30min subset is ~the slower retests;
immediate fills (no window) are reported as a separate base rate.

Run: python research/sweep_raid_postflow.py
Out: research/results/sweep_raid_postflow.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_attack as K  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_postflow.json"


def raids_with_fill(sym: str):
    """K.raids_with_level plus fill/exit bar timestamps (needed for the
    post-raid window). Same frozen rules, bookkeeping only."""
    bars = SC.load_csv(str(LT.CACHE / f"{sym}USDT_1h.csv"))
    n = len(bars)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    cl = [b[SC.C] for b in bars]
    a = SC.atr14(bars)
    P = SC.PIVOT
    pools = []
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            pools.append((i + P + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            pools.append((i + P + 1, lo[i], -1))
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        pools += list(lv.get(kind, []))
    out = []
    pending = sorted(pools)
    idx = 0
    live: list[tuple] = []
    for j in range(n - SC.W - SC.HOLD - 1):
        while idx < len(pending) and pending[idx][0] <= j:
            live.append(pending[idx][1:])
            idx += 1
        if a[j] is None or a[j] == 0:
            continue
        hit = [t for t in live if (h[j] > t[0] if t[1] == 1 else lo[j] < t[0])]
        if not hit:
            continue
        live = [t for t in live if t not in hit]
        for lvl, s in hit:
            kd, d = s, -s
            pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / a[j]
            fill = None
            for f in range(j + 1, j + 1 + SC.W):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    fill = f
                    break
            rec = {"sym": sym, "ts": bars[j][0], "pierce": pierce,
                   "side": kd, "lvl": lvl}
            if fill is None:
                rec.update({"cls": "BREAKOUT", "netR": None, "fill_ts": None})
            else:
                A_ = a[j]
                stop = lvl - d * SC.DIS * A_
                R, xb = None, min(fill + SC.HOLD, n - 1)
                for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
                    if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                        R, xb = -1.0, k
                        break
                if R is None:
                    R = d * (cl[xb] - lvl) / (SC.DIS * A_)
                net = LT.net(R, lvl, A_)
                rec.update({"cls": "REVERSAL" if net > 0 else "FAKE_RETEST",
                            "netR": net, "fill_ts": bars[fill][0]})
            out.append(rec)
    return out


def post_window(r, flow, m1, end_min):
    """Features over [sweep_hour_end, end_min). None if window too thin."""
    m_start = r["ts"] // 60 + 60
    if end_min - m_start < 30:
        return None
    s = r["side"]
    lvl = r["lvl"]
    fw = [flow[m] for m in range(m_start, end_min) if m in flow]
    pm = [m1[m] for m in range(m_start, end_min) if m in m1]
    if len(fw) < (end_min - m_start) * 0.6 or len(pm) < (end_min - m_start) * 0.6:
        return None
    vol = sum(w[0] for w in fw)
    if vol <= 0:
        return None
    att = r.get("_att")          # (att_vol_per_min,) stashed by caller
    f = {}
    f["pf_taker"] = s * sum(w[1] for w in fw) / vol
    f["pf_volrate"] = ((vol / len(fw)) / att) if att and att > 0 else None
    beyond = sum(1 for (hi, lo_, _c) in pm
                 if (hi > lvl if s == 1 else lo_ < lvl))
    f["pf_hold"] = beyond / len(pm)
    return f


def tercile_line(rows, key, target="netR"):
    vals = sorted(r[key] for r in rows if r.get(key) is not None)
    if len(vals) < 90:
        return None, f"  {key:<12} thin (n={len(vals)})"
    lo_c, hi_c = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    parts, rec = [], {}
    if lo_c == hi_c:
        return None, f"  {key:<12} degenerate distribution (tercile cuts equal) — skipped"
    for name, pred in (("low", lambda v: v <= lo_c),
                       ("mid", lambda v: lo_c < v < hi_c),
                       ("high", lambda v: v >= hi_c)):
        g = [r for r in rows if r.get(key) is not None and pred(r[key])]
        if not g:
            parts.append(f"{name} n=0")
            continue
        if target == "netR":
            xs = [r["netR"] for r in g if r["netR"] is not None]
            m = sum(xs) / len(xs) if xs else float("nan")
            wr = 100 * sum(1 for x in xs if x > 0) / len(xs) if xs else 0
            rec[name] = {"n": len(g), "netR": m, "wr": wr}
            parts.append(f"{name} {m:+.3f}/{wr:.0f}% (n={len(g)})")
        else:
            br = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
            rec[name] = {"n": len(g), "breakout_pct": br}
            parts.append(f"{name} 突破{br:.0f}% (n={len(g)})")
    return rec, f"  {key:<12}" + " | ".join(parts)


def main() -> int:
    print("=" * 78)
    print("  POST-RAID FLOW — 獵取後·回踩前的訂單流（6 looks; 判準=單調+雙幣+前後半）")
    print("=" * 78)
    f1_rows, f2_rows = [], []
    n_gap0 = n_gap = 0
    gap0_net, gap_net = [], []
    for sym, canon in A.SYMS.items():
        flow, _i, _c = A.load_flow(canon)
        m1 = K.fetch_1m(sym)
        rr = raids_with_fill(sym)
        # attack vol/min for volrate denominator (reuse round-4 machinery)
        att = {(x["ts"]): x for x in K.attach_attack(
            [dict(r) for r in rr if True], flow, m1)}
        for r in rr:
            a_ = att.get(r["ts"])
            if r["fill_ts"] is not None:
                gap_min = (r["fill_ts"] - r["ts"]) // 60 - 60
                if gap_min < 30:
                    n_gap0 += 1
                    if r["netR"] is not None:
                        gap0_net.append(r["netR"])
                    continue
                if a_ is None or a_.get("att_min", 0) <= 0:
                    continue
                r["_att"] = (sum(flow[m][0] for m in range(
                    r["ts"] // 60, r["ts"] // 60 + 60) if m in flow)
                    / max(a_["att_min"], 1))
                f = post_window(r, flow, m1, r["fill_ts"] // 60)
                if f:
                    n_gap += 1
                    if r["netR"] is not None:
                        gap_net.append(r["netR"])
                    r.update(f)
                    f1_rows.append(r)
            else:
                # F2: breakouts + (filled later than sweep+2h)
                pass
        for r in rr:
            filled_by_2h = (r["fill_ts"] is not None
                            and r["fill_ts"] <= r["ts"] + 7200)
            if filled_by_2h:
                continue
            a_ = att.get(r["ts"])
            if a_ is None or a_.get("att_min", 0) <= 0:
                continue
            r["_att"] = (sum(flow[m][0] for m in range(
                r["ts"] // 60, r["ts"] // 60 + 60) if m in flow)
                / max(a_["att_min"], 1))
            f = post_window(r, flow, m1, r["ts"] // 60 + 120)
            if f:
                r2 = dict(r)
                r2.update(f)
                f2_rows.append(r2)

    res = {}
    print(f"\n  [F1] 進場品質（fill 前窗口≥30min）n={len(f1_rows)}")
    print(f"  selection 注意: 立即回踩(無窗口) n={n_gap0} 均netR "
          f"{sum(gap0_net)/len(gap0_net):+.3f} vs 慢回踩 n={n_gap} 均 "
          f"{sum(gap_net)/len(gap_net):+.3f} — 兩群本來就不同")
    for k in ("pf_taker", "pf_volrate", "pf_hold"):
        rec, line = tercile_line(f1_rows, k, "netR")
        res[f"F1_{k}"] = rec
        print(line)
    print("  [F1 split] 雙幣")
    for k in ("pf_taker", "pf_volrate", "pf_hold"):
        for sym in A.SYMS:
            _, line = tercile_line(
                [r for r in f1_rows if r["sym"] == sym], k, "netR")
            print(f"  {sym:<4}" + line)
    rows_sorted = sorted(f1_rows, key=lambda r: r["ts"])
    half = len(rows_sorted) // 2
    print("  [F1 halves]")
    for k in ("pf_taker", "pf_volrate", "pf_hold"):
        for tag, seg in (("H1", rows_sorted[:half]), ("H2", rows_sorted[half:])):
            _, line = tercile_line(seg, k, "netR")
            print(f"  {tag} " + line)

    print(f"\n  [F2] 決議預測（sweep+2h 未回踩的倖存者）n={len(f2_rows)}, "
          f"基準突破率 {100*sum(1 for r in f2_rows if r['cls']=='BREAKOUT')/max(len(f2_rows),1):.0f}%")
    for k in ("pf_taker", "pf_volrate", "pf_hold"):
        rec, line = tercile_line(f2_rows, k, "cls")
        res[f"F2_{k}"] = rec
        print(line)
    print("  [F2 split] 雙幣 + 前後半（候選才看）")
    f2s = sorted(f2_rows, key=lambda r: r["ts"])
    h2 = len(f2s) // 2
    for k in ("pf_taker", "pf_volrate"):
        for sym in A.SYMS:
            _, line = tercile_line(
                [r for r in f2_rows if r["sym"] == sym], k, "cls")
            print(f"  {sym:<4}" + line)
        for tag, seg in (("H1", f2s[:h2]), ("H2", f2s[h2:])):
            _, line = tercile_line(seg, k, "cls")
            print(f"  {tag} " + line)

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  預先聲明: 反轉品質最好 = pf_taker 低(流翻轉) ∧ pf_volrate 低(衰竭)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
