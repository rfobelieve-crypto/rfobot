# -*- coding: utf-8 -*-
"""Raid microstructure, round 3 — the MINUTE-level dynamics the first two
rounds could not see, plus the causal context features from the user's list.

What is genuinely new here (everything else on the list is either already
tested-and-dead or impossible with retained data — see the mapping in the
conversation/commit):
  peak_delta    burstiness: max |1m delta| in the raid hour / hour volume
  absorption    aggression per unit of penetration: signed taker flow INTO
                the break divided by pierce depth — HIGH = heavy attack that
                barely moved price = absorbed (the user's 吸收 hypothesis)
  cvd_flip_30m  流 reversal speed: signed CVD drift in the FIRST 30 MINUTES
                after the raid hour. POST-RAID: characterisation only — the
                retest can begin during that window, so this is NOT strictly
                prior information for the entry. Tagged as such.
  touch_prior   pool quality: touches of the level (within 0.1 ATR) in the
                200 bars strictly BEFORE the raid — causal by construction,
                unlike the equal-levels count that died of lookahead
  age_h         pool age in hours (origin -> raid)
  session       Asia / London / NY / off-hours at the raid (categorical)
  htf_align     raid direction vs the 7-day price trend (with/against)

Universe: BTC+ETH raids with 1m flow coverage (~1.8k events, ~100 days).
Targets as before: resolution (BREAKOUT vs retest) and netR when retested.
Discipline: ~8 features x 2 targets ~ expect ~5 chance monotones; the bar
stays monotone + BOTH symbols + material magnitude. Descriptive only.

Run: python research/sweep_raid_micro.py
Out: research/results/sweep_raid_micro.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_micro.json"
SESS = {"asia": (0, 8), "london": (8, 13), "ny": (13, 21)}   # UTC, non-overlap


def raids_rich(sym: str) -> list[dict]:
    """A.raids() plus level/origin metadata needed for the context features."""
    bars = SC.load_csv(str(LT.CACHE / f"{sym}USDT_1h.csv"))
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)
    pools = []
    P = SC.PIVOT
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            pools.append((i + P + 1, h[i], 1, i))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            pools.append((i + P + 1, lo[i], -1, i))
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        pools += [(e, p, s, e) for (e, p, s) in lv.get(kind, [])]

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
        for lvl, s, origin in hit:
            kd, d = s, -s
            pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / a[j]
            fill = None
            for f in range(j + 1, j + 1 + SC.W):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    fill = f
                    break
            # context features, all strictly pre-raid
            tol = 0.1 * a[j]
            touch = sum(1 for k in range(max(0, j - 200), j)
                        if (abs(h[k] - lvl) <= tol) or (abs(lo[k] - lvl) <= tol))
            trend7 = cl[j] / cl[j - 168] - 1 if j >= 168 else None
            rec = {"sym": sym, "ts": bars[j][0], "pierce": pierce, "side": kd,
                   "open_j": bars[j][SC.O], "close_j": cl[j], "atr": a[j],
                   "touch_prior": touch, "age_h": j - origin,
                   "htf_align": (1 if trend7 is not None
                                 and (trend7 > 0) == (kd == 1) else 0)
                   if trend7 is not None else None}
            if fill is None:
                rec.update({"cls": "BREAKOUT", "netR": None})
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
                            "netR": net})
            out.append(rec)
    return out


def attach_micro(rows, flow):
    out = []
    for r in rows:
        m0 = r["ts"] // 60
        win = [(m, *flow[m]) for m in range(m0, m0 + 60) if m in flow]
        if len(win) < 40:
            continue
        base = [flow[m][0] for m in range(m0 - 1440, m0) if m in flow]
        if len(base) < 500:
            continue
        s = r["side"]
        vol = sum(w[1] for w in win)
        if vol <= 0:
            continue
        f = dict(r)
        f["peak_delta"] = max(abs(w[2]) for w in win) / vol
        into = s * sum(w[2] for w in win) / vol          # taker share into break
        f["absorption"] = into / max(r["pierce"], 0.10)   # attack per ATR gained
        post = [(m, *flow[m]) for m in range(m0 + 60, m0 + 90) if m in flow]
        if len(post) >= 20:
            pv = sum(w[1] for w in post)
            f["cvd_flip_30m"] = (s * sum(w[2] for w in post) / pv) if pv > 0 else None
        else:
            f["cvd_flip_30m"] = None
        hr = datetime.fromtimestamp(r["ts"], tz=timezone.utc).hour
        f["session"] = next((k for k, (a0, a1) in SESS.items() if a0 <= hr < a1),
                            "off")
        out.append(f)
    return out


def cat_profile(rows, key):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    parts = {}
    line = []
    for k in sorted(g):
        rs = g[k]
        n = len(rs)
        if n < 60:
            line.append(f"{k}: n={n} thin")
            continue
        br = 100 * sum(1 for x in rs if x["cls"] == "BREAKOUT") / n
        nets = [x["netR"] for x in rs if x["netR"] is not None]
        m = sum(nets) / len(nets) if nets else float("nan")
        parts[str(k)] = {"n": n, "breakout_pct": br, "netR": m}
        line.append(f"{k}: 突破{br:.0f}% netR{m:+.3f} (n={n})")
    return parts, f"  {key:<14}" + " | ".join(line)


def main() -> int:
    print("=" * 78)
    print("  RAID MICRO — 分鐘級動態 + 因果 context（新可測子集）")
    print("  (~8 特徵 x 2 目標, 純機率 ~5 個假單調; 判準 = 單調+雙幣一致+幅度)")
    print("=" * 78)
    allr = []
    for sym, canon in A.SYMS.items():
        flow, _imb, _c = A.load_flow(canon)
        rr = attach_micro(raids_rich(sym), flow)
        print(f"  {sym}: {len(rr)} raids")
        allr += rr
    res = {}
    print("\n  [terciles] 突破率 / 反轉率 / netR|回踩")
    for k in ("peak_delta", "absorption", "cvd_flip_30m",
              "touch_prior", "age_h"):
        rec, line = A.profile(allr, k)
        res[k] = rec
        print(line + ("   [POST-RAID 僅特徵化]" if k == "cvd_flip_30m" else ""))
    print("\n  [categorical]")
    for k in ("session", "htf_align"):
        rec, line = cat_profile(allr, k)
        res[k] = rec
        print(line)
    print("\n  [split] BTC vs ETH — 有型的才列")
    for k in ("peak_delta", "absorption", "cvd_flip_30m", "touch_prior"):
        for sym in A.SYMS:
            _, line = A.profile([r for r in allr if r["sym"] == sym], k)
            print(f"  {sym:<4}" + line)
    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
