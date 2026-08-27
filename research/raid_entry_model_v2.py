# -*- coding: utf-8 -*-
"""Raid entry meta-model, rebuilt on the AUTHORITATIVE population — §0.70b.

v1 (`raid_entry_model.py`) was built on `SC.backtest_symbol`. The operator,
after comparing the site's own liquidity chart against the verification
charts made for this study, said the model should be built on the site's
data instead. That is right, and v1 had four separate scope errors because
it was not:

  1. SWING ONLY. backtest_symbol walks detect_sweeps, so v1 saw 20% of what
     actually trades. The site chart labels every event with its family
     (波段/時段/昨日/上週) — the answer was on screen the whole time.
  2. WRONG COST MODEL. v1's target came from backtest_symbol's SLIP-based R.
     Gate F scores `SE.net_r(...)` — scenario-A per-symbol bps. The model was
     trained on a different quantity from the one that decides.
  3. LEVEL AGE MEASURED FROM THE WRONG BAR. v1 used the confirmation bar
     because `swing_levels()` discards the extreme. `rederive` carries BOTH
     `origin_ts` (the extreme) and the sweep, so age is now real age.
  4. SWEEP BAR RECOVERED BY PRICE MATCHING. v1 reverse-matched the sweep bar
     by level price inside a window — fragile, and it silently dropped
     events whose price appeared more than once. `rederive` returns it.

`rederive` is the function the published chart and the shadow ledger both
use: "Trade logic is byte-identical to the scorer; only bookkeeping is
added." Building on it means the model, the chart and the gate all describe
the same events.

WHAT DOES NOT CHANGE: the pre-registration. Same three baselines (with a
fourth, §0.71b's confluence rule), same four WQ101 gates, same forward
holdout, same fixed P>0.5. And the same standing limitation — the forward
window is still 96% data §0.59 declared spent, so a pass here designs a
pre-registration, it does not validate one.
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from shadow_review import rederive                          # noqa: E402
from research.crowd_battery2 import adx_state               # noqa: E402
from research.confluence_all_families import first_hits_batch  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "raid_entry_model_v2.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
HOME = {"RANGING", "TREND_DOWN"}
FREEZE = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
KIND_CODE = {"swing": 0, "session": 1, "pdh_pdl": 2, "pwh_pwl": 3}
REGIME_CODE = {"RANGING": 0, "TREND_UP": 1, "TREND_DOWN": 2, "NEUTRAL": 3}
TOL = 0.10
LB = 24
random.seed(211)

FEATURES = [
    "pierce",             # given by rederive, not recomputed
    "kind_code",          # NOW A REAL FEATURE — v1 had one family only
    "level_age_bars",     # from the EXTREME bar, not the confirmation bar
    "sweep_range_atr",
    "sweep_close_pos",
    "sweep_vol_z",
    "atr_pct",
    "approach_atr",
    "hour_utc",
    "dow",
    "regime_code",
    "ret24_atr",
    "side",
    "confluence_kinds",   # §0.71b
    "pools_ahead_3atr",   # D5
    "dist_next_pool_atr",  # D2
]


def clustered_ci(pairs, n_boot=2500):
    by = defaultdict(list)
    for d, v in pairs:
        by[d].append(v)
    days = list(by)
    if len(days) < 4:
        return None
    m = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        vals = [x for d in pick for x in by[d]]
        if vals:
            m.append(st.mean(vals))
    m.sort()
    return m[int(.025 * len(m))], m[int(.975 * len(m))]


def extract(sym: str) -> list[dict]:
    try:
        bars, trades, pool_rows = rederive(sym)
    except Exception:
        return []
    n = len(bars)
    if n < 400:
        return []
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    vol = [b[SC.V] for b in bars]
    a = SC.atr14(bars)
    idx = {b[0]: i for i, b in enumerate(bars)}
    adx = adx_state(bars)

    # full pool inventory, price-sorted per family, for the terrain and
    # confluence features. pool_rows already carries every pool's state.
    fam = defaultdict(list)
    for p in pool_rows:
        fam[p["kind"]].append((p["origin"], p["lvl"], p["side"]))
    live = {}
    for k, items in fam.items():
        hits = first_hits_batch(bars, items)
        arr = sorted((pr, est, sd, hh)
                     for (est, pr, sd), hh in zip(items, hits))
        live[k] = arr

    out = []
    for t in trades:
        if t["net"] is None or not t["b"]:
            continue                       # variant-B, settled only
        j = idx.get(t["sweep_ts"])
        oi = idx.get(t["origin_ts"])
        if j is None or oi is None or j < LB + 6:
            continue
        A = t["atr"]
        if not A or A <= 0 or a[j] is None:
            continue
        d = 1 if t["side"] == "LONG" else -1
        lvl = t["lvl"]
        rng = h[j] - lo[j]
        cp = ((c[j] - lo[j]) / rng) if rng > 0 else 0.5
        base = [vol[k] for k in range(max(0, j - LB), j) if vol[k] > 0]
        vz = (vol[j] / (sum(base) / len(base))) if base else 1.0
        ret24 = c[j] / c[j - LB] - 1
        lab = adx.get(bars[j][0] // 3600 * 3600)
        cell = ("RANGING" if lab == "RANGING" else
                "NEUTRAL" if lab != "TRENDING" else
                ("TREND_UP" if ret24 > 0 else "TREND_DOWN"))
        want = 1 if d == -1 else -1
        tol = TOL * A
        conf, ahead = 0, []
        for k2, arr in live.items():
            found = False
            for pr, est2, sd2, hh in arr:
                if sd2 != want or est2 > j or (hh is not None and hh < j):
                    continue
                if abs(pr - lvl) <= tol and k2 != t["kind"]:
                    found = True
                if (pr - lvl) * d > 0:
                    ahead.append(abs(pr - lvl) / A)
            conf += found
        dists = sorted(ahead)
        out.append({
            "sym": sym, "ts": int(t["fill_ts"]), "exit_ts": int(t["exit_ts"]),
            "R": float(t["net"]), "y": int(t["net"] > 0), "cell": cell,
            "kind": t["kind"],
            "pierce": float(t["pierce"]),
            "kind_code": KIND_CODE[t["kind"]],
            "level_age_bars": j - oi,
            "sweep_range_atr": rng / A,
            "sweep_close_pos": cp if d == 1 else 1.0 - cp,
            "sweep_vol_z": float(vz),
            "atr_pct": A / c[j],
            "approach_atr": (c[j] - c[j - 6]) * d / A,
            "hour_utc": datetime.fromtimestamp(bars[j][0], timezone.utc).hour,
            "dow": datetime.fromtimestamp(bars[j][0], timezone.utc).weekday(),
            "regime_code": REGIME_CODE[cell],
            "ret24_atr": (c[j] - c[j - LB]) / A,
            "side": d,
            "confluence_kinds": conf,
            "pools_ahead_3atr": sum(1 for x in dists if x <= 3.0),
            "dist_next_pool_atr": dists[0] if dists else 99.0,
        })
    return out


def arm(ev, label):
    if not ev:
        return {"label": label, "n": 0, "established": False}
    m = st.mean(x["R"] for x in ev)
    ci = clustered_ci([(x["ts"] // 86400, x["R"]) for x in ev])
    per = defaultdict(list)
    for x in ev:
        if x["sym"] in CORE9:
            per[x["sym"]].append(x["R"])
    br = sum(1 for s in per if st.mean(per[s]) > 0)
    return {"label": label, "n": len(ev), "meanR": round(m, 4),
            "wr": round(100 * sum(x["y"] for x in ev) / len(ev), 1),
            "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "breadth": f"{br}/{len(per)}",
            "established": bool(len(ev) >= 200 and ci and ci[0] > 0 and br >= 6)}


def show(a):
    if not a["n"]:
        print(f"  {a['label']:<26} 無樣本")
        return
    ci = f"[{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}]" if a["ci"] else "—"
    print(f"  {a['label']:<26} n={a['n']:<6} meanR {a['meanR']:+.4f}  "
          f"WR {a['wr']:5.1f}%  CI {ci:<20} 廣度 {a['breadth']:<6}"
          f"{'  ✓成立' if a['established'] else '  ·未成立'}")


def main() -> int:
    syms = sorted(p.name.replace("USDT_1h.csv", "")
                  for p in CACHE.glob("*USDT_1h.csv"))
    ev = []
    for s in syms:
        ev += extract(s)
    ev.sort(key=lambda x: x["ts"])
    if len(ev) < 1000:
        raise SystemExit(f"only {len(ev)} events")

    print("§0.70b 進場 meta-model —— 建在權威母體上（rederive，四家族、"
          "情境 A 成本）\n")
    kc = defaultdict(int)
    for x in ev:
        kc[x["kind"]] += 1
    print(f"  母體 {len(ev)} 筆（變體 B、已結算）、{len(syms)} 幣")
    for k in ("swing", "session", "pdh_pdl", "pwh_pwl"):
        print(f"    {k:<10} {kc[k]:6d}  ({100*kc[k]/len(ev):.0f}%)")
    print(f"  對照 v1 的 8,272 筆（僅 swing、SLIP 成本）\n")

    print("── 四條基準線 ──")
    b1 = arm(ev, "B1 全進")
    b3 = arm([x for x in ev if x["cell"] in HOME], "B3 §0.59 regime 濾網")
    b4 = arm([x for x in ev if x["confluence_kinds"] <= 1],
             "B4 §0.71b 堆疊≤1")
    for a in (b1, b3, b4):
        show(a)
    best = max((b3, b4), key=lambda z: z.get("meanR", -9))
    print(f"\n  最強的簡單規則：{best['label']}（{best['meanR']:+.4f}）"
          f"  ← **模型要贏的是這個**\n")

    X = np.array([[float(x[f]) for f in FEATURES] for x in ev])
    y = np.array([x["y"] for x in ev])
    ts = np.array([x["ts"] for x in ev])
    ex = np.array([x["exit_ts"] for x in ev])

    from xgboost import XGBClassifier
    params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                  min_child_weight=20, eval_metric="logloss",
                  n_jobs=4, verbosity=0)

    print("── (a) 時間 walk-forward（purge + embargo）──")
    EMB = 7 * 86400
    edges = np.linspace(ts.min() + (ts.max() - ts.min()) * .4, ts.max(), 9)
    lifts = []
    for k in range(8):
        a0, a1 = edges[k], edges[k + 1]
        te = np.where((ts >= a0) & (ts < a1))[0]
        tr = np.where(ex < a0 - EMB)[0]
        if len(te) < 100 or len(tr) < 800:
            continue
        m = XGBClassifier(**params).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        mdl = [ev[i] for i, pv in zip(te, p) if pv > 0.5]
        bl = [ev[i] for i in te
              if (ev[i]["cell"] in HOME if best is b3
                  else ev[i]["confluence_kinds"] <= 1)]
        if not mdl or not bl:
            continue
        lm, lb = st.mean(x["R"] for x in mdl), st.mean(x["R"] for x in bl)
        lifts.append(lm - lb)
        print(f"   fold {k+1}  測試 {len(te):5d}  模型取 {len(mdl):5d}  "
              f"模型 {lm:+.4f}  基準 {lb:+.4f}  差 {lm-lb:+.4f}")
    if not lifts:
        raise SystemExit("no usable folds")
    ci = clustered_ci([(i, v) for i, v in enumerate(lifts)], 3000)
    print(f"\n   per-fold 平均 {st.mean(lifts):+.4f}  "
          f"正折 {sum(1 for x in lifts if x > 0)}/{len(lifts)}  "
          f"CI {ci}")

    print("\n── (c) 前瞻留出（訓練 <FREEZE、評分 ≥FREEZE）──")
    tr = np.where(ex < FREEZE)[0]
    te = np.where(ts >= FREEZE)[0]
    fwd = None
    if len(te) >= 100 and len(tr) >= 1000:
        m = XGBClassifier(**params).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        allf = [ev[i] for i in te]
        mdl = [ev[i] for i, pv in zip(te, p) if pv > 0.5]
        bl = [x for x in allf
              if (x["cell"] in HOME if best is b3
                  else x["confluence_kinds"] <= 1)]
        a_all, a_mdl, a_bl = (arm(allf, "  前瞻 · 全進"),
                              arm(mdl, "  前瞻 · 模型取"),
                              arm(bl, f"  前瞻 · {best['label'][:12]}"))
        for a in (a_all, a_bl, a_mdl):
            show(a)
        fwd = {"all": a_all, "model": a_mdl, "baseline": a_bl,
               "lift": round(a_mdl.get("meanR", 0) - a_bl.get("meanR", 0), 4)}
        print(f"\n   模型 − 最強簡單規則 = {fwd['lift']:+.4f}R")
        if not a_mdl["established"] or not a_bl["established"]:
            print("   ⚠ 有臂未成立 —— 方向可看，不可當證據")
    else:
        print(f"   樣本不足（訓練 {len(tr)}／前瞻 {len(te)}）")

    g1 = st.mean(lifts) > 0
    g2 = sum(1 for x in lifts if x > 0) / len(lifts) > 0.55
    g3 = ci is not None and ci[0] > 0
    g4 = bool(fwd and fwd["lift"] > 0)
    print("\n── 四關 ──")
    for lab, ok in (("per-fold 平均 > 0", g1), ("正折 > 55%", g2),
                    ("lift CI 不含零", g3), ("前瞻贏過最強簡單規則", g4)):
        print(f"   {lab:<24} {'✓' if ok else '✗'}")
    v = ("**全過** —— 成為預註冊候選。但前瞻窗仍有 96% 是 §0.59 宣告作廢的"
         "樣本，所以這是「設計預註冊」不是「驗證」"
         if (g1 and g2 and g3 and g4) else
         "**未過** —— 記為負結果。母體已是權威口徑，v1 的四個範圍錯誤都已排除")
    print(f"\n判讀：{v}")

    m = XGBClassifier(**params).fit(X, y)
    imp = dict(sorted(zip(FEATURES, (float(z) for z in m.feature_importances_)),
                      key=lambda z: -z[1])[:8])
    print("\n  （全樣本重擬合的重要度，僅供理解）")
    for f, g in imp.items():
        print(f"    {f:<22} {g:.4f}")

    OUT.write_text(json.dumps({
        "n": len(ev), "by_kind": dict(kc), "features": FEATURES,
        "baselines": {"all": b1, "regime": b3, "confluence": b4},
        "lifts": [round(x, 4) for x in lifts],
        "lift_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "forward": fwd, "gates": {"per_fold": g1, "frac_pos": g2,
                                  "ci": g3, "forward": g4},
        "verdict": v, "importance": {k: round(v_, 4) for k, v_ in imp.items()},
    }, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
