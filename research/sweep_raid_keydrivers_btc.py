# -*- coding: utf-8 -*-
"""Key drivers, BTC supplement — the REAL derivatives flow (OI, CVD,
liquidations, funding) ranked head-to-head with the price-behaviour
features under the same methodology.

User question (2026-08-01): 都沒有用到真實訂單流 oi cvd liquidation L2
掛單之類的嗎? Answer: they were tested in their own rounds (Q quadrant and
liq_burst survived; most died), but the unified key-driver ranking pooled
BTC+ETH where Coinglass series don't exist. This closes that gap on the
BTC-only universe: same univariate + time-split multivariate permutation
methodology, price features and real flow in ONE table.

L2 order-book features remain untestable historically — depth_deltas
accumulates since 2026-07; that is the October checkpoint, not this file.

Run: python research/sweep_raid_keydrivers_btc.py
Out: research/results/sweep_raid_keydrivers_btc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import sweep_raid_menu2 as M  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402
from sweep_raid_keydrivers import auc  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_keydrivers_btc.json"
FEATS = ["pierce", "att_min", "reject_in_hour", "att_vshock",
         "oi_chg_raid", "fut_taker_signed", "liq_burst", "stop_fuel",
         "funding_signed"]
ZH = {"pierce": "穿越深度", "att_min": "攻擊分鐘", "reject_in_hour": "收回內側",
      "att_vshock": "量能倍數", "oi_chg_raid": "OI變化(獵取時)",
      "fut_taker_signed": "期貨CVD(順破向)", "liq_burst": "清算爆量",
      "stop_fuel": "被獵側清算佔比", "funding_signed": "資金費率(順破向)"}


def main() -> int:
    print("=" * 78)
    print("  KEY DRIVERS·BTC — 真實衍生品流 vs 價格行為，同場排名")
    print("=" * 78)
    rows = [r for r in M.build() if r["sym"] == "BTC"]
    S = D.load_state()
    dmap = {r["ts"]: r for r in D.attach(A.raids("BTC"), S)}
    merged = []
    for r in rows:
        d = dmap.get(r["ts"])
        if not d:
            continue
        rr = dict(r)
        for f in ("oi_chg_raid", "fut_taker_signed", "liq_burst",
                  "stop_fuel", "funding_signed"):
            rr[f] = d.get(f)
        if all(rr.get(f) is not None for f in FEATS):
            merged.append(rr)
    merged.sort(key=lambda r: r["ts"])
    n = len(merged)
    half = n // 2
    res = {}

    yA = np.array([1 if r["cls"] == "BREAKOUT" else 0 for r in merged])
    fills = [r for r in merged if r["netR"] is not None]
    yB = np.array([r["netR"] for r in fills])
    print(f"\n  [單變量·BTC] n={n}（fills={len(fills)}, 突破基準 {100*yA.mean():.0f}%）")
    print(f"  {'特徵':<12}{'解析AUC':>8}{'半1':>7}{'半2':>7} | {'netR-IC':>8}{'半1':>8}{'半2':>8}  判定")
    for f in FEATS:
        xA = np.array([r[f] for r in merged], dtype=float)
        a_full, a1, a2 = auc(yA, xA), auc(yA[:half], xA[:half]), auc(yA[half:], xA[half:])
        xB = np.array([r[f] for r in fills], dtype=float)
        hb = len(fills) // 2
        ic, _ = spearmanr(xB, yB)
        ic1, _ = spearmanr(xB[:hb], yB[:hb])
        ic2, _ = spearmanr(xB[hb:], yB[hb:])
        sa = (a1 - 0.5) * (a2 - 0.5) > 0 and abs(a_full - 0.5) > 0.03
        sb = ic1 * ic2 > 0 and abs(ic) > 0.05
        v = ("解析+品質" if sa and sb else "解析" if sa else "品質" if sb else "—")
        res[f] = {"auc": round(float(a_full), 3), "ic": round(float(ic), 3),
                  "verdict": v}
        print(f"  {ZH[f]:<12}{a_full:>8.3f}{a1:>7.3f}{a2:>7.3f} | "
              f"{ic:>+8.3f}{ic1:>+8.3f}{ic2:>+8.3f}  {v}")

    from xgboost import XGBClassifier, XGBRegressor
    rng = np.random.default_rng(7)
    cut = int(n * 0.7)
    Xa = np.array([[r[f] for f in FEATS] for r in merged], dtype=float)
    clf = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                        subsample=0.9, random_state=7, eval_metric="logloss")
    clf.fit(Xa[:cut], yA[:cut])
    base = auc(yA[cut:], clf.predict_proba(Xa[cut:])[:, 1])
    print(f"\n  [多變量A·解析] OOS AUC={base:.3f} · permutation:")
    imp = {}
    for i, f in enumerate(FEATS):
        drops = []
        for _ in range(20):
            Xp = Xa[cut:].copy()
            rng.shuffle(Xp[:, i])
            drops.append(base - auc(yA[cut:], clf.predict_proba(Xp)[:, 1]))
        imp[f] = float(np.mean(drops))
    for f, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {ZH[f]:<12} {v:+.4f}")
    res["mv_resolution"] = {"oos_auc": round(float(base), 3),
                            "perm": {k: round(v, 4) for k, v in imp.items()}}

    Xb = np.array([[r[f] for f in FEATS] for r in fills], dtype=float)
    cutb = int(len(fills) * 0.7)
    reg = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05,
                       subsample=0.9, random_state=7)
    reg.fit(Xb[:cutb], yB[:cutb])
    bic, _ = spearmanr(reg.predict(Xb[cutb:]), yB[cutb:])
    print(f"\n  [多變量B·品質] OOS IC={bic:+.3f} · permutation:")
    impb = {}
    for i, f in enumerate(FEATS):
        drops = []
        for _ in range(20):
            Xp = Xb[cutb:].copy()
            rng.shuffle(Xp[:, i])
            icp, _ = spearmanr(reg.predict(Xp), yB[cutb:])
            drops.append(bic - icp)
        impb[f] = float(np.mean(drops))
    for f, v in sorted(impb.items(), key=lambda x: -x[1]):
        print(f"    {ZH[f]:<12} {v:+.4f}")
    res["mv_quality"] = {"oos_ic": round(float(bic), 3),
                         "perm": {k: round(v, 4) for k, v in impb.items()}}

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
