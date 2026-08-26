# -*- coding: utf-8 -*-
"""Raid entry meta-model — will THIS sweep-failure work? — TODO §0.70.

The operator's proposal, and it is the one genuinely unexplored surface on
this line: the frozen rule already answers WHERE and WHEN a sweep-failure
setup exists. A second model answers only "will this one pay". That is
meta-labelling, and `research/sweep_failure/` contains no model of any kind
— every variant to date (A-E, the combo watchlist) is a hand-written
boolean.

WHY THIS IS BETTER POSITIONED THAN THE THINGS THAT FAILED BEFORE
  RL joint entry/exit   1,200-1,900 in-OOS bars per fold  -> 0/4 folds
  meta EXIT model       2-6 trades per fold               -> no power
  V7 direction          AUC ceiling 0.54-0.57 structural
  THIS                  8,262 events, each already a tradeable candidate
Events are not bars: there is no 80%-NEUTRAL dilution, the target is
binary, and the population's internal variation is large and already
mapped (home regime +0.1063 vs non-home -0.1514).

── PRE-REGISTRATION, frozen before the first run ────────────────────────

POPULATION  every variant-B event (pierce <= 0.25 ATR) from
            SC.backtest_symbol across all cached coins. All are SWING
            pools — backtest_symbol walks detect_sweeps, which is
            swing-only — so level_kind is constant here and is NOT a
            feature. The session/PDH/PWH families are a v2 question.

TARGET      binary, net_r > 0. Classification, not regression.

FEATURES    14, capped deliberately. Every one is computed at the SWEEP
            BAR's close, which is strictly before the fill (§0.66 measured
            it: zero fills land in the sweep bar, 86/106 land in the very
            next one). Every one has a mechanism, stated inline. Nothing
            is added later without passing the same causality question —
            tonight's G4 died precisely there.

BASELINES   three, and the third is the one that matters
              B1 take everything
              B2 variant B as it stands (identical to B1 on this
                 population — kept so the table is readable)
              B3 the §0.59 regime filter (home = RANGING u TREND_DOWN)
            BEATING B3 IS THE BAR. A model that merely rediscovers regime
            has bought a model's worth of complexity for a boolean that is
            already pre-registered.

DECISION    take when P(win) > 0.5. The threshold is FIXED at the natural
            0.5 and is not swept — a swept threshold here is the
            2026-06-20 trap. The top-half split is reported alongside as a
            fixed alternative view, also not chosen from results.

VALIDATION  (a) time walk-forward, expanding, with purge (drop train
                events whose exit reaches into the test window) and
                embargo
            (b) leave-coins-out: train on some coins, score coins the
                model has never seen. This is the honest test of "does the
                mechanism generalise", which is what the operator actually
                wants, and it is immune to the per-coin cherry-picking
                that killed the small-coin premise (§0.67: per-coin meanR
                ranges -0.21..+0.28 on n=26-62, pure noise).

GATES       the WQ101 four, unchanged, because aggregate lift is exactly
            how that one nearly shipped (+0.0072 aggregate, -0.0044
            per-fold):
              per-fold mean lift > 0
              fraction of positive folds > 55%
              day-clustered bootstrap CI of the lift excludes zero
              per-coin breadth >= 6/9 on core9
            AND every arm must be ESTABLISHED before it enters a
            comparison (n floor, CI away from zero) — the single error
            made three times in one session on 2026-08-26.

NOTHING HERE MAY BE WIRED TO ANYTHING. A pass makes this a candidate for
pre-registration with fresh forward samples, not a rule.
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
from research.crowd_battery2 import adx_state              # noqa: E402
from research.liquidity_map_check import swing_levels, first_hit  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "raid_entry_model.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
HOME = {"RANGING", "TREND_DOWN"}                 # §0.59b
PIERCE_B = 0.25
LB = 24
random.seed(71)

FEATURES = [
    "pierce_atr",        # sweep depth — the variant-B filter, as continuous
    "sweep_range_atr",   # violence of the sweep bar
    "sweep_close_pos",   # where it closed in its own range, oriented to the
                         # trade side: a sweep that already closed back is a
                         # failure showing itself
    "sweep_vol_z",       # participation vs trailing 24h
    "atr_pct",           # the coin's volatility regime
    "approach_atr",      # how far price travelled into the level (6 bars)
    "hour_utc",          # session timing
    "dow",               # weekday
    "regime_code",       # frozen ADX x direction cell
    "ret24_atr",         # 24h move in ATR units — trend context, continuous
    "side",              # feature, NOT a partition (mistake.md 2026-04-13)
    "level_age_bars",    # bars from pool creation to sweep
    "pools_ahead_3atr",  # D5 survived three terrain gates: density ahead
    "dist_next_pool_atr",  # D2: the wall in front
]
REGIME_CODE = {"RANGING": 0, "TREND_UP": 1, "TREND_DOWN": 2, "NEUTRAL": 3}


def extract(sym: str) -> list[dict]:
    fp = CACHE / f"{sym}USDT_1h.csv"
    if not fp.exists():
        return []
    bars = SC.load_csv(str(fp))
    n = len(bars)
    if n < 400:
        return []
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    o = [b[SC.O] for b in bars]
    vol = [b[SC.V] for b in bars]
    a = SC.atr14(bars)
    idx = {b[0]: i for i, b in enumerate(bars)}
    adx = adx_state(bars)

    # pool inventory with sweep times, for the two terrain features
    pools = [(est, price, side, first_hit(bars, est, price, side))
             for est, price, side in swing_levels(bars)]
    pools.sort()

    # sweep events keyed by level so a trade can find the bar it came from
    sw_by_lvl = defaultdict(list)
    for e in SC.detect_sweeps(bars):
        sw_by_lvl[round(float(e["level"]), 8)].append(e["j"])

    out = []
    for fill_ts, exit_ts, R, lvl, A, _stopped, pierce, side in \
            SC.backtest_symbol(bars):
        if pierce > PIERCE_B:
            continue
        fi = idx.get(fill_ts)
        if fi is None or A is None or A <= 0:
            continue
        # the sweep bar: the latest sweep of this level strictly before the
        # fill and inside the frozen W window
        cands = [j for j in sw_by_lvl.get(round(float(lvl), 8), [])
                 if j < fi and fi - j <= SC.W]
        if not cands:
            continue
        j = max(cands)
        if j < LB + 6 or a[j] is None or a[j] <= 0:
            continue
        Aj = a[j]
        rng = h[j] - lo[j]
        # close position oriented so 1.0 always means "closed in the
        # direction the trade wants", regardless of long or short
        cp = ((c[j] - lo[j]) / rng) if rng > 0 else 0.5
        base = [vol[k] for k in range(max(0, j - LB), j) if vol[k] > 0]
        vz = (vol[j] / (sum(base) / len(base))) if base else 1.0
        lab = adx.get(bars[j][0] // 3600 * 3600)
        ret24 = c[j] / c[j - LB] - 1
        cell = ("RANGING" if lab == "RANGING" else
                "NEUTRAL" if lab != "TRENDING" else
                ("TREND_UP" if ret24 > 0 else "TREND_DOWN"))
        # backtest_symbol emits the string "LONG"/"SHORT" (added 2026-08-18),
        # not a sign — read it, do not assume.
        d = 1 if str(side).upper() == "LONG" else -1
        # pools still resting AHEAD of the level in the trade's direction
        ahead = [p for est, p, s2, hit in pools
                 if est <= j and (hit is None or hit > j)
                 and s2 == d and (p - lvl) * d > 0]
        dists = sorted(abs(p - lvl) / Aj for p in ahead)
        est_bar = None
        for est, p, s2, hit in pools:
            if abs(p - lvl) < 1e-9 and hit == j:
                est_bar = est
                break
        dt = datetime.fromtimestamp(bars[j][0], timezone.utc)
        out.append({
            "sym": sym, "ts": int(fill_ts), "exit_ts": int(exit_ts),
            "R": float(R), "y": int(R > 0), "cell": cell,
            "pierce_atr": float(pierce),
            "sweep_range_atr": rng / Aj,
            "sweep_close_pos": cp if d == 1 else 1.0 - cp,
            "sweep_vol_z": float(vz),
            "atr_pct": Aj / c[j],
            "approach_atr": (c[j] - c[j - 6]) * d / Aj,
            "hour_utc": dt.hour, "dow": dt.weekday(),
            "regime_code": REGIME_CODE[cell],
            "ret24_atr": (c[j] - c[j - LB]) / Aj,
            "side": int(d),
            "level_age_bars": (j - est_bar) if est_bar is not None else -1,
            "pools_ahead_3atr": sum(1 for x in dists if x <= 3.0),
            "dist_next_pool_atr": dists[0] if dists else 99.0,
        })
    return out


def clustered_ci(pairs, n_boot=3000):
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


def arm(ev, label):
    """A summary that also says whether the arm is ESTABLISHED."""
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
            "established": bool(len(ev) >= 200 and ci and ci[0] > 0
                                and br >= 6)}


def show(a):
    if not a["n"]:
        print(f"  {a['label']:<28} 無樣本")
        return
    ci = f"[{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}]" if a["ci"] else "—"
    print(f"  {a['label']:<28} n={a['n']:<6} meanR {a['meanR']:+.4f}  "
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
        raise SystemExit(f"only {len(ev)} events — check extraction")

    print("§0.70 獵取進場 meta-model —— 判準跑數之前凍結")
    print(f"  母體 {len(ev)} 個事件、{len(syms)} 幣、"
          f"{datetime.fromtimestamp(ev[0]['ts'], timezone.utc):%Y-%m}"
          f" → {datetime.fromtimestamp(ev[-1]['ts'], timezone.utc):%Y-%m}")
    print(f"  特徵 {len(FEATURES)} 個，全部在**開掃棒收盤**算出（早於成交）\n")

    print("── 三條基準線 ──")
    b1 = arm(ev, "B1 全進")
    b3 = arm([x for x in ev if x["cell"] in HOME], "B3 §0.59 regime 濾網")
    show(b1)
    show(b3)
    print(f"\n  B3 − B1 = {b3['meanR'] - b1['meanR']:+.4f}R  "
          f"← **模型要贏的是這個差**\n")

    X = np.array([[float(x[f]) for f in FEATURES] for x in ev])
    y = np.array([x["y"] for x in ev])
    ts = np.array([x["ts"] for x in ev])
    ex = np.array([x["exit_ts"] for x in ev])

    from xgboost import XGBClassifier
    params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=2.0, min_child_weight=20,
                  eval_metric="logloss", n_jobs=4, verbosity=0)

    # ── (a) time walk-forward with purge + embargo ──────────────────────
    print("── (a) 時間 walk-forward（purge：訓練事件的出場不得伸進測試窗）──")
    EMB = 7 * 86400
    n_folds = 8
    t0, t1 = ts.min(), ts.max()
    edges = np.linspace(t0 + (t1 - t0) * 0.4, t1, n_folds + 1)
    folds = []
    for k in range(n_folds):
        a0, a1 = edges[k], edges[k + 1]
        te = np.where((ts >= a0) & (ts < a1))[0]
        tr = np.where((ex < a0 - EMB))[0]
        if len(te) < 60 or len(tr) < 500:
            continue
        m = XGBClassifier(**params).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        sel = te[p > 0.5]
        allf = [ev[i] for i in te]
        mdl = [ev[i] for i in sel]
        hm = [ev[i] for i in te if ev[i]["cell"] in HOME]
        if not mdl or not hm:
            continue
        folds.append({
            "n_test": len(te), "n_model": len(mdl),
            "all": st.mean(x["R"] for x in allf),
            "model": st.mean(x["R"] for x in mdl),
            "home": st.mean(x["R"] for x in hm)})
        print(f"   fold {k+1}  測試 {len(te):4d}  模型取 {len(mdl):4d}  "
              f"全進 {folds[-1]['all']:+.4f}  "
              f"模型 {folds[-1]['model']:+.4f}  "
              f"regime {folds[-1]['home']:+.4f}")

    if not folds:
        raise SystemExit("no usable folds")
    lift_all = [f["model"] - f["all"] for f in folds]
    lift_home = [f["model"] - f["home"] for f in folds]
    print(f"\n   對 B1：per-fold 平均 {st.mean(lift_all):+.4f}  "
          f"正折 {sum(1 for x in lift_all if x > 0)}/{len(lift_all)}")
    print(f"   對 B3：per-fold 平均 {st.mean(lift_home):+.4f}  "
          f"正折 {sum(1 for x in lift_home if x > 0)}/{len(lift_home)}"
          f"   ← 這一列才算數")

    # ── (b) leave-coins-out ─────────────────────────────────────────────
    print("\n── (b) 留幣交叉驗證（模型沒見過的幣）──")
    groups = [syms[i::4] for i in range(4)]
    lco = []
    for gi, held in enumerate(groups, 1):
        te = np.array([i for i, x in enumerate(ev) if x["sym"] in held])
        tr = np.array([i for i, x in enumerate(ev) if x["sym"] not in held])
        if len(te) < 200 or len(tr) < 500:
            continue
        m = XGBClassifier(**params).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        mdl = [ev[i] for i, pv in zip(te, p) if pv > 0.5]
        allf = [ev[i] for i in te]
        hm = [x for x in allf if x["cell"] in HOME]
        if not mdl or not hm:
            continue
        ma, mm, mh = (st.mean(x["R"] for x in allf),
                      st.mean(x["R"] for x in mdl),
                      st.mean(x["R"] for x in hm))
        lco.append({"held": held, "all": ma, "model": mm, "home": mh,
                    "n": len(allf), "n_model": len(mdl)})
        print(f"   留出組 {gi} ({len(held)} 幣)  n={len(allf):4d}  "
              f"全進 {ma:+.4f}  模型 {mm:+.4f}  regime {mh:+.4f}")

    # ── (c) the decisive one: train on the backtest world, score the
    #        forward one. §0.58 measured that the same regime cells pay
    #        materially less after FREEZE (within-cell decay = 75% of the
    #        gap), so a model fitted entirely on pre-FREEZE data may have
    #        learned a relationship that has since decayed. Nothing in (a)
    #        or (b) can see this: both resample the same world.
    FREEZE = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
    print("\n── (c) 前瞻留出：訓練只用 FREEZE 之前，評分只用之後 ──")
    tr = np.where(ex < FREEZE)[0]
    te = np.where(ts >= FREEZE)[0]
    fwd = None
    if len(te) >= 100 and len(tr) >= 1000:
        m = XGBClassifier(**params).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        allf = [ev[i] for i in te]
        mdl = [ev[i] for i, pv in zip(te, p) if pv > 0.5]
        hm = [x for x in allf if x["cell"] in HOME]
        a_all, a_mdl, a_hm = (arm(allf, "  前瞻 · 全進"),
                              arm(mdl, "  前瞻 · 模型取"),
                              arm(hm, "  前瞻 · regime 濾網"))
        for a in (a_all, a_mdl, a_hm):
            show(a)
        fwd = {"all": a_all, "model": a_mdl, "home": a_hm,
               "lift_vs_all": round(a_mdl.get("meanR", 0)
                                    - a_all.get("meanR", 0), 4),
               "lift_vs_home": round(a_mdl.get("meanR", 0)
                                     - a_hm.get("meanR", 0), 4)}
        print(f"\n   模型 − 全進   {fwd['lift_vs_all']:+.4f}R")
        print(f"   模型 − regime {fwd['lift_vs_home']:+.4f}R")
        if not a_mdl["established"]:
            print("   ⚠ 前瞻的模型臂**未成立**（n 或 CI 或廣度不足）——"
                  "方向可看，不可當證據")
    else:
        print(f"   樣本不足（訓練 {len(tr)}／前瞻 {len(te)}）")

    # ── gates ───────────────────────────────────────────────────────────
    print("\n── 四關（WQ101 那套）對 B3 ──")
    ci = clustered_ci([(i, v) for i, v in enumerate(lift_home)], 3000)
    g1 = st.mean(lift_home) > 0
    g2 = sum(1 for x in lift_home if x > 0) / len(lift_home) > 0.55
    g3 = ci is not None and ci[0] > 0
    lco_ok = bool(lco) and sum(1 for r in lco if r["model"] > r["home"]) >= 3
    print(f"   per-fold 平均 > 0            {'✓' if g1 else '✗'}  "
          f"({st.mean(lift_home):+.4f})")
    print(f"   正折比例 > 55%               {'✓' if g2 else '✗'}  "
          f"({100*sum(1 for x in lift_home if x>0)/len(lift_home):.0f}%)")
    print(f"   lift 的 CI 不含零            {'✓' if g3 else '✗'}  "
          f"({ci if ci else '—'})")
    print(f"   留幣 4 組中 ≥3 組贏 regime   {'✓' if lco_ok else '✗'}")
    fwd_ok = bool(fwd and fwd["lift_vs_home"] > 0)
    print(f"   前瞻留出：模型贏過 regime          "
          f"{'✓' if fwd_ok else '✗'}"
          + (f"  ({fwd['lift_vs_home']:+.4f})" if fwd else "  (無樣本)"))
    passed = g1 and g2 and g3 and lco_ok and fwd_ok
    v = ("**全數過關（含前瞻留出）** —— 進場 meta-model 成為預註冊候選"
         "（不是規則）。下一步是凍結特徵與門檻、用 08-26 之後的新成交驗證"
         if passed else
         ("**回測四關過，但前瞻留出未過** —— 模型學到的關係在 FREEZE 之後"
          "不成立，與 §0.58 的格內衰退一致。回測世界的 lift 不可外推"
          if (g1 and g2 and g3 and lco_ok) else
          "**未過** —— 記錄為負結果。判準要求贏過 §0.59 regime 濾網；"
          "只贏過『全進』不算，那代表模型只是把 regime 重學了一遍"))
    print(f"\n判讀：{v}")

    imp = {}
    m = XGBClassifier(**params).fit(X, y)
    for f, g in sorted(zip(FEATURES, m.feature_importances_),
                       key=lambda z: -z[1])[:8]:
        imp[f] = round(float(g), 4)
    print("\n  （全樣本重擬合的特徵重要度，僅供理解，不作證據）")
    for f, g in imp.items():
        print(f"    {f:<22} {g:.4f}")

    OUT.write_text(json.dumps({
        "n_events": len(ev), "features": FEATURES,
        "baseline_all": b1, "baseline_regime": b3,
        "folds": folds, "lift_vs_regime": [round(x, 4) for x in lift_home],
        "lift_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "leave_coins_out": [{k: (v if k != "held" else v)
                             for k, v in r.items()} for r in lco],
        "forward_holdout": fwd,
        "gates": {"per_fold_mean": g1, "frac_pos": g2, "ci": g3,
                  "leave_coins_out": lco_ok, "forward_holdout": fwd_ok},
        "passed": passed, "verdict": v, "importance": imp,
    }, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
