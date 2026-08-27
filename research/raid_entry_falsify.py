# -*- coding: utf-8 -*-
"""Two tests designed to KILL the compose model — TODO §0.70b.

Status this file starts from: `compose` passed all five gates, but its
forward window (FREEZE 07-28 -> today) is 96% data that §0.59 already
declared spent. On spent data no test can VALIDATE — the sample was used
to generate the hypothesis it is being asked to confirm. It can still
FALSIFY, which is worth doing now rather than after a two-month clock.

So both tests below are built to fail the model, and surviving them
changes nothing about its status (hypothesis, not result).

TEST A — IS THE SELECTION BETTER THAN RANDOM?
  The model picked 98 of the 147 forward home-regime events and scored
  +0.2006 against the full set's +0.1147. But ANY 98-subset of a
  right-skewed R distribution has a spread, and picking 2/3 of a set is a
  weak constraint. Draw 98 at random 20,000 times and see where +0.2006
  falls. If it sits inside the bulk, the "lift" is subset noise and the
  gate passed on nothing.

TEST B — DOES THE TOP FEATURE PREDICT THE OUTCOME, OR THE FILL?
  `sweep_close_pos` dominates importance (0.143). The population contains
  only sweeps that FILLED — price had to return to the level within W=8
  bars. A sweep bar closing far back in the trade's direction has moved
  price AWAY from the level, so it is less likely to fill at all. If
  sweep_close_pos separates filled from unfilled sweeps strongly, then its
  apparent skill is partly a survivorship artefact of the fill condition,
  not a statement about outcomes. Same family as mistake.md 2026-07-28:
  a variable that looks predictive because of what the sample conditioned
  on rather than what it forecasts.

Pre-committed reading:
  A: percentile of the observed mean among random subsets
       >= 95th -> selection is doing something; model survives this test
       <  95th -> the pass was subset noise; compose is dead
  B: filled-vs-unfilled separation in sweep_close_pos
       small (|d| < 0.2 std) -> not a fill artefact; survives
       large               -> the top feature is contaminated by the fill
                              condition and the feature set must be rebuilt
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

sys.argv = [sys.argv[0], "compose"]                # import-time CONFIG
from research.raid_entry_model import (                    # noqa: E402
    extract, FEATURES, HOME, CACHE,
)

OUT = ROOT / "research" / "results" / "raid_entry_falsify.json"
FREEZE = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
random.seed(83)
rng = np.random.default_rng(83)


def main() -> int:
    syms = sorted(p.name.replace("USDT_1h.csv", "")
                  for p in CACHE.glob("*USDT_1h.csv"))
    ev = []
    for s in syms:
        ev += extract(s)
    ev.sort(key=lambda x: x["ts"])
    home = [x for x in ev if x["cell"] in HOME]
    tr = [x for x in home if x["exit_ts"] < FREEZE]
    te = [x for x in home if x["ts"] >= FREEZE]
    print("§0.70b 兩個設計來殺掉 compose 的檢定")
    print(f"  主場母體 {len(home)}｜訓練 {len(tr)}｜前瞻 {len(te)}")
    print("  提醒：前瞻窗 96% 是 §0.59 宣告作廢的樣本 —— "
          "以下檢定只能證偽，不能證實\n")

    from xgboost import XGBClassifier
    params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                  min_child_weight=20, eval_metric="logloss",
                  n_jobs=4, verbosity=0)
    Xtr = np.array([[float(x[f]) for f in FEATURES] for x in tr])
    ytr = np.array([x["y"] for x in tr])
    Xte = np.array([[float(x[f]) for f in FEATURES] for x in te])
    m = XGBClassifier(**params).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    sel = [x for x, pv in zip(te, p) if pv > 0.5]
    k, obs = len(sel), st.mean(x["R"] for x in sel)
    base = st.mean(x["R"] for x in te)
    print(f"── TEST A  選 {k}/{len(te)} 筆，meanR {obs:+.4f}（全體 {base:+.4f}）──")

    Rs = np.array([x["R"] for x in te])
    draws = np.array([rng.choice(Rs, size=k, replace=False).mean()
                      for _ in range(20000)])
    pct = 100.0 * float((draws < obs).mean())
    print(f"   隨機抽 {k} 筆 20,000 次：均值 {draws.mean():+.4f}、"
          f"5~95% 區間 [{np.percentile(draws,5):+.4f},"
          f"{np.percentile(draws,95):+.4f}]")
    print(f"   觀測值落在第 **{pct:.1f}** 百分位")
    a_ok = pct >= 95.0
    print(f"   → {'✓ 選擇優於隨機' if a_ok else '✗ 落在隨機分佈之內 —— lift 是子集雜訊'}")

    # ── TEST B ──────────────────────────────────────────────────────────
    print(f"\n── TEST B  sweep_close_pos 預測的是結果，還是「有沒有成交」？──")
    fills, misses = [], []
    for sym in syms:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        h = [b[SC.H] for b in bars]
        lo = [b[SC.L] for b in bars]
        c = [b[SC.C] for b in bars]
        a = SC.atr14(bars)
        n = len(bars)
        for e in SC.detect_sweeps(bars):
            j, lvl = e["j"], e["level"]
            if a[j] is None or a[j] <= 0:
                continue
            kd = 1 if e["kind"] == "buy" else -1
            d = -kd
            pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / a[j]
            if pierce > 0.25:
                continue
            filled = False
            for f in range(j + 1, min(j + 1 + SC.W, n)):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    filled = True
                    break
            rng_ = h[j] - lo[j]
            cp = ((c[j] - lo[j]) / rng_) if rng_ > 0 else 0.5
            cp = cp if d == 1 else 1.0 - cp
            (fills if filled else misses).append(cp)

    mf, mm = st.mean(fills), st.mean(misses)
    sd = st.pstdev(fills + misses)
    dcoh = (mf - mm) / sd if sd else 0.0
    print(f"   有成交 n={len(fills):5d}  sweep_close_pos 均值 {mf:.4f}")
    print(f"   未成交 n={len(misses):5d}  sweep_close_pos 均值 {mm:.4f}")
    print(f"   標準化差 (Cohen's d) = **{dcoh:+.3f}**")
    b_ok = abs(dcoh) < 0.2
    print(f"   → {'✓ 分離小，不是成交條件造成的假象' if b_ok else '✗ 分離大 —— 這個特徵有很大一部分在說「會不會成交」而不是「成交後賺不賺」'}")

    print()
    if a_ok and b_ok:
        v = ("兩個殺手測試都沒殺掉它。**狀態不變：仍是假設，不是結果**"
             "——前瞻窗是作廢資料，這一點沒有任何檢定能改變。")
    else:
        bad = ([] + (["A 選擇不優於隨機"] if not a_ok else [])
               + (["B 頭號特徵被成交條件污染"] if not b_ok else []))
        v = f"**被殺掉**：{'、'.join(bad)}。compose 不再是候選。"
    print(f"判讀：{v}")
    OUT.write_text(json.dumps({
        "test_a": {"k": k, "n": len(te), "observed": round(obs, 4),
                   "base": round(base, 4), "percentile": round(pct, 1),
                   "passed": a_ok},
        "test_b": {"filled_n": len(fills), "missed_n": len(misses),
                   "filled_mean": round(mf, 4), "missed_mean": round(mm, 4),
                   "cohens_d": round(dcoh, 3), "passed": b_ok},
        "verdict": v}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
