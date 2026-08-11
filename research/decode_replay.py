"""Two-tail decode replay — which decoding rule survives a drifting model?

WRITTEN BEFORE LOOKING AT ANY RESULT.  The 2026-08-08 refresh shipped with
four gates and still landed a decode that cannot fire DOWN at all; the gates
were not too few, they were the wrong shape (G4 compared the buffer's
DISPERSION RATIO and passed at 0.78, while nobody compared its LOCATION or
asked whether the live output could physically reach the cutoffs).  So this
file fixes the criteria first and reports every variant against them,
pass or fail.

── THE DEFECT BEING FIXED ────────────────────────────────────────────────
Decoding is "rank today's prediction against a 500-bar buffer, fire on the
2.5%/7.5% tails".  That silently assumes the buffer and the live stream are
the same distribution.  Measured 2026-08-11 on the deployed model:

    buffer (seeded from in-sample preds)  mean -0.000005  std 0.001196
    live stream since deploy              mean +0.000716  std 0.001057

A +0.60-sigma location offset.  Consequence, and this is arithmetic rather
than statistics: the loosest DOWN cutoff was -0.001786 while the model's
all-time-lowest live output was -0.001480.  NO DOWN signal of ANY tier could
fire.  9 UP : 0 DOWN, and the last six real positions were all LONG.

Two distinct failure modes share this one root — the reference distribution
not matching the thing being judged:
  fresh after a retrain : buffer is in-sample, live can't reach its tails
  long after a retrain  : buffer is all-live, but the model's level drifts
                          and the percentile reads the drift as direction
                          (that is the 2026-08-08 entry in mistake.md)
A fix that only addresses one of them is not a fix.

── STIMULUS ──────────────────────────────────────────────────────────────
The real production prediction stream from indicator_history, with the real
forward TWAP returns from its own close column.  Not a simulation: this
stream contains the actual drift, the actual regime turns and the actual
model-version changes that any decode has to survive.  Historical bars are
never re-scored (feedback_no_signal_overwrite), so these are the predictions
as they were made at the time.

Hysteresis and cooldown are NOT modelled.  They sit downstream of the
direction decision and apply identically to every variant, so including
them would only add noise to a comparison.  Absolute counts here are
therefore higher than what the live system fires.

── TWO TESTS, AND AN INSTRUMENT CHECK THAT COMES FIRST ───────────────────
A first version of this file replayed the whole stream in one pass and
reported the current decode as balanced (Aug 14 UP : 18 DOWN).  Live fired
9 UP : 0 DOWN over the same days.  The replay was wrong, not the live
system: a deploy REPLACES the buffer with the seed, and a single forward
pass had long since rolled the seed out.  A harness that cannot reproduce
the known failure cannot be trusted to rank fixes for it (mistake.md
2026-07-29), so the reset is now modelled and CHECKED:

  INSTRUMENT CHECK  replay V0 from the 2026-08-08 reset and require it to
                    reproduce the observed one-sidedness.  If it does not,
                    everything below is void and is reported as void.
  TEST A drift      full stream, no reset — the slow failure mode, where a
                    frozen model's level walks away from its own buffer.
  TEST B reset      buffer := seed at a cut point, then replay forward —
                    the fast failure mode observed on 2026-08-08.  Run at
                    several cut points so the answer is not one date's luck.

── PRE-REGISTERED CRITERIA (fixed before the first run) ──────────────────
  P1 BALANCE        UP share of fired signals within [30%, 70%] overall,
                    AND within [25%, 75%] in >= 80% of calendar months.
                    Pooled balance alone is what let a LONG-only quarter
                    hide inside a balanced year.
  P2 REACHABILITY   no month may fire >= 5 signals on one side and 0 on the
                    other.  This is the specific defect: a cutoff that
                    cannot physically be reached.
  P3 NON-INFERIORITY  directional accuracy of fired signals >= the current
                    decode's minus 2.0pp.  Balance is worthless if it is
                    bought by firing garbage in the other direction.
  P4 RATE           fires on 5%-25% of bars.  Below 5% the strategy starves
                    (the 2026-08 low-frequency problem); above 25% the tier
                    stops meaning anything.
  P5 CAUSALITY      bar i may use bars <= i only.  Asserted in code, not
                    trusted to review.

A variant must pass P1, P2, P4, P5 and not fail P3 to be a candidate.
Passing does NOT mean deploy — it means it earns a forward shadow window.

Usage:  python research/decode_replay.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ART = ROOT / "indicator" / "model_artifacts" / "dual_model"
FLOOR_STRONG, FLOOR_MOD = 0.0008, 0.0005
STRONG_FRAC, MOD_FRAC = 0.05, 0.15
WINDOW, WARMUP = 500, 100

# P-criteria
P1_POOLED = (0.30, 0.70)
P1_MONTHLY = (0.25, 0.75)
P1_MONTH_SHARE = 0.80
P2_ONE_SIDED_MIN = 5
P3_MAX_ACC_DROP = 0.020
P4_RATE = (0.05, 0.25)


# ── data ────────────────────────────────────────────────────────────────
def load():
    import shared.db as sdb
    conn = sdb.get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT dt, close, pred_return_4h p FROM indicator_history "
                "WHERE pred_return_4h IS NOT NULL AND close IS NOT NULL "
                "ORDER BY dt")
            rows = cur.fetchall()
    finally:
        conn.close()
    dt = [r["dt"] for r in rows]
    cl = np.array([float(r["close"]) for r in rows])
    pr = np.array([float(r["p"]) for r in rows])
    # target the model was trained on: 4h TWAP path return
    n = len(cl)
    y = np.full(n, np.nan)
    for i in range(n - 4):
        y[i] = cl[i + 1:i + 5].mean() / cl[i] - 1.0
    return dt, pr, y


def wf_quantiles():
    """The model's OOS output shape, from the walk-forward run.

    This is the one honest description of how the deployed model behaves on
    data it never saw.  mistake.md 2026-04-19 forbids WF fold predictions as
    a BUFFER seed because their scale is wrong; using their SHAPE as a fixed
    cutoff calibration is a different use and is what the shipped 'fallback'
    thresholds already do.
    """
    import pandas as pd
    p = ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
    v = pd.read_parquet(p)["pred_ret"].to_numpy()
    v = v[np.isfinite(v)]
    return {"s_dn": float(np.quantile(v, STRONG_FRAC / 2)),
            "m_dn": float(np.quantile(v, MOD_FRAC / 2)),
            "m_up": float(np.quantile(v, 1 - MOD_FRAC / 2)),
            "s_up": float(np.quantile(v, 1 - STRONG_FRAC / 2))}


def seed_buffer():
    st = json.loads((ART / "training_stats.json").read_text())
    return [float(x) for x in st.get("dir_pred_history", [])]


# ── decode variants ─────────────────────────────────────────────────────
# Every variant is a generator over bars yielding (direction, tier).  Each
# may look at index i and anything before it, never after (P5).

def _tier(p, s_up, m_up, m_dn, s_dn):
    if p >= max(s_up, FLOOR_STRONG):
        return "UP", "Strong"
    if p <= min(s_dn, -FLOOR_STRONG):
        return "DOWN", "Strong"
    if p >= max(m_up, FLOOR_MOD):
        return "UP", "Moderate"
    if p <= min(m_dn, -FLOOR_MOD):
        return "DOWN", "Moderate"
    return "NEUTRAL", "Weak"


def v_rolling(pr, seed, window=WINDOW, warmup=WARMUP, fb=None, center=False):
    """The shipped decode, parameterised so every variant is one code path.

    seed   : list pre-loaded into the buffer (current behaviour), or []
    fb     : fixed fallback cutoffs used while the buffer is under warmup
    center : subtract the buffer's trailing median from BOTH the prediction
             and the cutoffs, making the decode immune to level drift by
             construction rather than by hoping the buffer keeps up
    """
    buf = deque(seed, maxlen=window)
    out = []
    for p in pr:
        buf.append(float(p))              # push-then-read: "now" included,
        if len(buf) < warmup:             # matches inference.py
            if fb is None:                # warmup silence: fire nothing until
                out.append(("NEUTRAL", "Weak"))   # the buffer is live-grown
                continue
            out.append(_tier(p, fb["s_up"], fb["m_up"], fb["m_dn"], fb["s_dn"]))
            continue
        b = np.fromiter(buf, dtype=float)
        med = float(np.median(b)) if center else 0.0
        out.append(_tier(
            p - med,
            float(np.quantile(b, 1 - STRONG_FRAC / 2)) - med,
            float(np.quantile(b, 1 - MOD_FRAC / 2)) - med,
            float(np.quantile(b, MOD_FRAC / 2)) - med,
            float(np.quantile(b, STRONG_FRAC / 2)) - med))
    return out


def v_fixed_centered(pr, fb, window=WINDOW, warmup=WARMUP):
    """De-mean on a trailing window, then apply the model's OOS shape.

    Separates the two things the rolling percentile conflates: LEVEL (which
    drifts and must be removed) and SHAPE (which is a property of the model
    and is better estimated from the whole walk-forward than from 500 bars
    of whatever the market just did).
    """
    buf = deque(maxlen=window)
    out = []
    for p in pr:
        buf.append(float(p))
        if len(buf) < warmup:
            out.append(_tier(p, fb["s_up"], fb["m_up"], fb["m_dn"], fb["s_dn"]))
            continue
        med = float(np.median(np.fromiter(buf, dtype=float)))
        out.append(_tier(p - med, fb["s_up"], fb["m_up"], fb["m_dn"], fb["s_dn"]))
    return out


# ── scoring ─────────────────────────────────────────────────────────────
def score(name, dec, dt, y):
    fired = [(i, d) for i, (d, t) in enumerate(dec)
             if d in ("UP", "DOWN") and np.isfinite(y[i])]
    n_bar = int(np.isfinite(y).sum())
    if not fired:
        return {"name": name, "n": 0, "rate": 0.0, "up_share": float("nan"),
                "acc": float("nan"), "ret": float("nan"), "months": {},
                "p1": False, "p2": False, "p4": False}
    up = sum(1 for _, d in fired if d == "UP")
    hit = [(y[i] > 0) == (d == "UP") for i, d in fired]
    ret = [y[i] if d == "UP" else -y[i] for i, d in fired]
    months = defaultdict(lambda: [0, 0])
    for i, d in fired:
        months[str(dt[i])[:7]][0 if d == "UP" else 1] += 1
    ok_m = [m for m, (u, dn) in months.items() if u + dn >= 10]
    bal = [P1_MONTHLY[0] <= u / (u + dn) <= P1_MONTHLY[1]
           for m, (u, dn) in months.items() if m in ok_m]
    p1 = (P1_POOLED[0] <= up / len(fired) <= P1_POOLED[1]
          and (not bal or np.mean(bal) >= P1_MONTH_SHARE))
    p2 = not any((u >= P2_ONE_SIDED_MIN and dn == 0)
                 or (dn >= P2_ONE_SIDED_MIN and u == 0)
                 for u, dn in months.values())
    rate = len(fired) / max(n_bar, 1)
    return {"name": name, "n": len(fired), "rate": rate,
            "up_share": up / len(fired), "acc": float(np.mean(hit)),
            "ret": float(np.mean(ret)), "months": dict(months),
            "p1": p1, "p2": p2, "p4": P4_RATE[0] <= rate <= P4_RATE[1],
            "bal_months": (float(np.mean(bal)) if bal else float("nan"))}


def build(pr, seed, fb):
    """The variant set.  seed is what the buffer is initialised with.

    A 'de-mean the prediction before ranking it' variant was tried and
    dropped: quantiles are already location-invariant, so subtracting the
    buffer's median from both sides is algebraically a no-op (it reproduced
    V0 to the signal).  Centring only bites when the cutoffs are FIXED, and
    fixed WF-calibrated cutoffs over-fire (the WF stream's spread is 0.67x
    the production stream's).  The live lever is therefore how fast the
    buffer tracks the live stream: seed or not, and how long the window.
    """
    return [
        ("V0 現行（seed 取代 buffer）", v_rolling(pr, seed, fb=fb)),
        ("V1 不 seed · 窗 500", v_rolling(pr, [], fb=fb)),
        ("V2 不 seed · 窗 200", v_rolling(pr, [], window=200, fb=fb)),
        ("V3 不 seed · 窗 200 · 暖機不開火",
         v_rolling(pr, [], window=200, fb=None)),
        ("V4 不 seed · 窗 120 · 暖機不開火",
         v_rolling(pr, [], window=120, warmup=120, fb=None)),
    ]


def instrument_check(dt, pr, y, seed, fb) -> bool:
    """V0 replayed from the real 2026-08-08 reset must reproduce 9 UP : 0 DOWN
    (one-sidedness, not the exact count — hysteresis/cooldown are not
    modelled so the absolute number is expected to be higher)."""
    cut = next((i for i, d in enumerate(dt) if str(d) >= "2026-08-08 21"), None)
    if cut is None:
        print("  儀器檢查：找不到 08-08 切點 → 本檔結果作廢")
        return False
    dec = v_rolling(pr[cut:], seed, fb=fb)
    u = sum(1 for d, _ in dec if d == "UP")
    dn = sum(1 for d, _ in dec if d == "DOWN")
    ok = dn == 0 and u > 0
    print(f"  儀器檢查：從 08-08 reset 重放 V0 → UP {u} / DOWN {dn}   "
          f"（實際開火 9:0）  {'重現成功 ✓' if ok else '未重現 ✗ → 以下結果作廢'}")
    return ok


def main() -> int:
    dt, pr, y = load()
    fb = wf_quantiles()
    seed = seed_buffer()
    print(f"生產 pred 流 {len(pr)} 根 · {str(dt[0])[:10]} → {str(dt[-1])[:10]}")
    print(f"WF OOS 校準的固定切點: Strong[{fb['s_dn']:+.6f},{fb['s_up']:+.6f}]"
          f"  Mod[{fb['m_dn']:+.6f},{fb['m_up']:+.6f}]")
    print(f"seed buffer {len(seed)} 根 (mean {np.mean(seed):+.6f})\n")

    print("═" * 88)
    print("  儀器檢查 —— 重現不了已知病灶的 harness 不能拿來排序修法")
    print("═" * 88)
    valid = instrument_check(dt, pr, y, seed, fb)
    if not valid:
        print("\n作廢：harness 無法重現 2026-08-08 的單邊失效，不輸出變體排名。")
        return 1

    print("\n" + "═" * 88)
    print("  TEST B ── reset 後 30 天的方向平衡（快速失效模式，08-08 那種）")
    print("═" * 88)
    seen, cuts = set(), []
    for i, d in enumerate(dt):
        day = str(d)[:10]
        if day in ("2026-05-15", "2026-06-15", "2026-07-01",
                   "2026-07-15", "2026-08-08") and day not in seen:
            seen.add(day)
            cuts.append(i)
    horizon, warm = 24 * 30, 100
    print("  每格 = reset 後 30 天的 UP:DOWN。分成暖機期（前 100 根，buffer 還沒")
    print("  長成 live 的）和暖機後 —— 偏斜住在哪一段，決定了要修哪裡。\n")
    print(f"  {'變體':<28}" + "".join(f"{str(dt[c])[5:10]:>14}" for c in cuts))
    for seg, lo, hi in (("暖機期", 0, warm), ("暖機後", warm, horizon)):
        print(f"  ── {seg} ──")
        rr = defaultdict(list)
        for c in cuts:
            for name, dec in build(pr[c:c + horizon], seed, fb):
                s = dec[lo:hi]
                rr[name].append((sum(1 for d, _ in s if d == "UP"),
                                 sum(1 for d, _ in s if d == "DOWN")))
        for name, vals in rr.items():
            line = f"  {name:<28}"
            for u, dn in vals:
                sh = u / (u + dn) if (u + dn) else float("nan")
                mark = "!" if (u + dn) and not (0.25 <= sh <= 0.75) else " "
                line += f"{f'{u}:{dn}{mark}':>14}"
            print(line)
    print("  ! = UP 佔比落在 25%-75% 之外（單邊）")

    print("\n" + "═" * 88)
    print("  TEST A ── 全程重放（慢速失效模式：模型水平漂離自己的 buffer）")
    print("═" * 88)
    variants = build(pr, seed, fb)
    rows = [score(n, d, dt, y) for n, d in variants]
    base = rows[0]

    print("═" * 88)
    print(f"{'變體':<26}{'開火數':>7}{'開火率':>8}{'UP佔':>8}{'方向命中':>9}"
          f"{'均報酬':>10}{'月度平衡':>9}  判定")
    print("═" * 88)
    for r in rows:
        if r["n"] == 0:
            print(f"{r['name']:<26}  (無開火)")
            continue
        p3 = np.isnan(base["acc"]) or r["acc"] >= base["acc"] - P3_MAX_ACC_DROP
        flags = "".join([" P1" if not r["p1"] else "", " P2" if not r["p2"] else "",
                         " P3" if not p3 else "", " P4" if not r["p4"] else ""])
        verdict = "候選 ✓" if not flags else f"FAIL:{flags}"
        bm = r["bal_months"]
        print(f"{r['name']:<26}{r['n']:>7}{r['rate']*100:>7.1f}%"
              f"{r['up_share']*100:>7.0f}%{r['acc']*100:>8.1f}%"
              f"{r['ret']*10000:>+9.1f}bp"
              f"{'  n/a' if np.isnan(bm) else f'{bm*100:>8.0f}%'}  {verdict}")

    print("\n── 最近 6 個月的逐月 UP:DOWN（P2 就是在抓這個）──")
    allm = sorted({m for r in rows for m in r["months"]})[-6:]
    print(f"  {'月份':<9}" + "".join(f"{r['name'][:12]:>14}" for r in rows))
    for m in allm:
        line = f"  {m:<9}"
        for r in rows:
            u, dn = r["months"].get(m, [0, 0])
            line += f"{f'{u}:{dn}':>14}"
        print(line)

    print(f"\n判準（跑之前寫死）：P1 UP佔 {P1_POOLED[0]:.0%}-{P1_POOLED[1]:.0%} 且"
          f" ≥{P1_MONTH_SHARE:.0%} 的月份落在 {P1_MONTHLY[0]:.0%}-{P1_MONTHLY[1]:.0%}"
          f"；P2 不得有單側 ≥{P2_ONE_SIDED_MIN} 而另一側 0 的月份；"
          f"\n              P3 命中率不低於 V0 減 {P3_MAX_ACC_DROP:.1%}；"
          f"P4 開火率 {P4_RATE[0]:.0%}-{P4_RATE[1]:.0%}")
    print("通過 ≠ 可部署 —— 通過的意思是它有資格進 forward shadow 觀察窗。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
