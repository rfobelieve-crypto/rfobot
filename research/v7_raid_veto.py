# -*- coding: utf-8 -*-
"""V7 raid-chase veto — robustness deep-dive before any registration.

Context (2026-08-02): v7_raid_context.py found live Strong signals that
FOLLOW a fresh raid break win 52% vs 64% with no recent raid, gradient
consistent in both halves. Before this becomes a registered veto rule it
has to survive the slicing that killed weaker ideas:

  1. lookback sensitivity  — 2/4/6/8h windows (4h is the named default;
     the others show whether the effect is a cliff or a curve)
  2. direction split       — UP vs DOWN signals (the known 6pp asymmetry
     must not be the whole story)
  3. regime split          — the veto must not be a regime proxy
  4. thirds stability      — three time slices, gradient direction each
  5. counterfactual        — portfolio effect of applying the veto:
     kept-WR vs vetoed-WR, and the signal-count cost
  6. actual_return_4h view — average realized 4h return per bucket (the
     money-adjacent number, not just hit rate)

`--clock` prints one line for the weekly PortfolioClocks report: trailing
90d kept vs vetoed WR — the forward-confirmation clock that must stay
separated before production adoption is discussed.

Run: python research/v7_raid_veto.py [--clock]
Out: research/results/v7_raid_veto.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

from shared.db import get_db_conn  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_raid_veto.json"


def load(tier: str = "Strong"):
    """tier defaults to Strong — the frozen registration. 'Moderate' feeds
    the PARALLEL clock registered 2026-08-02 (TODO 0.49); it shares every
    definition and threshold, only the population differs."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, regime, correct, "
                "actual_return_4h FROM tracked_signals "
                "WHERE strength=%s AND correct IS NOT NULL "
                "ORDER BY signal_time", (tier,))
            sigs = cur.fetchall()
    finally:
        conn.close()
    by_hh: dict[int, list] = defaultdict(list)
    for r in raids_with_fill("BTC"):
        by_hh[r["ts"] // 3600].append(r["side"])
    return sigs, by_hh


TRIGGER_START = "2026-08-02"   # both clocks start the same instant
TRIGGER_TARGET = 60
GAP_THRESHOLD_PP = 8.0


def clock_block(rows, tier):
    """90d kept-vs-vetoed gap + forward count since TRIGGER_START.

    Identical arithmetic for both tiers — the ONLY difference between the
    frozen Strong clock and the parallel Moderate one is which signals go
    in. Counting starts at TRIGGER_START so neither clock can be fed by
    data that existed when it was registered."""
    if not rows:
        return {"tier": tier, "kept_wr": None, "veto_wr": None,
                "gap_pp": None, "n_kept": 0, "n_veto": 0,
                "since_trigger": 0, "trigger_target": TRIGGER_TARGET,
                "gap_threshold_pp": GAP_THRESHOLD_PP}
    cut = max(r["ts"] for r in rows) - 90 * 86400
    rec = [r for r in rows if r["ts"] >= cut]
    kept = [r for r in rec if r["b"] != "follow"]
    veto = [r for r in rec if r["b"] == "follow"]
    k, v = wr(kept), wr(veto)
    t0 = datetime.strptime(TRIGGER_START, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp()
    return {"tier": tier,
            "kept_wr": round(k, 1) if k is not None else None,
            "veto_wr": round(v, 1) if v is not None else None,
            "gap_pp": round(k - v, 1) if (k is not None and v is not None)
            else None,
            "n_kept": len(kept), "n_veto": len(veto),
            "since_trigger": sum(1 for r in rows if r["ts"] >= t0),
            "trigger_target": TRIGGER_TARGET,
            "gap_threshold_pp": GAP_THRESHOLD_PP}


def bucket(by_hh, ts_h, direction, look):
    for k in range(0, look + 1):
        sides = by_hh.get(ts_h - k)
        if sides:
            s = sides[0]
            fade = ((s == 1 and direction == "DOWN")
                    or (s == -1 and direction == "UP"))
            return "fade" if fade else "follow"
    return "none"


def wr(rows):
    return 100 * sum(r["c"] for r in rows) / len(rows) if rows else None


def ret(rows):
    xs = [r["ret"] for r in rows if r["ret"] is not None]
    # realized 4h return IN THE SIGNAL'S DIRECTION (positive = signal paid)
    return 100 * sum(xs) / len(xs) if xs else None


def fmt(rows, tag):
    parts = []
    for b in ("none", "fade", "follow"):
        g = [r for r in rows if r["b"] == b]
        if len(g) >= 15:
            parts.append(f"{b} {wr(g):.0f}%/{ret(g):+.2f}% (n={len(g)})")
        else:
            parts.append(f"{b} thin({len(g)})")
    return f"  {tag:<14}" + " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", action="store_true")
    args = ap.parse_args()
    sigs, by_hh = load()
    rows = []
    for s in sigs:
        ts = int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        d = s["direction"]
        sgn = 1 if d == "UP" else -1
        rr = (float(s["actual_return_4h"]) * sgn
              if s["actual_return_4h"] is not None else None)
        rows.append({"ts": ts, "dir": d, "regime": s["regime"] or "?",
                     "c": int(s["correct"]), "ret": rr,
                     "b": bucket(by_hh, ts // 3600, d, 4)})

    if args.clock:
        blocks = {"strong": clock_block(rows, "Strong")}
        # PARALLEL Moderate clock (registered 2026-08-02, TODO 0.49):
        # identical dims, identical +60 / >=8pp thresholds, 3-5x the fire
        # rate. Indirect evidence — the executor trades Strong only — and
        # it must be labelled as such wherever it is displayed.
        try:
            msigs, mby = load("Moderate")
            mrows = [{"ts": int(x["signal_time"].replace(
                          tzinfo=timezone.utc).timestamp()),
                      "dir": x["direction"], "c": int(x["correct"]),
                      "b": bucket(mby, int(x["signal_time"].replace(
                          tzinfo=timezone.utc).timestamp()) // 3600,
                          x["direction"], 4)}
                     for x in msigs]
            blocks["moderate"] = clock_block(mrows, "Moderate")
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] moderate clock failed: {e}")
        for tag, b in blocks.items():
            if b["kept_wr"] is None or b["veto_wr"] is None:
                print(f"V7 raid-veto [{tag}]: thin")
                continue
            print(f"V7 raid-veto [{tag}] (90d): kept {b['kept_wr']:.0f}% "
                  f"(n={b['n_kept']}) vs vetoed {b['veto_wr']:.0f}% "
                  f"(n={b['n_veto']}) — gap {b['gap_pp']:+.1f}pp | "
                  f"trigger {b['since_trigger']}/{b['trigger_target']}")
        k = blocks["strong"]["kept_wr"]
        v = blocks["strong"]["veto_wr"]
        # adoption-trigger progress (TODO 0.483): +60 Strong fired since
        # 2026-08-02, then gap >=8pp tables the informed decision. Written
        # as JSON so the agent/site can display the countdown (refreshes
        # into the Railway image on each push; asof stamped for honesty).
        try:
            from datetime import datetime as _dt
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    # Two counts per tier, because they answer different
                    # questions and only one of them ticks immediately:
                    #   resolved = the EVIDENCE count (correct backfilled
                    #              ~4h after the bar) — this is what the
                    #              trigger is actually about
                    #   fired    = signals emitted, which moves the moment
                    #              a signal prints
                    # Reporting only `resolved` made the counter look
                    # frozen for four hours after every signal (operator
                    # noticed on 2026-08-02, a Moderate fired and the
                    # board still read 0/60).
                    for tier_key, tier_name in (("strong", "Strong"),
                                                ("moderate", "Moderate")):
                        cur.execute(
                            "SELECT SUM(correct IS NOT NULL) resolved, "
                            "COUNT(*) fired FROM tracked_signals "
                            "WHERE strength=%s AND signal_time >= %s",
                            (tier_name, TRIGGER_START))
                        row = cur.fetchone() or {}
                        blk = blocks.get(tier_key)
                        if blk is not None:
                            blk["since_trigger"] = int(row.get("resolved") or 0)
                            blk["since_trigger_fired"] = int(row.get("fired") or 0)
                    since = blocks.get("strong", {}).get("since_trigger", 0)
            finally:
                conn.close()
            out = {"kept_wr": round(k, 1) if k else None,
                   "veto_wr": round(v, 1) if v else None,
                   "gap_pp": round(k - v, 1) if k and v else None,
                   "n_kept": blocks["strong"]["n_kept"],
                   "n_veto": blocks["strong"]["n_veto"],
                   "strong_since_trigger": since, "trigger_target": 60,
                   "gap_threshold_pp": 8.0,
                   "clocks": blocks,
                   "asof_utc": f"{_dt.utcnow():%Y-%m-%d %H:%M}"}
            (ROOT / "research/results/v7_veto_clock.json").write_text(
                json.dumps(out, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] veto clock json failed: {e}")
        return 0

    print("=" * 78)
    print("  V7 RAID-CHASE VETO — 註冊前的穩健性切片（勝率% / 方向報酬%）")
    print("=" * 78)
    res = {}
    n = len(rows)
    print(f"  Strong n={n} · 整體 {wr(rows):.0f}% / {ret(rows):+.2f}%\n")

    print("  [1] 回看窗口敏感度")
    for look in (2, 4, 6, 8):
        rr = [dict(r, b=bucket(by_hh, r["ts"] // 3600, r["dir"], look))
              for r in rows]
        print(fmt(rr, f"look={look}h"))
        res[f"look_{look}"] = {b: wr([x for x in rr if x["b"] == b])
                               for b in ("none", "fade", "follow")}

    print("\n  [2] 方向分拆（4h 窗）")
    for d in ("UP", "DOWN"):
        print(fmt([r for r in rows if r["dir"] == d], f"{d}"))

    print("\n  [3] regime 分拆")
    for g in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == g]
        if len(sub) >= 60:
            print(fmt(sub, g[:13]))

    print("\n  [4] 三等分時間切片")
    third = n // 3
    for i, tag in ((0, "T1"), (1, "T2"), (2, "T3")):
        seg = rows[i * third: (i + 1) * third if i < 2 else n]
        print(fmt(seg, tag))
        res[f"third_{tag}"] = {b: wr([x for x in seg if x["b"] == b])
                               for b in ("none", "fade", "follow")}

    print("\n  [5] 反事實：套用否決的組合效果")
    kept = [r for r in rows if r["b"] != "follow"]
    veto = [r for r in rows if r["b"] == "follow"]
    print(f"    保留 {len(kept)} 筆: WR {wr(kept):.0f}% / {ret(kept):+.2f}%"
          f"  |  否決 {len(veto)} 筆 ({100*len(veto)/n:.0f}%): "
          f"WR {wr(veto):.0f}% / {ret(veto):+.2f}%")
    res["counterfactual"] = {"kept_n": len(kept), "kept_wr": wr(kept),
                             "kept_ret": ret(kept), "veto_n": len(veto),
                             "veto_wr": wr(veto), "veto_ret": ret(veto)}
    OUT.write_text(json.dumps(res, indent=1, default=float),
                   encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
