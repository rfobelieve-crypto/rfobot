# -*- coding: utf-8 -*-
"""Is drv_q causal at decision time? — TODO §0.66.

The audit found that every Coinglass-derived combo on the raid line has a
CI low above zero (R∧Q +0.5206, R∧V∧Q +0.6145, R∧快∧Q +0.3789) and that
variant E — the ladder's only strongly positive variant, +0.3389R at 71%
WR — is built entirely on Coinglass flags. The obvious move is to extend
those flags from BTC to all nine coins, since the paid data covers 11.

BEFORE extending anything, the flag has to be causal. Tonight already
killed one Coinglass gauge for exactly this: G4's liquidation effect was
+0.1480R same-bar and −0.0218R lagged one bar, i.e. it was measuring the
trade's own hour, not an environment.

drv_q = oi[hh] < oi[hh-1] AND side*(taker_buy[hh]-taker_sell[hh]) > 0,
where hh is the SWEEP hour. Those sums are only complete at the sweep
hour's CLOSE. So:

  fill_hh == hh        -> the flag uses the bar the fill happened in.
                          NOT causal.
  fill_hh == hh + 1    -> sweep bar closed before the fill bar opened.
                          Causal.
  fill_hh >= hh + 2    -> unambiguously causal.

FIRST VERSION OF THIS FILE WAS WRONG, and the correction is the point.
It used drv_gap_oi ("na" when fill_hh < hh+2) as a proxy for "possibly
same-bar", found the effect concentrated in the "na" group, and declared
the whole family dead. But "na" also covers fill_hh == hh+1, which IS
causal — the sweep bar closes before that fill bar opens. Measuring the
real gap with find_sweep shows the actual distribution:

    gap 1 bar : 86    gap 2-7 bars : 20    gap 0 (same bar) : ZERO

Not one fill lands in the sweep bar. drv_q is causal for every row, and
the "definitely causal" subset was small (n=20) only because it excluded
the 86 rows at gap==1. A no-power comparison was used to kill a family —
the same "cannot measure treated as does not exist" error caught twice
already today.

So this file now splits on the MEASURED gap, and adds the gate that
actually matters instead:

PROSPECTIVE vs BACKFILLED. drv_q was added to the recorder on 2026-08-02
as a survivor of earlier BTC research. Rows before that date were
backfilled, so the flag was chosen with knowledge of them — they are
in-sample for this question no matter how causal the arithmetic is.
Causality and freshness are different failure modes and both have to pass.

Pre-committed reading:
  * effect survives on genuinely prospective rows -> extending drv_q to
    the other eight coins is worth pre-registering
  * effect exists only in backfilled rows -> it is a selection artefact of
    the 2026-08-02 survivor screen, and variant E goes with it
  * too few prospective rows to tell -> say exactly that, and wait
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
OUT = ROOT / "research" / "results" / "drv_q_causality.json"
random.seed(43)


def ci(vals_pairs, n_boot=4000):
    by = defaultdict(list)
    for d, v in vals_pairs:
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


SURVIVOR_EPOCH = "2026-08-02"       # the day drv_q entered the recorder


def main() -> int:
    sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
    import sweep_core as SC
    from shadow_engine import find_sweep
    bars = SC.load_csv(str(ROOT / "research/sweep_failure/.cache/BTCUSDT_1h.csv"))

    rows = []
    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "CLOSED" or r.get("symbol") != "BTC":
                continue
            q = str(r.get("drv_q", ""))
            if q not in ("0", "1"):
                continue                    # "na"/blank: flag never computed
            try:
                ts, R = int(float(r["fill_ts"])), float(r["net_r"])
                px, atr = float(r["entry_px"]), float(r["atr"])
            except (ValueError, TypeError, KeyError):
                continue
            sw = find_sweep(bars, ts, px, atr, float(r.get("pierce_atr") or 0))
            gap = None if sw is None else ts // 3600 - sw[0] // 3600
            rows.append({"ts": ts, "R": R, "q": q == "1", "gap": gap,
                         "seen": str(r.get("first_seen_utc", "") or "")})

    print("§0.66 drv_q：因果性與新鮮度，兩關分開查\n")
    print(f"  BTC 已結算且 drv_q 有值：n={len(rows)}")
    same = sum(1 for r in rows if r["gap"] == 0)
    print(f"  成交落在開掃同一根的筆數：**{same}** "
          f"（gap=1 有 {sum(1 for r in rows if r['gap'] == 1)} 筆）")
    print("  → 開掃那根一律先收盤，**drv_q 對每一列都是因果的**\n")

    fresh = [r for r in rows if r["seen"] >= SURVIVOR_EPOCH]
    stale = [r for r in rows if r["seen"] < SURVIVOR_EPOCH]
    print(f"  真正前瞻（first_seen ≥ {SURVIVOR_EPOCH}）：{len(fresh)}"
          f"｜回填（旗標挑選時已知）：{len(stale)}\n")

    res = {"n": len(rows), "same_bar_fills": same,
           "prospective_n": len(fresh), "backfilled_n": len(stale)}
    print(f"{'子集':<26} {'Q=1 n':>7} {'Q=1 meanR':>11} {'Q=0 n':>7} "
          f"{'Q=0 meanR':>11} {'差':>9} {'Q=1 的日聚類CI':>22}")
    for lab, sub in (("全部", rows),
                     ("**真正前瞻**", fresh),
                     ("回填（in-sample）", stale)):
        q1 = [r for r in sub if r["q"]]
        q0 = [r for r in sub if not r["q"]]
        if not q1 or not q0:
            print(f"{lab:<26} 樣本不足")
            continue
        m1, m0 = st.mean(r["R"] for r in q1), st.mean(r["R"] for r in q0)
        c = ci([(r["ts"] // 86400, r["R"]) for r in q1])
        cs = f"[{c[0]:+.3f},{c[1]:+.3f}]" if c else "—"
        print(f"{lab:<26} {len(q1):7d} {m1:+11.4f} {len(q0):7d} {m0:+11.4f} "
              f"{m1-m0:+9.4f} {cs:>22}")
        res[lab] = {"q1_n": len(q1), "q1_R": round(m1, 4),
                    "q0_n": len(q0), "q0_R": round(m0, 4),
                    "delta": round(m1 - m0, 4),
                    "q1_ci": [round(c[0], 4), round(c[1], 4)] if c else None}

    a = res.get("**真正前瞻**")
    print()
    if not a or min(a["q1_n"], a["q0_n"]) < 15:
        got = f"（Q=1 {a['q1_n']}／Q=0 {a['q0_n']}）" if a else ""
        v = (f"**因果關通過、新鮮度關無法判定**{got}。前瞻樣本太薄，"
             "不足以分辨 drv_q 是真旗標還是 2026-08-02 存活者篩選的產物。"
             "**這正是不能現在擴到九幣的理由**——擴充會用同一批已知資料"
             "重新確認一個從它挑出來的旗標。要等前瞻樣本。")
    elif a["delta"] > 0 and a["q1_ci"] and a["q1_ci"][0] > 0:
        v = (f"**兩關皆過**：前瞻樣本上 Q=1 仍為 {a['q1_R']:+.4f}"
             f"（CI {a['q1_ci']} 離零），較 Q=0 高 {a['delta']:+.4f}。"
             "擴充到九幣值得預註冊。")
    else:
        v = (f"因果關過、前瞻關未過：前瞻差 {a['delta']:+.4f}"
             + (f"、CI {a['q1_ci']}" if a["q1_ci"] else "")
             + " —— 列觀察不列證據。")
    print(f"判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
