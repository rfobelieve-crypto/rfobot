# -*- coding: utf-8 -*-
"""Cross-sectional crypto — the one edge family this project has never tested.

Every one of the ~19 documented NO-GOs in TODO.md is the same shape: predict
ONE asset's own future direction from its own history (V7, cancel flow, WQ101,
DVOL, subhourly, and #3 itself, which runs per-coin independently). Ranking
coins AGAINST EACH OTHER is a different problem: it never needs a view on
where the market goes, only on who outperforms whom, and it is dollar-neutral
by construction.

PRE-REGISTERED (written before running):
  signal      past return over lookback L (simple, no tuning knobs)
  portfolio   long the top K, short the bottom K, equal weight, dollar-neutral
  rebalance   every H hours; hold exactly H
  grid        L in {24, 72, 168}h x H in {24, 72, 168}h — ALL NINE reported,
              no cell dropped. Positive spread = momentum, negative = reversal;
              a family is only interesting if the SIGN IS CONSISTENT across
              the grid, since one cell out of nine at p<0.05 is expected.
  costs       measured turnover per rebalance x taker bps, both legs
  decision    this is a SCREEN. Anything that survives earns a pre-registered
              walk-forward, never a deployment (mistake.md 2026-06-02).

KNOWN BIAS, stated up front: the 29-coin universe is today's liquid set, so
coins that died are absent. Survivorship inflates cross-sectional MOMENTUM
(the dead ones were the losers you would have been short... or long) and the
direction of the bias depends on the leg. Treat a momentum result with far
more suspicion than a reversal result here.

Run: python research/cross_sectional_crypto.py
Out: research/results/cross_sectional_crypto.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "research/sweep_failure/.cache"
OUT = ROOT / "research/results/cross_sectional_crypto.json"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX",
         "TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
         "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
         "SAND", "AXS"]
LOOKBACKS = [24, 72, 168]
HOLDS = [24, 72, 168]
K = 6                      # top/bottom sixth of 29
TAKER_BPS = 5.0


def load_panel() -> pd.DataFrame:
    cols = {}
    for c in COINS:
        p = CACHE / f"{c}USDT_1h.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        df = df.drop_duplicates(subset=df.columns[0], keep="last")
        s = pd.Series(df["close"].astype(float).values,
                      index=pd.to_datetime(df.iloc[:, 0].astype("int64"), unit="s"))
        cols[c] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols).sort_index()
    return panel.dropna(thresh=int(0.8 * len(cols)))


def boot_t(x: np.ndarray, nb: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    m = x.mean()
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)])
    sd = x.std(ddof=1)
    t = m / (sd / math.sqrt(len(x))) if sd > 0 else 0.0
    return m, t, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_cell(panel: pd.DataFrame, L: int, H: int) -> dict:
    px = panel
    past = px / px.shift(L) - 1.0
    fwd = px.shift(-H) / px - 1.0
    spreads, turnovers = [], []
    prev_long, prev_short = set(), set()
    for i in range(L, len(px) - H, H):
        p = past.iloc[i].dropna()
        f = fwd.iloc[i]
        p = p[f.reindex(p.index).notna()]
        if len(p) < 2 * K + 4:
            continue
        order = p.sort_values(ascending=False)
        longs, shorts = set(order.index[:K]), set(order.index[-K:])
        spreads.append(f[list(longs)].mean() - f[list(shorts)].mean())
        # turnover as a fraction of gross exposure that must be traded
        chg = len(longs - prev_long) + len(shorts - prev_short)
        turnovers.append(chg / (2.0 * K))
        prev_long, prev_short = longs, shorts
    if len(spreads) < 30:
        return {"n": len(spreads)}
    s = np.array(spreads)
    turn = float(np.mean(turnovers))
    # both legs pay taker on the traded fraction, entry and exit
    cost = 2.0 * turn * 2.0 * TAKER_BPS / 1e4
    net = s - cost
    m, t, lo, hi = boot_t(net)
    gm = s.mean()
    per_year = 365 * 24 / H
    return {"n": len(s), "gross_bps": gm * 1e4, "cost_bps": cost * 1e4,
            "net_bps": m * 1e4, "t": t, "ci_lo_bps": lo * 1e4,
            "ci_hi_bps": hi * 1e4, "turnover": turn,
            "ann_net_pct": ((1 + m) ** per_year - 1) * 100,
            "hit_rate": float((net > 0).mean() * 100)}


def main() -> int:
    panel = load_panel()
    print(f"panel: {panel.shape[1]} coins x {len(panel)} hourly bars  "
          f"{panel.index[0]:%Y-%m-%d} -> {panel.index[-1]:%Y-%m-%d}")
    print("\nlong top-6 / short bottom-6 by past return, dollar-neutral")
    print(f"{'L(h)':>5}{'H(h)':>6}{'n reb':>7}{'gross':>9}{'cost':>8}"
          f"{'net bps':>9}{'t':>7}{'CI95 (bps)':>20}{'ann%':>9}{'turn':>7}")
    res = {}
    signs = []
    for L in LOOKBACKS:
        for H in HOLDS:
            r = run_cell(panel, L, H)
            res[f"L{L}_H{H}"] = r
            if "net_bps" not in r:
                print(f"{L:>5}{H:>6}{r['n']:>7}   (too few rebalances)")
                continue
            signs.append(1 if r["net_bps"] > 0 else -1)
            print(f"{L:>5}{H:>6}{r['n']:>7}{r['gross_bps']:>+9.1f}"
                  f"{r['cost_bps']:>8.1f}{r['net_bps']:>+9.1f}{r['t']:>+7.2f}"
                  f"  [{r['ci_lo_bps']:>+7.1f},{r['ci_hi_bps']:>+7.1f}]"
                  f"{r['ann_net_pct']:>+9.1f}{r['turnover']:>7.2f}")
    pos = sum(1 for s in signs if s > 0)
    print(f"\nsign consistency: {pos}/{len(signs)} cells positive "
          f"({'momentum' if pos > len(signs)/2 else 'reversal'} leaning)")
    sig = [k for k, v in res.items()
           if "ci_lo_bps" in v and (v["ci_lo_bps"] > 0 or v["ci_hi_bps"] < 0)]
    print(f"cells with CI excluding 0: {len(sig)}/{len(signs)}"
          + (f"  -> {', '.join(sig)}" if sig else ""))
    print("\nREAD: 9 cells means ~0.45 false positives at p<0.05 by chance; "
          "only a CONSISTENT sign across the grid is evidence. Survivorship "
          "in the 29-coin universe biases the momentum reading specifically.")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
