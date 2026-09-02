"""Nested-martingale LP ladder: what the allocation shape actually buys you.

The spec under study (operator-supplied, 2026-09-03):

    total T, ratio r, m big intervals, n small bins each (m*n bins over a
    fixed price range, geometric in log price).
    big interval j base    X_j = X * r^(j-1),  X = T(r-1)/(r^m - 1)
    bin k inside interval j: c(j,k) = Y_1 * r^(j+k-2),  Y_1 = X(r-1)/(r^n - 1)

Two things this file answers, both without fitting anything:

1.  ALGEBRA.  c(j,k) = Y_1 * r^(j+k-2) is separable (outer geometric in j
    times inner geometric in k) and therefore NOT monotone in depth: at every
    big boundary the per-bin size drops by r^-(n-2) (26x at r=1.5, n=10)
    between two adjacent bins ~1.4% apart in price. The "sawtooth" IS that
    reset. Net effect versus a monotone ladder with the same endpoints: the
    resets move money OUT of the deep tail (38% vs 66% of capital in the
    deepest 20% of the range) -- the nesting SOFTENS the martingale, but with
    a discontinuity no price level justifies, and a gentler single ratio
    reaches the same softness smoothly.

2.  PATH REPLAY on real BTC 1h (research/sweep_failure/.cache/BTCUSDT_1h.csv).
    A static ladder is placed at every rolling start; over the following window
    each bin fills when price crosses its lower edge and unwinds when price
    crosses back up. Two outcomes are reported, because an LP earns from one
    and loses from the other:

      fee side       capital-weighted traversals -- fees ~ (capital in bin) x
                     (times price crossed the bin) x fee_rate. Deep bins are
                     rarely crossed, so a martingale profile parks its money
                     where the fee clock does not tick.
      inventory side terminal mark-to-market of bins still holding base at the
                     end of the window (the martingale tail).

    Fees are NOT modelled venue-side (we have no per-bin volume/competing
    liquidity data); `--fee-bps` only converts traversals into a comparable
    yield so the two sides can be put on one ruler.

Profiles compared on identical paths and identical total capital:
    nested     the spec above
    single_mf  monotone ladder with the SAME money-weighted fill price -- the
               like-for-like control: same average bet depth, no sawtooth
    single_ep  monotone ladder with the same first/last bin ratio
    uniform    equal capital per bin
    inverse    geometric the other way (most capital at the top of the range)

single_mf's ratio is solved for a 50% range, so keep --drop 0.5 when using it.

Usage:
    python research/lp_ladder/nested_martingale.py                 # table + replay
    python research/lp_ladder/nested_martingale.py --no-replay     # algebra only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "sweep_failure" / ".cache" / "BTCUSDT_1h.csv"


# ---------------------------------------------------------------- allocation

def nested_alloc(T: float, r: float, m: int, n: int) -> np.ndarray:
    """Capital per bin, spec order (bin 1 = shallowest)."""
    X = T * (r - 1) / (r ** m - 1)
    Y1 = X * (r - 1) / (r ** n - 1)
    return np.array([Y1 * r ** (j + k - 2)
                     for j in range(1, m + 1) for k in range(1, n + 1)])


def geometric(T: float, rho: float, N: int) -> np.ndarray:
    w = rho ** np.arange(N)
    return T * w / w.sum()


def single_ep_alloc(T: float, r: float, m: int, n: int) -> np.ndarray:
    """Monotone ladder matched to the nested ENDPOINTS (first/last bin ratio)."""
    N = m * n
    return geometric(T, r ** ((m + n - 2) / (N - 1)), N)


def single_mf_alloc(T: float, r: float, m: int, n: int,
                    p_hi: float = 100_000, p_lo: float = 50_000) -> np.ndarray:
    """Monotone ladder matched to the nested profile's MONEY-WEIGHTED FILL.

    The economically comparable control: same total, same range, same average
    entry price -- so any difference in the replay is the shape, not the depth
    of the average bet. rho solved by bisection.
    """
    N = m * n
    mid = np.sqrt(grid(p_hi, p_lo, N)[:-1] * grid(p_hi, p_lo, N)[1:])
    target = float((nested_alloc(T, r, m, n) * mid).sum() / T)

    def avg(rho):
        c = geometric(T, rho, N)
        return float((c * mid).sum() / T)

    lo, hi = 1.0, 3.0
    for _ in range(80):
        midr = (lo + hi) / 2
        if avg(midr) > target:      # too shallow -> steepen
            lo = midr
        else:
            hi = midr
    return geometric(T, (lo + hi) / 2, N)


def uniform_alloc(T: float, r: float, m: int, n: int) -> np.ndarray:
    N = m * n
    return np.full(N, T / N)


def inverse_alloc(T: float, r: float, m: int, n: int) -> np.ndarray:
    return single_ep_alloc(T, r, m, n)[::-1]


PROFILES = {"nested": nested_alloc, "single_mf": single_mf_alloc,
            "single_ep": single_ep_alloc, "uniform": uniform_alloc,
            "inverse": inverse_alloc}


def grid(p_hi: float, p_lo: float, N: int) -> np.ndarray:
    """N+1 edges, equal in log price (what the spec's table actually does)."""
    return np.exp(np.linspace(np.log(p_hi), np.log(p_lo), N + 1))


# ------------------------------------------------------------------ algebra

def describe(T, r, m, n, p_hi, p_lo):
    N = m * n
    edges = grid(p_hi, p_lo, N)
    mid = np.sqrt(edges[:-1] * edges[1:])
    print(f"\n=== allocation shape  T={T:,.0f}  r={r}  m={m}  n={n}  "
          f"range {p_hi:,.0f} -> {p_lo:,.0f} ({N} bins) ===")
    head = f"{'profile':8s} {'sum':>10s} {'bin1':>8s} {'binN':>9s} " \
           f"{'last/first':>10s} {'bot20%cap':>10s} {'avgFill':>9s} {'jumps':>6s}"
    print(head)
    for name, fn in PROFILES.items():
        c = fn(T, r, m, n)
        bot = c[int(N * 0.8):].sum() / c.sum()          # deepest 20% of bins
        avg_fill = (c * mid).sum() / c.sum()            # money-weighted entry
        # count adjacent pairs where capital DROPS as price falls
        jumps = int((np.diff(c) < -1e-9).sum())
        print(f"{name:8s} {c.sum():10,.0f} {c[0]:8.1f} {c[-1]:9,.1f} "
              f"{c[-1]/c[0]:10.1f} {bot:10.1%} {avg_fill:9,.0f} {jumps:6d}")
    c = nested_alloc(T, r, m, n)
    print(f"\nsawtooth: at every big boundary the next bin is "
          f"{c[n]/c[n-1]:.3f}x the previous one "
          f"(= r^-(n-2) = {r ** -(n - 2):.3f}), while the price gap between "
          f"those two bins is {(mid[n-1]/mid[n] - 1):.2%}.")
    print(f"deepest single bin holds {c[-1]/T:.1%} of T; "
          f"the deepest big interval holds {c[-n:].sum()/T:.1%}.")
    print(f"money-weighted average fill (if fully traversed) "
          f"{(c*mid).sum()/c.sum():,.0f} = {(c*mid).sum()/c.sum()/p_lo - 1:.1%} "
          f"above the range floor.")


# ------------------------------------------------------------------- replay

def load_btc():
    ts, o, h, l, cl = [], [], [], [], []
    with open(CACHE, newline="") as fh:
        for row in csv.DictReader(fh):
            ts.append(int(row["time"])); o.append(float(row["open"]))
            h.append(float(row["high"])); l.append(float(row["low"]))
            cl.append(float(row["close"]))
    return (np.array(ts), np.array(o), np.array(h), np.array(l), np.array(cl))


def replay_one(low, high, close, edges, alloc, fee_bps):
    """One ladder over one window. Returns (fee_yield, inventory_pnl, deployed).

    bin i owns [edges[i+1], edges[i]]; it is BUY-filled the first time the bar
    low pierces edges[i+1] and unwound when a later bar high regains edges[i].
    Every completed down-up traversal turns the bin's capital over once, which
    is the notional the fee is charged on.
    """
    N = len(alloc)
    lo_edge, hi_edge = edges[1:], edges[:-1]
    filled = np.zeros(N, bool)
    entry = np.zeros(N)
    traversals = np.zeros(N)
    for t in range(len(close)):
        hit = (~filled) & (low[t] <= lo_edge)
        if hit.any():
            filled |= hit
            # a range order fills across the bin: average ~ geometric mid
            entry[hit] = np.sqrt(lo_edge[hit] * hi_edge[hit])
        back = filled & (high[t] >= hi_edge)
        if back.any():
            traversals[back] += 1
            filled[back] = False
    fee = (alloc * traversals).sum() * fee_bps / 1e4
    px = close[-1]
    inv = (alloc[filled] * (px / entry[filled] - 1)).sum() if filled.any() else 0.0
    touched = filled | (traversals > 0)
    return fee, inv, alloc[touched].sum(), (alloc * traversals).sum()


def replay(T, r, m, n, drop, window_days, step_days, fee_bps):
    ts, o, h, l, c = load_btc()
    N = m * n
    bars = window_days * 24
    step = step_days * 24
    starts = range(0, len(c) - bars, step)
    out = {k: {"fee": [], "inv": [], "dep": [], "turn": []} for k in PROFILES}
    maxdrop = []
    for s in starts:
        p0 = c[s]
        edges = grid(p0, p0 * (1 - drop), N)
        w = slice(s + 1, s + 1 + bars)
        maxdrop.append(1 - l[w].min() / p0)
        for name, fn in PROFILES.items():
            fee, inv, dep, turn = replay_one(l[w], h[w], c[w], edges,
                                             fn(T, r, m, n), fee_bps)
            out[name]["fee"].append(fee / T)
            out[name]["inv"].append(inv / T)
            out[name]["dep"].append(dep / T)
            out[name]["turn"].append(turn / T)
    print(f"\n=== path replay: BTC 1h, {len(list(starts))} rolling starts "
          f"(every {step_days}d), {window_days}d windows, "
          f"range = spot -> spot-{drop:.0%}, fee {fee_bps}bps/traversal ===")
    print(f"{'profile':8s} {'feeYield':>9s} {'invPnL':>9s} {'net':>9s} "
          f"{'net p5':>9s} {'net med':>9s} {'deployed':>9s} {'fee/risk':>9s}")
    for name in PROFILES:
        d = out[name]
        fee, inv = np.array(d["fee"]), np.array(d["inv"])
        net = fee + inv
        risk = -np.percentile(net, 5)
        print(f"{name:8s} {fee.mean():9.3%} {inv.mean():9.3%} {net.mean():9.3%} "
              f"{np.percentile(net, 5):9.3%} {np.median(net):9.3%} "
              f"{np.mean(d['dep']):9.1%} "
              f"{(fee.mean()/risk if risk > 0 else float('nan')):9.2f}")
    print("\nfeeYield/invPnL/net are fractions of TOTAL capital over the window "
          "(not annualised). fee/risk = mean fee yield per unit of 5th-pct loss.")

    md = np.array(maxdrop)
    print(f"\nhow deep the window actually went (max drop from the start "
          f"price): median {np.median(md):.1%}, p75 {np.percentile(md,75):.1%}, "
          f"p95 {np.percentile(md,95):.1%}, max {md.max():.1%} -- bins below "
          f"that never fill, and idle capital earns nothing.")
    yr = 365 / window_days
    print(f"\n{'profile':10s} {'idle':>7s} {'turnover':>9s} {'needFeeAPR':>11s}"
          f" {'beFee_bps':>10s}")
    for name in PROFILES:
        d = out[name]
        dep = float(np.mean(d["dep"]))
        inv = float(np.mean(d["inv"]))
        turn = float(np.mean(d["turn"]))
        need = (-inv / dep) * yr if dep > 0 else float("nan")
        # break-even fee charged per bin traversal: turnover * f = -inv
        be = (-inv / turn) * 1e4 if turn > 0 else float("nan")
        print(f"{name:10s} {1-dep:7.1%} {turn:9.2f} {need:11.1%} {be:10.0f}")
    print("idle = capital never touched by price in the window. turnover = "
          "capital-weighted bin traversals (the fee clock, in units of T). "
          "needFeeAPR = fee yield ON DEPLOYED capital needed to offset the mean "
          "inventory loss. beFee_bps = fee that would have to be charged on "
          "EVERY bin traversal to break even -- compare with the venue's actual "
          "tier (5 / 30 / 100 bps). That comparison is the whole decision.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=float, default=100_000)
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p-hi", type=float, default=100_000)
    ap.add_argument("--p-lo", type=float, default=50_000)
    ap.add_argument("--drop", type=float, default=0.5)
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    describe(a.total, a.r, a.m, a.n, a.p_hi, a.p_lo)
    if not a.no_replay:
        replay(a.total, a.r, a.m, a.n, a.drop, a.window_days, a.step_days, a.fee_bps)


if __name__ == "__main__":
    main()
