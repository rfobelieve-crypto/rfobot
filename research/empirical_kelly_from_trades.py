"""Empirical Kelly from ACTUAL closed trades (paper archive + live OKX).

Motivated by the @RuujSs "math that runs every hedge fund" piece (2026-06-05):
Kelly f* = mu/sigma^2, use fractional Kelly for estimation error. This computes
f* from the REAL per-trade net returns (unlevered), not the assumed mu=5%/sigma=30%
in CLAUDE.md, and compares against the leverage ladder.

net_pct = unlevered per-trade return (price move in trade direction - round-trip
cost). Leverage-independent, so paper (1x) and live (10x) pool cleanly. admin_heal
exits (zeroed, not real outcomes) are excluded.

HONESTY: n is tiny (~18). Kelly is hypersensitive to mu/sigma at small n, so the
bootstrap CI on f* will be very wide — which is itself the article's argument for
quarter-Kelly. Point estimate is a sanity check on CLAUDE.md's 0.56x, NOT a
mandate to change live sizing.
"""
from __future__ import annotations

import numpy as np
from shared.db import get_db_conn


def fetch_net_returns():
    rows = []
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT direction, entry_tier, net_pct, exit_reason, 'paper' AS src "
                "FROM v7_paper_positions_archive "
                "WHERE status='CLOSED' AND exit_reason <> 'admin_heal'")
            rows += list(cur.fetchall())
            cur.execute(
                "SELECT direction, entry_tier, net_pct, exit_reason, 'live' AS src "
                "FROM v7_okx_positions "
                "WHERE status IN ('CLOSED','DEMOTED') AND exit_reason <> 'admin_heal'")
            rows += list(cur.fetchall())
    finally:
        conn.close()
    return rows


def exact_kelly(r: np.ndarray) -> float:
    """argmax_f mean(log(1 + f*r)) on the empirical sample (handles asymmetry).
    f is bounded so 1+f*r > 0 for the worst loss."""
    worst = r.min()
    f_hi = (0.999 / -worst) if worst < 0 else 50.0  # cap before ruin on worst trade
    fs = np.linspace(0.0, min(f_hi, 50.0), 4000)
    g = np.array([np.mean(np.log1p(f * r)) for f in fs])
    return float(fs[int(np.argmax(g))])


def kelly_stats(r: np.ndarray) -> dict:
    mu, sig = r.mean(), r.std(ddof=1)
    f_approx = mu / sig**2 if sig > 0 else float("nan")
    f_exact = exact_kelly(r)
    wr = (r > 0).mean() * 100
    return dict(n=len(r), mu=mu, sig=sig, wr=wr,
                f_approx=f_approx, f_exact=f_exact)


def boot_f(r: np.ndarray, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    fs = []
    for _ in range(n):
        s = rng.choice(r, len(r), replace=True)
        if s.std(ddof=1) > 0:
            fs.append(s.mean() / s.var(ddof=1))
    fs = np.array(fs)
    return np.percentile(fs, 2.5), np.percentile(fs, 50), np.percentile(fs, 97.5)


def growth_curve(r: np.ndarray, levs):
    """Geometric growth rate per trade at each leverage L (article's G(f))."""
    out = []
    for L in levs:
        if np.all(1 + L * r > 0):
            out.append((L, float(np.mean(np.log1p(L * r)))))
        else:
            out.append((L, float("nan")))  # ruin: some trade wipes the book
    return out


def report(label, r):
    if len(r) < 2:
        print(f"\n## {label}: n={len(r)} too small")
        return
    s = kelly_stats(r)
    lo, med, hi = boot_f(r)
    print(f"\n## {label}")
    print(f"  n={s['n']}  win%={s['wr']:.0f}  mu={s['mu']*100:+.3f}%/trade  "
          f"sigma={s['sig']*100:.3f}%  (mu/sigma per-trade Sharpe={s['mu']/s['sig']:.3f})")
    print(f"  Kelly f* (mu/sigma^2 approx): {s['f_approx']:.2f}x")
    print(f"  Kelly f* (exact, geo-growth): {s['f_exact']:.2f}x")
    print(f"  Half-Kelly: {s['f_exact']/2:.2f}x   Quarter-Kelly: {s['f_exact']/4:.2f}x")
    print(f"  Bootstrap f* (mu/sigma^2)  95% CI: [{lo:.2f}, {hi:.2f}]  median {med:.2f}x")
    print(f"  -> CI width {hi-lo:.1f}x leverage units = estimate is "
          f"{'UNRELIABLE' if (hi-lo) > 5 else 'usable'} at this n")


def main():
    rows = fetch_net_returns()
    r_all = np.array([float(x["net_pct"]) for x in rows])
    n_paper = sum(1 for x in rows if x["src"] == "paper")
    n_live = sum(1 for x in rows if x["src"] == "live")
    print(f"trades: {len(rows)} total ({n_paper} paper + {n_live} live, "
          f"admin_heal excluded)")
    print(f"net_pct per trade: {[round(x*100,2) for x in sorted(r_all)]}  (%)")

    report("ALL trades (paper+live, v7.1 strategy)", r_all)

    strong = np.array([float(x["net_pct"]) for x in rows if x["entry_tier"] == "Strong"])
    mod = np.array([float(x["net_pct"]) for x in rows if x["entry_tier"] == "Moderate"])
    report("STRONG tier only", strong)
    report("MODERATE tier only", mod)

    # Growth curve vs ladder
    print("\n## Geometric growth per trade G(L) vs leverage ladder")
    print("   (peak = empirical Kelly; NaN = a real trade in the sample would")
    print("    have wiped the account at that leverage)")
    levs = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    for L, g in growth_curve(r_all, levs):
        tag = ""
        for name, lv in [("Stage4a", 1.0), ("Stage4b", 1.2), ("Stage4c", 1.5),
                         ("Stage4d cap", 2.0), ("Stage3 NOW", 10.0)]:
            if abs(L - lv) < 1e-6:
                tag = f"  <- {name}"
        gtxt = f"{g*1e4:+.0f} bps/trade" if not np.isnan(g) else "RUIN (NaN)"
        print(f"   L={L:5.1f}x : G={gtxt}{tag}")

    print("\n## vs CLAUDE.md assumption")
    print("   CLAUDE.md leverage ladder assumes f* = mu/sigma^2 ~ 0.56x "
          "(mu=+5%, sigma=30% on some horizon).")
    s = kelly_stats(r_all)
    print(f"   Empirical per-trade: mu={s['mu']*100:+.2f}%, sigma={s['sig']*100:.2f}% "
          f"-> f*~{s['f_approx']:.2f}x (approx) / {s['f_exact']:.2f}x (exact)")


if __name__ == "__main__":
    main()
