# -*- coding: utf-8 -*-
"""Strategy #3 under two pieces of published quant mathematics, actually computed.

Both address questions the plain bootstrap cannot answer.

1) DEFLATED SHARPE RATIO  (Bailey & Lopez de Prado, 2014)
   A bootstrap CI answers "is this Sharpe distinguishable from zero given
   THIS sample". It cannot see that the strategy was chosen after many
   trials. Under the null, the MAXIMUM Sharpe over N independent trials is
   already positive; the expected max is

       SR0 = sqrt(V) * [ (1-gamma)*Z^-1(1 - 1/N) + gamma*Z^-1(1 - 1/(N*e)) ]

   with gamma = Euler-Mascheroni, and V the variance of trial Sharpes
   (under the null, V = 1/T). The deflated Sharpe is then

       DSR = Phi( (SR - SR0) * sqrt(T-1)
                  / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) )

   which also corrects for non-normal returns (skew g3, kurtosis g4) —
   negative skew and fat tails make a given Sharpe less trustworthy, which
   is exactly the shape of a stop-loss strategy's return distribution.

   T matters enormously here: the 9-coin sample has 6995 trades but they
   are not independent (day-clustered VIF ~ 3), so both readings are
   reported and the clustered one is the honest one.

2) ANYTIME-VALID CONFIDENCE SEQUENCE  (Howard et al. 2021, Gaussian-mixture
   boundary; the Robbins/Ville line of work)
   Gate F currently uses a fixed-n test: fix the sample size, look once.
   Looking early inflates type-I error, which is why the runway is dead
   time. A confidence sequence holds SIMULTANEOUSLY at every t:

       P( exists t : |mean_t - mu| >= u(V_t)/t ) <= alpha
       u(V) = sqrt( (V + rho) * log( (V + rho) / (rho * alpha^2) ) )

   so the gate can be checked every week, forever, and stop the moment the
   lower bound clears zero. The price is a wider band at any fixed t (~1.9x
   here); the gain is that a stronger-than-estimated edge is caught early
   instead of waiting out the whole fixed budget.

   Applied to DAILY aggregates, which handles the cross-coin correlation
   structurally (one day = one observation) rather than by a VIF fudge.

Run: python research/sweep_failure/dsr_and_confseq.py
Out: research/results/sweep_dsr_confseq.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_dsr_confseq.json"
CACHE = HERE / ".cache"
SCEN_A = {"entry": 7.0, "texit": 3.0, "sexit": 10.0}
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
GAMMA = 0.5772156649015329          # Euler-Mascheroni
ALPHA = 0.05

# Documented strategy/feature-family trials in this repo (mistake.md + TODO):
# WQ101, liquidity proxies, orderbook features, resistance map, DVOL,
# options positioning, RL joint exit, meta-labeling exit, exit-variant sweep,
# asymmetric cutoffs, long-horizon trend, on-chain/ETF, churn overlay,
# subhourly, cancel playbooks, intra-bar volume, sweep-failure itself.
N_TRIALS_BASE = 17


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation (|err| < 1.15e-9) — no scipy needed."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def moments(x: list[float]) -> tuple[float, float, float, float]:
    n = len(x)
    m = sum(x) / n
    v = sum((y - m) ** 2 for y in x) / (n - 1)
    s = math.sqrt(v)
    g3 = sum((y - m) ** 3 for y in x) / n / s ** 3
    g4 = sum((y - m) ** 4 for y in x) / n / s ** 4
    return m, s, g3, g4


def expected_max_sr(n_trials: int, T: float) -> float:
    """E[max SR] over n_trials independent null strategies of length T."""
    sqrtV = 1.0 / math.sqrt(T)          # null sampling sd of an SR estimate
    return sqrtV * ((1 - GAMMA) * norm_ppf(1 - 1.0 / n_trials)
                    + GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def dsr(sr: float, sr0: float, T: float, g3: float, g4: float) -> float:
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    return norm_cdf((sr - sr0) * math.sqrt(T - 1) / denom)


def min_trl(sr: float, sr0: float, g3: float, g4: float,
            conf: float = 0.95) -> float:
    """Observations needed for DSR >= conf, holding SR/moments fixed."""
    if sr <= sr0:
        return float("inf")
    z = norm_ppf(conf)
    return 1 + (1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) * (z / (sr - sr0)) ** 2


def conf_seq_radius(t: int, sigma: float, rho: float, alpha: float) -> float:
    """Gaussian-mixture anytime-valid radius on the mean of t observations."""
    V = t * sigma ** 2
    return math.sqrt((V + rho) * math.log((V + rho) / (rho * alpha ** 2))) / t


def main() -> int:
    trades = []
    for s in CORE9:
        for fill_ts, _, r, lvl, atr, stopped, _pierce in SC.backtest_symbol(
                SC.load_csv(str(CACHE / f"{s}USDT_1h.csv"))):
            legs = SCEN_A["entry"] + (SCEN_A["sexit"] if stopped else SCEN_A["texit"])
            trades.append((fill_ts, r - legs / 1e4 * lvl / (SC.DIS * atr)))
    rs = [r for _, r in trades]
    m, sd, g3, g4 = moments(rs)
    sr = m / sd
    T_raw = len(rs)

    byd = defaultdict(float)
    for ts, r in trades:
        byd[datetime.fromtimestamp(ts, tz=timezone.utc).date()] += r
    daily = [byd[d] for d in sorted(byd)]
    dm, dsd, dg3, dg4 = moments(daily)
    n_days = len(daily)
    T_eff = T_raw / 3.04                       # measured day-clustered VIF

    print("=" * 76)
    print("  STRATEGY #3 — DEFLATED SHARPE + ANYTIME-VALID CONFIDENCE SEQUENCE")
    print("=" * 76)
    print(f"  per-trade: n={T_raw}  mean={m:+.5f}R  sd={sd:.4f}  "
          f"SR={sr:+.5f}  skew={g3:+.2f}  kurt={g4:.2f}")
    print(f"  daily    : n={n_days}  mean={dm:+.4f}R/day  sd={dsd:.4f}  "
          f"skew={dg3:+.2f}  kurt={dg4:.2f}")
    print(f"  negative skew / fat tails inflate the DSR denominator — a "
          f"stop-loss strategy is penalised here, correctly.")

    print(f"\n  [1] DEFLATED SHARPE — is the edge real given N trials?")
    print(f"  {'N trials':>9}{'SR0 (raw T)':>13}{'DSR raw':>10}"
          f"{'SR0 (clustered)':>17}{'DSR clustered':>15}")
    rows = {}
    for N in (5, 10, N_TRIALS_BASE, 30, 50, 100):
        s0r = expected_max_sr(N, T_raw)
        s0c = expected_max_sr(N, T_eff)
        dr = dsr(sr, s0r, T_raw, g3, g4)
        dc = dsr(sr, s0c, T_eff, g3, g4)
        rows[N] = {"sr0_raw": s0r, "dsr_raw": dr,
                   "sr0_clustered": s0c, "dsr_clustered": dc}
        mark = "  <- this repo" if N == N_TRIALS_BASE else ""
        print(f"  {N:>9}{s0r:>13.5f}{dr:>10.3f}{s0c:>17.5f}{dc:>15.3f}{mark}")
    print(f"  (DSR = P(true SR > 0 | N trials, skew, kurtosis). >0.95 = deploy-grade.)")

    s0c = expected_max_sr(N_TRIALS_BASE, T_eff)
    need = min_trl(sr, s0c, g3, g4)
    print(f"\n  [2] MIN TRACK RECORD LENGTH for DSR>=0.95 at N={N_TRIALS_BASE}"
          f" (clustered):")
    if math.isinf(need):
        print(f"      UNREACHABLE at this SR — observed SR {sr:.5f} <= "
              f"E[max null SR] {s0c:.5f}")
    else:
        print(f"      {need:,.0f} effective observations "
              f"(have {T_eff:,.0f}) -> need {need*3.04/229/12:.1f} more years "
              f"at the 9-coin rate")

    print(f"\n  [3] ANYTIME-VALID CONFIDENCE SEQUENCE on DAILY PnL")
    print(f"      (correlation handled structurally: 1 day = 1 observation)")
    rho = n_days * dsd ** 2
    print(f"  {'days':>7}{'CS lower bound':>17}{'fixed-n lower':>16}{'ratio':>8}")
    cs_rows = {}
    for t in (n_days, 1500, 2000, 3000, 4000):
        rad_cs = conf_seq_radius(t, dsd, rho, ALPHA)
        rad_fx = 1.96 * dsd / math.sqrt(t)
        cs_rows[t] = {"cs_lo": dm - rad_cs, "fixed_lo": dm - rad_fx}
        tag = "  <- today" if t == n_days else ""
        print(f"  {t:>7}{dm - rad_cs:>+17.4f}{dm - rad_fx:>+16.4f}"
              f"{rad_cs/rad_fx:>8.2f}{tag}")
    # when does the CS clear zero at the observed daily mean?
    t = n_days
    while t < 20000 and dm - conf_seq_radius(t, dsd, rho, ALPHA) <= 0:
        t += 10
    yrs = (t - n_days) / 365.0
    print(f"\n      at the observed +{dm:.4f}R/day the CS clears 0 at "
          f"t={t} days -> {yrs:.1f} more years")
    for mult, label in ((1.5, "50% hotter"), (2.0, "2x hotter")):
        t2 = n_days
        while t2 < 20000 and dm * mult - conf_seq_radius(t2, dsd, rho, ALPHA) <= 0:
            t2 += 10
        print(f"      if forward runs {label:<12} -> t={t2} days "
              f"({(t2-n_days)/365.0:.1f} more years)")

    OUT.write_text(json.dumps({
        "per_trade": {"n": T_raw, "mean": m, "sd": sd, "sr": sr,
                      "skew": g3, "kurt": g4, "T_effective": T_eff},
        "daily": {"n": n_days, "mean": dm, "sd": dsd},
        "dsr": {str(k): v for k, v in rows.items()},
        "min_trl_effective_obs": need,
        "conf_seq": {str(k): v for k, v in cs_rows.items()},
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
