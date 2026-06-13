# V7.1 Exit Logic Variant Backtest

Generated: 2026-05-16 12:44 UTC

OOS span: 2025-11-27 04:00 → 2026-04-30 03:00  (3696 bars)
v7.1 signals (Strong+Moderate, UP+DOWN): 1112  (Strong 417, Moderate 695)
Signal regime mix (at signal bar): {'CHOPPY': 721, 'BEAR': 198, 'BULL': 193}
Disaster stop = max(3% , 4×ATR14)

## Variant comparison

| Variant | n | WR | gross/tr (bps) | net/tr (bps) | |ret| (bps) | hold (h) | Sharpe | MaxDD | Cum net |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| V0  fixed 4h | 1112 | 60.9% | +38.0 | +30.0 | 82 | 4.0 | 5.36 | 7.07% | +99.5% |
| V1  fixed 8h | 1112 | 58.3% | +43.0 | +35.0 | 103 | 8.0 | 3.65 | 11.68% | +63.5% |
| V2  fixed 12h | 1112 | 56.2% | +48.2 | +40.2 | 125 | 12.0 | 3.64 | 17.00% | +66.2% |
| V3  fixed 24h | 1112 | 60.2% | +71.0 | +63.0 | 181 | 24.0 | 3.36 | 16.36% | +60.2% |
| V4  8h + opp-Strong | 1112 | 60.4% | +50.4 | +42.4 | 103 | 7.4 | 4.59 | 10.04% | +99.2% |
| V5  8h + opp-Strong + wide stop | 1112 | 60.4% | +49.1 | +41.1 | 104 | 7.3 | 4.54 | 10.92% | +97.5% |
| V6  12h + opp-Strong | 1112 | 61.6% | +61.2 | +53.2 | 123 | 10.6 | 4.95 | 15.37% | +119.8% |
| V7  pure signal exit (72h cap) | 1112 | 67.1% | +85.2 | +77.2 | 186 | 27.9 | 3.48 | 10.81% | +82.7% |
| V8  pure signal exit + wide stop | 1112 | 66.0% | +79.7 | +71.7 | 186 | 24.1 | 3.78 | 10.20% | +89.7% |

Sharpe = annualised daily-MTM equal-weight portfolio (same as Method C verification). MaxDD / Cum net from the compounded daily MTM equity. net/tr is per-trade after 8bps round-trip fee.

## Win rate by v7.1 tier

| Variant | Strong n | Strong WR | Strong net | Moderate n | Moderate WR | Moderate net |
|---|--:|--:|--:|--:|--:|--:|
| V0  fixed 4h | 417 | 65.7% | +48.4 | 695 | 58.0% | +18.9 |
| V1  fixed 8h | 417 | 60.2% | +53.5 | 695 | 57.1% | +23.9 |
| V2  fixed 12h | 417 | 60.2% | +62.0 | 695 | 53.8% | +27.1 |
| V3  fixed 24h | 417 | 63.3% | +78.2 | 695 | 58.3% | +53.8 |
| V4  8h + opp-Strong | 417 | 61.9% | +60.7 | 695 | 59.6% | +31.5 |
| V5  8h + opp-Strong + wide stop | 417 | 61.9% | +59.5 | 695 | 59.6% | +30.1 |
| V6  12h + opp-Strong | 417 | 65.0% | +72.4 | 695 | 59.6% | +41.6 |
| V7  pure signal exit (72h cap) | 417 | 68.3% | +87.7 | 695 | 66.3% | +71.0 |
| V8  pure signal exit + wide stop | 417 | 67.9% | +89.4 | 695 | 64.9% | +61.1 |

## Exit-reason breakdown & trigger diagnostics

- **V0  fixed 4h**: time_exit 1112 (100.0%, net +30.0bps)
- **V1  fixed 8h**: time_exit 1112 (100.0%, net +35.0bps)
- **V2  fixed 12h**: time_exit 1112 (100.0%, net +40.2bps)
- **V3  fixed 24h**: time_exit 1112 (100.0%, net +63.0bps)
- **V4  8h + opp-Strong**: opp_signal 207 (18.6%, net +91.7bps); time_exit 905 (81.4%, net +31.2bps)
    opposite-signal trigger rate = 18.6%
- **V5  8h + opp-Strong + wide stop**: disaster_stop 24 (2.2%, net -352.9bps); opp_signal 205 (18.4%, net +95.3bps); time_exit 883 (79.4%, net +39.2bps)
    disaster-stop trigger rate = 2.2%  [OK]
    opposite-signal trigger rate = 18.4%
- **V6  12h + opp-Strong**: opp_signal 274 (24.6%, net +92.2bps); time_exit 838 (75.4%, net +40.4bps)
    opposite-signal trigger rate = 24.6%
- **V7  pure signal exit (72h cap)**: opp_signal 914 (82.2%, net +98.6bps); time_exit 198 (17.8%, net -21.5bps)
    opposite-signal trigger rate = 82.2%
- **V8  pure signal exit + wide stop**: disaster_stop 146 (13.1%, net -325.3bps); opp_signal 869 (78.1%, net +109.5bps); time_exit 97 (8.7%, net +330.8bps)
    disaster-stop trigger rate = 13.1%  [OUT OF 1-5% TARGET]
    opposite-signal trigger rate = 78.1%

## Regime breakdown (regime at entry, from is_trending_bull/bear)

### BULL

| Variant | n | WR | net/tr (bps) | Sharpe | MaxDD |
|---|--:|--:|--:|--:|--:|
| V0  fixed 4h | 194 | 60.8% | +17.6 | 4.37 | 3.32% |
| V1  fixed 8h | 194 | 57.2% | +35.7 | 4.46 | 2.72% |
| V2  fixed 12h | 194 | 57.7% | +41.0 | 3.48 | 4.28% |
| V3  fixed 24h | 194 | 50.5% | +8.4 | 0.66 | 13.92% |
| V4  8h + opp-Strong | 194 | 60.3% | +45.6 | 4.94 | 2.72% |
| V5  8h + opp-Strong + wide stop | 194 | 60.3% | +45.6 | 4.94 | 2.72% |
| V6  12h + opp-Strong | 194 | 61.9% | +52.4 | 4.09 | 3.98% |
| V7  pure signal exit (72h cap) | 194 | 68.0% | +87.2 | 2.84 | 10.66% |
| V8  pure signal exit + wide stop | 194 | 66.5% | +66.7 | 3.00 | 7.63% |

### BEAR

| Variant | n | WR | net/tr (bps) | Sharpe | MaxDD |
|---|--:|--:|--:|--:|--:|
| V0  fixed 4h | 194 | 64.9% | +49.8 | 3.26 | 7.69% |
| V1  fixed 8h | 194 | 61.3% | +65.3 | 3.42 | 7.11% |
| V2  fixed 12h | 194 | 67.5% | +94.3 | 3.83 | 8.46% |
| V3  fixed 24h | 194 | 72.2% | +183.1 | 4.50 | 10.76% |
| V4  8h + opp-Strong | 194 | 62.9% | +70.4 | 3.62 | 7.14% |
| V5  8h + opp-Strong + wide stop | 194 | 62.9% | +62.6 | 3.37 | 8.92% |
| V6  12h + opp-Strong | 194 | 69.6% | +101.4 | 3.97 | 8.55% |
| V7  pure signal exit (72h cap) | 194 | 74.7% | +176.2 | 4.36 | 9.50% |
| V8  pure signal exit + wide stop | 194 | 72.2% | +177.6 | 4.61 | 9.27% |

### CHOPPY

| Variant | n | WR | net/tr (bps) | Sharpe | MaxDD |
|---|--:|--:|--:|--:|--:|
| V0  fixed 4h | 724 | 59.8% | +28.0 | 5.40 | 5.69% |
| V1  fixed 8h | 724 | 57.7% | +26.7 | 2.79 | 9.78% |
| V2  fixed 12h | 724 | 52.8% | +25.5 | 3.71 | 8.96% |
| V3  fixed 24h | 724 | 59.5% | +45.4 | 3.54 | 9.43% |
| V4  8h + opp-Strong | 724 | 59.8% | +34.1 | 3.76 | 7.99% |
| V5  8h + opp-Strong + wide stop | 724 | 59.8% | +34.1 | 3.92 | 8.71% |
| V6  12h + opp-Strong | 724 | 59.4% | +40.5 | 5.27 | 7.79% |
| V7  pure signal exit (72h cap) | 724 | 64.8% | +48.1 | 4.36 | 7.50% |
| V8  pure signal exit + wide stop | 724 | 64.2% | +44.7 | 4.72 | 6.02% |

## Decision

**Q1 — does extending fixed horizon (4→8→12→24) monotonically improve net Sharpe?**
- Sharpe 4h→8h→12h→24h: 5.36 → 3.65 → 3.64 → 3.36  → NOT monotone
- net/trade: +30.0 → +35.0 → +40.2 → +63.0 bps

**Q2 — does opposite-signal early exit help or hurt? (V4 vs V1)**
- Sharpe 3.65 → 4.59; net/tr +35.0 → +42.4 bps; WR 58.3% → 60.4%; MaxDD 11.68% → 10.04%

**Q3 — does wide disaster stop cost edge or add insurance? (V5 vs V4)**
- Sharpe 4.59 → 4.54; net/tr +42.4 → +41.1 bps; MaxDD 10.04% → 10.92%; stop fired 2.2% of trades

**Q4 — which variant is robust across regimes?**
- V0  fixed 4h: BULL=4.4  BEAR=3.3  CHOPPY=5.4
- V1  fixed 8h: BULL=4.5  BEAR=3.4  CHOPPY=2.8
- V2  fixed 12h: BULL=3.5  BEAR=3.8  CHOPPY=3.7
- V3  fixed 24h: BULL=0.7  BEAR=4.5  CHOPPY=3.5
- V4  8h + opp-Strong: BULL=4.9  BEAR=3.6  CHOPPY=3.8
- V5  8h + opp-Strong + wide stop: BULL=4.9  BEAR=3.4  CHOPPY=3.9
- V6  12h + opp-Strong: BULL=4.1  BEAR=4.0  CHOPPY=5.3
- V7  pure signal exit (72h cap): BULL=2.8  BEAR=4.4  CHOPPY=4.4
- V8  pure signal exit + wide stop: BULL=3.0  BEAR=4.6  CHOPPY=4.7

**Q5 — clock vs signal: does a pure signal-decay exit beat the time-based variants?**
- V7 (pure signal): Sharpe 3.48, WR 67.1%, net +77.2bps, hold 27.9h, MaxDD 10.81%, cum +82.7%
- vs V0 (clock 4h): Sharpe 5.36, net +30.0bps, MaxDD 7.07%
- vs V4 (8h + opp-Strong): Sharpe 4.59, net +42.4bps, MaxDD 10.04%
- V7 safety-cap (72h) bind rate: 17.8% of trades (low = signal genuinely drives the exit)
- V8 (V7 + wide stop): Sharpe 3.78, net +71.7bps, MaxDD 10.20%, stop fired 13.1%
- **Verdict: clock still wins** (V7 Sharpe 3.48 vs V0 5.36)

### Recommendation

(The line above is the script's mechanical auto-pick. Analyst narrative below.)

### Analyst recommendation

**Risk-adjusted default: V0 — fixed 4h.** Best Sharpe (5.36) AND lowest MaxDD
(7.07%) of all 9 variants; only variant with no regime weakness; and its Sharpe
is the most trustworthy (least trade overlap → least autocorrelation distortion
in the sqrt(365) annualisation).

**If higher absolute return per trade is wanted: V4 (8h + opp-Strong)** —
+42 bps/trade, Sharpe 4.59, MaxDD 10%, regime-balanced. V6 (12h+opp) pushes to
+53 bps/trade but MaxDD 15.4% (aggressive end).

**Do NOT deploy V3 (fixed 24h)** — BULL Sharpe 0.66, structural failure.

### Clock vs signal — answer to "must the exit be time-based?"

V7/V8 test a PURE signal-decay exit (hold until v7.1 gives any opposite
reading; 72h safety cap only; no fixed horizon). Findings:

- **V7 has the best win rate (67.1%) and best net/trade (+77 bps) of every
  variant** — the signal genuinely knows good exit points (its opp-signal
  exits average +99 bps).
- **But V7's Sharpe is only 3.48 — among the worst of the credible variants.**
  The v7.1 signal is a SLOW exit trigger: left alone it produces 27.9h average
  holds. Long, lumpy, heavily-overlapping holds tank risk-adjusted return.
- **V7 still needs a clock.** The 72h cap binds on 17.8% of trades, and those
  cap-bound trades average **−21.5 bps** — i.e. when the signal never decays
  you bleed, and only the clock stops it.
- **V7 fails in BULL (Sharpe 2.8).** In an uptrend v7.1 rarely emits an
  opposite reading, so longs run into the 72h cap — the same stale-position
  failure as V3 (fixed 24h).
- **V8's disaster stop fired on 13.1% of trades** (vs the 1-5% target). On a
  long-hold strategy a "wide" stop stops being rare insurance and becomes an
  active strategy component — a design smell.

**Conclusion:** time is not removable. It does a job the signal cannot —
capping holding length, and capped holds give better Sharpe. The signal is
good at exit QUALITY (when it fires, +99 bps) but bad at SPEED (slow to fire →
long holds). The best result is neither pure-clock nor pure-signal but the
COMBINATION — V4 / V6 — clock for speed, signal for quality. Pure signal exit
(V7) is NOT recommended as the primary exit despite its headline WR.

### Caveats / sample-size warnings

- All Sharpes are WF-OOS BACKTEST figures; live degrades (CLAUDE.md: live WR
  69% → 60%). Trust the RANKING, not absolute levels.
- Long-hold variants (V2/V3/V6/V7/V8) have heavy trade overlap → autocorrelated
  daily returns → sqrt(365) annualisation OVER-states their Sharpe. V7's true
  risk-adjusted return is therefore even worse than the 3.48 shown.
- BULL/BEAR regime sub-samples ≈194 trades each, heavily overlapping; effective
  independent n is much smaller — treat regime Sharpe as indicative.
- 5-month OOS contains no extreme crash; disaster-stop tail value untestable.
- "Opposite" reading uses the rolling-percentile decode tier as the
  confidence proxy; production calibration drift would shift trigger rates.
- Re-running v71_exit_variants.py regenerates the tables above and overwrites
  this narrative section.
