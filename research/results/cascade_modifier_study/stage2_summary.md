# Cascade Modifier Study — Stage 2 Summary

Walk-forward backtest of the cascade confidence modifier on
**3515 OOS bars** (2025-11-17 18:00:00+00:00 → 2026-04-13 04:00:00+00:00).

## Modifier config
- Activation gate: **rolling top 5% of p_cascade** over 500-bar window (warmup 100, cold-start fallback p=0.3)
- Dynamic gate threshold range: [0.060, 0.440], median 0.122
- Boost (aligned): confidence × **1.15** (capped at 100)
- Damp (conflicted): confidence × **0.7**
- Tier thresholds: Strong ≥ 80.0, Moderate ≥ 65.0

## Modifier activity

- Modifier active on **32 bars** (0.9% of backtest)
- Aligned (direction-reg agrees with contrarian): 18
- Conflicted (disagree): 14
- Aligned : Conflicted ratio = **1.29**
- Sanity (inactive bars unchanged tier): **True**

## Overall tier comparison

### Strong tier
| Metric | Original | V2 | Δ |
|---|---|---|---|
| Count | 266 | 265 | -1 (-0.4%) |
| Win rate | 65.8% | 65.3% | -0.51pp |
| Wilson 95% CI | [59.9%, 71.2%] | [59.4%, 70.8%] | — |
| Mean \|ret\| | 0.577% | 0.571% | — |

### Moderate tier
| Metric | Original | V2 | Δ |
|---|---|---|---|
| Count | 130 | 127 | -3 (-2.3%) |
| Win rate | 63.8% | 64.6% | +0.72pp |
| Wilson 95% CI | [55.3%, 71.6%] | [55.9%, 72.3%] | — |
| Mean \|ret\| | 0.545% | 0.563% | — |

## By regime

### CHOPPY (2383 bars)

| Tier | Metric | Original | V2 | Δ |
|---|---|---|---|---|
| Strong | Count | 175 | 172 | -3 |
| Strong | WR | 66.9% | 65.7% | -1.16pp |
| Moderate | Count | 80 | 78 | -2 |
| Moderate | WR | 63.7% | 66.7% | +2.92pp |

### TRENDING_BULL (530 bars)

| Tier | Metric | Original | V2 | Δ |
|---|---|---|---|---|
| Strong | Count | 28 | 29 | +1 |
| Strong | WR | 60.7% | 62.1% | +1.35pp |
| Moderate | Count | 24 | 23 | -1 |
| Moderate | WR | 50.0% | 47.8% | -2.17pp |

### TRENDING_BEAR (602 bars)

| Tier | Metric | Original | V2 | Δ |
|---|---|---|---|---|
| Strong | Count | 63 | 64 | +1 |
| Strong | WR | 65.1% | 65.6% | +0.55pp |
| Moderate | Count | 26 | 26 | +0 |
| Moderate | WR | 76.9% | 73.1% | -3.85pp |

## Modifier-active bars only (focused view)

n = 32

| Tier | Original n | V2 n | Original WR | V2 WR | Δ WR |
|---|---|---|---|---|---|
| Strong | 17 | 16 | 52.9% | 43.8% | -9.19pp |
| Moderate | 9 | 6 | 66.7% | 83.3% | +16.67pp |

## Modifier-subgroup breakdown

- Aligned bars (n=18): WR = 44.4% (CI [24.6%, 66.3%])
- Conflicted bars (n=14): WR = 64.3% (CI [38.8%, 83.7%])

The aligned vs conflicted WR gap is the core evidence for the modifier's
logic. A wide gap (aligned WR ≫ conflicted WR) supports the
boost/damp scheme; a narrow gap means the modifier is labeling noise.

## Recommendation

**DO NOT ADOPT**

Strong WR degraded -0.5pp.

### Decision thresholds (per study spec)

- Strong WR ↑ ≥ 3pp **and** Count ↓ ≤ 30% → ADOPT
- Strong WR ↑ 0-3pp → MARGINAL, accumulate more live data
- Strong WR ↓ → DO NOT ADOPT

### Caveats

1. Baseline direction-reg OOS IC ≈ 0.18 (production-validated), but this
   backtest replays the full production decoding pipeline on WF-OOS
   predictions — the Strong-tier baseline WR here is the most honest
   estimate of what production actually delivers today.
2. Modifier active rate is driven by cascade classifier precision at
   p ≥ 0.70 threshold; if cascade classifier precision shifts in the
   next 3 months, the modifier's impact will shift accordingly.
3. The aligned/conflicted sub-group WR gap may not be stable across
   market regimes. Recommend re-running Stage 2 after 3-6 months of
   post-launch data accumulates.
