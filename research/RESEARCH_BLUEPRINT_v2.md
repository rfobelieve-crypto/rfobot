# Research Blueprint v2 — Next-Round Research Plan

## Baseline (Audited, 2026-04-02)

| Metric | Value | Note |
|--------|-------|------|
| Raw OOS IC | +0.022 | Per-fold feature selection, no lookahead |
| Z-score OOS IC | +0.011 | Z-score hurts, not helps |
| ICIR | +0.16 | Very low — unstable across folds |
| Positive IC folds | 2/5 | Fold 2 (+0.16) and Fold 4 (+0.05) |
| Strong dir_acc | 53.5% | Barely above random |
| Weekly IC positive | 50% | Coin-flip level |
| Data | 3985 bars x 71 features | 166 days, 1h native |

Previous IC=+0.071 was inflated by global feature selection lookahead.
All research below uses this audited baseline as the starting point.

---

## Question 1: Feature Selection — How to Fix

### Option A: Fixed Feature Set (Burn-in Window)

**Design:**
- Use first 25% of data (~1000 bars, ~42 days) as burn-in
- Run feature selection ONCE on burn-in data
- Lock the selected features for ALL subsequent folds
- Never re-select

**Pros:**
- Zero feature selection lookahead, period
- Most stable — features don't drift between folds
- Simpler to audit and reproduce

**Cons:**
- Burn-in period is wasted (no OOS predictions for those bars)
- If market regime shifts, early-selected features may not capture it
- 1000 bars may be too few for stable importance ranking

### Option B: Expanding-Window Feature Selection

**Design:**
- Fold k selects features using ONLY data from bar 0 to fold (k-1) end
- Each fold may have slightly different feature set
- Minimum 1000 bars before first selection

**Pros:**
- Adapts to evolving feature importance over time
- More data for later folds = more stable selection

**Cons:**
- Fold 1 has smallest training set → noisiest selection
- Feature set instability (audit showed only 16/40 features common to all folds)
- Harder to reason about "which features matter"

### Recommendation for current data size (166 days)

**Use Option A (Fixed Feature Set).** Reasons:
1. 166 days is too short for stable expanding-window selection
2. Audit showed per-fold selection creates high variance (16/40 common features)
3. Simpler to debug, audit, and reason about
4. When data reaches 365+ days, revisit Option B

**Implementation:**
```
burn_in = first 1000 bars (42 days)
select_features(df[:1000], feat_cols, top_k=40)
walk_forward on remaining 2985 bars with locked features
```

---

## Question 2: Extracting Higher-Quality Signals from Weak Alpha

The core insight: overall IC=+0.02 is an average. Some conditions
have real signal, others are noise. The goal is to ABSTAIN in
noise conditions and only emit signals in high-quality windows.

### Approach: Selective Signaling with Abstention

**Layer 1: Regime Gate**
- Compute regime label (trailing only, no lookahead)
- If model historically has IC < 0 in current regime → ABSTAIN
- This removes the regime that is actively dragging overall IC down

**Layer 2: Prediction Magnitude Gate**
- Only emit signal when |raw_pred| > threshold
- Threshold = e.g. 75th percentile of |pred| in trailing window
- Rationale: weak predictions are noise; strong predictions carry signal

**Layer 3: Feature Familiarity Check**
- Compare current feature vector to training distribution
- If Mahalanobis distance > threshold → ABSTAIN (OOD risk)
- Simple approximation: if >3 features are >3 sigma from training mean

**Layer 4: Local Performance Tracker**
- Track rolling hit rate of recent N predictions
- If recent 20 predictions have dir_acc < 45% → reduce confidence
- Adaptive: automatically reduces signal in bad streaks

**Validation requirement:**
Each layer must be validated independently:
- Compare IC with and without each gate
- Verify gate improves precision (correct signals / total signals)
- Verify gate does NOT just reduce N to trivially small

**Expected outcome:**
- Fewer signals (maybe 30-40% of bars)
- Higher per-signal accuracy (target: 55%+ dir_acc in emitted signals)
- This matches product goal: "indicator" not "signal every bar"

---

## Question 3: Regime-Aware Analysis Framework

### Regime Definition (no lookahead)

```python
# Trailing-only regime detection
ret_24h = close.pct_change(24)     # 24h trailing return
vol_24h = log_return.rolling(24).std()  # 24h realized vol

# Percentile rank (expanding window — no future data)
vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)

# Classification
TRENDING_BULL  = (vol_pct > 0.6) & (ret_24h > 0.005)
TRENDING_BEAR  = (vol_pct > 0.6) & (ret_24h < -0.005)
CHOPPY         = ~TRENDING_BULL & ~TRENDING_BEAR
```

Key: `expanding().rank()` ensures percentile is computed using only
past data. No forward information.

### Research Steps

**Step 1: Regime-sliced OOS IC**
For each regime, compute:
- OOS IC (per fold and overall)
- OOS direction accuracy
- Number of bars (is there enough data?)

**Step 2: Identify drag regime**
The audit showed:
- Fold 3 (2026-01-08 ~ 2026-02-05): IC=-0.11
- March monthly IC: -0.025
Question: which regime dominates these bad periods?

**Step 3: Decision — filter vs separate model**

| Approach | When to use |
|----------|------------|
| Regime-aware filtering | If one regime has IC<0 and others have IC>0. Simply abstain in bad regime. |
| Regime-specific threshold | If all regimes have IC>0 but different magnitudes. Adjust confidence by regime. |
| Regime-specific model | NOT recommended with 166 days. Each regime has <1000 bars — too few for separate XGBoost. |

**Recommended: Regime-aware filtering (abstention in drag regime)**

---

## Question 4: Z-Score Normalization — What to Do

### Why Z-Score Hurts IC

The audit showed:
- Raw pred IC = +0.022
- After z-score IC = +0.011 (halved)

**Root cause analysis:**

1. **Information destruction**: Raw pred contains absolute magnitude.
   Z-score converts to "how unusual vs recent history." If the model
   correctly predicts a persistent trend (e.g., sustained bearish),
   z-score would normalize it to 0 (because recent predictions are
   all similar), destroying the directional signal.

2. **Regime-change penalty**: When market shifts regime, the new
   predictions differ from recent history → z-score shows large
   values. But these are exactly the moments where the model is
   least reliable (new regime, limited training data).

3. **Short window problem**: Window=48 (2 days) means z-score
   is dominated by very recent predictions, creating
   mean-reversion bias in what should be a momentum signal.

### Recommendation

**Remove z-score from the core prediction layer entirely.**

Use raw prediction directly for:
- Direction: sign(pred) with deadzone on |pred| magnitude
- Confidence: based on |pred| percentile in TRAINING distribution
  (not rolling z-score of recent predictions)

Z-score can be kept ONLY for visualization:
- Chart display of "how unusual is current prediction"
- Not used for direction or confidence computation

**Specific change:**
```
# BEFORE (current)
direction = sign(zscore(pred))  # z-score destroys info
confidence = zscore_percentile  # relative to recent, unstable

# AFTER (proposed)
direction = sign(pred) with |pred| > deadzone  # raw signal
confidence = percentile(|pred|, training_distribution)  # stable reference
```

---

## Question 5: Confidence System v2

### Current System Problems
- Based on z-score percentile (which hurts IC)
- Agreement weight is always 1.0 (single model)
- Regime dampening uses a fixed 0.6x multiplier (not data-driven)

### Proposed Confidence v2 Architecture

**Component 1: Prediction Magnitude (implement NOW)**
```
mag_score = percentile_rank(|pred|, training_pred_distribution)
```
- Range: 0-100
- Stable reference: training distribution doesn't change
- High magnitude = model is more "opinionated" = higher confidence
- Validation: check if high mag_score correlates with higher dir_acc

**Component 2: Regime Compatibility (implement NOW)**
```
regime_score = historical_IC_in_current_regime
```
- If current regime historically has IC>0.03: score = 1.0
- If IC is 0-0.03: score = 0.6
- If IC<0: score = 0.0 (ABSTAIN)
- Must use only trailing data for regime IC calculation

**Component 3: Feature Familiarity (implement NEXT)**
```
familiarity = 1 - (n_extreme_features / total_features)
```
- Count features with |zscore| > 3 (vs training distribution)
- If >5 features are extreme: current state is OOD, reduce confidence
- Requires: training feature mean/std (store in training_stats.json)

**Component 4: Local Track Record (implement NEXT)**
```
local_score = rolling_dir_accuracy(last 20 predictions)
```
- If recent 20 predictions have <45% accuracy: halve confidence
- Requires: storing actual outcomes as they become available
- Natural fit for the live system (check 4h later)

**Component 5: Model Agreement (implement LATER, needs 2nd model)**
- Only meaningful with 2+ models
- Currently always 1.0 (skip)

### Composite Confidence v2
```
confidence = mag_score * regime_score * familiarity_weight
```

**Validation protocol:**
1. Compute confidence for all OOS bars
2. Bin into quintiles
3. Check monotonic relationship: higher confidence → higher |actual_return|
4. Check monotonic relationship: higher confidence → higher dir_acc
5. If not monotonic, the confidence system is not calibrated

---

## Question 6: Experiment Matrix

### MUST DO (Priority 1 — do these first, in order)

| # | Experiment | Hypothesis | Metric | Est. Effort |
|---|-----------|-----------|--------|-------------|
| E1 | Fixed feature set (burn-in 1000 bars) | Reduces variance, stable baseline | OOS IC, ICIR, fold variance | 1 hour |
| E2 | Remove z-score from prediction layer | Raw pred has higher IC than z-scored | OOS IC improvement | 30 min |
| E3 | Regime-sliced OOS analysis | Identify which regime drags IC | Per-regime IC table | 1 hour |
| E4 | Regime-aware abstention | Abstaining in bad regime improves precision | Precision of emitted signals | 1 hour |
| E5 | Confidence v2 (mag + regime only) | Better calibrated than z-score percentile | Confidence-accuracy monotonicity | 2 hours |

### OPTIONAL (Priority 2 — do if P1 results are promising)

| # | Experiment | Hypothesis | Metric |
|---|-----------|-----------|--------|
| E6 | Feature familiarity gate (OOD detection) | Abstaining in OOD bars improves precision | Precision improvement |
| E7 | Local track record dampening | Recent accuracy predicts near-future accuracy | Rolling hit rate autocorrelation |
| E8 | LightGBM comparison | Different tree algorithm may capture different patterns | OOS IC delta |
| E9 | Horizon scan (1h, 2h, 4h, 8h) | 4h may not be optimal horizon | IC at each horizon |

### DO NOT DO (yet)

| Experiment | Why not |
|-----------|---------|
| LSTM / TFT | Not enough data (166 days). Minimum 1-2 years for sequence models. |
| Regime-specific models | Each regime has <1000 bars. Separate XGBoost will overfit. |
| Ensemble with Ridge | Already tested — Ridge has negative IC on 1h data. |
| More feature engineering | Tested 42 new features, all degraded walk-forward. Data-limited. |
| Multi-asset (ETH) | Fix BTC model quality first. Adding ETH won't fix weak IC. |
| On-chain data | Good idea but separate research track. Don't mix into current audit. |

---

## Data Format Consistency Audit

### Issues Found

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| coinglass_backfill.py saves RangeIndex | HIGH | line 268: `index=False` | Change to `df.set_index("dt").to_parquet()` |
| Timestamp column naming varies | MEDIUM | ts_open vs time vs create_time | Standardize: always use `dt` as DatetimeIndex |
| metrics parquet isolated | LOW | BTCUSDT_metrics.parquet has RangeIndex | Not used in current pipeline, defer |
| inference.py zero-fills missing features | MEDIUM | line 58: `features[c] = 0` | Change to forward-fill from previous bar |
| 15m legacy files still exist | LOW | BTC_USD_15m_*.parquet | Clean up when confirmed unused |
| EXCLUDE defined in 2 places | MEDIUM | prediction_indicator.py + feature_config.py | Single source of truth in feature_config.py |

### Recommended Fix Order
1. Fix zero-fill in inference.py (affects live predictions)
2. Unify EXCLUDE to single source
3. Fix coinglass_backfill.py index
4. Clean up 15m legacy files

---

## Top 3 Actions — Starting from IC=+0.02

### 1. Fix Feature Selection + Remove Z-Score (Experiments E1+E2)

**Why first:** These are the two known issues directly distorting our
measurement. E1 gives us a clean, auditable baseline. E2 stops the
z-score from destroying what little signal exists. Together they
establish the TRUE baseline we're building on.

**Expected outcome:** IC stays ~+0.02 but ICIR should improve
(less fold variance with fixed features). Direction accuracy may
improve by 1-2% from removing z-score distortion.

### 2. Regime-Aware Abstention (Experiments E3+E4)

**Why second:** The audit showed 3/5 folds have negative IC. If we
can identify that the negative folds correspond to a specific regime,
we can simply ABSTAIN in that regime. This converts a mixed-signal
model into a selective high-quality indicator.

**Expected outcome:** Fewer signals (maybe 40-60% of bars), but
the emitted signals have IC > +0.03 and dir_acc > 55%. This is
the product-level improvement that matters most.

### 3. Confidence v2 with Magnitude + Regime (Experiment E5)

**Why third:** Once we have clean prediction (E1+E2) and know which
regimes work (E3+E4), we can build a confidence score that actually
means something. The current confidence is based on z-score
percentile which we've shown hurts IC.

**Expected outcome:** Monotonic calibration — "Strong" signals
actually have higher accuracy than "Moderate" which has higher
accuracy than "Weak". This is the minimum viable quality for
a published indicator.

---

## Success Criteria for v2

Before declaring v2 ready:

- [ ] Fixed feature set, no lookahead, fully auditable
- [ ] Raw pred OOS IC > +0.02 (maintained, not regressed)
- [ ] Emitted signals (after abstention) IC > +0.03
- [ ] Emitted signals dir_acc > 55%
- [ ] Confidence monotonicity verified (Strong > Moderate > Weak)
- [ ] No negative IC regimes in the active signal set
- [ ] All folds with emitted signals have positive IC
