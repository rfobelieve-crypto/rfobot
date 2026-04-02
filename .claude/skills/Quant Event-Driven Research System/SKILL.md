name: quant_event_driven_research
description: >
  Build a production-grade event-driven quantitative research system for crypto perpetual futures.
  Focus on liquidity sweeps, taker flow, OI, and funding to discover statistically robust trading edges.

---

core_principles:

  - event_driven:
      description: >
        All analysis must revolve around liquidity sweep events.
        NOT generic time-series prediction.
      rule: >
        Every dataset must answer: "What happens AFTER the event?"

  - no_lookahead:
      description: >
        No future data leakage is allowed.
      rule: >
        All features must be computable using only data available at the event timestamp.

  - tradability_constraint:
      description: >
        Signals must be executable in real market conditions.
      rule: >
        Avoid microsecond-level or HFT-dependent features.
        All signals must survive 1–2 minute latency.

  - robustness_over_complexity:
      description: >
        Prefer simple, stable features over complex models.
      rule: >
        Focus on statistical edge, not model sophistication.

---

system_architecture:

  raw_layer:
    description: >
      Store original data with no transformation.
    requirements:
      - trades (taker side, price, size, timestamp)
      - open interest
      - funding rate
      - no aggregation

  feature_layer:
    description: >
      Clean and aligned dataset using 15-minute buckets.
    requirements:
      - strict time alignment (15m)
      - deduplication
      - outlier filtering
      - no mixed granularity

  research_layer:
    description: >
      Event-driven dataset for edge discovery.
    requirements:
      - liquidity_events table
      - pre-event features
      - post-event outcomes
      - label generation

---

time_standard:

  bucket: 15m
  rule: >
    All data must be aligned to 15-minute intervals.
    No mixing of 1m / 5m / tick data in feature layer.

---

event_dataset:

  table: liquidity_events

  event_info:
    - event_id
    - symbol
    - event_side (BSL / SSL)
    - trigger_timestamp
    - trigger_price
    - trigger_bucket_15m

  pre_event_features:
    - delta_1bar
    - delta_2bar
    - delta_4bar
    - OI_change_1bar
    - OI_change_2bar
    - funding_rate
    - funding_zscore

  interaction_features:
    - pressure: normalized_delta * OI_change
    - delta_price_divergence
    - flow_acceleration

  post_event_outcomes:
    - return_1bar
    - return_2bar
    - return_4bar

  label_definition:

    BSL:
      reversal: return_4bar <= -0.5%
      continuation: return_4bar >= +0.5%
      neutral: otherwise

    SSL:
      reversal: return_4bar >= +0.5%
      continuation: return_4bar <= -0.5%
      neutral: otherwise

---

feature_engineering:

  delta:
    - raw_delta = buy_volume - sell_volume
    - normalized_delta = (buy - sell) / (buy + sell)

  oi_features:
    - oi_change
    - oi_change_pct

  interaction:
    - pressure = normalized_delta * oi_change_pct

  dynamics:
    - flow_acceleration = delta_now - delta_prev

---

data_quality_rules:

  - no_early_aggregation
  - strict_time_alignment
  - trade_deduplication
  - outlier_filtering (extreme trade size)
  - handle_missing_OI_funding

---

research_objective:

  description: >
    Identify statistically significant trading edges.

  required_outputs:
    - win_rate
    - average_return
    - return_distribution
    - condition-based grouping

  example_queries:
    - BSL + positive delta + rising OI → reversal probability
    - high funding + strong delta → continuation vs trap

---

execution_constraints:

  - avoid sub-second features
  - avoid orderbook microstructure dependency
  - signals must remain valid under delay

---

output:

  - feature_table_15m
  - liquidity_events_table
  - queryable research interface
  - model-ready dataset (for XGBoost / LGBM)

---

final_goal:

  description: >
    Build a market behavior classification system that discovers repeatable trading edges.

  emphasis:
    - statistical validity
    - interpretability
    - real-world tradability

  avoid:
    - overfitting
    - black-box models without edge
    - purely predictive systems without structure