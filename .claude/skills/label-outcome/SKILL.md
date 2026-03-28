# Skill: Label Outcome

Classify liquidity sweep events as reversal, continuation, or neutral.

## Trigger
When analyzing sweep outcomes, building edge statistics, or modifying outcome classification logic.

## Context
- Main bot outcome tracking: `outcome_tracker.py` (±0.5% first-hit, 15m/1h)
- Event-level tracking: `BTC_perp_data.py` → `create_event()`, `event_watchdog()`
- DB tables: `sweep_outcomes`, `liquidity_events`

## Classification Rules

### outcome_tracker.py (simple first-hit)
- BSL (buy-side sweep): price hits lower target first → reversal, upper → continuation
- SSL (sell-side sweep): price hits upper target first → reversal, lower → continuation
- Threshold: ±0.5% from trigger price
- Windows: 15m, 1h
- Timeout → unresolved

### BTC_perp_data.py (return-based)
- 1h: ±0.5% = reversal/continuation, < 0.3% = neutral
- 4h: ±1.0% = reversal/continuation, < 0.5% = neutral
- BSL: negative return = reversal, positive = continuation
- SSL: positive return = reversal, negative = continuation

## Data Fields (liquidity_events)
- `liquidity_side`: "buy" (BSL) / "sell" (SSL)
- `first_hit_side`: "upper" / "lower" / "none"
- `first_hit_delta`: cumulative delta at hit time
- `return_15m`, `return_1h`, `return_4h`: forward returns (%)
- `result_1h`, `result_4h`: "reversal" / "continuation" / "neutral"
- `session`: "Asia" / "London" / "NY" / "Off-hours"

## Edge Analysis
Combine with flow_bars_1m data to find:
- Does pre-sweep delta predict outcome?
- Does delta divergence increase reversal probability?
- Session-dependent patterns?
