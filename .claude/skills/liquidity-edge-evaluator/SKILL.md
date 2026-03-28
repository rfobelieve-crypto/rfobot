---
name: liquidity-edge-evaluator
description: Evaluate liquidity sweep events, check whether the event matches the user's trading edge, and classify the expected bias as reversal or continuation. Use this when analyzing webhook events, sweep logs, or labeled market event data for BTC or similar instruments.
---

# Purpose
This skill evaluates liquidity events produced by the user's liquidity bot.

It should be used when:
- A liquidity sweep event is detected
- The user asks whether the event matches their edge
- The user wants a classification of reversal vs continuation
- The user wants structured event analysis for later validation

# Core workflow
When invoked:

1. Read the available event data
2. Identify:
   - timestamp
   - symbol
   - liquidity type
   - event price
3. Extract or compute edge-related features if present:
   - delta_1m
   - delta_5m
   - volume spike
   - rejection / acceptance around sweep level
   - taker buy / taker sell imbalance
4. Determine whether the event matches the configured edge
5. Output:
   - edge_match: true/false
   - bias: reversal/continuation/unclear
   - reason list
   - missing fields if any
6. If forward price data is available, also classify actual outcome:
   - reversal
   - continuation
   - unresolved

# Output format
Always return:

## Event
- Symbol:
- Time:
- Liquidity type:
- Event price:

## Edge evaluation
- Edge match:
- Bias:
- Confidence:
- Reasons:

## Outcome
- Actual outcome:
- Validation window:
- Notes:

# Rules
- Do not invent missing data
- If required fields are missing, explicitly list them
- Prefer structured reasoning over vague commentary
- Separate predicted bias from actual validated outcome
- If event data is incomplete, say "unclear" instead of forcing a conclusion

# Required definitions to respect
Use the user's project definitions for:
- what counts as a sweep
- what counts as reversal
- what counts as continuation
- what delta threshold is considered extreme

If those definitions are not found in project docs or prompt context, ask the user to define them or state that they are missing.