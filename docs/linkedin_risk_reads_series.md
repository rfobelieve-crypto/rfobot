# LinkedIn Series B: "Risk Reads"

> **Why this exists**: Parallel content series alongside the EP autobiographical
> journey. EP builds personality + story; Risk Reads builds **technical
> credibility**. Together: 2-3x inbound DM rate from quant recruiters.
>
> Filed 2026-06-20 from a deep-dive conversation. Ship date TBD — first one to
> draft once EP1 has 5-7 days of organic reach.

---

## Positioning vs Series A (EP)

| | Series A: EP (Journey) | Series B: Risk Reads (this) |
|---|---|---|
| Tone | First-person, story | Third-person, analytical |
| Length | 800-1200 words | 600-1000 words |
| Audience | Anyone curious about my path | Senior quants / risk engineers |
| Career signal | Personality + execution | **Technical credibility** |
| Visual style | ▍ subheadings (shared with EP) | ▍ subheadings (shared with EP) |

Cadence proposal: alternate every two weeks.
- Week 1: EP{N}
- Week 2: Risk Reads #{N}
- ⇒ One post per week, no audience fatigue.

---

## Series naming

Final: **"Risk Reads"** — short, sequential ("Risk Reads #1, #2, ..."),
fits LinkedIn UI well. Each post title format:

```
Risk Reads #N: {Hook Question or Counterintuitive Claim}
```

---

## Content backlog (7 posts, ~3-4 months of ammo)

### Risk Reads #1: Why Spreads Widen — and Why That Means Smart Money Is Pulling Back

**Core thesis**: Spread widening isn't random — it's market makers pricing in
adverse-selection risk. When MMs widen the spread, they're saying "we suspect
informed flow is coming and we don't want to be on the other side."

5 mechanisms to cover:
- Adverse selection risk
- Inventory imbalance
- Vol-based compensation
- Funding cost
- News-window pre-positioning

Visual: a market-maker decision flow (when do they widen vs tighten).

---

### Risk Reads #2: CVD Divergence Is Not About Volume — It's About Order Type Asymmetry

**Core thesis**: CVD measures market orders (taker), not limit orders. So CVD
divergence is the visible fingerprint of an invisible whale operating on the
opposite side via limit orders.

Cover:
- The market order vs limit order asymmetry
- Why retail uses market, whales use limit
- The two divergence regimes (price up + CVD down = whale buying;
  price down + CVD up = whale distributing)
- How this connects to absorption patterns

Visual: a side-by-side of (visible flow / hidden flow) at a turning point.

---

### Risk Reads #3: How to Read Open Interest in 5 Regimes — and Tell a Real Trend from a Short Squeeze

**Core thesis**: OI tells you whether the move is real new money or just
existing positions closing. Memorize the 2x2 (OI direction × price direction)
and you can never confuse trend with squeeze again.

Cover:
- OI = new positions opened, not total positions
- The 5 regimes (Up+Up, Up+Down, Up+Flat, Down+Up, Down+Down)
- Why OI ↑ + price flat is the most dangerous (breakout primed)
- How to use this as an entry filter

Visual: the 2x2 regime matrix with crypto chart examples for each cell.

---

### Risk Reads #4: The Real Meaning of Funding Rate — The Price of Leveraged Long Exposure

**Core thesis**: Funding isn't a "fee" — it's the auction-cleared price of
borrowing long exposure on perp. Persistent positive funding means demand
for leveraged long structurally exceeds supply of shorts. That's a fragility
signal, not a momentum signal.

Cover:
- The funding mechanism (mark-index premium)
- The arbitrage path that normally pulls funding to zero
- Why persistent extreme funding = arbitrageurs can't absorb demand
- The squeeze setup it creates

Visual: funding cycle diagram with arbitrageur loop annotated.

---

### Risk Reads #5: Anatomy of a Liquidation Cascade — Why Crypto Is Structurally Vulnerable

**Core thesis**: A cascade isn't volatility — it's a feedback loop between
forced sellers and withdrawn liquidity. The 7 steps from first liquidation
to bottom.

Cover the 7-step cascade:
1. Price hits first liq level
2. Forced market sells
3. Cascade triggers next layer
4. MMs withdraw
5. Thin bids → bigger slippage per sell
6. Cross-exchange contagion
7. Cascade ends when forced sellers exhausted

Why crypto specifically: high leverage availability, transparent positions,
aggressive liq engines, automated MMs, cross-exchange arbitrage links.

Visual: the cascade feedback diagram (forced selling → MM withdrawal →
worse fills → more forced selling).

---

### Risk Reads #6: 5 Channels, One Theme — Risk Is Aggregate Positioning Getting Too One-Sided

**Core thesis (synthesis post)**: All 5 channels are windows into the same
underlying phenomenon — when aggregate positioning becomes too skewed,
the system becomes fragile and any small trigger cascades.

Cover the unifying frame:
- Spread widening = MM info asymmetry
- CVD divergence = retail/whale order-type asymmetry
- OI buildup = position-side asymmetry
- Funding extreme = leverage demand/supply asymmetry
- Liq cascade = leverage density asymmetry at price bands

**Conclusion**: "Every risk signal is a measure of imbalance. The trader
who understands what is imbalanced can act before the snap-back."

Visual: the 5-channel synthesis diagram (one common root → 5 manifestations).

---

### Risk Reads #7 (Applied): What I Watch Before Every Entry — 30-Second Pre-Trade Checklist

**Core thesis**: Practical applied post — show readers the actual 30-second
checklist I run before any entry, mapped to the 5 channels.

Cover:
- Spread sanity (≤ 2× 60min avg)
- Funding direction filter (don't long with funding > +0.05%)
- Liq cascade status (no entry within 60min of cascade)
- OI/price regime (avoid OI buildup + price flat)
- CVD/price alignment (no divergence > 2σ at entry candle)

Visual: the checklist as a clean infographic.

---

## Style guide (shared with EP series)

### Visual / formatting

- ▍ visual marker for section breaks (looks like Markdown blockquote but
  carries forward to LinkedIn plain-text correctly)
- 6-8 subheadings per post (matches EP rhythm)
- Risk disclaimer in short form at the bottom (above hashtags)
- 4-6 hashtags: subset of {#TradingSystems #QuantitativeTrading
  #AlgorithmicTrading #MarketMicrostructure #RiskManagement #Crypto #Web3}
- One concept-diagram image per post (not data screenshot — clean
  illustration of the mechanism)

### Narrative structure — 5-beat PAS framework

Every Risk Reads post (and most technical EP posts) should follow
the Problem-Agitation-Solution copywriting structure. Friend-given
advice 2026-06-21, validated against EP2 which already executes this
cleanly:

```
[1] Pain         — the hook. State the concrete problem in 1-2 lines.
                   Must be specific enough that the target reader
                   thinks "yes, that's me."
[2] Resonance    — show you've lived this. "I also fell into this"
                   beats "many people experience X."
[3] Curiosity    — gap statement: name what's not obvious about why
                   it happens. Force a "how" question in the reader.
[4] Solution +   — the mechanism + the takeaway. Reader should leave
    Benefit       with one usable mental model or threshold.
[5] Next         — open question, series tease, or call-to-think.
                   Drives the engagement loop the algorithm rewards.
```

Pre-write self-check (ask before drafting):

1. [Pain]       Who am I solving what for?
2. [Resonance]  Will 90% of target readers feel this?
3. [Curiosity]  Have I made them want to know "how"?
4. [Benefit]    What concrete thing do they walk away with?
5. [Next]       Why would they follow me for more?

If any of these 5 is "no" or "weak," the post is not ready.

Worked example (Risk Reads #1 outline):

| Beat | Content |
|---|---|
| Pain | "Backtest looked clean, live PnL is 30 bps short per trade." |
| Resonance | "First 100 live trades I blamed luck. Trade 200 it clicked." |
| Curiosity | "When does spread widen, and why?" |
| Solution + Benefit | MM widens spread when adverse-selection risk rises. 5 triggers. Rule: spread > 3× 60min avg → don't enter. |
| Next | "Part 1 of a series. Next: CVD divergence." |

EP posts (autobiographical) may bend this structure when the genre
is origin-narrative (EP1 is a chapter, not a case study). But any
technical EP (e.g., trail-bug postmortem) should still hit all 5
beats — pain is the engine.

---

## When to ship #1

Conditions:
- EP1 has had 5-7 days of organic reach (don't compete with own content)
- I (rfobelieve) am still in "actively posting" mode (so algorithm
  treats Risk Reads as continuation of activity, not a comeback post)
- Have at least one concept diagram ready

---

## Invocation note for future Claude sessions

When user says "Risk Reads #N" or "the next mechanism post" or
"that risk series", load this doc as context. The 5-channel mechanism
content was already taught in the original conversation (2026-06-20)
and the thesis-per-post is locked in above.

Default action when invoked:
1. Write the post body in EP-style English (~800 words, 6-8 ▍ headings)
2. Generate the concept diagram (1200×630, PNG, dark IDE aesthetic
   matching trail bug onion image)
3. Produce docx + clean Markdown for LinkedIn paste
4. Suggest pairing with the corresponding Coinglass / TradingView
   reference visual the user could include alongside
