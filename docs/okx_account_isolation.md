# OKX Account Isolation (Stage 3+ manual-blowup protection)

**Why this exists.** On 2026-06-05 the live account went from ~$188 to $0.05
because the operator **manually** opened ~12 contracts (~48x) and got
liquidated. None of the executor's kill switches helped — they only protect
the *executor's own* order path, not orders a human places by hand on the
same account.

The only real fix is **isolation**: the executor trades a dedicated OKX
sub-account that the operator **never** logs into to trade manually. Code can
*detect* interference (see below) but cannot *prevent* a human from trading an
account they can log into. Isolation is an operational control, not a code one.

---

## Ops setup (do this in the OKX UI — one time)

1. **Create a sub-account** (OKX → Assets → Sub-accounts → Create).
2. **Move the trading capital** (e.g. the current ~$105) into the sub-account.
   The main account keeps the rest; the sub-account holds only what the bot is
   allowed to risk.
3. **Create an API key scoped to the sub-account.** Permissions:
   - ✅ Read
   - ✅ Trade
   - ❌ **Withdraw / Transfer — must be OFF** (the executor's E4 check
     DEMOTES on startup if withdraw is present).
4. **Set Railway env** (Service 1) to the **sub-account** key:
   - `OKX_API_KEY_LIVE`, `OKX_API_SECRET_LIVE`, `OKX_PASSPHRASE_LIVE`
   - `OKX_INITIAL_CAPITAL_USD` ≈ the sub-account balance (e.g. `105`). This is
     the baseline for the total-loss cap (CAP-4) — if it's stale (e.g. 155
     while the account holds 105), the executor DEMOTES on boot thinking it
     already lost the difference.
   - `STAGE=live` (required for `is_simulated=0`).
5. **On your phone / OKX app, only ever use the MAIN account.** Treat the
   sub-account as the bot's — never place a manual order on it.

---

## What the code does (backstop, not prevention)

The reconciler compares the local DB against OKX every cycle. If OKX carries a
position the executor never opened, the kill check escalates:

| Reconciliation result | Meaning | Action |
|---|---|---|
| `orphan_exchange` / `size_diff` / `direction_diff` / `multiple_exchange_positions` | OKX has something the executor didn't create | **`MANUAL-INTERFERENCE` → DEMOTE** (sticky). Executor stops, sends a loud alert, and **does NOT close the foreign position** (it must not touch a position it didn't open). |
| `orphan_local` | DB has a position OKX doesn't (WS missed a fill) | `A4` → HALT, then auto-heal to CONSISTENT after a streak |
| `UNAVAILABLE` | OKX query failed | `A4` → HALT (recoverable) |

DEMOTE is **terminal** — the executor will not auto-resume. Re-entry requires a
deliberate restart (Railway redeploy / service restart), and cold-start
reconciliation must come back CONSISTENT, so you must clear the foreign
position first.

### Limits of the backstop
- Detection is **per-cycle**. A manual trade opened *and* closed/liquidated
  between two cycles may leave no position to reconcile. Large equity drops
  from such an event are still caught by the daily/total loss caps (CAP-3 /
  CAP-4), but the per-trade interference itself may go unattributed.
- This is exactly why **isolation (ops) is the primary control** and the code
  detection is only a secondary net.

---

## Governance reminder

Per `CLAUDE.md`: hitting the total-loss cap means **return to Stage 1 and
re-validate** — not "this time is an exception". The 2026-06-05 event was
manual, so resuming live is defensible, **but only after isolation is in place**
so the same vector can't recur.
