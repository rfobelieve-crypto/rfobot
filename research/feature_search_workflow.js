export const meta = {
  name: 'feature-search-newfamilies',
  description: 'Engineer novel trailing-only feature families, screen by conditional IC vs V7 residual, A/B survivors, adversarially verify any lift (incl. dvol), synthesize honest AUC-ceiling verdict',
  phases: [
    { title: 'Engineer+Screen', detail: '6 novel order-flow feature families, conditional-IC screen via featsearch_lib' },
    { title: 'A/B survivors', detail: 'ensemble A/B + 4-cond per-fold sanity on screen passers' },
    { title: 'Adversarial verify', detail: 'skeptics attack each lift claim (outlier / regime-confound / redundancy)' },
    { title: 'Synthesize', detail: 'honest verdict: edge change + AUC ceiling' },
  ],
}

const ROOT = 'C:/Users/rfo/Desktop/flowbot/flow_system'

// Precise, leak-safe specs. Each family draws only from order-flow / price-behavior
// channels already in features_all.parquet. NO TA indicators (MACD/EMA/RSI/Boll).
const FAMILIES = [
  {
    key: 'xchan_div',
    title: 'Cross-channel divergence & lead/lag',
    idea: 'Spot-CVD vs perp-CVD divergence dynamics (cg_scvd_* vs cg_fcvd_*), OI-vs-price sign-agreement streaks (trapped positioning), funding-vs-coinbase-premium divergence (leverage crowd vs spot allocators), ETF-net-flow vs funding divergence. Build multi-window divergence, sign-streaks, and z-scored gaps.',
  },
  {
    key: 'liq_aftermath',
    title: 'Liquidation-cascade aftermath',
    idea: 'Bars-since-last cascade (cg_liq_cascade / cg_liq_surge), post-long-liq vs post-short-liq forward-window behavior encoded as trailing state, liq-imbalance regime transitions, cascade magnitude interacted with subsequent absorption_net. All state computed trailing.',
  },
  {
    key: 'vol_termstructure',
    title: 'DVOL term-structure & vol-risk-premium',
    idea: 'This is the ONE channel that passed the existing-feature screen (dvol level). Test if there is DYNAMIC alpha beyond the collinear level: dvol vs realized-vol spread dynamics (dvol_rv_spread), dvol acceleration/curvature, dvol*funding (vol premium vs carry), low-vol compression-then-expansion setup flags, dvol percentile regime.',
  },
  {
    key: 'regime_masked',
    title: 'Correctly-formed regime conditioning',
    idea: 'mistake.md 2026-04-13: NEVER feat*sparse_indicator. Use feat*(1-is_dead_regime) form, masking only where a base flow feature plausibly dies. e.g. funding/OI/CVD features masked OUTSIDE high-DVOL or extreme-FNG; also add the regime indicators themselves as features so XGB learns the split. Verify each mask keeps >70% non-zero.',
  },
  {
    key: 'tail_jump',
    title: 'Higher-moment & jump microstructure',
    idea: 'Rolling return kurtosis, downside semivariance, jump indicator (|ret|/ATR over threshold) carrying sign, gap-fill tendency, consecutive-wick exhaustion sequences, body/range compression. Pure price-behavior stats (allowed per mistake.md 2026-04-01), NOT TA pattern indicators.',
  },
  {
    key: 'flow_accel',
    title: 'Flow acceleration & 2nd-order microstructure',
    idea: 'CVD 2nd-difference (acceleration), taker-imbalance regime persistence beyond imb_*, trade-intensity * avg-trade-size (informed-vs-noise), absorption-then-reversal sequences, large-vs-small-bar divergence persistence. Trailing-only.',
  },
]

const SPEC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['family', 'n_built', 'passers', 'parquet_path', 'top5', 'notes'],
  properties: {
    family: { type: 'string' },
    n_built: { type: 'integer', description: 'how many candidate features built' },
    passers: {
      type: 'array',
      description: 'features passing the full screen_candidates() screen',
      items: {
        type: 'object', additionalProperties: false,
        required: ['name', 'cond_ic', 'consist', 'max_corr', 'nonzero'],
        properties: {
          name: { type: 'string' },
          cond_ic: { type: 'number' },
          consist: { type: 'number' },
          max_corr: { type: 'number' },
          nonzero: { type: 'number' },
        },
      },
    },
    parquet_path: { type: 'string', description: 'path to saved passer parquet, or "" if no passers' },
    top5: { type: 'string', description: 'top-5 candidates by |cond_ic| as a compact text table (even if they failed), for the synthesis' },
    notes: { type: 'string', description: 'leak-safety self-check result + any caveat' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['claim', 'verdict', 'reason'],
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['real_lift', 'mirage', 'inconclusive'] },
    reason: { type: 'string' },
  },
}

// ---- Phase 1: engineer + screen, in parallel ----
phase('Engineer+Screen')
const familyResults = await parallel(FAMILIES.map((fam) => () =>
  agent(
    `You are a quant feature engineer on a BTC 4h direction model. Working dir: ${ROOT}.

GOAL: build a NOVEL feature family "${fam.key}" (${fam.title}) of trailing-only candidate
features, then screen them for MARGINAL information the deployed 136-feature V7 model has
not already captured. Idea: ${fam.idea}

HARD RULES (from the project's mistake.md — violating these invalidates the work):
1. Trailing-only. A feature at bar t may use ONLY data at bars <= t. No .shift(-k), no
   centered/forward windows. State features must be cumulative-from-past.
2. NO technical-analysis indicators (MACD/EMA/RSI/Bollinger/Stoch). Order-flow + raw
   price-behavior statistics ONLY (returns, vol, skew, kurtosis, wicks are allowed).
3. NEVER feat*sparse_indicator where the indicator's non-zero fraction < 30% (it collapses
   to the indicator's sparsity pattern). For regime conditioning use feat*(1-is_dead_regime).
4. Daily alt-data (etf_*, fear_greed_*) is ALREADY lagged in the parquet — do not re-shift,
   but do not build new daily features that peek same-day.

DATA & TOOLING:
- Load the production frame: \`from research.featsearch_lib import load_features, screen_candidates\`
  then \`df = load_features()\` (282 cols: raw OHLCV close/high/low/open/volume/taker_buy_vol/
  taker_buy_quote/trade_count/quote_vol + all cg_*/etf_*/dvol_*/fg_*/tox_*/dfp_*/init_*/imb_*
  channels). FIRST print [c for c in df.columns] to see exactly what raw inputs exist for your family.
- Build your candidate features into a DataFrame \`cand\` indexed by df.index (trailing-only).
- Screen: \`res = screen_candidates(cand, existing_df=df)\`. This computes, vs the CLEAN V7
  OOS residual: conditional IC, per-fold sign consistency, max |corr| to any existing column,
  nonzero fraction, and SCREEN_PASS = (|cond_ic|>0.03 & consist>0.60 & max_corr<0.90 &
  nonzero>0.30). DO NOT write your own IC/screen code — use screen_candidates so the
  methodology is identical across families.
- Save ONLY the passing columns: if any res.SCREEN_PASS, write those candidate columns to
  ${ROOT}/research/results/newfeat_${fam.key}.parquet (index=df.index). Else parquet_path="".

WRITE your script to ${ROOT}/research/newfeat_${fam.key}.py and RUN it with
\`PYTHONUTF8=1 python research/newfeat_${fam.key}.py\` (cwd=${ROOT}). Iterate until it runs clean.
Build 8-15 genuinely distinct candidates (not 8 transforms of one thing).

Report the REAL screen numbers you observed (not hypothetical). Most families are expected to
FAIL the screen (V7 is documented-saturated on these channels) — reporting an honest FAIL with
the top candidates is a valid, valuable result. Do not fabricate passers.`,
    { label: `fam:${fam.key}`, phase: 'Engineer+Screen', agentType: 'general-purpose', schema: SPEC_SCHEMA }
  )
))

const families = familyResults.filter(Boolean)
const allPassers = families.flatMap((f) => (f.passers || []).map((p) => p.name))
const passerParquets = families.filter((f) => f.parquet_path).map((f) => f.parquet_path)
log(`Screen done: ${families.length} families, ${allPassers.length} new screen-passers: ${allPassers.join(', ') || '(none)'}`)

// ---- Phase 2: A/B survivors (only if any new feature passed the screen) ----
phase('A/B survivors')
let newFamilyAB = null
if (allPassers.length > 0) {
  newFamilyAB = await agent(
    `Working dir: ${ROOT}. New candidate features PASSED the conditional-IC screen: ${allPassers.join(', ')}.
Their columns are saved in these parquets: ${passerParquets.join(', ')}.

Run the ensemble A/B with the production trainer + 4-condition per-fold sanity:
\`PYTHONUTF8=1 python research/feature_search_ab.py --extra-parquet "${passerParquets.join(',')}" --add "${allPassers.join(',')}" --label new_families\`

This trains V7-baseline vs V7+these-features over 77 walk-forward folds (clean, no early-stop
leak) and reports: clean pooled AUC base->new, aggregate lift, per-fold mean lift, frac_positive
folds, bootstrap p(lift<=0). DEPLOY requires ALL of: agg>+0.005, mean>+0.001, frac_pos>0.55,
boot_p<0.05 (on clean AUC). Read research/results/feature_search_ab.json for the new_families entry.

Report the exact clean-AUC verdict block and whether it is DEPLOY or NO-GO. Do not spin a NO-GO
as positive.`,
    { label: 'ab:new_families', phase: 'A/B survivors', agentType: 'general-purpose',
      schema: { type: 'object', additionalProperties: false,
        required: ['deploy', 'clean_auc_base', 'clean_auc_new', 'agg_lift', 'mean_fold_lift', 'frac_pos', 'boot_p', 'summary'],
        properties: {
          deploy: { type: 'boolean' },
          clean_auc_base: { type: 'number' }, clean_auc_new: { type: 'number' },
          agg_lift: { type: 'number' }, mean_fold_lift: { type: 'number' },
          frac_pos: { type: 'number' }, boot_p: { type: 'number' },
          summary: { type: 'string' },
        } } }
  )
} else {
  log('No new family feature passed the screen — skipping new-family A/B (screen-fail across all 6 families IS the result).')
}

// ---- Phase 3: adversarial verification ----
// Always verify the dvol battery result (the only existing-feature screen-passer).
// Also verify any new-family A/B that claims DEPLOY.
phase('Adversarial verify')
const battery = args && args.battery ? args.battery : '(battery summary not provided)'

const verifyTasks = []
// dvol: three distinct lenses
const dvolLenses = [
  { lens: 'outlier-fold', q: 'Is any dvol clean-AUC lift propped up by 1-2 extreme folds rather than broad-based? Read research/results/feature_search_ab.json (labels dvol_level_all / dvol_best_single / dvol_dynamics): inspect frac_pos, mean_fold_lift vs agg_lift, boot_ci. If agg_lift>0.005 but mean_fold_lift is near 0 or frac_pos<0.55, it is outlier-driven (mistake.md 2026-06-02).' },
  { lens: 'regime-confound', q: 'dvol_* is a slow-moving daily-ish implied-vol LEVEL. A persistent conditional IC of a low-frequency level vs residual is often just a vol-REGIME proxy that will not generalize forward (the sample happens to have a vol trend aligned with outcomes). Argue whether the dvol signal is genuine 4h alpha or a regime artifact. Consider that max_corr_deployed≈0.88 (already half-captured by bvol_*).' },
  { lens: 'redundancy', q: 'dvol_close/open/high/low/ma_24h are ~collinear with each other AND ~0.88 corr to a deployed feature. Run/inspect: does dvol add anything the deployed bvol_*/dvol_oi_interaction features do not already encode? Is the A/B lift (if any) within noise of that redundancy?' },
]
for (const dl of dvolLenses) {
  verifyTasks.push(() => agent(
    `Working dir: ${ROOT}. Adversarially evaluate the DVOL feature result through the "${dl.lens}" lens.
Battery summary provided: ${battery}
Question: ${dl.q}
Read research/results/feature_search_ab.json and research/results/feature_search_screen.csv as needed.
Default to skepticism: only call real_lift if the evidence genuinely survives this lens. Be specific with numbers.`,
    { label: `verify:dvol:${dl.lens}`, phase: 'Adversarial verify', agentType: 'general-purpose', schema: VERDICT_SCHEMA }
  ))
}
// new-family survivor verification (if it claimed deploy)
if (newFamilyAB && newFamilyAB.deploy) {
  for (const lens of ['outlier-fold', 'redundancy', 'leak-recheck']) {
    verifyTasks.push(() => agent(
      `Working dir: ${ROOT}. A new feature family A/B claimed DEPLOY: ${newFamilyAB.summary}.
Adversarially verify through the "${lens}" lens. For leak-recheck, re-read the family's
research/newfeat_*.py and confirm every passing feature is strictly trailing (no shift(-k),
no centered window, no same-day daily peek). For outlier-fold, inspect per-fold lift dist in
research/results/feature_search_ab.json (label new_families). For redundancy, check max_corr
to existing cols in the screen output. Default to skepticism.`,
      { label: `verify:newfam:${lens}`, phase: 'Adversarial verify', agentType: 'general-purpose', schema: VERDICT_SCHEMA }
    ))
  }
}
const verdicts = (await parallel(verifyTasks)).filter(Boolean)

// ---- Phase 4: synthesis ----
phase('Synthesize')
const synth = await agent(
  `Working dir: ${ROOT}. You are writing the FINAL honest verdict of a 2026-06-14 feature-engineering
re-run on the BTC 4h direction model. Be rigorous and do not overstate.

CONTEXT:
- Today's clean V7 baseline (de-leaked, no early-stop): sign-AUC=0.5412, Spearman IC=+0.063.
  Documented structural ceiling is AUC 0.54-0.57. The leaky "canonical 0.59/0.17" is early-stop inflated.
- Existing-feature screen: of 137 already-computed non-deployed candidates, only 5 collinear dvol_*
  (level) passed the conditional-IC screen; everything else failed on fold-consistency (the high
  |cond_ic| ones like dfp_fcvd_cum_24h_rank +0.107 had consist≈0.39 = outlier-driven).
- Existing-feature A/B battery verdict (read research/results/feature_search_ab.json): ${battery}
- New-family screen results (this workflow): ${JSON.stringify(families.map((f) => ({ family: f.family, n_built: f.n_built, n_pass: (f.passers || []).length, top5: f.top5 })))}
- New-family A/B: ${newFamilyAB ? JSON.stringify(newFamilyAB) : 'no new family passed the screen'}
- Adversarial verdicts: ${JSON.stringify(verdicts)}

Read research/results/feature_search_ab.json and research/results/feature_search_screen.csv to confirm numbers.

Write a markdown report answering the user's two questions DIRECTLY:
  (1) 重新跑特徵工程，有沒有新特徵能提高整體 edge?
  (2) AUC 能否突破天花板?
Include: the honest today-AUC, what passed/failed and why, the dvol question resolution (real vs
regime artifact, per the adversarial verdicts), whether ANY config achieved the 4-condition deploy
gate, and a clear recommendation. If the answer is "no breakthrough, V7 confirmed saturated, ceiling
intact," say so plainly with the evidence — that is the most likely and most valuable honest outcome.
End with the ONE concrete next lever (almost certainly: only异源 data — options GEX / on-chain whale /
sentiment — can move AUC; same-source is exhausted), and note this should be saved to mistake.md/memory.`,
  { label: 'synthesize', phase: 'Synthesize', agentType: 'general-purpose' }
)

return { today_auc: 0.5412, new_screen_passers: allPassers, new_family_deploy: !!(newFamilyAB && newFamilyAB.deploy), verdicts, report: synth }
