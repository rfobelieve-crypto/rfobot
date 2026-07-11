//@version=2
// ═══════════════════════════════════════════════════════════════════
// Liquidity Vacuum 撤單壓力 — MMT custom indicator (v3, 2026-07-11)
//
// 把「巨量=機構撤單→真空→掃止損的被迫流」世界觀做成即時可視化:
//   1) 撤單偏斜直方圖  ← near-touch 深度快照差分 − 成交(cancel proxy)
//   2) 被迫流徽章      ← 巨量×清算共現(flow_system 實測 92%/35x)
//   3) 止損區逼近警示  ← P-wall-pull 即時視圖
//
// ⚠️ 紀律:監視工具,非已驗證 edge(正式判決 = 2026-08-10 cancel_lead_ic);
//   只放 Personal 不發 Community;門檻是顯示分類非調參。
//
// API 已在 MMT 編輯器 autocomplete 實地確認(2026-07-11,app.mmt.gg V5.118):
//   OHLCV(OHLCVAccessor): open() high() low() close() volume()
//        buyVolume(off?) sellVolume(off?) buyCount() sellCount() unix() calc()
//   BOOK (BookAccessor): bestBid() bestAsk() bidPrice(lvl) askPrice(lvl)
//        bidSize(lvl) askSize(lvl) binSize() calc() depthMany() imbalanceMany()
//        depth(refPrice: number, ...)
//        imbalance(refPrice: number, outerPct: number, opts?: BookHelperOptions,
//                  off?: number): BookImbalanceResult   ← 高階,未用(簽名已存)
//   STAT (StatAccessor): buyLiq(off?) sellLiq(off?) fundingRate() markPrice() unix()
// 本版刻意只用「無歧義的簡單取值器」手算 near-touch 深度;未來若要換內建
// book.imbalance(refPrice, outerPct) 可省迴圈,但需先確認 BookImbalanceResult 欄位。
// ═══════════════════════════════════════════════════════════════════

indicator("Liquidity Vacuum 撤單壓力", false)   // false = 副圖

// ── inputs(顯示用分類,非調參)──────────────────────────────
const nearBps   = input.int("Near-touch 範圍 (bps)", 10)
const maxLevels = input.int("掃描檔位上限", 50)
const extremeLb = input.int("止損區極值回看 (bars)", 1440)   // 1m 圖 = 24h
const volLb     = input.int("巨量基準回看 (bars)", 1440)
const spikeMult = input.float("巨量倍數 (vs 均量)", 8.0)      // ≈ top 1%
const skewGate  = input.float("警示偏斜門檻", 0.3)

const candles = subscribe(data.OHLCV)
const book    = subscribe(data.BOOK)
const stat    = subscribe(data.STAT)

// 上一根 bar 的 near-touch 名目深度(快照差分用)
let prevBid = null
let prevAsk = null

// near-touch 名目深度:從 best 往外掃檔位,累加落在 ±nearBps 內的 price*size
function nearTouchDepth() {
    const bb = book.bestBid()
    const ba = book.bestAsk()
    if (bb == null || ba == null) return null
    const mid = (bb + ba) / 2
    const lo = mid * (1 - nearBps / 10000)
    const hi = mid * (1 + nearBps / 10000)
    let bid = 0
    let ask = 0
    for (let i = 0; i < maxLevels; i++) {
        const bp = book.bidPrice(i)
        if (bp != null && bp >= lo) bid += bp * (book.bidSize(i) || 0)
        const ap = book.askPrice(i)
        if (ap != null && ap <= hi) ask += ap * (book.askSize(i) || 0)
        if ((bp == null || bp < lo) && (ap == null || ap > hi)) break
    }
    return { bid: bid, ask: ask, mid: mid }
}

function onBar(index) {
    const c = candles.close()
    const v = candles.volume()
    if (c == null) return

    // ── 1) 撤單偏斜(near-touch 深度淨減少 − 該側成交)───────────
    const buyNotional  = (candles.buyVolume()  || 0) * c
    const sellNotional = (candles.sellVolume() || 0) * c

    const d = nearTouchDepth()
    let skew = null
    if (d != null && prevBid != null) {
        const bidPull = Math.max(0, prevBid - d.bid - sellNotional)  // 買側被抽
        const askPull = Math.max(0, prevAsk - d.ask - buyNotional)   // 賣側被抽
        const tot = bidPull + askPull
        skew = tot > 0 ? (askPull - bidPull) / tot : 0   // +1=賣側被抽(向上真空)
    }
    if (d != null) { prevBid = d.bid; prevAsk = d.ask }

    if (skew != null)
        plot("撤單偏斜", skew, {
            style: "histogram",
            color: skew > 0 ? color.green : color.red
        })

    // ── 2) 被迫流徽章(巨量 × 清算)──────────────────────────
    const liq = (stat.buyLiq() || 0) + (stat.sellLiq() || 0)
    const volAvg = ta.sma(candles.volume, volLb)
    if (volAvg != null && v > spikeMult * volAvg) {
        marker(index, liq > 0 ? "⚡被迫流" : "巨量", { color: color.orange })
    }

    // ── 3) 止損區逼近警示(P-wall-pull 即時視圖)──────────────
    const hi24 = ta.highest(candles.high, extremeLb)
    const lo24 = ta.lowest(candles.low, extremeLb)
    if (skew != null && hi24 != null && lo24 != null) {
        const nearHi = c >= hi24 * (1 - 0.001) && c < hi24
        const nearLo = c <= lo24 * (1 + 0.001) && c > lo24
        if (nearHi && skew > skewGate)
            marker(index, "🎯上方止損區+賣側撤離", { color: color.green })
        if (nearLo && skew < -skewGate)
            marker(index, "🎯下方止損區+買側撤離", { color: color.red })
    }
}
