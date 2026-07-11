//@version=2
// ═══════════════════════════════════════════════════════════════════
// Liquidity Vacuum 撤單壓力 — MMT custom indicator (v4, 2026-07-11)
//
// 「巨量=機構撤單→真空→掃止損的被迫流」世界觀可視化。
//   1) 撤單偏斜         ← near-touch 深度快照差分 − 成交(cancel proxy)
//   2) 被迫流旗標       ← 巨量×清算共現(flow_system 實測 92%/35x)
//   3) 止損區掃離風險   ← P-wall-pull 即時視圖
//
// ⚠️ 監視工具,非已驗證 edge(判決 2026-08-10);只放 Personal;門檻=顯示分類。
//
// v4 修正(依 MMT Problems 面板實測):
//   - ta.sma/highest/lowest 第一參數要「數值」→ 傳 candles.volume() 呼叫值,非參照
//   - plot 的 style 不吃字串("histogram" 報 BaseLineStyleValue)→ 暫移除,畫線
//   - marker() 不可呼叫(它是形狀列舉 circle/square/up/down...)→ 旗標改用 plot 輸出
//     ↑ 這兩個花俏 API 的正確名稱待 autocomplete 補查後再升級(見末尾 TODO)
// API 已確認:OHLCV close/high/low/volume/buyVolume/sellVolume;
//   BOOK bestBid/bestAsk/bidPrice(lvl)/askPrice(lvl)/bidSize/askSize;
//   STAT buyLiq/sellLiq。
// ═══════════════════════════════════════════════════════════════════

indicator("Liquidity Vacuum 撤單壓力", false)   // false = 副圖

const nearBps   = input.int("Near-touch 範圍 (bps)", 10)
const maxLevels = input.int("掃描檔位上限", 50)
const extremeLb = input.int("止損區極值回看 (bars)", 1440)
const volLb     = input.int("巨量基準回看 (bars)", 1440)
const spikeMult = input.float("巨量倍數 (vs 均量)", 8.0)
const skewGate  = input.float("警示偏斜門檻", 0.3)

const candles = subscribe(data.OHLCV)
const book    = subscribe(data.BOOK)
const stat    = subscribe(data.STAT)

let prevBid = null
let prevAsk = null

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

    // ta.* 每根都要呼叫以維持滾動狀態 → 傳「呼叫後的數值」
    const volAvg = ta.sma(v == null ? 0 : v, volLb)
    const hi24 = ta.highest(candles.high() == null ? 0 : candles.high(), extremeLb)
    const lo24 = ta.lowest(candles.low() == null ? 1e12 : candles.low(), extremeLb)
    if (c == null) return

    // ── 1) 撤單偏斜(near-touch 深度淨減少 − 該側成交)───────────
    const buyNotional  = (candles.buyVolume()  || 0) * c
    const sellNotional = (candles.sellVolume() || 0) * c
    const d = nearTouchDepth()
    let skew = 0
    if (d != null && prevBid != null) {
        const bidPull = Math.max(0, prevBid - d.bid - sellNotional)  // 買側被抽
        const askPull = Math.max(0, prevAsk - d.ask - buyNotional)   // 賣側被抽
        const tot = bidPull + askPull
        skew = tot > 0 ? (askPull - bidPull) / tot : 0   // +1=賣側被抽(向上真空)
    }
    if (d != null) { prevBid = d.bid; prevAsk = d.ask }
    plot("撤單偏斜", skew, { color: skew > 0 ? color.green : color.red })

    // ── 2) 被迫流旗標(巨量 × 清算)= 0 / 1 / 2 ────────────────
    const liq = (stat.buyLiq() || 0) + (stat.sellLiq() || 0)
    const isSpike = volAvg != null && v != null && v > spikeMult * volAvg
    const forcedFlag = isSpike ? (liq > 0 ? 2 : 1) : 0   // 2=被迫流(巨量+清算) 1=純巨量
    plot("被迫流旗標", forcedFlag, { color: color.blue })

    // ── 3) 止損區掃離風險 = +1(上緣被賣側撤離)/ -1(下緣買側撤離)──
    let stopRisk = 0
    if (hi24 != null && lo24 != null) {
        const nearHi = c >= hi24 * (1 - 0.001) && c < hi24
        const nearLo = c <= lo24 * (1 + 0.001) && c > lo24
        if (nearHi && skew > skewGate) stopRisk = 1
        if (nearLo && skew < -skewGate) stopRisk = -1
    }
    plot("止損掃離風險", stopRisk, { color: color.blue })
}

// ── TODO(升級成原設計的視覺,需 autocomplete 補查兩個 API 名)─────
//  A) histogram 樣式:plot 的 style 欄位吃 BaseLineStyleValue 列舉。
//     在編輯器打 `plot("x", 0, { style: ` 後看 autocomplete 跳出的列舉名,
//     或打 `style.` / `plotStyle.` 看有沒有 histogram。找到後把撤單偏斜那條
//     plot 補回 { style: <histogram列舉>, color: ... }。
//  B) 標記函式:marker 是形狀列舉(circle/square/diamond/up/down/left/right/
//     cross/plus/asterisk),畫標記的「函式」另有其名。在編輯器打 `plot`
//     開頭看有沒有 plotMarker / plotShape / drawMarker 之類,確認簽名後
//     把 forcedFlag==2 與 stopRisk!=0 改回 emoji 標記(⚡ / 🎯)。
