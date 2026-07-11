//@version=2
// ═══════════════════════════════════════════════════════════════════
// Liquidity Vacuum 撤單壓力 — MMT custom indicator (draft v2, 2026-07-11)
//
// 把「巨量=機構撤單→真空→掃止損的被迫流」世界觀做成即時可視化:
//   1) 撤單偏斜直方圖  ← cancel_skew 的 bar 級快照差分 proxy
//   2) 被迫流徽章      ← 巨量×清算共現(flow_system 實測 92%/35x)
//   3) 止損區逼近警示  ← P-wall-pull 即時視圖
//
// ⚠️ 紀律:監視工具,非已驗證 edge(正式判決 = 2026-08-10 cancel_lead_ic);
//   只放 Personal 不發 Community;門檻是顯示分類非調參。
//
// 語法基準:MMT 官方模板(accessor 風格 candles.close())。
// [T1][T2][T3] = 待確認的取值器名——在編輯器裡打 `book.` / `stat.` /
// `candles.` 看自動補全清單即可揭曉,然後照補全改名。
// ═══════════════════════════════════════════════════════════════════

indicator("Liquidity Vacuum 撤單壓力", false)   // false = 副圖

// ── inputs(顯示用分類,非調參)──────────────────────────────
const nearBps   = input.int("Near-touch 範圍 (bps)", 10)
const extremeLb = input.int("止損區極值回看 (bars)", 1440)   // 1m 圖 = 24h
const volLb     = input.int("巨量基準回看 (bars)", 1440)
const spikeMult = input.float("巨量倍數 (vs 均量)", 8.0)      // ≈ top 1%
const liqMult   = input.float("重清算倍數 (vs 均清算)", 10.0)
const skewGate  = input.float("警示偏斜門檻", 0.3)

const candles = subscribe(data.OHLCV)
const book    = subscribe(data.BOOK)
const stat    = subscribe(data.STAT)

// 上一根 bar 的 near-touch 深度(快照差分用)
let prevBid = null
let prevAsk = null

// [T1] BOOK 取值器待確認:假設 book.bids() / book.asks() 回傳
//      [{price, size}, ...];若補全顯示別的名字(levels/depth 等)照改。
function nearTouchDepth(mid) {
    const lo = mid * (1 - nearBps / 10000)
    const hi = mid * (1 + nearBps / 10000)
    let bid = 0
    let ask = 0
    for (const lv of (book.bids() || [])) if (lv.price >= lo) bid += lv.price * lv.size
    for (const lv of (book.asks() || [])) if (lv.price <= hi) ask += lv.price * lv.size
    return { bid: bid, ask: ask }
}

function onBar(index) {
    const c = candles.close()
    const v = candles.volume()
    if (c == null) return

    // ── 1) 撤單偏斜(快照差分 − 成交)────────────────────────
    // [T2] 買賣量拆分待確認:docs 說 OHLCV 含 "trade splits"——
    //      在編輯器打 `candles.` 找類似 buyVolume()/sellVolume() 的方法;
    //      找到前先設 0(= 撤單估計偏保守,只看深度淨減少)。
    const buyNotional  = 0   // ← [T2] e.g. candles.buyVolume()  * c
    const sellNotional = 0   // ← [T2] e.g. candles.sellVolume() * c

    const d = nearTouchDepth(c)
    let skew = null
    if (prevBid != null) {
        const bidPull = Math.max(0, prevBid - d.bid - sellNotional)
        const askPull = Math.max(0, prevAsk - d.ask - buyNotional)
        const tot = bidPull + askPull
        skew = tot > 0 ? (askPull - bidPull) / tot : 0   // +1=賣側被抽(向上真空)
    }
    prevBid = d.bid
    prevAsk = d.ask

    if (skew != null)
        plot("撤單偏斜", skew, {
            style: "histogram",
            color: skew > 0 ? color.green : color.red
        })

    // ── 2) 被迫流徽章(巨量 × 清算)──────────────────────────
    // [T3] 清算取值器待確認:在編輯器打 `stat.` 找 liquidation 相關方法
    const liq = 0            // ← [T3] e.g. stat.liqBuy() + stat.liqSell()
    const volAvg = ta.sma(candles.volume, volLb)   // 若 ta.sma 簽名不同照補全改
    if (volAvg != null && v > spikeMult * volAvg) {
        // 清算欄位補上前,先只標「巨量」;補上後區分 ⚡被迫流 / 一般巨量
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
