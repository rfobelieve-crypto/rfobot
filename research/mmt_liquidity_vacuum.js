//@version=2
// ═══════════════════════════════════════════════════════════════════
// Liquidity Vacuum 撤單壓力 — MMT custom indicator (draft 2026-07-11)
//
// 把「巨量=機構撤單→真空→掃止損的被迫流」世界觀做成即時可視化。
// 三個組件,一一對應 flow_system 的三條研究線:
//   1) 撤單偏斜直方圖  ← cancel_skew(depth_deltas 的 bar 級 proxy)
//   2) 被迫流徽章      ← 巨量×清算共現(實測 92%/35x 濃縮)
//   3) 止損區逼近警示  ← P-wall-pull(逼近 24h 極值 + 對側被抽)
//
// ⚠️ 紀律(與 flow_system 一致):
//   - 這是「監視工具」不是「已驗證 edge」——撤單領先性的正式判決在
//     2026-08-10 的 cancel_lead_ic checkpoint,在那之前它只是眼睛。
//   - 只放 Personal,不發 Community(edge 不分享,2026-07-10 決策)。
//   - 所有門檻都是「顯示用分類」,不是調參出來的交易訊號。
//   - MMT 的 BOOK 是 bar 級快照:撤單估計 = 相鄰快照深度差 − 該側成交量
//     (bar 內「抽走又掛回」互相抵銷看不見;比自建 collector 粗,方向一致)。
//
// TODO(3 處):在 MMT 文件/範例裡確認實際欄位名後替換:
//   [T1] data.BOOK 的 bid/ask 檔位陣列取法(價格、數量)
//   [T2] data.OHLCV 的買/賣量拆分欄位(trade splits)
//   [T3] data.STAT 的清算欄位(多空清算額)
// ═══════════════════════════════════════════════════════════════════

indicator("Liquidity Vacuum 撤單壓力", false);   // false = 副圖

// ── inputs(顯示用分類參數,非調參)─────────────────────────────
const nearBps   = input.int("Near-touch 深度範圍 (bps)", 10);
const extremeLb = input.int("止損區極值回看 (bars)", 1440);   // 1m 圖 = 24h
const volLb     = input.int("巨量基準回看 (bars)", 1440);
const spikeMult = input.float("巨量倍數 (vs 均量)", 8.0);     // ≈ top 1%
const liqMult   = input.float("重清算倍數 (vs 均清算)", 10.0);

const ohlcv = subscribe(data.OHLCV);
const book  = subscribe(data.BOOK);
const stat  = subscribe(data.STAT);

// 上一根 bar 的 near-touch 深度(供快照差分)
let prevBidNear = null, prevAskNear = null;

function nearTouchDepth(bookBar, mid) {
  // [T1] 依實際 BOOK 結構取檔位。假設形如
  //      bookBar.bids = [{price, size}, ...], bookBar.asks = [...]
  const lo = mid * (1 - nearBps / 10000);
  const hi = mid * (1 + nearBps / 10000);
  let bid = 0, ask = 0;
  for (const l of (bookBar.bids || [])) if (l.price >= lo) bid += l.size * l.price;
  for (const l of (bookBar.asks || [])) if (l.price <= hi) ask += l.size * l.price;
  return { bid, ask };
}

function onBar(i) {
  const c = ohlcv.close[i], h = ohlcv.high[i], l = ohlcv.low[i];
  const v = ohlcv.volume[i];
  if (c == null || book[i] == null) return;

  // ── 1) 撤單偏斜(bar 級快照差分 proxy)─────────────────────
  const d = nearTouchDepth(book[i], c);
  let skew = null;
  if (prevBidNear != null) {
    // [T2] 買/賣主動成交量拆分;先用 volume/2 佔位(有 splits 後替換)
    const buyVol  = (ohlcv.buyVolume  && ohlcv.buyVolume[i])  || v / 2;
    const sellVol = (ohlcv.sellVolume && ohlcv.sellVolume[i]) || v / 2;
    // 撤單下限估計 = 深度淨減少 − 被成交吃掉的部分(不為負)
    const bidPull = Math.max(0, prevBidNear - d.bid - sellVol * c * 0);
    const askPull = Math.max(0, prevAskNear - d.ask - buyVol  * c * 0);
    // 註:成交扣減需以「該側名目」為單位,拿到 [T2] 真欄位後把 *0 改成 *1
    const tot = bidPull + askPull;
    skew = tot > 0 ? (askPull - bidPull) / tot : 0;   // +1=賣側被抽(向上真空)
  }
  prevBidNear = d.bid; prevAskNear = d.ask;

  if (skew != null)
    plot("vacuum_skew", skew, { style: "histogram",
      color: skew > 0 ? "#36ffae" : "#ff5f6d" });

  // ── 2) 被迫流徽章(巨量 × 清算共現)────────────────────────
  const volAvg = sma(ohlcv.volume, i, volLb);
  // [T3] 清算欄位:假設 stat[i].liqLong / liqShort(USD)
  const liq = ((stat[i] && (stat[i].liqLong || 0) + (stat[i].liqShort || 0)) || 0);
  const liqAvg = smaLiq(i, volLb);
  if (volAvg && v > spikeMult * volAvg) {
    const forced = liqAvg != null && liq > liqMult * Math.max(liqAvg, 1);
    marker(forced ? "⚡被迫流" : "巨量", i,
           { color: forced ? "#f5b544" : "#9aa0a6", position: "bottom" });
  }

  // ── 3) 止損區逼近警示(P-wall-pull 即時視圖)────────────────
  const hi24 = highest(ohlcv.high, i, extremeLb);
  const lo24 = lowest(ohlcv.low,  i, extremeLb);
  const nearHi = hi24 && c >= hi24 * (1 - 0.001) && c < hi24;
  const nearLo = lo24 && c <= lo24 * (1 + 0.001) && c > lo24;
  if (skew != null) {
    if (nearHi && skew > 0.3)
      marker("🎯上方止損區+賣側撤離", i, { color: "#36ffae", position: "top" });
    if (nearLo && skew < -0.3)
      marker("🎯下方止損區+買側撤離", i, { color: "#ff5f6d", position: "top" });
  }
}

// ── 小工具(若 MMT 內建 ta.* 則用內建替換)────────────────────
function sma(series, i, n) {
  let s = 0, k = 0;
  for (let j = Math.max(0, i - n + 1); j <= i; j++) { s += series[j] || 0; k++; }
  return k ? s / k : null;
}
let liqHist = [];
function smaLiq(i, n) {
  const cur = ((stat[i] && (stat[i].liqLong || 0) + (stat[i].liqShort || 0)) || 0);
  liqHist.push(cur); if (liqHist.length > n) liqHist.shift();
  return liqHist.length ? liqHist.reduce((a, b) => a + b, 0) / liqHist.length : null;
}
function highest(series, i, n) {
  let m = null;
  for (let j = Math.max(0, i - n + 1); j < i; j++)
    if (series[j] != null && (m == null || series[j] > m)) m = series[j];
  return m;
}
function lowest(series, i, n) {
  let m = null;
  for (let j = Math.max(0, i - n + 1); j < i; j++)
    if (series[j] != null && (m == null || series[j] < m)) m = series[j];
  return m;
}
