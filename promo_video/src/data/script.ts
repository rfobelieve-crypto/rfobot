/**
 * Voiceover script for V7 promo video.
 *
 * Each scene: start/end in frames (30fps), voiceover_zh, on-screen text,
 * and a hint of what visuals belong on-screen.
 *
 * Total: 14 scenes ≈ 12:30 @ 30fps = 22,500 frames
 *
 * VO production workflow:
 *   1. Extract `voiceover_zh` from each scene into ElevenLabs / Azure
 *   2. Save as assets/audio/scene_01.mp3, scene_02.mp3, ...
 *   3. Remotion picks them up automatically via Audio component
 */

export type Scene = {
  id: number;
  title: string;
  start: number;        // frame
  end: number;          // frame
  vo_zh: string;        // voiceover (Chinese, TTS-ready)
  on_screen: string;    // big text on screen
  visual_hint: string;  // for designer / dev reference
};

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

const s = (sec: number) => Math.round(sec * FPS);

export const SCENES: Scene[] = [
  {
    id: 1,
    title: 'Cold Open',
    start: s(0.0), end: s(14.0),
    vo_zh:
      '凌晨三點，BTC 突然跳水百分之五。' +
      '你還在睡。' +
      '當你醒來，倉位是早就被強平了，' +
      '還是被一個寫好風控的系統自動處理掉？',
    on_screen: '凌晨 3:00 AM',
    visual_hint:
      'Dark scene. BTC chart silhouette dropping. Clock 03:00 ticking. ' +
      'Slow zoom on a sleeping smartphone. Heart-rate-style red bar.',
  },
  {
    id: 2,
    title: 'Hook',
    start: s(14.0), end: s(33.2),
    vo_zh:
      '我花了四個月，把一個訊號指標系統，' +
      '一步一步演化成可以自動下單的量化交易機器人。' +
      '從盯盤、到 paper trading 驗證，' +
      '到上週剛接上 OKX 真實帳戶。' +
      '今天帶你看完整套流程怎麼運作。',
    on_screen: 'V7 BTC 量化交易系統',
    visual_hint:
      'Big title card fade-in. Subtitle: "從訊號到自動下單的完整工程實踐". ' +
      'Soft gradient background (deep blue → black).',
  },
  {
    id: 3,
    title: 'Problem',
    start: s(33.2), end: s(62.3),
    vo_zh:
      '為什麼自己做？' +
      '市面上的量化產品有三個共通問題：' +
      '第一，訊號黑盒，你不知道它為什麼說多或空。' +
      '第二，沒有 staged rollout，新策略上線即實盤，賠了才知道。' +
      '第三，沒有 hard kill switch，' +
      '黑天鵝來的時候，紀律永遠輸給情緒。' +
      '我想做一套自己看得懂、能信、能慢慢加碼的東西。',
    on_screen: '為什麼自己做',
    visual_hint:
      'Three problem cards slide in left→right:\n' +
      '1. 黑盒訊號  2. 上線即實盤  3. 沒 kill switch\n' +
      'Each card flips revealing the issue with a red X.',
  },
  {
    id: 4,
    title: 'Architecture Overview',
    start: s(62.3), end: s(88.0),
    vo_zh:
      '系統分兩個獨立服務：' +
      '左邊是市場數據收集，' +
      '直接連 Binance 跟 OKX 的 WebSocket 收原始 trades。' +
      '右邊是 indicator 服務，' +
      '跑 V7 dual XGBoost 模型，輸出方向跟強度。' +
      '兩邊用同一個 MySQL 同步狀態。' +
      '中間掛 Telegram bot 接 webhook，' +
      '所有訊號跟 alert 推到我的手機。',
    on_screen: '兩服務 + 共用 MySQL',
    visual_hint:
      'Animated architecture diagram: two boxes (market_data | indicator) ' +
      'with arrows flowing through MySQL in the middle. ' +
      'Telegram bot at the top. Use the mermaid diagram from assets/.',
  },
  {
    id: 5,
    title: 'V7 Model',
    start: s(88.0), end: s(113.6),
    vo_zh:
      'V7 模型是核心。' +
      '兩百個工程特徵分十二個群組，' +
      '都是 trailing-only 計算，嚴格避免 look-ahead bias。' +
      'Direction model 輸出未來四小時的 path return 預測，' +
      '用 rolling percentile 解碼成 UP DOWN 或 NEUTRAL。' +
      'Magnitude model 預測 absolute return。' +
      '兩個獨立 XGBoost regressor，每週重訓一次。',
    on_screen: 'V7 Dual XGBoost\n200 features',
    visual_hint:
      'Feature importance horizontal bars animate in. ' +
      'Top features: cg_oi_close_pctchg_8h, vol_kurtosis, impact_asymmetry... ' +
      'Then show two model boxes (Direction | Magnitude) connecting.',
  },
  {
    id: 6,
    title: 'Stage Framework',
    start: s(113.6), end: s(141.4),
    vo_zh:
      '上線分五個階段，不是一次到位。' +
      'Stage 1 paper trading 0 風險虛擬下單。' +
      'Stage 2 在真實環境跑 read-only smoke test 驗證 code。' +
      'Stage 3 拿一百塊美金跑 live，輸光不痛。' +
      'Stage 4 累積到一千、五千、一萬美金分階段放大。' +
      '每個 stage 進階都要 hit 硬條件，不是覺得最近表現好就跳。',
    on_screen: 'Stage 0 → 1 → 2 → 3 → 4',
    visual_hint:
      'Progress staircase animation. Each step lights up with money amount: ' +
      '$0 paper / $0 testnet / $100 live / $1k / $5k / $10k+. ' +
      'Highlight Stage 3 in green (current).',
  },
  {
    id: 7,
    title: 'Kill Switches',
    start: s(141.4), end: s(172.2),
    vo_zh:
      'Kill switch 是這套系統的最後防線。' +
      '單日虧損百分之二十、累計虧損百分之三十、' +
      '與交易所失聯超過五分鐘、reconcile 不一致，' +
      '都會自動 halt 或 demote。' +
      '為什麼槓桿上限訂二倍？' +
      'Kelly criterion 算出來 optimal 是零點五六倍，' +
      '一倍已經超過 Kelly。' +
      '二倍 vol drag 還可以被 edge 蓋過，' +
      '三倍以上 drag 大過期望報酬，長期數學上必虧。',
    on_screen: 'Kelly < 1x\n2x Hard Cap',
    visual_hint:
      'Math formula reveal: f* = μ/σ². Then leverage vs drag bar chart: ' +
      '1x: -5%, 2x: -18%, 3x: -40%, 5x: -112% (ruin). Last bar in red.',
  },
  {
    id: 8,
    title: 'Telegram Demo',
    start: s(172.2), end: s(201.1),
    vo_zh:
      'Telegram 是主要互動介面。' +
      '我用按鈕設計把所有功能擺在 thumb-reach。' +
      'Chart 是即時 K 線疊三角形訊號。' +
      'Perf 是當前績效報告。' +
      'OKX Perf 是 live 帳戶餘額 跟 Sharpe。' +
      '第一筆真實下單，bot 會推 PENDING 訊息給我，' +
      '我回 yes 才會真的執行。' +
      '一筆人工確認過後切自動，' +
      '量化交易就回到該有的樣子。',
    on_screen: 'Telegram = 主控台',
    visual_hint:
      'Mock TG conversation: signal alert with inline buttons. ' +
      'Show PENDING #1 → user types /yes_1 → entry alert appears. ' +
      'Phone-frame mockup, finger-tap animation on buttons.',
  },
  {
    id: 9,
    title: 'Paper Track Record',
    start: s(201.1), end: s(228.6),
    vo_zh:
      '五月十八號開始 paper trading。' +
      '十三天，九筆 trade，累計報酬八點五個百分點。' +
      '勝率六成二，平均每筆淨九十六個基點。' +
      '結構是少數大贏家撐起大部分績效，' +
      '這是 trailing stop 策略的典型樣貌。' +
      'Edge 有跡象，但 N 等於八樣本太少，' +
      '統計上還不能定論，需要更多 trades 才篤定。',
    on_screen: '13 天 +8.5%',
    visual_hint:
      'Equity curve from $1000 → $1085 animating left-to-right. ' +
      'Each trade marker shows the result. ' +
      'Sidebar stats: 9 trades, WR 62.5%, +96bps avg, σ 130bps.',
  },
  {
    id: 10,
    title: 'OKX Live Launch',
    start: s(228.6), end: s(259.5),
    vo_zh:
      '上週把 OKX 接上來，' +
      '初始保證金一百美金，十倍槓桿。' +
      '為什麼十倍？因為 BTC 一張合約名義價值約七百五十美金，' +
      '一百保證金加一倍槓桿連一張合約都開不了。' +
      '所以我做了 informed override：' +
      '保證金壓低、槓桿放寬到十倍、' +
      '同時把單日虧損上限從五十收緊到二十、' +
      '累計上限從五十收緊到三十，補償風險放大。',
    on_screen: '$100 + 10x leverage',
    visual_hint:
      'Math reveal: 1 contract = 0.01 BTC × $73k = $730 notional. ' +
      'Then show the trade-off: leverage ↑ → kill switch tightened ↓. ' +
      'Highlight the explicit "informed override" decision date.',
  },
  {
    id: 11,
    title: 'Engineering Hardening',
    start: s(259.5), end: s(291.2),
    vo_zh:
      '從本地測試過到 Railway 真實上線，' +
      '抓了七個如果沒實際連線根本不會發現的 bug。' +
      'OKX 用 read_only 不是 read。' +
      'GET 帶 query string 但簽章沒包含。' +
      'API secret 貼到 Railway 時最後一個字被吃掉。' +
      '帳戶 posMode 跟系統假設不符。' +
      'WS reconnect counter 錯誤計入正常 close。' +
      '所有 bug 都寫進 mistake log，' +
      '同類錯誤未來抓 10 秒解決。',
    on_screen: '7 個真實 bug + mistake.md',
    visual_hint:
      'Bug list scrolls upward. Each bug ✗ → fix ✓ animation. ' +
      'End with the mistake.md file opening — show "Rule:" lines as the moral.',
  },
  {
    id: 12,
    title: 'Dashboard Tour',
    start: s(291.2), end: s(315.7),
    vo_zh:
      'Dashboard 分三個 endpoint。' +
      '/perf 看 indicator 的訊號績效跟 IC 衰退。' +
      '/paper-perf 看 paper cohort 的逐筆紀錄與曲線。' +
      '/okx-perf 看 live 帳戶餘額、勝率、Sharpe、' +
      '當前部位跟 kill log 歷史。' +
      '所有數據從 MySQL 即時計算，' +
      '沒任何 cache 騙我自己。',
    on_screen: '/perf  /paper-perf  /okx-perf',
    visual_hint:
      'Three browser tabs animate in showing screenshots/mock-ups of ' +
      'each dashboard. Highlight Sharpe / equity / open position panels.',
  },
  {
    id: 13,
    title: 'Why Different',
    start: s(315.7), end: s(346.0),
    vo_zh:
      '這套跟市售量化的差別：' +
      '第一，訊號全開源，' +
      '每個特徵的 IC validation 紀錄都在 commit history 裡。' +
      '第二，staged rollout 寫進 production code，' +
      'hit hard rule 自動降階，不靠紀律。' +
      '第三，258 個 unit tests 覆蓋 OKX 整合，' +
      '每個 kill switch 都 testnet 驗證過能真的觸發。' +
      '第四，所有踩過的坑都寫進 mistake.md，' +
      '不重複同樣的錯。',
    on_screen: '透明 / 紀律 / 測試 / 學習',
    visual_hint:
      '4-quadrant comparison: "Off-the-shelf 量化" vs "V7". ' +
      'Each row: transparency / rollout / tests / learning. ' +
      'V7 column all green checkmarks; competitor column ?/✗.',
  },
  {
    id: 14,
    title: 'Current Status + Roadmap',
    start: s(346.0), end: s(378.7),
    vo_zh:
      '坦白現況：' +
      'OKX live 上週才啟動，目前零筆 live trade，' +
      'paper 13 天 +8.5% 是初步信號不是定論。' +
      '我的計畫是先累積三十筆 live trades，' +
      '看 Sharpe、看跨 regime 表現、看黑天鵝怎麼處理。' +
      '如果 forward window 撐住，' +
      '會開放有限名額 beta 給有興趣的朋友。' +
      '不會今天就推訂閱。想加入 wait-list 留言我會私訊你。' +
      '感謝看到這裡。',
    on_screen: 'beta 開放需 30+ live trades',
    visual_hint:
      'Honest timeline: 5/27 OKX integration → 5/31 live ACTIVE → ' +
      '0 trades today → "30+ trades to validate" → "beta open if Sharpe > 1.5". ' +
      'End on contact info / wait-list QR code.',
  },
];

export const TOTAL_FRAMES = SCENES[SCENES.length - 1].end;
export const TOTAL_DURATION_SEC = TOTAL_FRAMES / FPS;
