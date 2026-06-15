/**
 * Subtitle cues — RETIMED to compact scene windows by
 * scripts/retime-subtitles.py (proportional per-scene stretch).
 */
import {FPS} from './script';

export type Cue = {
  start: number;
  end: number;
  text: string;
};

const s = (sec: number) => Math.round(sec * FPS);

export const SUBTITLES: Cue[] = [
  // Scene 1
  {start: s(0.00), end: s(2.80), text: '凌晨三點，BTC 突然跳水 5%。'},
  {start: s(2.80), end: s(4.90), text: '你還在睡。'},
  {start: s(4.90), end: s(9.10), text: '當你醒來，倉位是早就被強平了，'},
  {start: s(9.10), end: s(14.00), text: '還是被寫好風控的系統自動處理掉？'},
  // Scene 2
  {start: s(14.00), end: s(17.83), text: '我花了四個月，把一個訊號指標系統，'},
  {start: s(17.83), end: s(21.67), text: '一步步演化成可以自動下單的量化機器人。'},
  {start: s(21.67), end: s(25.53), text: '從盯盤、到 paper trading 驗證，'},
  {start: s(25.53), end: s(29.37), text: '到上週剛接上 OKX 真實帳戶。'},
  {start: s(29.37), end: s(33.20), text: '今天帶你看完整套流程怎麼運作。'},
  // Scene 3
  {start: s(33.20), end: s(36.10), text: '為什麼自己做？'},
  {start: s(36.10), end: s(40.47), text: '市面上量化產品有三個共通問題：'},
  {start: s(40.47), end: s(45.57), text: '一、訊號黑盒，不知道為什麼說多空。'},
  {start: s(45.57), end: s(50.67), text: '二、沒 staged rollout，上線即實盤。'},
  {start: s(50.67), end: s(55.77), text: '三、沒 hard kill switch，紀律輸給情緒。'},
  {start: s(55.77), end: s(62.30), text: '我想要看得懂、能信、能慢慢加碼的東西。'},
  // Scene 4
  {start: s(62.30), end: s(64.43), text: '系統分兩個獨立服務：'},
  {start: s(64.43), end: s(67.43), text: '左邊市場數據，連 Binance / OKX WebSocket。'},
  {start: s(67.43), end: s(70.87), text: '右邊 indicator，跑 V7 dual XGBoost 模型。'},
  {start: s(70.87), end: s(74.30), text: '輸出方向跟強度。'},
  {start: s(74.30), end: s(77.30), text: '兩邊用 MySQL 同步狀態。'},
  {start: s(77.30), end: s(81.57), text: 'Telegram bot 接 webhook，alert 推到手機。'},
  {start: s(81.57), end: s(88.00), text: ''},
  // Scene 5
  {start: s(88.00), end: s(90.13), text: 'V7 模型是核心。'},
  {start: s(90.13), end: s(93.53), text: '200 個工程特徵分 12 個群組。'},
  {start: s(93.53), end: s(97.40), text: '全部 trailing-only 計算，無 look-ahead。'},
  {start: s(97.40), end: s(101.67), text: 'Direction model 預測 4 小時 path return。'},
  {start: s(101.67), end: s(105.07), text: 'Rolling percentile 解碼 UP / DOWN / NEUTRAL。'},
  {start: s(105.07), end: s(108.47), text: 'Magnitude model 預測 absolute return。'},
  {start: s(108.47), end: s(113.60), text: '兩個 XGBoost regressor，每週重訓。'},
  // Scene 6
  {start: s(113.60), end: s(116.37), text: '上線分 5 個階段，不是一次到位。'},
  {start: s(116.37), end: s(119.63), text: 'Stage 1: paper trading 0 風險。'},
  {start: s(119.63), end: s(123.33), text: 'Stage 2: read-only live smoke 驗證 code。'},
  {start: s(123.33), end: s(127.03), text: 'Stage 3: $100 live，輸光不痛。'},
  {start: s(127.03), end: s(130.73), text: 'Stage 4: $1k / $5k / $10k+ 分階段。'},
  {start: s(130.73), end: s(135.37), text: '每個 stage 進階要 hit 硬條件，'},
  {start: s(135.37), end: s(141.40), text: '不是覺得最近表現好就跳。'},
  // Scene 7
  {start: s(141.40), end: s(144.47), text: 'Kill switch 是最後防線。'},
  {start: s(144.47), end: s(148.60), text: '單日 -20% / 累計 -30% / WS 斷 5min'},
  {start: s(148.60), end: s(151.67), text: '/ reconcile 不一致 → 自動 halt。'},
  {start: s(151.67), end: s(155.27), text: '為什麼槓桿上限訂 2x？'},
  {start: s(155.27), end: s(159.37), text: 'Kelly optimal 算出來 0.56x，1x 已超過。'},
  {start: s(159.37), end: s(164.50), text: '2x vol drag 還能被 edge 覆蓋。'},
  {start: s(164.50), end: s(172.20), text: '3x 以上 drag 大過期望報酬，數學上必虧。'},
  // Scene 8
  {start: s(172.20), end: s(174.60), text: 'Telegram 是主要互動介面。'},
  {start: s(174.60), end: s(178.47), text: '按鈕設計把功能擺在 thumb-reach。'},
  {start: s(178.47), end: s(181.83), text: 'Chart：即時 K 線疊三角形訊號。'},
  {start: s(181.83), end: s(184.73), text: 'Perf：當前績效報告。'},
  {start: s(184.73), end: s(188.57), text: 'OKX Perf：live 帳戶餘額跟 Sharpe。'},
  {start: s(188.57), end: s(192.43), text: '第一筆 live 下單會推 PENDING 給我，'},
  {start: s(192.43), end: s(196.27), text: '回 /yes 才執行。一筆人工確認後切自動。'},
  {start: s(196.27), end: s(201.10), text: '量化交易就回到該有的樣子。'},
  // Scene 9
  {start: s(201.10), end: s(203.87), text: '5/18 開始 paper trading。'},
  {start: s(203.87), end: s(207.53), text: '13 天 9 筆 trade，累計 +8.5%。'},
  {start: s(207.53), end: s(211.20), text: '勝率 62%，平均每筆淨 +96 bps。'},
  {start: s(211.20), end: s(215.30), text: '少數大贏家撐起大部分績效，'},
  {start: s(215.30), end: s(219.43), text: 'trailing stop 策略典型樣貌。'},
  {start: s(219.43), end: s(224.00), text: 'Edge 有跡象，但 N=8 樣本太少，'},
  {start: s(224.00), end: s(228.60), text: '統計上還不能定論。'},
  // Scene 10
  {start: s(228.60), end: s(232.73), text: '上週把 OKX 接上來，'},
  {start: s(232.73), end: s(236.33), text: '初始保證金 $100，10x 槓桿。'},
  {start: s(236.33), end: s(240.43), text: '為什麼 10x？BTC 1 contract = $730，'},
  {start: s(240.43), end: s(244.57), text: '$100 + 1x 連 1 contract 都開不了。'},
  {start: s(244.57), end: s(249.20), text: '所以做了 informed override：'},
  {start: s(249.20), end: s(253.33), text: '槓桿放寬到 10x、'},
  {start: s(253.33), end: s(259.50), text: 'kill switch 從 -50% 收緊到 -20%。'},
  // Scene 11
  {start: s(259.50), end: s(263.73), text: '從本地到 Railway 上線，抓了 7 個 bug。'},
  {start: s(263.73), end: s(267.43), text: 'OKX 用 read_only 不是 read。'},
  {start: s(267.43), end: s(271.67), text: 'GET 帶 query 但簽章沒包含。'},
  {start: s(271.67), end: s(275.87), text: 'API secret 貼到 Railway 最後一字被吃。'},
  {start: s(275.87), end: s(280.10), text: 'WS counter 錯誤計入正常 close。'},
  {start: s(280.10), end: s(285.40), text: '所有 bug 都寫進 mistake log。'},
  {start: s(285.40), end: s(291.20), text: '同類錯誤未來抓 10 秒解決。'},
  // Scene 12
  {start: s(291.20), end: s(294.47), text: 'Dashboard 分 3 個 endpoint。'},
  {start: s(294.47), end: s(297.73), text: '/perf：indicator 訊號績效跟 IC 衰退。'},
  {start: s(297.73), end: s(301.40), text: '/paper-perf：paper cohort 逐筆紀錄。'},
  {start: s(301.40), end: s(305.50), text: '/okx-perf：live 餘額、WR、Sharpe、'},
  {start: s(305.50), end: s(309.57), text: '當前部位、kill log 歷史。'},
  {start: s(309.57), end: s(315.70), text: '即時從 MySQL 算，沒 cache 騙自己。'},
  // Scene 13
  {start: s(315.70), end: s(318.73), text: '這套跟市售量化的差別：'},
  {start: s(318.73), end: s(323.27), text: '一、訊號全開源，IC validation 在 git。'},
  {start: s(323.27), end: s(328.83), text: '二、staged rollout 寫進 production code。'},
  {start: s(328.83), end: s(333.87), text: 'Hit hard rule 自動降階，不靠紀律。'},
  {start: s(333.87), end: s(338.43), text: '三、258 unit tests 覆蓋 OKX 整合。'},
  {start: s(338.43), end: s(342.47), text: '每個 kill switch testnet 驗證過。'},
  {start: s(342.47), end: s(346.00), text: '四、踩過的坑都寫進 mistake.md。'},
  // Scene 14
  {start: s(346.00), end: s(348.17), text: '坦白現況：'},
  {start: s(348.17), end: s(352.53), text: 'OKX live 上週啟動，0 筆 live trade。'},
  {start: s(352.53), end: s(357.43), text: 'Paper 13 天 +8.5% 是初步信號不是定論。'},
  {start: s(357.43), end: s(362.33), text: '計畫先累積 30+ 筆 live trades，'},
  {start: s(362.33), end: s(366.70), text: '看 Sharpe、跨 regime、黑天鵝。'},
  {start: s(366.70), end: s(371.07), text: 'Forward window 撐住才開放 beta。'},
  {start: s(371.07), end: s(375.97), text: '不會今天推訂閱。'},
  {start: s(375.97), end: s(750.00), text: '想加 wait-list 留言。感謝。'},
];
