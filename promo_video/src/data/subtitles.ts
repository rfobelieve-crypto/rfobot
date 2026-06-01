/**
 * Subtitle cues — broken into chunks short enough to read.
 *
 * Each cue: start_frame, end_frame, text (Chinese, max ~18 chars/line).
 * Auto-generated structure; tweak timing once you have the actual VO mp3.
 *
 * Remotion renders these via the <Subtitles> component in Composition.tsx.
 */
import {FPS} from './script';

export type Cue = {
  start: number;
  end: number;
  text: string;
};

const s = (sec: number) => Math.round(sec * FPS);

// Pacing: ~3-4s per cue; max 28 chars per cue for two-line Chinese display.
export const SUBTITLES: Cue[] = [
  // Scene 1: Cold Open (0-20s)
  {start: s(0), end: s(4), text: '凌晨三點，BTC 突然跳水 5%。'},
  {start: s(4), end: s(7), text: '你還在睡。'},
  {start: s(7), end: s(13), text: '當你醒來，倉位是早就被強平了，'},
  {start: s(13), end: s(20), text: '還是被寫好風控的系統自動處理掉？'},

  // Scene 2: Hook (20-50s)
  {start: s(20), end: s(26), text: '我花了四個月，把一個訊號指標系統，'},
  {start: s(26), end: s(32), text: '一步步演化成可以自動下單的量化機器人。'},
  {start: s(32), end: s(38), text: '從盯盤、到 paper trading 驗證，'},
  {start: s(38), end: s(44), text: '到上週剛接上 OKX 真實帳戶。'},
  {start: s(44), end: s(50), text: '今天帶你看完整套流程怎麼運作。'},

  // Scene 3: Problem (50-90s)
  {start: s(50), end: s(54), text: '為什麼自己做？'},
  {start: s(54), end: s(60), text: '市面上量化產品有三個共通問題：'},
  {start: s(60), end: s(67), text: '一、訊號黑盒，不知道為什麼說多空。'},
  {start: s(67), end: s(74), text: '二、沒 staged rollout，上線即實盤。'},
  {start: s(74), end: s(81), text: '三、沒 hard kill switch，紀律輸給情緒。'},
  {start: s(81), end: s(90), text: '我想要看得懂、能信、能慢慢加碼的東西。'},

  // Scene 4: Architecture (90-150s)
  {start: s(90), end: s(95), text: '系統分兩個獨立服務：'},
  {start: s(95), end: s(102), text: '左邊市場數據，連 Binance / OKX WebSocket。'},
  {start: s(102), end: s(110), text: '右邊 indicator，跑 V7 dual XGBoost 模型。'},
  {start: s(110), end: s(118), text: '輸出方向跟強度。'},
  {start: s(118), end: s(125), text: '兩邊用 MySQL 同步狀態。'},
  {start: s(125), end: s(135), text: 'Telegram bot 接 webhook，alert 推到手機。'},
  {start: s(135), end: s(150), text: ''},

  // Scene 5: V7 Model (150-210s)
  {start: s(150), end: s(155), text: 'V7 模型是核心。'},
  {start: s(155), end: s(163), text: '200 個工程特徵分 12 個群組。'},
  {start: s(163), end: s(172), text: '全部 trailing-only 計算，無 look-ahead。'},
  {start: s(172), end: s(182), text: 'Direction model 預測 4 小時 path return。'},
  {start: s(182), end: s(190), text: 'Rolling percentile 解碼 UP / DOWN / NEUTRAL。'},
  {start: s(190), end: s(198), text: 'Magnitude model 預測 absolute return。'},
  {start: s(198), end: s(210), text: '兩個 XGBoost regressor，每週重訓。'},

  // Scene 6: Stage Framework (210-270s)
  {start: s(210), end: s(216), text: '上線分 5 個階段，不是一次到位。'},
  {start: s(216), end: s(223), text: 'Stage 1: paper trading 0 風險。'},
  {start: s(223), end: s(231), text: 'Stage 2: read-only live smoke 驗證 code。'},
  {start: s(231), end: s(239), text: 'Stage 3: $100 live，輸光不痛。'},
  {start: s(239), end: s(247), text: 'Stage 4: $1k / $5k / $10k+ 分階段。'},
  {start: s(247), end: s(257), text: '每個 stage 進階要 hit 硬條件，'},
  {start: s(257), end: s(270), text: '不是覺得最近表現好就跳。'},

  // Scene 7: Kill Switches (270-330s)
  {start: s(270), end: s(276), text: 'Kill switch 是最後防線。'},
  {start: s(276), end: s(284), text: '單日 -20% / 累計 -30% / WS 斷 5min'},
  {start: s(284), end: s(290), text: '/ reconcile 不一致 → 自動 halt。'},
  {start: s(290), end: s(297), text: '為什麼槓桿上限訂 2x？'},
  {start: s(297), end: s(305), text: 'Kelly optimal 算出來 0.56x，1x 已超過。'},
  {start: s(305), end: s(315), text: '2x vol drag 還能被 edge 覆蓋。'},
  {start: s(315), end: s(330), text: '3x 以上 drag 大過期望報酬，數學上必虧。'},

  // Scene 8: Telegram Demo (330-390s)
  {start: s(330), end: s(335), text: 'Telegram 是主要互動介面。'},
  {start: s(335), end: s(343), text: '按鈕設計把功能擺在 thumb-reach。'},
  {start: s(343), end: s(350), text: 'Chart：即時 K 線疊三角形訊號。'},
  {start: s(350), end: s(356), text: 'Perf：當前績效報告。'},
  {start: s(356), end: s(364), text: 'OKX Perf：live 帳戶餘額跟 Sharpe。'},
  {start: s(364), end: s(372), text: '第一筆 live 下單會推 PENDING 給我，'},
  {start: s(372), end: s(380), text: '回 /yes 才執行。一筆人工確認後切自動。'},
  {start: s(380), end: s(390), text: '量化交易就回到該有的樣子。'},

  // Scene 9: Paper Track Record (390-450s)
  {start: s(390), end: s(396), text: '5/18 開始 paper trading。'},
  {start: s(396), end: s(404), text: '13 天 9 筆 trade，累計 +8.5%。'},
  {start: s(404), end: s(412), text: '勝率 62%，平均每筆淨 +96 bps。'},
  {start: s(412), end: s(421), text: '少數大贏家撐起大部分績效，'},
  {start: s(421), end: s(430), text: 'trailing stop 策略典型樣貌。'},
  {start: s(430), end: s(440), text: 'Edge 有跡象，但 N=8 樣本太少，'},
  {start: s(440), end: s(450), text: '統計上還不能定論。'},

  // Scene 10: OKX Live (450-510s)
  {start: s(450), end: s(458), text: '上週把 OKX 接上來，'},
  {start: s(458), end: s(465), text: '初始保證金 $100，10x 槓桿。'},
  {start: s(465), end: s(473), text: '為什麼 10x？BTC 1 contract = $730，'},
  {start: s(473), end: s(481), text: '$100 + 1x 連 1 contract 都開不了。'},
  {start: s(481), end: s(490), text: '所以做了 informed override：'},
  {start: s(490), end: s(498), text: '槓桿放寬到 10x、'},
  {start: s(498), end: s(510), text: 'kill switch 從 -50% 收緊到 -20%。'},

  // Scene 11: Engineering (510-570s)
  {start: s(510), end: s(518), text: '從本地到 Railway 上線，抓了 7 個 bug。'},
  {start: s(518), end: s(525), text: 'OKX 用 read_only 不是 read。'},
  {start: s(525), end: s(533), text: 'GET 帶 query 但簽章沒包含。'},
  {start: s(533), end: s(541), text: 'API secret 貼到 Railway 最後一字被吃。'},
  {start: s(541), end: s(549), text: 'WS counter 錯誤計入正常 close。'},
  {start: s(549), end: s(559), text: '所有 bug 都寫進 mistake log。'},
  {start: s(559), end: s(570), text: '同類錯誤未來抓 10 秒解決。'},

  // Scene 12: Dashboard Tour (570-630s)
  {start: s(570), end: s(578), text: 'Dashboard 分 3 個 endpoint。'},
  {start: s(578), end: s(586), text: '/perf：indicator 訊號績效跟 IC 衰退。'},
  {start: s(586), end: s(595), text: '/paper-perf：paper cohort 逐筆紀錄。'},
  {start: s(595), end: s(605), text: '/okx-perf：live 餘額、WR、Sharpe、'},
  {start: s(605), end: s(615), text: '當前部位、kill log 歷史。'},
  {start: s(615), end: s(630), text: '即時從 MySQL 算，沒 cache 騙自己。'},

  // Scene 13: Why Different (630-690s)
  {start: s(630), end: s(636), text: '這套跟市售量化的差別：'},
  {start: s(636), end: s(645), text: '一、訊號全開源，IC validation 在 git。'},
  {start: s(645), end: s(656), text: '二、staged rollout 寫進 production code。'},
  {start: s(656), end: s(666), text: 'Hit hard rule 自動降階，不靠紀律。'},
  {start: s(666), end: s(675), text: '三、258 unit tests 覆蓋 OKX 整合。'},
  {start: s(675), end: s(683), text: '每個 kill switch testnet 驗證過。'},
  {start: s(683), end: s(690), text: '四、踩過的坑都寫進 mistake.md。'},

  // Scene 14: Status + Roadmap (690-750s)
  {start: s(690), end: s(694), text: '坦白現況：'},
  {start: s(694), end: s(702), text: 'OKX live 上週啟動，0 筆 live trade。'},
  {start: s(702), end: s(711), text: 'Paper 13 天 +8.5% 是初步信號不是定論。'},
  {start: s(711), end: s(720), text: '計畫先累積 30+ 筆 live trades，'},
  {start: s(720), end: s(728), text: '看 Sharpe、跨 regime、黑天鵝。'},
  {start: s(728), end: s(736), text: 'Forward window 撐住才開放 beta。'},
  {start: s(736), end: s(745), text: '不會今天推訂閱。'},
  {start: s(745), end: s(750), text: '想加 wait-list 留言。感謝。'},
];
