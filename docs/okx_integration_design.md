# OKX Integration Design — Stage 2 V7 Testnet Executor

> Version: 0.1 (initial draft, 2026-05-25)
> Owner: rfo
> Companion docs: `docs/stage2_kill_criteria.md`, `docs/stage2_testnet_checklist.md`
> 本文件只設計**架構與 interface**,不寫具體 SDK 呼叫。Stage 1 期間禁寫 execution code。

## 0. 文件性質

這份是 Stage 2 進場前的「設計凍結」(design freeze)文件。寫完之後:
- 80% 的架構決策已做完,Stage 2 開工時專注實作不爭論
- 每個模組的責任邊界清楚 — code review 有 ground truth
- 與既有系統(`indicator/app.py`, `v7_paper_executor.py`)的接觸面定清楚 — 不會 surprise

寫**架構與介面**,不寫:
- 具體 OKX SDK 呼叫(那是 Stage 2 implementation)
- UI / dashboard 細節(A2 checklist 涵蓋了)
- ML 模型本身(這份是 execution layer,模型 untouched)

---

## 1. 範圍與限制

### Stage 2 明確 in-scope

- 一個交易所:OKX
- 一個 instrument:`BTC-USDT-SWAP`(perpetual swap)
- 一個策略:V7(signal-exit + 3×ATR trailing stop)
- 一個方向:**net mode 單向倉**,任何時刻最多 1 個 open position
- 模式:**testnet 模擬交易**(`x-simulated-trading: 1`)
- 槓桿:**1x**(`leverage=1`,即使 td_mode 是 cross)

### Stage 2 明確 out-of-scope(寫死禁止)

| 項目 | 為什麼禁 |
|---|---|
| Live trading | Stage 2 定義就是 testnet 0 risk |
| Multi-asset(ETH 等) | 一次只驗證一個變量 |
| 槓桿 > 1 | CLAUDE.md Stage 4 之前不上 leverage |
| Spot 模式 | 與 paper 的 perp 假設不一致(funding / contract size) |
| 智慧訂單路由 / iceberg / TWAP | Stage 2 用 market entry + algo stop,簡單到底 |
| 訂單簿分析 / maker 優化 | 同上,Stage 2 不追求 fill quality 優化 |
| 多執行緒競爭部位 | 單 position 設計上避開 race |

---

## 2. 設計原則(non-negotiable)

這 8 條原則貫穿所有模組設計。任何模組與某條衝突 → 設計改,不是原則改。

### P1. Fail-safe 優先,即時性其次
不確定狀態 → 停止下新單。寧可錯過 signal 也不下盲單。

### P2. 真實狀態在 exchange,不是 local DB
Local DB 是 cache。**對帳結果是 source of truth**。任何衝突 OKX 端贏。

### P3. 所有副作用 idempotent
- Order 用 `clOrdId`(client order ID)確保重送不重複下單
- Reconciliation 可以重跑任意次,結果一致
- Restart 後系統能從 DB + OKX 推導出正確狀態

### P4. 死狀態優於髒狀態
進程 crash > 進程繼續但 state 不一致。
偵測到不一致 → halt 自己,等人工。**絕不** 自動「修復」 state。

### P5. WS push first,REST poll only as fallback
Order/position 變化用 WS private channel 即時推送。
REST 只用於:cold start 對帳、WS 失聯時 fallback、定期 cross-check。

### P6. 永遠不要 cancel-then-new
Trailing stop 更新用 `amend-algo-order`(atomic)。
cancel + new 中間有 race window — order 在裸奔。

### P7. 所有 kill trigger 直接對應代碼分支
每個 kill check function 的 docstring 標 `kill_criteria.md` 的 trigger ID。
不能有「實作了但沒掛在某個 trigger 上」的 code path。

### P8. Paper executor 不修改,並行 shadow 運行
Stage 2 testnet executor 與 paper executor **同時** 跑,吃同一個 signal。
- Paper 結果是 baseline
- Testnet 結果是 reality
- 兩者 fill price delta 即是 slippage histogram(A2 checklist T-E2E-04)
- Paper 邏輯不動 = baseline 不漂移

---

## 3. 系統架構

### 3.1 模組分層

```
┌─────────────────────────────────────────────────────────┐
│  indicator/app.py update_cycle()  (existing)            │
│   └─ 餵 klines + signal 給 router                       │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  ExecutorRouter (new — indicator/executor_router.py)    │
│  根據 config 同時呼叫多個 executor:                       │
│    - paper(永遠開,Stage 1+ baseline)                   │
│    - okx(Stage 2 / Stage 3 / Stage 4)                  │
└──────────┬────────────────────────┬─────────────────────┘
           ▼                        ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ V7PaperExecutor          │  │ V7OkxExecutor           │
│ (existing, unchanged)    │  │ (new)                   │
│ indicator/               │  │ indicator/okx/          │
│   v7_paper_executor.py   │  │   executor.py           │
└─────────────────────────┘  └────────────┬────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │ OkxClient               │
                             │ indicator/okx/          │
                             │   client.py             │
                             │ - rest.py               │
                             │ - ws_public.py          │
                             │ - ws_private.py         │
                             └────────────┬────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │ PositionReconciler      │
                             │ indicator/okx/          │
                             │   reconciler.py         │
                             └────────────┬────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │ OkxStateStore           │
                             │ indicator/okx/state.py  │
                             │ (MySQL via shared/db.py)│
                             └─────────────────────────┘
```

### 3.2 依賴方向(嚴格 top-down)

```
app.py
  └─ ExecutorRouter
       ├─ V7PaperExecutor  →  v7_paper_positions table
       └─ V7OkxExecutor
            ├─ OkxClient(無上游依賴,純通訊)
            ├─ PositionReconciler
            │    ├─ OkxClient(read-only)
            │    └─ OkxStateStore(read-only 比對)
            └─ OkxStateStore  →  v7_okx_* tables
```

**嚴禁**:
- 下層模組 import 上層模組
- 任何模組直接呼叫 OKX(必須走 OkxClient)
- 任何模組直接寫 v7_okx_* table(必須走 OkxStateStore)

### 3.3 新增的檔案結構

```
indicator/
├── executor_router.py          # new
├── okx/                        # new package
│   ├── __init__.py
│   ├── client.py               # OkxClient 集合介面
│   ├── rest.py                 # REST low-level (auth / retry / rate limit)
│   ├── ws_public.py            # public WS(若需要,目前 market data 已有)
│   ├── ws_private.py           # private WS(orders/positions/balance)
│   ├── executor.py             # V7OkxExecutor 核心 cycle 邏輯
│   ├── reconciler.py           # PositionReconciler
│   ├── state.py                # OkxStateStore (DB layer)
│   ├── types.py                # dataclasses(Order, Position, etc.)
│   ├── kill_checks.py          # 31 個 kill trigger check function
│   └── config.py               # 載入 / 驗證 config
├── v7_paper_executor.py        # unchanged
└── app.py                      # 小幅修改:wire ExecutorRouter

docs/
├── stage2_kill_criteria.md     # 已存在
├── stage2_testnet_checklist.md # 已存在
├── okx_integration_design.md   # 本文件
├── api_key_management.md       # 下一份(A4)
└── stage_progression_log.md    # Stage 2 啟動時新建

migrations/
└── 0XX_v7_okx_tables.sql       # new (見 §7)
```

---

## 4. State Machine

### 4.1 Executor 狀態

```
       ┌────────┐
       │  INIT  │  config 載入、DB schema check
       └───┬────┘
           │ config OK
           ▼
     ┌─────────────┐
     │ CONNECTING  │  WS public + private 建連,REST auth 驗證
     └─────┬───────┘
           │ both WS subscribed + auth OK
           ▼
       ┌────────┐
       │ READY  │  cold start 對帳,等 signal
       └───┬────┘
           │ cold-start reconciliation: consistent
           ▼
       ┌────────┐
       │ ACTIVE │  正常運作:接 signal、下單、管 stop、對帳
       └───┬────┘
           │
     ┌─────┴──────────────────────┐
     │ (transient error)          │ (kill trigger)
     ▼                            ▼
 ┌────────┐                  ┌─────────┐
 │ HALTED │ <──auto──────────│ DEMOTED │  終止,等人工
 └───┬────┘                  └─────────┘
     │ recovered             (Stage 1 paper executor 繼續)
     ▼
 ┌────────┐
 │ ACTIVE │
 └────────┘
```

### 4.2 各狀態的語意

| 狀態 | 可下新單? | 可管 open 倉? | 對帳? | 持久化? |
|---|---|---|---|---|
| INIT | ❌ | n/a | ❌ | save status |
| CONNECTING | ❌ | n/a | ❌ | save status |
| READY | ❌ | (no open) | ✅ cold start | save status |
| ACTIVE | ✅ | ✅ | ✅ every cycle | save status + state |
| HALTED | ❌ | ✅(trailing stop 繼續保護)| ✅ | save status + reason |
| DEMOTED | ❌ | ❌(已平倉)| ✅(只記錄)| save status + trigger ID |

### 4.3 狀態遷移觸發條件

- `INIT → CONNECTING`:config valid
- `CONNECTING → READY`:WS public + private 都 subscribed,REST `/api/v5/account/balance` 成功
- `READY → ACTIVE`:cold-start reconciliation = consistent(無未預期倉位)
- `ACTIVE → HALTED`:某些 trigger(C5 NTP 5s drift、B4 stop 未掛、A3 heartbeat lag、A4 對帳不一致)
- `HALTED → ACTIVE`:trigger 條件消失 + 連續 N cycles 對帳 OK + 自動 resume(僅針對 transient 類)
- `ACTIVE → DEMOTED` / `HALTED → DEMOTED`:任一 kill trigger(A1/A2/A4-A6/B1-B6/C1/C4/C6/D2/D3 等);詳見 kill_criteria.md
- `DEMOTED → READY`:**只能人工**(走 stage2_kill_criteria.md §5 流程)

---

## 5. 模組契約

### 5.1 `OkxClient`(低階通訊)

```python
class OkxClient:
    """Pure communication layer. No business logic.
    Every method is idempotent w.r.t. clOrdId / algoClOrdId.
    """

    # ── Orders ──────────────────────────────────────
    def submit_market_order(
        self, *, inst_id: str, side: Side, sz: int,
        td_mode: str, cl_ord_id: str,
    ) -> OrderResult:
        """Submit market order. clOrdId for idempotency.
        Returns: OrderResult(ord_id, cl_ord_id, status='submitted'|'filled'|'rejected', error?)
        Retry policy: 3x exponential backoff on 5xx; do NOT retry on 4xx.
        """

    def submit_algo_stop(
        self, *, inst_id: str, side: Side, sz: int,
        trigger_px: float, td_mode: str, algo_cl_ord_id: str,
    ) -> AlgoOrderResult: ...

    def amend_algo_stop(
        self, *, algo_id: str, new_trigger_px: float,
    ) -> AmendResult:
        """ATOMIC amend. Never cancel-then-new. (Principle P6)"""

    def cancel_algo_stop(self, *, algo_id: str) -> CancelResult: ...

    # ── Reads ───────────────────────────────────────
    def get_positions(self, *, inst_id: str) -> list[Position]: ...
    def get_balance(self) -> Balance: ...
    def get_account_config(self) -> AccountConfig:
        """Returns posMode, tdMode capabilities, etc."""

    # ── WS subscriptions(in ws_private.py) ────────
    def subscribe_orders(self, callback: Callable[[OrderEvent], None]): ...
    def subscribe_positions(self, callback: Callable[[PositionEvent], None]): ...
    def subscribe_balance(self, callback: Callable[[BalanceEvent], None]): ...

    # ── Health ──────────────────────────────────────
    def is_connected(self) -> ConnectivityStatus:
        """Returns: public_ws_ok, private_ws_ok, last_heartbeat_age_sec, rest_last_latency_ms"""
```

### 5.2 `PositionReconciler`

```python
class PositionReconciler:
    """Source of truth = exchange. Never auto-mutate local state to match;
    only flag mismatch."""

    def reconcile_cycle(self) -> ReconciliationResult:
        """Called at start of every executor cycle.
        Compare OkxStateStore.get_open() vs OkxClient.get_positions().

        Returns:
          - CONSISTENT
          - MISMATCH(detail: size/direction/orphan_local/orphan_exchange)
          - UNAVAILABLE(reason: WS_DOWN | REST_TIMEOUT)
        """

    def daily_reconcile(self) -> DailyReconciliationResult:
        """EOD: compare every fill in OKX vs every trade in local DB. T-RC-06."""

    def get_consecutive_clean_days(self) -> int:
        """For graduation checks(GR-04)."""
```

### 5.3 `V7OkxExecutor`

```python
class V7OkxExecutor:
    """Mirror of V7PaperExecutor's cycle logic, but talks to OKX."""

    def __init__(self, client: OkxClient, store: OkxStateStore,
                 reconciler: PositionReconciler, config: OkxConfig): ...

    def cycle(self, *, klines: pd.DataFrame,
              signal_direction: str, signal_strength: str,
              model_version: str | None) -> CycleResult:
        """One cycle on the latest closed bar.

        Steps:
          1. Status guard(skip if HALTED/DEMOTED/CONNECTING)
          2. Reconciliation guard(skip + halt if MISMATCH)
          3. Kill checks(NTP, heartbeat, etc.)
          4. If open position: manage exit / amend trailing stop
          5. If flat + actionable signal: submit entry + algo stop

        Returns: CycleResult(action, ...)
        """

    def halt(self, *, reason: str, trigger_id: str | None = None): ...
    def demote(self, *, reason: str, trigger_id: str): ...
    def get_status(self) -> ExecutorStatus: ...
```

### 5.4 `OkxStateStore`

```python
class OkxStateStore:
    """DB layer for v7_okx_* tables. Pure read/write, no logic."""

    # positions
    def insert_open(self, *, ...) -> int: ...
    def get_open_position(self) -> Position | None: ...
    def update_trail(self, *, position_id: int, ...): ...
    def close_position(self, *, ...): ...

    # orders
    def map_order(self, *, cl_ord_id: str, ord_id: str, ...): ...
    def get_pending_orders(self) -> list[OrderMapping]: ...

    # status
    def save_executor_status(self, *, status: ExecutorStatus): ...
    def load_executor_status(self) -> ExecutorStatus: ...

    # kill log
    def log_kill_trigger(self, *, trigger_id: str, context: dict): ...
    def get_recent_triggers(self, *, since: datetime) -> list[KillEvent]: ...

    # reconciliation log
    def log_reconciliation(self, *, result: ReconciliationResult): ...
```

### 5.5 `ExecutorRouter`

```python
class ExecutorRouter:
    """Wires multiple executors. Per-cycle, fans out signal to each."""

    def __init__(self, paper_exec: V7PaperExecutor,
                 okx_exec: V7OkxExecutor | None,
                 config: RouterConfig): ...

    def cycle(self, *, klines, signal_direction, signal_strength,
              model_version) -> RouterResult:
        """Always invokes paper. Invokes okx iff:
          - config.okx_enabled = True
          - okx_exec.status not in {DEMOTED, INIT, CONNECTING}
        Catches per-executor exceptions; one failure does not block the
        other. Returns per-executor results."""
```

---

## 6. 並發模型

### 6.1 Thread map

| Thread | 責任 | 既有 / 新 |
|---|---|---|
| Main(Flask) | HTTP / webhook | 既有 |
| APScheduler | update_cycle 每小時觸發 | 既有 |
| OKX WS public | 行情(目前 market_data 用)| 既有 |
| **OKX WS private** | order / position / balance push | **新** |
| **Reconciler daemon** | 每 60s 一次背景對帳 | **新** |
| **Health monitor** | NTP / heartbeat / connectivity check | **新** |

### 6.2 共享狀態與 lock

```python
# OkxStateStore 是唯一可變共享狀態
# 所有寫透過 with self._lock:
class OkxStateStore:
    def __init__(self):
        self._lock = threading.RLock()
    
    def insert_open(self, ...):
        with self._lock:
            # DB write
```

WS callback 不直接改 in-memory state — 全部走 OkxStateStore(DB persisted)。
這樣即使 callback 在另一 thread 也安全。

### 6.3 Cycle 內的順序保證

`V7OkxExecutor.cycle()` 是 single-threaded(APScheduler 排程)。
但 cycle 進行中可能有 WS event 來:
- 已透過 OkxStateStore 寫入 DB
- cycle 下次讀 store 時自然拿到最新

**不要** 在 cycle 中靠 in-memory state 做決策 — 永遠重新讀 store。

---

## 7. 持久化 Schema(新增 MySQL tables)

### 7.1 `v7_okx_positions`(parallel to v7_paper_positions)

與 paper 表結構 mirror,額外加 OKX 識別欄位:

```sql
CREATE TABLE v7_okx_positions (
  id              BIGINT PRIMARY KEY AUTO_INCREMENT,
  entry_time      DATETIME NOT NULL,
  direction       VARCHAR(8) NOT NULL,         -- LONG / SHORT
  entry_tier      VARCHAR(16) NOT NULL,
  entry_price     DOUBLE NOT NULL,
  atr_at_entry    DOUBLE NOT NULL,
  stop_dist       DOUBLE NOT NULL,
  trail_extreme   DOUBLE NOT NULL,
  current_stop    DOUBLE NOT NULL,
  size_contracts  INT NOT NULL,                -- OKX 用 contract 數
  size_frac       DOUBLE NOT NULL,
  notional_usd    DOUBLE NOT NULL,
  equity_before   DOUBLE NOT NULL,
  model_version   VARCHAR(64),
  paused_at_signal TINYINT DEFAULT 0,
  -- OKX-specific
  entry_cl_ord_id VARCHAR(64) NOT NULL,
  entry_ord_id    VARCHAR(64),
  stop_algo_cl_ord_id VARCHAR(64) NOT NULL,
  stop_algo_id    VARCHAR(64),
  -- close
  status          VARCHAR(16) DEFAULT 'OPEN',  -- OPEN / CLOSED / DEMOTED
  exit_time       DATETIME,
  exit_price      DOUBLE,
  exit_reason     VARCHAR(32),
  exit_fees_usd   DOUBLE,
  gross_pct       DOUBLE,
  net_pct         DOUBLE,
  equity_ret_pct  DOUBLE,
  equity_after    DOUBLE,
  -- audit
  created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_entry_cl_ord (entry_cl_ord_id),
  INDEX idx_status (status),
  INDEX idx_entry_time (entry_time)
);
```

### 7.2 `v7_okx_kill_log`

```sql
CREATE TABLE v7_okx_kill_log (
  id           BIGINT PRIMARY KEY AUTO_INCREMENT,
  ts           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  trigger_id   VARCHAR(8) NOT NULL,            -- e.g. "A4", "B2"
  severity     VARCHAR(16) NOT NULL,           -- HALT / DEMOTE / HARD_FREEZE
  context      JSON NOT NULL,                  -- structured details
  resolved_at  DATETIME,
  resolution   TEXT,
  INDEX idx_ts (ts),
  INDEX idx_trigger (trigger_id)
);
```

### 7.3 `v7_okx_reconciliation_log`

```sql
CREATE TABLE v7_okx_reconciliation_log (
  id           BIGINT PRIMARY KEY AUTO_INCREMENT,
  ts           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  result       VARCHAR(16) NOT NULL,           -- CONSISTENT / MISMATCH / UNAVAILABLE
  detail       JSON,                           -- mismatch details if any
  INDEX idx_ts (ts),
  INDEX idx_result (result)
);
```

### 7.4 `v7_okx_executor_status`(單列,持久化 state machine 狀態)

```sql
CREATE TABLE v7_okx_executor_status (
  id              INT PRIMARY KEY,             -- always 1
  status          VARCHAR(16) NOT NULL,        -- INIT / CONNECTING / READY / ACTIVE / HALTED / DEMOTED
  last_changed_at DATETIME NOT NULL,
  reason          TEXT,
  trigger_id      VARCHAR(8),
  context         JSON
);
```

---

## 8. 配置管理

### 8.1 Config schema

```yaml
# config/okx_executor.yaml(或在 Railway env vars)
v7_executor:
  mode: paper_only | paper_plus_testnet | paper_plus_live
  # Stage 2 期間永遠 paper_plus_testnet

okx:
  rest_base: https://www.okx.com
  ws_public: wss://ws.okx.com:8443/ws/v5/public
  ws_private: wss://ws.okx.com:8443/ws/v5/private
  is_simulated: 1                              # Stage 2: 1, Stage 3+: 0
  api_key: ${OKX_API_KEY}                      # env var
  api_secret: ${OKX_API_SECRET}
  passphrase: ${OKX_PASSPHRASE}
  inst_id: BTC-USDT-SWAP
  td_mode: cross                               # cross w/ leverage=1
  pos_mode: net_mode
  leverage: 1                                  # hard cap

risk:
  initial_capital_usd: 100                     # Stage 3 default; Stage 2 用 testnet 預設
  risk_frac: 0.02
  max_position_count: 1

monitoring:
  reconciliation_interval_sec: 60
  ntp_check_interval_sec: 300
  heartbeat_timeout_sec: 30
  health_alert_telegram_chat_id: ${ALERT_CHAT_ID}
```

### 8.2 Config 驗證(啟動時 fail-fast)

```python
def validate_config(cfg: OkxConfig) -> None:
    # E1: tdMode/leverage 衝突
    assert cfg.leverage == 1, "Stage 2-3: leverage must be 1"
    # E2: posMode
    assert cfg.pos_mode == "net_mode", "Stage 2: posMode must be net_mode"
    # E4: API key 權限(啟動時 query OKX 確認)
    perms = client.get_account_config().api_permissions
    assert "withdraw" not in perms, "API key MUST NOT have withdraw permission"
    assert "transfer" not in perms, "API key MUST NOT have transfer permission"
    assert "trade" in perms and "read" in perms
    # Mode sanity
    assert cfg.is_simulated in (0, 1)
    if cfg.is_simulated == 0:
        # live key — extra checks
        assert os.environ.get("STAGE", "") == "live", \
            "is_simulated=0 requires STAGE=live env var"
```

---

## 9. 整合既有系統(接觸面)

### 9.1 `indicator/app.py`(修改)

```python
# 既有
from indicator import v7_paper_executor

# 新增
from indicator.executor_router import ExecutorRouter
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.client import OkxClient
# ...

# 初始化(在現有 IndicatorEngine init 之後)
def _init_router():
    paper = v7_paper_executor  # module-level, 既有
    okx_cfg = load_okx_config()
    if okx_cfg.enabled:
        okx_client = OkxClient(okx_cfg)
        okx_store = OkxStateStore()
        okx_reconciler = PositionReconciler(okx_client, okx_store)
        okx_exec = V7OkxExecutor(okx_client, okx_store, okx_reconciler, okx_cfg)
        okx_exec.start()  # CONNECTING → READY → ACTIVE
    else:
        okx_exec = None
    return ExecutorRouter(paper, okx_exec, RouterConfig.from_env())

# update_cycle 內(取代既有的直接呼叫 v7_paper_executor.cycle)
def update_cycle():
    # ... 既有的 signal 計算
    router_result = _router.cycle(
        klines=klines,
        signal_direction=signal_direction,
        signal_strength=signal_strength,
        model_version=model_version,
    )
    # router_result 包含 paper + okx 兩個 result
```

### 9.2 既有 `v7_paper_executor.py`

**不修改**。Stage 1 baseline 保持不動。

### 9.3 既有 `paper_trading.py` dashboard

**不修改**。新增 `okx_trading.py` 提供 `/testnet-perf`(A2 T-RP-01)。

### 9.4 既有 `app.py` Telegram 告警

**修改**:新增 channel 區分(`alert_chat_id` for routine, `critical_chat_id` for kill triggers)。

### 9.5 `shared/db.py`

**不修改**。新 tables 透過既有 `get_db_conn()` 連線。

---

## 10. 失敗處理模式

### 10.1 Idempotency

```python
# 每個訂單在系統內先 generate clOrdId,然後送 OKX
import uuid
cl_ord_id = f"v7_{int(bar_ts.timestamp())}_{uuid.uuid4().hex[:8]}"
# OkxStateStore 寫入 PENDING + cl_ord_id
# OkxClient.submit_market_order(cl_ord_id=cl_ord_id)
# 若 timeout 或 unknown error → 用 cl_ord_id 查 OKX 該訂單真實狀態
```

### 10.2 Retry 政策

| 操作 | 重試 | Backoff | 失敗後 |
|---|---|---|---|
| Submit order | 3x | 1s/2s/4s | 用 cl_ord_id 查狀態,確定未成交才放棄 |
| Amend algo | 2x | 1s/2s | 失敗則 halt + alert(trailing 沒更新風險高) |
| Cancel algo | 3x | 1s/2s/4s | 失敗則 halt + 對帳 |
| Get positions | 5x | 0.5s/1s/2s/4s/8s | 全失敗則 reconciler 回 UNAVAILABLE → executor halt |
| Get balance | 3x | 0.5s/1s/2s | 失敗則 demote(無法算 size) |

### 10.3 Circuit breaker

```python
class CircuitBreaker:
    """If N failures within W seconds, halt the executor for cool_down period."""
    def __init__(self, threshold: int, window_sec: int, cooldown_sec: int): ...
    def record_failure(self): ...
    def is_tripped(self) -> bool: ...

# 使用例:OKX REST 連續 5 次 5xx within 60s → halt
rest_breaker = CircuitBreaker(threshold=5, window_sec=60, cooldown_sec=300)
```

### 10.4 結構化 logging

```python
# 不要 print。所有 log 用 logger 帶 structured fields。
logger.info("order_submitted", extra={
    "cl_ord_id": cl_ord_id, "side": "LONG", "sz_contracts": 1,
    "trigger": "v7_entry", "bar_ts": bar_ts.isoformat(),
})
# 這樣 log aggregator(Railway logs)可以 query。
```

---

## 11. 測試策略

### 11.1 Unit tests(`tests/okx/`)

- 每個模組的純邏輯
- OkxClient 用 mock REST/WS 測 retry / parsing
- PositionReconciler 用 fixture 餵不同 OKX response,測 MATCH/MISMATCH 判定
- V7OkxExecutor cycle 邏輯用 mock client + mock store

### 11.2 Integration tests(對 OKX demo)

- CI on-demand(不每次 push 跑,因為依賴 OKX 服務)
- 涵蓋 A2 checklist 的 [Auto] 項目
- 有獨立的 test API key(不與生產共用)

### 11.3 Recorded fixtures

```
tests/okx/fixtures/
├── ws_order_filled.json
├── ws_position_update.json
├── rest_balance_response.json
├── rest_positions_response.json
└── rest_error_*.json
```

從 OKX demo 抓 real response 存檔,unit test 用。

### 11.4 Property tests

`PositionReconciler` 的不變量用 hypothesis:
```python
@given(local_positions=..., okx_positions=...)
def test_reconciler_invariant(local, okx):
    result = reconciler.compare(local, okx)
    # 任何 local has 1 不在 okx → result.has_orphan_local
    # ...
```

---

## 12. 反模式(明確禁止)

寫在 code review checklist 裡,任一出現拒絕 merge:

| 反模式 | 為什麼禁 |
|---|---|
| `cancel_algo + submit_algo`(取代 amend)| 中間 race window 倉位裸奔(P6) |
| `position = ...` 不寫入 store | in-memory state,kill -9 後遺失(P3) |
| `try: ... except: pass` | 吞例外是 silent failure 根源(mistake log §2026-04-22) |
| 在 WS callback 內 block I/O | 卡住 WS thread,heartbeat 失敗 |
| Optimistic 標 OPEN(submit 後立刻記 OPEN) | 應等 fill confirm 才標 |
| 從 OKX 拉狀態反向覆蓋 local | source of truth = exchange,但**不自動覆蓋**(P4) |
| 跨 cycle 用 in-memory cache | 重讀 store 才能保證一致 |
| Sleep + poll for fill | 用 WS push(P5) |
| 沒有 clOrdId 的 order | 失敗重送會重複下單(P3) |

---

## 13. 開發里程碑(Stage 2 進場後的工作分解)

進場前(現在做):**A1 + A2 + A3 + A4 文件就緒**

進場後預估工作量(2 人週):

| 階段 | 預估 | 產出 |
|---|---|---|
| M1: 骨架 + OkxClient REST | 3 天 | rest.py + auth + 5 個基本 endpoint |
| M2: OkxClient WS private | 3 天 | ws_private.py + 3 個 channel 訂閱 + reconnect |
| M3: OkxStateStore + schema | 1 天 | state.py + migration SQL |
| M4: PositionReconciler | 2 天 | reconciler.py + 涵蓋 T-RC 系列 |
| M5: V7OkxExecutor cycle | 3 天 | executor.py + kill_checks.py |
| M6: ExecutorRouter + app.py wire | 1 天 | router.py + app.py 修改 |
| M7: /testnet-perf dashboard | 1 天 | okx_trading.py + HTML render |
| M8: Unit + integration tests | 3 天 | tests/okx/ 全部 |
| M9: A2 checklist testnet 驗證 | 14 天 | 44 個 items 全跑 + result doc |

**總計 ~5 週工作 + 14 天 testnet 觀察 = ~7-8 週**

加上 Stage 1 等待累積 trades 的 3 個月,**現實上 Stage 2 啟動 → Stage 3 啟動約 5-6 個月**。這是給期望管理用的數字,不是承諾。

---

## 14. 開放問題(進場前要解決)

進 Stage 2 之前以下問題要有答案,寫進本 doc 或 issue tracker:

1. **OKX testnet demo 是否需要 KYC?** — 影響註冊流程
2. **Demo API key 是否有 IP whitelist?** — Railway egress IP 需確認
3. **Demo WS heartbeat 機制是否與 live 一致?** — 影響 T-CN-04 模擬
4. **是否所有 algo order type 在 demo 都支援?** — 影響 T-SO-06 fallback 設計
5. **Funding settlement 在 demo 是否真的扣款?** — 影響 T-RT-03 驗證
6. **Telegram alert critical channel 是否與 routine channel 分流?** — 影響 monitoring 設計
7. **是否需要獨立 process(separate Railway service)還是同進程?** — 影響架構(目前設計假設同進程)
8. **Reconciler interval 60s 是否足夠?** — 與 cycle 1h 比過於頻繁?可調為 5min 視 testnet 結果

每個問題在進 Stage 2 之前要 resolve,在本 doc 對應位置加註答案 + 決策者。

---

## 15. 文件版本管理

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-05-25 | Initial draft |

修改本 doc 流程:
- 任何架構變動 → bump minor(0.1 → 0.2),change log 寫進 §15
- Stage 2 進場時 → freeze 為 1.0
- Stage 2 進場後再改 → 須有 design review + Stage 2 doc supplement
