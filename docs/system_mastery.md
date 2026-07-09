# rfobot 系统精通手册 — 面试就绪度

> 目的：面试前彻底读熟自己的系统。每层含「原理 → 面试官会问 → 标准答案」。
> 完整详细版（含所有标准答案）见 docx：Claude 对话 2026-07-07 生成的
> `rfobot_系統精通手冊.docx`。本文是可版控的精简索引 + 灵魂拷问清单。
>
> 读法：遮住答案自己回答问题，答不出来的就是知识债，重点补。

## 系统一句话
一套 3 年演化的 BTC 订单流量化交易系统，跑在 OKX 永续合约实盘（Stage 3, $155）。
双 XGBoost（方向 + 幅度）基于 200+ 订单流特征，77-fold walk-forward 验证，
31-trigger kill matrix 风控，纪律写进 code。

数据流：数据层 → 特征层 → 模型层 → 信号层 → 执行层 → 监控层

## 六层灵魂拷问（答不出 = 知识债）

### 层 1 — 数据层
- [ ] Binance/OKX 合约量相加为什么错？（答：contract size 差 100x；修法是
      notional_usd = price×size×contract_size，不是统一标签）
- [ ] 秒级时间戳被当毫秒会怎样、为什么 silent（答：塌到 1970 年、合法日期不报错、
      去重看成新数据）；1e12 门槛为什么有效（秒 10 位 / 毫秒 13 位，中间分界）
- [ ] 为什么拆三个服务？share data not code 解决什么？

### 层 2 — 特征层
- [ ] 怎么防 look-ahead bias？（trailing-only + merge_asof backward + build_live_features
      训练/生产共用）
- [ ] 加新特征流程？（先 IC 回测 + conditional IC + ensemble A/B，OOS AUC +0.005 才加）
- [ ] 为什么不用 MACD/RSI？（纯订单流范畴，price pattern 排除）

### 层 3 — 模型层
- [ ] 为什么双 regressor 分开不用 classifier？
- [ ] walk-forward + purge/embargo 为什么需要？（label 跨界泄漏）
- [ ] IC 0.063 好不好？（贴 AUC 0.57 的天花板，top-5% precision 上限 68-72%）
- [ ] 怎么判断模型衰退？（必须 walk-forward，禁 in-sample，栽过差 0.50 IC）

### 层 4 — 信号层
- [ ] 为什么 rolling percentile 不用固定阈值？（自适应 vol regime + absolute floor 兜底）
- [ ] 为什么 Strong-only？（Gate A: Strong CI 下缘 56% 显著，Moderate 51.7% 不显著）
- [ ] Gate A 739/59.5%/CI 怎么算？（tracked_signals 回填 + bootstrap）

### 层 5 — 执行层（Trading Systems 核心）
- [ ] 5 种 reconcile mismatch 分别是什么、怎么处理？（orphan_local/exchange/
      size_diff/direction_diff/price_diff；orphan_exchange DEMOTE、orphan_local HALT）
- [ ] 状态机四态？为什么 HALTED 和 DEMOTED 两种停？（可逆 vs 不可逆）
- [ ] 举 3 个 kill trigger + 边界？（CAP-2/3/4、C5/C6 NTP、E4 提币权限）
- [ ] 为什么 fail-closed？为什么不用 CCXT？

### 层 6 — 风控数学
- [ ] 2.0x 上限怎么算？（Kelly 0.56x + vol drag E[r]-0.5σ²L² + stress test 三交叉）
- [ ] 什么时候能放宽？（24 个月 Sharpe≥3.0 + MDD 从未超 -10%）
- [ ] hit kill 之后怎么办？（降阶重验，不准「这次例外」）

## 面试金矿：mistake.md 失败故事（讲成 30 秒）
- 5.5 天毒数据 → schema 纪律 + 自动侦测
- in-sample 诊断差 0.50 IC → walk-forward 铁律
- aggregate AUC 被 outlier fold 撑起 → 4 条部署门槛
- facade drift 两次 → AST signature-parity 测试
- 6/5 手动爆仓 + admin_heal 孤儿仓 → 破坏性 endpoint 改 POST
- 差点 threshold sweep overfit → 守住纪律，categorical > continuous

## 自我介绍三角形
会手动交易（Topstep + 4 年）× 会建系统（3 年 live rfobot）× 懂产业（Mighty DAO + 链上）。
单项都有人比你强，三合一在 junior 市场极稀有。

## 进度追踪（逐层攻破后打勾）
- [ ] 层 1 数据层 — 能不看笔记回答全部 Q
- [ ] 层 2 特征层
- [ ] 层 3 模型层
- [ ] 层 4 信号层
- [ ] 层 5 执行层
- [ ] 层 6 风控数学
- [ ] mistake.md 六个故事能讲
- [ ] 回 Claude 做面试官压力测试（跨层追问）
