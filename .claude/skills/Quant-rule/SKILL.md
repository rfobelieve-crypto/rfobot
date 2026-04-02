# SKILL.md - Claude 使用最佳實務（針對本量化專案）

## 基本使用原則
1. 每次主要任務前，先參考 CLAUDE.md 中的專案規則。
2. 採用 **Plan Mode**：先要求 step-by-step 思考，不要直接輸出程式碼。
3. 使用 **XML 結構化 Prompt** 提升精準度與一致性。
4. 採用小任務迭代：一次只完成一個小模組，驗證通過後再串接。
5. 每次輸出後，強制進行 self-check（自審核）。

## 推薦 Prompt Template（直接複製使用）

```xml
<role>你是資深量化交易開發者，專精高頻訂單流與永續合約資料處理。</role>

<task>[具體任務，例如：設計 OrderFlowProcessor 類別]</task>

<constraints>
- 必須使用 Polars
- 同時支援歷史 batch 模式與即時 single-tick 模式
- 嚴格遵守 CLAUDE.md 中的所有原則（歷史/即時一致性、stateful 設計等）
- 處理 Funding 的離散更新與 forward-fill
- 考慮 edge cases：資料缺失、時間跳動、高波動時段
</constraints>

<output_format>
1. 先用 step-by-step 思考（Plan Mode），列出：
   - 輸入輸出定義
   - 需要維護的狀態變數
   - 歷史與即時一致性的實現方式
   - 潛在 edge cases 與處理方法
2. 確認理解後，輸出完整、可直接執行的程式碼（包含 type hints 與 docstring）
3. 附上使用範例：
   - 歷史 batch 處理範例
   - single tick 測試範例
   - replay 驗證腳本
4. 最後列出 3-5 個測試案例與預期行為
</output_format>{\rtf1}