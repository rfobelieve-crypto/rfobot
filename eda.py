import pandas as pd

# =========================
# 1. 讀資料
# =========================
df = pd.read_csv("liquidity_events_simulated.csv")

print("==== 基本資訊 ====")
print(df.info())
print("\n==== 前5筆 ====")
print(df.head())


# =========================
# 2. Outcome 分布
# =========================
print("\n==== Reversal / Continuation 比例 ====")
print(df["outcome_label"].value_counts())
print(df["outcome_label"].value_counts(normalize=True))


# =========================
# 3. liquidity_side vs outcome
# =========================
print("\n==== Liquidity Side vs Outcome ====")
print(pd.crosstab(df["liquidity_side"], df["outcome_label"], normalize="index"))


# =========================
# 4. Flow 分析（最重要）
# =========================
print("\n==== Flow Until Hit Delta（平均） ====")
print(df.groupby("outcome_label")["flow_until_hit_delta"].mean())

print("\n==== Flow 1H Delta（平均） ====")
print(df.groupby("outcome_label")["flow_1h_delta"].mean())


# =========================
# 5. Hit latency 分析
# =========================
print("\n==== Hit Latency（平均秒） ====")
print(df.groupby("outcome_label")["hit_latency_seconds"].mean())


# =========================
# 6. 基本統計
# =========================
print("\n==== 數值欄位統計 ====")
print(df.describe())