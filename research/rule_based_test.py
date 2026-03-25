import pandas as pd

# =========================
# 1. 讀取資料
# =========================
df = pd.read_csv("liquidity_events_simulated.csv")

# label: reversal=1, continuation=0
df["label"] = df["outcome_label"].map({
    "reversal": 1,
    "continuation": 0
})

print("==== 基本資料 ====")
print(df[["liquidity_side", "flow_until_hit_delta", "flow_1h_delta", "outcome_label"]].head())


# =========================
# 2. 規則1：只看 flow_until_hit_delta 正負
# =========================
print("\n==== 規則1：delta 正負 vs reversal 機率 ====")

df["delta_sign"] = df["flow_until_hit_delta"].apply(lambda x: "negative" if x < 0 else "positive_or_zero")

sign_stats = df.groupby("delta_sign")["label"].agg(["count", "mean"])
sign_stats = sign_stats.rename(columns={"mean": "reversal_rate"})
print(sign_stats)


# =========================
# 3. 規則2：delta 分區間
# =========================
print("\n==== 規則2：delta 區間 vs reversal 機率 ====")

bins = [-10**9, -100000, -50000, 0, 50000, 100000, 10**9]
labels = [
    "< -100k",
    "-100k ~ -50k",
    "-50k ~ 0",
    "0 ~ 50k",
    "50k ~ 100k",
    "> 100k"
]

df["delta_bucket"] = pd.cut(df["flow_until_hit_delta"], bins=bins, labels=labels)

bucket_stats = df.groupby("delta_bucket", observed=False)["label"].agg(["count", "mean"])
bucket_stats = bucket_stats.rename(columns={"mean": "reversal_rate"})
print(bucket_stats)


# =========================
# 4. 規則3：依 liquidity_side 拆開看
# =========================
print("\n==== 規則3：buy / sell 各自看 delta 正負 ====")

side_sign_stats = df.groupby(["liquidity_side", "delta_sign"])["label"].agg(["count", "mean"])
side_sign_stats = side_sign_stats.rename(columns={"mean": "reversal_rate"})
print(side_sign_stats)


# =========================
# 5. 規則4：測幾個簡單 threshold
# =========================
print("\n==== 規則4：測試幾個 threshold ====")

thresholds = [-100000, -50000, -30000, -10000, 0, 10000, 30000, 50000]

results = []
for th in thresholds:
    subset = df[df["flow_until_hit_delta"] <= th]
    if len(subset) > 0:
        reversal_rate = subset["label"].mean()
        results.append({
            "rule": f"delta <= {th}",
            "count": len(subset),
            "reversal_rate": round(reversal_rate, 4)
        })

threshold_df = pd.DataFrame(results)
print(threshold_df)


# =========================
# 6. 規則5：建立最簡單 rule prediction
#    如果 delta < 0 預測 reversal，否則 continuation
# =========================
print("\n==== 規則5：最簡單 rule model 準確率 ====")

df["rule_pred"] = (df["flow_until_hit_delta"] < 0).astype(int)

accuracy = (df["rule_pred"] == df["label"]).mean()
print(f"Rule accuracy (delta < 0 => reversal): {accuracy:.4f}")

confusion = pd.crosstab(
    df["label"],
    df["rule_pred"],
    rownames=["Actual"],
    colnames=["Predicted"]
)
print("\nConfusion Matrix:")
print(confusion)