import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 1. 讀資料
# =========================
df = pd.read_csv("liquidity_events_simulated.csv")


# =========================
# 2. Feature Engineering
# =========================

# 類別轉數值
df["liquidity_side"] = df["liquidity_side"].map({
    "buy": 1,
    "sell": 0
})

df["symbol"] = df["symbol"].map({
    "BTCUSDT": 1,
    "ETHUSDT": 0
})

# 核心 feature（你剛剛EDA驗證過的）
df["flow_ratio"] = df["flow_until_hit_buy"] / (
    df["flow_until_hit_sell"] + 1e-6
)

df["abs_delta"] = df["flow_until_hit_delta"].abs()

# Label
df["label"] = df["outcome_label"].map({
    "reversal": 1,
    "continuation": 0
})


# =========================
# 3. 選擇特徵（重點！）
# =========================

features = [
    "liquidity_side",
    "flow_until_hit_delta",
    "flow_1h_delta",
    "flow_ratio",
    "abs_delta",
]

X = df[features]
y = df["label"]


# =========================
# 4. Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# 5. 建立模型
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 6. 預測
# =========================
pred = model.predict(X_test)


# =========================
# 7. 評估（非常重要）
# =========================
print("==== Classification Report ====")
print(classification_report(y_test, pred))

print("\n==== Confusion Matrix ====")
print(confusion_matrix(y_test, pred))


# =========================
# 8. Feature Importance
# =========================
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n==== Feature Importance ====")
print(importance)