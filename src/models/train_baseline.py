import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# Load processed data
df = pd.read_csv("data/processed/transaction_features.csv")

# Features used by the model
features = [
    "amount",
    "hour",
    "day_of_week",
    "sender_transaction_count",
    "sender_total_amount",
    "sender_avg_amount",
    "receiver_transaction_count",
    "receiver_total_amount",
    "receiver_avg_amount"
]

X = df[features]
y = df["is_suspicious"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train baseline model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Baseline Model Results")
print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

# Save model
joblib.dump(model, "models/baseline_model.pkl")

print("Model saved to: models/baseline_model.pkl")