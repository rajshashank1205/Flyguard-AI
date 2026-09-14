import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("data/processed/final_features.csv")

features = [
    "amount",
    "hour",
    "day_of_week",
    "sender_transaction_count",
    "sender_total_amount",
    "sender_avg_amount",
    "receiver_transaction_count",
    "receiver_total_amount",
    "receiver_avg_amount",
    "sender_in_degree",
    "sender_out_degree",
    "sender_total_degree",
    "sender_in_amount",
    "sender_out_amount",
    "receiver_in_degree",
    "receiver_out_degree",
    "receiver_total_degree",
    "receiver_in_amount",
    "receiver_out_amount"
]

X = df[features]
y = df["is_suspicious"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Graph-Enhanced Model Results")
print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

joblib.dump(model, "models/graph_model.pkl")

print("Model saved to: models/graph_model.pkl")