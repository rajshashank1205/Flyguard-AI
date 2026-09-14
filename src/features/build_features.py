import pandas as pd

# Load data
df = pd.read_csv("data/raw/transactions.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

# Sender behavior
sender_stats = df.groupby("sender").agg(
    sender_transaction_count=("transaction_id", "count"),
    sender_total_amount=("amount", "sum"),
    sender_avg_amount=("amount", "mean")
).reset_index()

# Receiver behavior
receiver_stats = df.groupby("receiver").agg(
    receiver_transaction_count=("transaction_id", "count"),
    receiver_total_amount=("amount", "sum"),
    receiver_avg_amount=("amount", "mean")
).reset_index()

# Merge features
df = df.merge(sender_stats, on="sender", how="left")
df = df.merge(receiver_stats, on="receiver", how="left")

# Save processed features
df.to_csv("data/processed/transaction_features.csv", index=False)

print("Feature engineering completed.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved to: data/processed/transaction_features.csv")