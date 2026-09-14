import pandas as pd

# Load transaction features
transactions = pd.read_csv(
    "data/processed/transaction_features.csv"
)

# Load graph features
graph = pd.read_csv(
    "data/processed/graph_features.csv"
)

# Rename account column for merging
sender_graph = graph.add_prefix("sender_")
receiver_graph = graph.add_prefix("receiver_")

# Merge sender graph features
df = transactions.merge(
    sender_graph,
    left_on="sender",
    right_on="sender_account",
    how="left"
)

# Merge receiver graph features
df = df.merge(
    receiver_graph,
    left_on="receiver",
    right_on="receiver_account",
    how="left"
)

# Save combined dataset
df.to_csv(
    "data/processed/final_features.csv",
    index=False
)

print("Feature merging completed.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved to: data/processed/final_features.csv")