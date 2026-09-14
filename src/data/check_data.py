import pandas as pd

df = pd.read_csv("data/raw/transactions.csv")

print(df.head())
print("\nColumns:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["is_suspicious"].value_counts())
print("\nMissing Values:")
print(df.isnull().sum())
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(df.dtypes)
print("\nData validation:")
print("Negative amounts:", (df["amount"] < 0).sum())
print("Zero amounts:", (df["amount"] == 0).sum())
print("Duplicate transaction IDs:", df["transaction_id"].duplicated().sum())
print("Invalid timestamps:", df["timestamp"].isna().sum())