import pandas as pd

df = pd.read_csv("data/processed/final_features.csv")

print("PRISM Weakness Analysis")
print("=======================")

print("\nClass distribution:")
print(df["is_suspicious"].value_counts(normalize=True))

print("\nSuspicious transaction statistics:")
print(
    df[df["is_suspicious"] == 1]["amount"].describe()
)

print("\nNormal transaction statistics:")
print(
    df[df["is_suspicious"] == 0]["amount"].describe()
)