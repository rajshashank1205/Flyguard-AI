import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path


# -----------------------------
# CONFIGURATION
# -----------------------------

NUM_ACCOUNTS = 500
NUM_NORMAL_TRANSACTIONS = 5000
NUM_SUSPICIOUS_ACCOUNTS = 30

random.seed(42)
np.random.seed(42)


# -----------------------------
# CREATE FOLDERS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
LABEL_DIR = BASE_DIR / "data" / "labels"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# CREATE ACCOUNTS
# -----------------------------

accounts = [
    f"ACC{i:04d}"
    for i in range(1, NUM_ACCOUNTS + 1)
]

print(f"Created {len(accounts)} accounts")


# -----------------------------
# SELECT SUSPICIOUS ACCOUNTS
# -----------------------------

suspicious_accounts = random.sample(
    accounts,
    NUM_SUSPICIOUS_ACCOUNTS
)

normal_accounts = [
    acc for acc in accounts
    if acc not in suspicious_accounts
]


# -----------------------------
# CREATE NORMAL TRANSACTIONS
# -----------------------------

transactions = []

start_time = datetime(
    2026, 1, 1, 8, 0, 0
)

for i in range(NUM_NORMAL_TRANSACTIONS):

    sender = random.choice(normal_accounts)
    receiver = random.choice(normal_accounts)

    # Prevent sending to itself
    while receiver == sender:
        receiver = random.choice(normal_accounts)

    amount = round(
        np.random.lognormal(
            mean=7,
            sigma=1
        ),
        2
    )

    timestamp = start_time + timedelta(
        minutes=random.randint(
            0,
            60 * 24 * 30
        )
    )

    transactions.append({
        "transaction_id": f"TX{i:06d}",
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "timestamp": timestamp
    })


# -----------------------------
# CREATE SUSPICIOUS PATTERNS
# -----------------------------

transaction_counter = NUM_NORMAL_TRANSACTIONS


for mule in suspicious_accounts:

    # Multiple accounts send money to mule
    senders = random.sample(
        normal_accounts,
        random.randint(5, 15)
    )

    # Mule sends money to other accounts
    receivers = random.sample(
        normal_accounts,
        random.randint(3, 10)
    )

    base_time = start_time + timedelta(
        days=random.randint(1, 25)
    )

    total_received = 0


    # Incoming transactions
    for sender in senders:

        amount = random.randint(
            5000,
            30000
        )

        timestamp = base_time + timedelta(
            minutes=random.randint(0, 30)
        )

        total_received += amount

        transactions.append({
            "transaction_id":
                f"TX{transaction_counter:06d}",

            "sender": sender,
            "receiver": mule,
            "amount": amount,
            "timestamp": timestamp
        })

        transaction_counter += 1


    # Outgoing transactions shortly after
    split_amount = total_received / len(receivers)

    for receiver in receivers:

        amount = round(
            split_amount * random.uniform(
                0.8,
                1.1
            ),
            2
        )

        timestamp = base_time + timedelta(
            minutes=random.randint(
                31,
                90
            )
        )

        transactions.append({
            "transaction_id":
                f"TX{transaction_counter:06d}",

            "sender": mule,
            "receiver": receiver,
            "amount": amount,
            "timestamp": timestamp
        })

        transaction_counter += 1


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

transactions_df = pd.DataFrame(
    transactions
)

transactions_df = transactions_df.sort_values(
    "timestamp"
)

transactions_df.reset_index(
    drop=True,
    inplace=True
)


# -----------------------------
# CREATE LABELS
# -----------------------------

labels = []

for account in accounts:

    label = 1 if account in suspicious_accounts else 0

    labels.append({
        "account_id": account,
        "label": label
    })


labels_df = pd.DataFrame(
    labels
)


# -----------------------------
# SAVE FILES
# -----------------------------

transactions_path = (
    RAW_DIR / "transactions.csv"
)

labels_path = (
    LABEL_DIR / "account_labels.csv"
)


transactions_df.to_csv(
    transactions_path,
    index=False
)

labels_df.to_csv(
    labels_path,
    index=False
)


# -----------------------------
# SUMMARY
# -----------------------------

print("\n==============================")
print("DATASET GENERATED SUCCESSFULLY")
print("==============================")

print(
    f"Accounts: {len(accounts)}"
)

print(
    f"Transactions: {len(transactions_df)}"
)

print(
    f"Suspicious Accounts: "
    f"{len(suspicious_accounts)}"
)

print(
    f"\nTransactions saved to:"
)

print(transactions_path)

print(
    f"\nLabels saved to:"
)

print(labels_path)