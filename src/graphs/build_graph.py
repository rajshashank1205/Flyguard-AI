import pandas as pd
import networkx as nx

# Load transactions
df = pd.read_csv("data/raw/transactions.csv")

print("Building financial transaction graph...")

# Create directed graph
G = nx.DiGraph()

# Add accounts as nodes
accounts = set(df["sender"]).union(set(df["receiver"]))

for account in accounts:
    G.add_node(account)

# Add transaction relationships as edges
for _, row in df.iterrows():
    sender = row["sender"]
    receiver = row["receiver"]
    amount = row["amount"]

    if G.has_edge(sender, receiver):
        G[sender][receiver]["transaction_count"] += 1
        G[sender][receiver]["total_amount"] += amount
    else:
        G.add_edge(
            sender,
            receiver,
            transaction_count=1,
            total_amount=amount
        )

# Save graph
nx.write_graphml(G, "data/processed/financial_graph.graphml")

print("\nFINANCIAL GRAPH CREATED")
print("=======================")
print("Nodes (Accounts):", G.number_of_nodes())
print("Edges (Relationships):", G.number_of_edges())
print("Transactions Processed:", len(df))
print("Graph saved to: data/processed/financial_graph.graphml")