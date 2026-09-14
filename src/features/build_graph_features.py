import pandas as pd
import networkx as nx

# Load graph
G = nx.read_graphml("data/processed/financial_graph.graphml")

# Calculate graph features
graph_features = []

for account in G.nodes():
    graph_features.append({
        "account": account,
        "in_degree": G.in_degree(account),
        "out_degree": G.out_degree(account),
        "total_degree": G.degree(account),
        "in_amount": sum(
            data["total_amount"]
            for _, _, data in G.in_edges(account, data=True)
        ),
        "out_amount": sum(
            data["total_amount"]
            for _, _, data in G.out_edges(account, data=True)
        )
    })

features_df = pd.DataFrame(graph_features)

features_df.to_csv(
    "data/processed/graph_features.csv",
    index=False
)

print("Graph feature engineering completed.")
print("Accounts:", len(features_df))
print("Saved to: data/processed/graph_features.csv")