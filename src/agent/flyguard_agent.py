import os
from typing import TypedDict

import joblib
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from prismtrace import PRISMtraceLangGraphHandler, wrap_langgraph

load_dotenv()


class FlyGuardState(TypedDict, total=False):
    transaction_id: str
    transaction: dict
    risk_score: float
    decision: str
    explanation: str


FEATURES = [
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
    "receiver_out_amount",
]


def load_transaction(state: FlyGuardState):
    df = pd.read_csv("data/processed/final_features.csv")

    transaction = df[
        df["transaction_id"] == state["transaction_id"]
    ]

    if transaction.empty:
        raise ValueError("Transaction ID not found")

    return {
        "transaction": transaction.iloc[0].to_dict()
    }


def predict_risk(state: FlyGuardState):
    model = joblib.load("models/graph_model.pkl")
    transaction = state["transaction"]

    input_data = pd.DataFrame(
        [[transaction[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    risk_score = float(
        model.predict_proba(input_data)[0][1]
    )

    decision = (
        "HIGH RISK — REVIEW REQUIRED"
        if risk_score >= 0.5
        else "LOWER RISK"
    )

    return {
        "risk_score": risk_score,
        "decision": decision,
    }


def explain_decision(state: FlyGuardState):
    transaction = state["transaction"]

    explanation = (
        f"Amount: {transaction['amount']:.2f}; "
        f"Sender outgoing connections: "
        f"{transaction['sender_out_degree']}; "
        f"Receiver incoming connections: "
        f"{transaction['receiver_in_degree']}; "
        f"Model decision: {state['decision']}"
    )

    return {
        "explanation": explanation
    }


# Build the LangGraph workflow
builder = StateGraph(FlyGuardState)

builder.add_node("load_transaction", load_transaction)
builder.add_node("predict_risk", predict_risk)
builder.add_node("explain_decision", explain_decision)

builder.add_edge(START, "load_transaction")
builder.add_edge("load_transaction", "predict_risk")
builder.add_edge("predict_risk", "explain_decision")
builder.add_edge("explain_decision", END)

flyguard_agent = builder.compile()


# Connect the workflow to PRISM
prism_handler = PRISMtraceLangGraphHandler(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
    host=os.environ["PRISMTRACE_HOST"],
    agent_name="flyguard-risk-agent",
    session_id="flyguard-demo-1",
)

prism_agent = wrap_langgraph(
    flyguard_agent,
    prism_handler,
)


if __name__ == "__main__":
    try:
        result = prism_agent.invoke({
            "transaction_id": "T0004292"
        })

        print("\nFlyGuard AI Agent Result")
        print("=======================")
        print("Transaction:", result["transaction_id"])
        print("Risk score:", result["risk_score"])
        print("Decision:", result["decision"])
        print("Explanation:", result["explanation"])

    finally:
        prism_handler.flush()