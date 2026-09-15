import os
from typing import Annotated
from typing_extensions import TypedDict

import joblib
import pandas as pd
from dotenv import load_dotenv

from prismtrace import (
    PRISMtraceLangGraphHandler,
    wrap_langgraph,
)

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# FILE PATHS
# ============================================================

FEATURES_PATH = "data/processed/final_features.csv"
MODEL_PATH = "models/graph_model.pkl"


# ============================================================
# TRANSACTION ANALYSIS TOOL
# ============================================================

@tool
def analyze_transaction(transaction_id: str) -> str:
    """Analyze one transaction and return its suspicious-risk score."""

    df = pd.read_csv(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)

    row = df[df["transaction_id"] == transaction_id]

    if row.empty:
        return f"Transaction {transaction_id} was not found."

    model_features = [
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

    X = row[model_features]

    risk_score = float(
        model.predict_proba(X)[0][1]
    )

    decision = (
        "HIGHER RISK"
        if risk_score >= 0.5
        else "LOWER RISK"
    )

    return (
        f"Transaction: {transaction_id}\n"
        f"Amount: {float(row.iloc[0]['amount']):.2f}\n"
        f"Risk score: {risk_score:.4f}\n"
        f"Decision: {decision}\n"
        f"Sender: {row.iloc[0]['sender']}\n"
        f"Receiver: {row.iloc[0]['receiver']}\n"
    )


# ============================================================
# ACCOUNT INVESTIGATION TOOL
# ============================================================

@tool
def investigate_account(account_id: str) -> str:
    """Investigate an account's transaction activity and suspicious activity."""

    df = pd.read_csv(FEATURES_PATH)

    sent = df[df["sender"] == account_id]
    received = df[df["receiver"] == account_id]

    if sent.empty and received.empty:
        return f"Account {account_id} was not found."

    sent_suspicious = int(sent["is_suspicious"].sum())
    received_suspicious = int(received["is_suspicious"].sum())

    return (
        f"Account: {account_id}\n"
        f"Transactions sent: {len(sent)}\n"
        f"Transactions received: {len(received)}\n"
        f"Total sent amount: {sent['amount'].sum():.2f}\n"
        f"Total received amount: {received['amount'].sum():.2f}\n"
        f"Suspicious sent transactions: {sent_suspicious}\n"
        f"Suspicious received transactions: {received_suspicious}\n"
        f"Unique senders: {received['sender'].nunique()}\n"
        f"Unique receivers: {sent['receiver'].nunique()}\n"
    )


# ============================================================
# AGENT STATE
# ============================================================

tools = [
    analyze_transaction,
    investigate_account,
]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    final_output: str


# ============================================================
# OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CHATBOT NODE
# ============================================================

def chatbot_node(state: AgentState):
    system_message = SystemMessage(
        content=(
            "You are FlyGuard AI, a financial-risk investigation assistant. "
            "You analyze suspicious transaction and mule-account behavior. "
            "Use tools whenever the user asks about a transaction or account. "
            "Do not claim that a person is criminal. Use terms like risk, "
            "suspicion, and investigation priority. Explain results clearly "
            "in simple language."
        )
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    content = response.content

    if isinstance(content, list):
        content = "".join(
            str(item.get("text", ""))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return {
        "messages": [response],
        "final_output": str(content),
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "chatbot")

app = builder.compile()


# ============================================================
# PRISM SETUP
# ============================================================

prism_api_key = os.getenv("PRISMTRACE_API_KEY")
prism_project_id = os.getenv("PRISMTRACE_PROJECT_ID")
prism_host = os.getenv("PRISMTRACE_HOST")

if not prism_api_key:
    raise RuntimeError(
        "PRISMTRACE_API_KEY is missing from the .env file."
    )

if not prism_project_id:
    raise RuntimeError(
        "PRISMTRACE_PROJECT_ID is missing from the .env file."
    )

if not prism_host:
    raise RuntimeError(
        "PRISMTRACE_HOST is missing from the .env file."
    )


prism_handler = PRISMtraceLangGraphHandler(
    api_key=prism_api_key,
    project_id=prism_project_id,
    host=prism_host,
    agent_name="flyguard-conversational-agent",
    session_id="flyguard-conversation-demo",
)

prism_app = wrap_langgraph(
    app,
    prism_handler,
)


# ============================================================
# CONVERSATIONAL LOOP
# ============================================================

if __name__ == "__main__":
    print("FlyGuard conversational agent started.")
    print("Type 'exit' to stop.\n")

    messages = []

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Agent stopped.")
                break

            if not user_input:
                continue

            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            result = prism_app.invoke(
                {
                    "messages": messages,
                    "final_output": "",
                }
            )

            messages = result["messages"]

            final_message = messages[-1]

            final_response = getattr(
                final_message,
                "content",
                str(final_message),
            )

            if isinstance(final_response, list):
                final_response = "".join(
                    str(item.get("text", ""))
                    if isinstance(item, dict)
                    else str(item)
                    for item in final_response
                )

            print(f"\nFlyGuard: {final_response}\n")

            # Send the completed trace to PRISM.
            prism_handler.flush()

    except KeyboardInterrupt:
        print("\nAgent stopped by user.")

    finally:
        # Flush any remaining PRISM spans before exit.
        prism_handler.flush()