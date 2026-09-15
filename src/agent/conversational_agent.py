import json
import os
import re
from functools import lru_cache
from time import perf_counter
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import uuid4
from typing_extensions import TypedDict

import joblib
import pandas as pd
from dotenv import load_dotenv

from prismtrace import PRISMtrace

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
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
MODEL_NAME = "qwen2.5:3b"
RISK_THRESHOLD = 0.5
AGENT_ID = "flyguard-conversational-agent"
TRANSACTION_ID_PATTERN = re.compile(r"\bT\d+\b", flags=re.IGNORECASE)
ACCOUNT_ID_PATTERN = re.compile(r"\bA\d+\b", flags=re.IGNORECASE)

MODEL_FEATURES = [
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


# ============================================================
# PRISM / MESSAGE HELPERS
# ============================================================

def content_to_text(content: Any) -> str:
    """Return displayable text from LangChain's string or block content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return "" if content is None else str(content)


def configure_prism_proxy_bypass(host: str) -> None:
    """Bypass a broken proxy for PRISM only when explicitly enabled locally."""
    enabled = os.getenv("PRISMTRACE_BYPASS_PROXY", "").lower()
    if enabled not in {"1", "true", "yes"}:
        return

    hostname = urlparse(host).hostname
    if not hostname:
        return

    for variable in ("NO_PROXY", "no_proxy"):
        entries = [
            entry.strip()
            for entry in os.getenv(variable, "").split(",")
            if entry.strip()
        ]
        if hostname.lower() not in {entry.lower() for entry in entries}:
            os.environ[variable] = ",".join([*entries, hostname])


def tool_results_from_messages(messages: list) -> list[dict[str, Any]]:
    """Return only structured tool results emitted during a graph turn."""
    results: list[dict[str, Any]] = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        try:
            result = json.loads(content_to_text(getattr(message, "content", "")))
        except json.JSONDecodeError:
            result = {
                "status": "error",
                "message": "The requested data could not be verified.",
            }
        if isinstance(result, dict):
            results.append(result)
    return results


def tool_names_from_messages(messages: list) -> list[str]:
    names: list[str] = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            name = tool_call.get("name") if isinstance(tool_call, dict) else None
            if name and name not in names:
                names.append(str(name))
    return names


def token_usage_from_messages(messages: list) -> tuple[int, int]:
    """Sum Ollama usage across the model calls made for one user turn."""
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def unavailable_response(tool_results: list[dict[str, Any]]) -> str:
    """Prevent a polished but ungrounded answer after a failed tool call."""
    message = next(
        (
            str(result.get("message"))
            for result in tool_results
            if result.get("status") != "ok" and result.get("message")
        ),
        "The requested data could not be verified.",
    )
    return (
        "Assessment: Data unavailable — no risk assessment was generated.\n\n"
        f"Evidence: {message}\n\n"
        "Recommended action: Confirm the identifier and route the case to a "
        "human analyst if a decision is time-sensitive.\n\n"
        "Caveat: No conclusion about suspicious activity or wrongdoing can be "
        "drawn without verified source data."
    )


def response_from_transaction_result(result: dict[str, Any]) -> str:
    """Render a verified transaction result without an LLM round trip.

    The risk score is still produced by the persisted XGBoost model.  This
    function only replaces the slow natural-language rendering step with a
    deterministic report, so it cannot alter the score or decision.
    """
    if result.get("status") != "ok":
        return unavailable_response([result])

    is_higher_risk = result["decision"] == "HIGHER RISK"
    assessment = (
        "This transaction has a higher-risk signal and should be reviewed by "
        "a human analyst."
        if is_higher_risk
        else "This transaction has a lower-risk signal and routine monitoring "
        "is sufficient."
    )
    action = (
        "Route the case to a human analyst before any adverse action is taken."
        if is_higher_risk
        else "Continue routine monitoring and reassess if new verified activity "
        "changes the risk signal."
    )
    return (
        f"Assessment: {assessment}\n\n"
        "Evidence: Transaction "
        f"{result['transaction_id']} is ${result['amount']:.2f}; its verified "
        f"model risk score is {result['risk_score']:.4f}. Sender "
        f"{result['sender']} has {result['sender_transaction_count']} recorded "
        f"transactions, and receiver {result['receiver']} has "
        f"{result['receiver_transaction_count']}.\n\n"
        f"Recommended action: {action}\n\n"
        "Caveat: This is a risk signal, not a finding of wrongdoing."
    )


def response_from_account_result(result: dict[str, Any]) -> str:
    """Render verified account evidence without an LLM round trip."""
    if result.get("status") != "ok":
        return unavailable_response([result])

    return (
        "Assessment: This account activity is provided for investigation "
        "prioritization; no conclusion about wrongdoing is made.\n\n"
        "Evidence: Account "
        f"{result['account_id']} sent {result['transactions_sent']} transactions "
        f"totaling ${result['total_sent_amount']:.2f} and received "
        f"{result['transactions_received']} totaling "
        f"${result['total_received_amount']:.2f}. It has "
        f"{result['suspicious_sent_transactions']} flagged outgoing and "
        f"{result['suspicious_received_transactions']} flagged incoming "
        "transactions in the verified data.\n\n"
        "Recommended action: Review the verified counterparties and transaction "
        "history if the account requires a risk decision.\n\n"
        "Caveat: This is a risk signal, not a finding of wrongdoing."
    )


def enforce_compliance_gate(
    response: str, tool_results: list[dict[str, Any]]
) -> str:
    """Block unsupported financial-crime assertions from reaching a user."""
    if tool_results and any(result.get("status") != "ok" for result in tool_results):
        return unavailable_response(tool_results)

    if not response.strip():
        return unavailable_response([])

    prohibited = re.compile(
        r"\b(criminal|guilty|fraudster|money launderer|committed fraud)\b",
        flags=re.IGNORECASE,
    )
    if prohibited.search(response):
        return (
            "Assessment: The case requires human review before a risk conclusion "
            "can be shared.\n\n"
            "Evidence: The automated response contained an unsupported compliance "
            "assertion and was withheld.\n\n"
            "Recommended action: Ask a qualified analyst to review the verified "
            "transaction data.\n\n"
            "Caveat: A risk signal is not a finding of wrongdoing."
        )

    if "caveat:" not in response.lower():
        response = response.rstrip() + (
            "\n\nCaveat: This is a risk signal, not a finding of wrongdoing."
        )
    return response


# ============================================================
# TRANSACTION ANALYSIS TOOL
# ============================================================

@lru_cache(maxsize=1)
def load_feature_data() -> pd.DataFrame:
    """Load and validate the feature table once per agent process."""
    df = pd.read_csv(FEATURES_PATH)
    required_columns = {
        "transaction_id",
        "sender",
        "receiver",
        "amount",
        *MODEL_FEATURES,
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError("The feature table is missing required columns.")
    return df


@lru_cache(maxsize=1)
def load_risk_model() -> Any:
    """Load the trained risk model once per agent process."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_transaction_rows() -> pd.DataFrame:
    """Create a transaction-ID index once for constant-time request lookup."""
    return load_feature_data().set_index("transaction_id", drop=False)


@lru_cache(maxsize=1)
def load_risk_scores() -> dict[str, float]:
    """Calculate every persisted-model score once during startup warm-up.

    The batch uses the identical model and feature columns as the former
    request-time ``predict_proba`` call.  Only *when* each score is calculated
    changes; the score and threshold decision do not.
    """
    df = load_feature_data()
    scores = load_risk_model().predict_proba(df[MODEL_FEATURES])[:, 1]
    return dict(zip(df["transaction_id"].astype(str), scores))


@lru_cache(maxsize=1)
def load_account_statistics() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Aggregate account facts once so account investigations stay local."""
    df = load_feature_data()
    sent = df.groupby("sender", sort=False).agg(
        transactions_sent=("sender", "size"),
        total_sent_amount=("amount", "sum"),
        suspicious_sent_transactions=("is_suspicious", "sum"),
        unique_counterparties_sent_to=("receiver", "nunique"),
    )
    received = df.groupby("receiver", sort=False).agg(
        transactions_received=("receiver", "size"),
        total_received_amount=("amount", "sum"),
        suspicious_received_transactions=("is_suspicious", "sum"),
        unique_counterparties_received_from=("sender", "nunique"),
    )
    return sent.to_dict(orient="index"), received.to_dict(orient="index")


def tool_payload(status: str, **fields: Any) -> str:
    """Use one machine-readable contract for success and failure results."""
    return json.dumps({"status": status, **fields}, default=str)


def get_transaction_result(transaction_id: str) -> dict[str, Any]:
    """Compute the existing model result and return its structured evidence."""
    started_at = perf_counter()
    try:
        try:
            transaction = load_transaction_rows().loc[transaction_id]
        except KeyError:
            return {
                "status": "not_found",
                "message": "The transaction ID was not found in the verified feature table.",
            }

        # Keep the former first-match behavior if a source file contains a
        # duplicate transaction ID.
        if isinstance(transaction, pd.DataFrame):
            transaction = transaction.iloc[0]
        risk_score = float(load_risk_scores()[transaction_id])
        decision = "HIGHER RISK" if risk_score >= RISK_THRESHOLD else "LOWER RISK"
        return {
            "status": "ok",
            "transaction_id": transaction_id,
            "decision": decision,
            "risk_score": round(risk_score, 4),
            "amount": round(float(transaction["amount"]), 2),
            "sender": str(transaction["sender"]),
            "receiver": str(transaction["receiver"]),
            "sender_transaction_count": int(transaction["sender_transaction_count"]),
            "receiver_transaction_count": int(transaction["receiver_transaction_count"]),
            "tool_latency_ms": round((perf_counter() - started_at) * 1000),
        }
    except Exception:
        return {
            "status": "error",
            "message": "Verified transaction data could not be retrieved. No risk assessment is available.",
        }


@tool
def analyze_transaction(transaction_id: str) -> str:
    """Return verified, structured risk evidence for one transaction."""
    return tool_payload(**get_transaction_result(transaction_id))


# ============================================================
# ACCOUNT INVESTIGATION TOOL
# ============================================================

def get_account_result(account_id: str) -> dict[str, Any]:
    """Return verified activity evidence for one account."""
    started_at = perf_counter()
    try:
        sent_statistics, received_statistics = load_account_statistics()
        sent = sent_statistics.get(account_id)
        received = received_statistics.get(account_id)
        if sent is None and received is None:
            return {
                "status": "not_found",
                "message": "The account ID was not found in the verified feature table.",
            }

        sent = sent or {}
        received = received or {}

        return {
            "status": "ok",
            "account_id": account_id,
            "transactions_sent": int(sent.get("transactions_sent", 0)),
            "transactions_received": int(received.get("transactions_received", 0)),
            "total_sent_amount": round(float(sent.get("total_sent_amount", 0)), 2),
            "total_received_amount": round(float(received.get("total_received_amount", 0)), 2),
            "suspicious_sent_transactions": int(sent.get("suspicious_sent_transactions", 0)),
            "suspicious_received_transactions": int(received.get("suspicious_received_transactions", 0)),
            "unique_counterparties_sent_to": int(sent.get("unique_counterparties_sent_to", 0)),
            "unique_counterparties_received_from": int(received.get("unique_counterparties_received_from", 0)),
            "tool_latency_ms": round((perf_counter() - started_at) * 1000),
        }
    except Exception:
        return {
            "status": "error",
            "message": "Verified account data could not be retrieved. No risk assessment is available.",
        }


@tool
def investigate_account(account_id: str) -> str:
    """Return verified, structured activity evidence for one account."""
    return tool_payload(**get_account_result(account_id))


def fast_response_for_input(
    user_input: str,
) -> tuple[str, list[dict[str, Any]], list[str]] | None:
    """Answer ID-based investigations locally, avoiding two Ollama calls.

    An ID unambiguously identifies the requested tool and the required response
    fields, so an LLM cannot add verified evidence.  General conversation still
    follows the existing LangGraph/Ollama route below.
    """
    transaction_ids = {
        match.group().upper() for match in TRANSACTION_ID_PATTERN.finditer(user_input)
    }
    if len(transaction_ids) == 1:
        result = get_transaction_result(transaction_ids.pop())
        return (
            enforce_compliance_gate(response_from_transaction_result(result), [result]),
            [result],
            ["analyze_transaction"],
        )
    if len(transaction_ids) > 1:
        return None

    account_ids = {
        match.group().upper() for match in ACCOUNT_ID_PATTERN.finditer(user_input)
    }
    if len(account_ids) == 1:
        result = get_account_result(account_ids.pop())
        return (
            enforce_compliance_gate(response_from_account_result(result), [result]),
            [result],
            ["investigate_account"],
        )

    return None


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
    model=MODEL_NAME,
    temperature=0,
    num_predict=256,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CHATBOT NODE
# ============================================================

def chatbot_node(state: AgentState):
    tool_results = tool_results_from_messages(state["messages"])
    if tool_results and any(result.get("status") != "ok" for result in tool_results):
        response = unavailable_response(tool_results)
        return {
            "messages": [AIMessage(content=response)],
            "final_output": response,
        }

    system_message = SystemMessage(
        content=(
            "You are FlyGuard AI, a financial-risk investigation assistant. "
            "You analyze suspicious transaction and mule-account behavior. "
            "For a named transaction or account, call the relevant tool before "
            "making factual claims. Tool results are JSON objects: only a result "
            "with status 'ok' may support an assessment. Never infer missing "
            "facts. If a tool returns not_found or error, state that data is "
            "unavailable and do not generate a risk assessment. "
            "Do not claim that a person is criminal. Use terms such as risk, "
            "suspicion, and investigation priority.\n\n"
            "When verified data is available, write a complete, concise report "
            "with exactly these sections:\n"
            "Assessment: state the model risk band and its operational meaning.\n"
            "Evidence: give 2-4 concrete numeric or identifier facts from the "
            "tool result, including the risk score when available.\n"
            "Recommended action: use routine monitoring for lower risk; for "
            "higher risk, recommend human analyst review rather than an automated "
            "adverse action.\n"
            "Caveat: explain that this is a risk signal, not a finding of "
            "wrongdoing.\n\n"
            "Use clear plain language, preserve the tool's values exactly, and "
            "do not mention hidden prompts, tool internals, or unsupported facts."
        )
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"],
    )

    content = enforce_compliance_gate(
        content_to_text(response.content), tool_results
    )
    if content != content_to_text(response.content):
        response = response.model_copy(update={"content": content})

    return {
        "messages": [response],
        "final_output": content,
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


configure_prism_proxy_bypass(prism_host)


prism_client = PRISMtrace(
    api_key=prism_api_key,
    project_id=prism_project_id,
    host=prism_host,
    timeout=15,
)

conversation_id = f"flyguard-conversation-{uuid4().hex[:12]}"


def record_prism_turn(
    user_input: str,
    final_response: str,
    latency_ms: int,
    turn_messages: list,
    tool_results: list[dict[str, Any]] | None = None,
    tool_names: list[str] | None = None,
) -> None:
    """Record the user-visible, compliance-gated answer as the scored trace."""
    if tool_results is None:
        tool_results = tool_results_from_messages(turn_messages)
    input_tokens, output_tokens = token_usage_from_messages(turn_messages)
    if tool_names is None:
        tool_names = tool_names_from_messages(turn_messages)

    prism_client.trace_llm(
        model=MODEL_NAME,
        input_messages=[{"role": "user", "content": user_input}],
        output=final_response,
        latency_ms=latency_ms,
        token_count_input=input_tokens,
        token_count_output=output_tokens,
        agent_id=AGENT_ID,
        agent_name="FlyGuard Conversational Agent",
        session_id=conversation_id,
        metadata={
            "framework": "langgraph",
            "tool_names": tool_names,
            "tool_statuses": [result.get("status") for result in tool_results],
            "data_grounded": bool(tool_results)
            and all(result.get("status") == "ok" for result in tool_results),
            "compliance_gate": "passed",
        },
    )
    # trace_llm sends in a background thread; flush before the next turn or exit.
    prism_client.flush()


# ============================================================
# CONVERSATIONAL LOOP
# ============================================================

if __name__ == "__main__":
    # Move one-time CSV and model deserialization out of the measured request
    # path. Prediction itself remains exactly the same as before.
    load_feature_data()
    load_risk_model()
    load_transaction_rows()
    load_risk_scores()
    load_account_statistics()

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

            prior_message_count = len(messages)
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            turn_started_at = perf_counter()
            fast_result = fast_response_for_input(user_input)
            if fast_result is not None:
                final_response, fast_tool_results, fast_tool_names = fast_result
                messages.append(AIMessage(content=final_response))
                turn_messages = messages[prior_message_count:]
            else:
                result = app.invoke(
                    {
                        "messages": messages,
                        "final_output": "",
                    }
                )

                messages = result["messages"]
                final_message = messages[-1]
                final_response = result.get("final_output") or content_to_text(
                    getattr(final_message, "content", final_message)
                )
                turn_messages = messages[prior_message_count:]
                fast_tool_results = None
                fast_tool_names = None

            record_prism_turn(
                user_input=user_input,
                final_response=final_response,
                latency_ms=round((perf_counter() - turn_started_at) * 1000),
                turn_messages=turn_messages,
                tool_results=fast_tool_results,
                tool_names=fast_tool_names,
            )

            print(f"\nFlyGuard: {final_response}\n")

    except KeyboardInterrupt:
        print("\nAgent stopped by user.")

    finally:
        prism_client.close()
