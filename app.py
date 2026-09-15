"""FlyGuard AI demonstration dashboard."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
TRANSACTIONS_PATH = ROOT / "data" / "raw" / "transactions.csv"
MODEL_PATH = ROOT / "models" / "graph_model.pkl"
MODEL_FEATURES_PATH = ROOT / "data" / "processed" / "final_features.csv"

st.set_page_config(
    page_title="FlyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* The app uses a light canvas, so explicitly set dark foreground colors.
         This also prevents a dark Streamlit theme from rendering white text on it. */
      :root {
        --text-color: #172033;
        --secondary-background-color: #ffffff;
      }
      /* Keep Streamlit's header available: it owns the button that reopens a
         collapsed sidebar. The rest of the header remains visually quiet. */
      header[data-testid="stHeader"] {
        background: transparent !important;
        height: 3.5rem;
      }
      header[data-testid="stHeader"] [data-testid="stToolbar"] {
        background: transparent !important;
        padding: .45rem .65rem;
      }
      [data-testid="stExpandSidebarButton"],
      [data-testid="stSidebarCollapseButton"] button {
        background: #0b1f3a !important;
        border: 1px solid #0d4f67 !important;
        border-radius: 9px !important;
        color: #ffffff !important;
      }
      [data-testid="stExpandSidebarButton"]:hover,
      [data-testid="stSidebarCollapseButton"] button:hover {
        background: #08716f !important;
        border-color: #08716f !important;
      }
      [data-testid="stExpandSidebarButton"] svg,
      [data-testid="stSidebarCollapseButton"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
      }
      .stApp { background: #f7f9fc; color: #172033; }
      .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li,
      .stApp [data-testid="stMarkdownContainer"], .stApp label,
      .stApp .stText, .stApp .stCaption { color: #172033; }
      .block-container { max-width: 1280px; padding-top: 2.25rem; padding-bottom: 2.5rem; }
      .hero { background: linear-gradient(120deg, #0b1f3a 0%, #0d4f67 55%, #08716f 100%);
              border-radius: 18px; color: #ffffff; padding: 2.15rem 2.4rem; margin-bottom: 1.5rem;
              box-shadow: 0 14px 30px rgba(12, 31, 58, .13); }
      .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -.035em; color: #ffffff !important; }
      .hero p { margin: .55rem 0 0; color: #d9f2f0 !important; font-size: 1.04rem; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e5eaf1;
              border-radius: 14px; padding: .95rem 1.1rem; box-shadow: 0 2px 7px rgba(15, 23, 42, .035); }
      [data-testid="stMetricLabel"] { color: #475569 !important; font-size: .86rem; }
      [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 650; }
      button[data-baseweb="tab"] { color: #334155 !important; font-weight: 600; padding-inline: .1rem; }
      button[data-baseweb="tab"][aria-selected="true"] { color: #dc2626 !important; }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.5rem; }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #dc2626; height: 3px; }
      .stApp h2, .stApp h3 { letter-spacing: -.018em; }
      [data-testid="stExpander"], [data-testid="stExpander"] details {
        background: #ffffff !important;
        border: 1px solid #d8e1eb !important;
        border-radius: 12px !important;
        color: #172033 !important;
      }
      [data-testid="stExpander"] summary,
      [data-testid="stExpander"] summary * {
        background: #ffffff !important;
        color: #172033 !important;
        font-weight: 600;
      }
      [data-testid="stExpander"] summary:hover,
      [data-testid="stExpander"] summary:hover * {
        background: #f1f7f8 !important;
        color: #0b4f67 !important;
      }
      [data-testid="stDownloadButton"] button {
        background: #111827 !important;
        border: 1px solid #111827 !important;
        color: #ffffff !important;
        font-weight: 600;
      }
      [data-testid="stDownloadButton"] button * { color: #ffffff !important; }
      [data-testid="stDownloadButton"] button:hover {
        background: #000000 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
      }
      [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: #111827 !important;
        border: 1px solid #64748b !important;
        color: #ffffff !important;
        font-weight: 600;
      }
      [data-testid="stSidebar"] [data-testid="stButton"] button * { color: #ffffff !important; }
      [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: #000000 !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
      }
      .stApp [data-testid="stDataFrame"] {
        --ag-background-color: #ffffff;
        --ag-foreground-color: #172033;
        --ag-header-background-color: #eaf0f6;
        --ag-header-foreground-color: #172033;
        --ag-border-color: #d8e1eb;
        --ag-row-border-color: #e5eaf1;
        --ag-odd-row-background-color: #f8fafc;
      }
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] label *,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
      }
      [data-testid="stSidebar"] .sidebar-note,
      [data-testid="stSidebar"] .sidebar-note * { color: #dbeafe !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-radius: 10px;
      }
      /* Dropdown menus are rendered outside the sidebar, so give their options
         an explicit light surface and dark foreground. */
      div[data-baseweb="popover"] [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid #d8e1eb;
        border-radius: 10px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, .16);
      }
      div[data-baseweb="popover"] [role="option"] {
        color: #172033 !important;
        background: #ffffff !important;
      }
      div[data-baseweb="popover"] [role="option"]:hover,
      div[data-baseweb="popover"] [aria-selected="true"] {
        color: #0b4f67 !important;
        background: #e6f4f5 !important;
      }
      .risk-chip { display: inline-block; padding: .28rem .7rem; border-radius: 99px;
                   font-weight: 650; font-size: .86rem; }
      .caption { color: #607084; }
      @media (max-width: 700px) {
        .block-container { padding-top: 1rem; padding-inline: 1rem; }
        .hero { padding: 1.55rem; border-radius: 14px; }
        .hero h1 { font-size: 1.85rem; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .9rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    """Load and normalize source transactions used by the dashboard."""
    if not TRANSACTIONS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["timestamp"])
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    frame["is_suspicious"] = pd.to_numeric(
        frame["is_suspicious"], errors="coerce"
    ).fillna(0).astype(int)
    return frame


@st.cache_resource(show_spinner=False)
def load_graph_model(model_modified: float):
    """Load the saved graph-enhanced transaction classifier."""
    try:
        return joblib.load(MODEL_PATH)
    except (FileNotFoundError, ImportError, ModuleNotFoundError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_model_predictions(
    model_modified: float, feature_data_modified: float
) -> pd.DataFrame:
    """Return saved-model predictions keyed to the source transaction ID."""
    empty_result = pd.DataFrame(
        columns=["transaction_id", "ml_prediction", "ml_suspicious_probability"]
    )
    model = load_graph_model(model_modified)
    if model is None or not MODEL_FEATURES_PATH.exists():
        return empty_result

    feature_frame = pd.read_csv(MODEL_FEATURES_PATH)
    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features or any(
        feature not in feature_frame.columns for feature in model_features
    ):
        return empty_result

    probabilities = model.predict_proba(feature_frame[model_features])
    suspicious_class_index = list(model.classes_).index(1)
    return pd.DataFrame(
        {
            "transaction_id": feature_frame["transaction_id"],
            "ml_prediction": model.predict(feature_frame[model_features]).astype(int),
            "ml_suspicious_probability": probabilities[:, suspicious_class_index],
        }
    )


@st.cache_data(show_spinner=False)
def account_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create account-level metrics from both incoming and outgoing payments."""
    outgoing = frame.rename(columns={"sender": "account"})[
        ["account", "receiver", "amount", "is_suspicious"]
    ]
    incoming = frame.rename(columns={"receiver": "account", "sender": "counterparty"})[
        ["account", "counterparty", "amount", "is_suspicious"]
    ]
    outgoing = outgoing.rename(columns={"receiver": "counterparty"})
    activity = pd.concat([outgoing, incoming], ignore_index=True)
    summary = activity.groupby("account").agg(
        transactions=("amount", "size"),
        total_volume=("amount", "sum"),
        avg_transaction=("amount", "mean"),
        suspicious_transactions=("is_suspicious", "sum"),
        counterparties=("counterparty", "nunique"),
    )
    summary["mule_risk_score"] = (
        summary["suspicious_transactions"] / summary["transactions"] * 100
    ).fillna(0)
    summary["risk_category"] = pd.cut(
        summary["mule_risk_score"],
        bins=[-0.01, 5, 20, 100],
        labels=["Low", "Moderate", "High"],
    ).astype(str)
    return summary.reset_index().sort_values("mule_risk_score", ascending=False)


def account_model_prediction(
    predictions: pd.DataFrame, frame: pd.DataFrame, account: str
) -> dict[str, float | int | str] | None:
    """Summarize transaction-level model outputs for one account."""
    account_transaction_ids = frame.loc[
        (frame["sender"] == account) | (frame["receiver"] == account),
        "transaction_id",
    ]
    account_predictions = predictions[
        predictions["transaction_id"].isin(account_transaction_ids)
    ]
    if account_predictions.empty:
        return None

    flagged_predictions = int(account_predictions["ml_prediction"].sum())
    highest_probability = account_predictions["ml_suspicious_probability"].max()
    return {
        "label": "Suspicious" if flagged_predictions else "No alert",
        "flagged_predictions": flagged_predictions,
        "transactions_scored": len(account_predictions),
        "highest_probability": highest_probability,
    }


def risk_style(category: str) -> tuple[str, str]:
    return {
        "High": ("#fee2e2", "#b91c1c"),
        "Moderate": ("#fef3c7", "#a16207"),
        "Low": ("#dcfce7", "#15803d"),
    }.get(category, ("#e5e7eb", "#374151"))


def feature_rows(account: pd.Series) -> pd.DataFrame:
    values = {
        "Suspicious transaction rate": account["mule_risk_score"],
        "Transaction volume (scaled)": min(account["total_volume"] / 10000, 100),
        "Network reach (scaled)": min(account["counterparties"] / 2, 100),
        "Transaction frequency (scaled)": min(account["transactions"] / 2, 100),
    }
    return pd.DataFrame({"Feature": values.keys(), "Relative signal": values.values()})


def draw_network(frame: pd.DataFrame, account: str) -> None:
    related = frame[(frame["sender"] == account) | (frame["receiver"] == account)].copy()
    related = related.nlargest(18, "amount")
    graph = nx.DiGraph()
    for row in related.itertuples():
        graph.add_edge(row.sender, row.receiver, amount=row.amount, suspicious=row.is_suspicious)

    if not graph.nodes:
        st.info("No financial relationships are available for this account.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    positions = nx.spring_layout(graph, seed=8, k=1.1)
    colors = ["#115e72" if node == account else "#dbeafe" for node in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, node_color=colors, node_size=1300, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_weight="bold", ax=ax)
    safe_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d["suspicious"]]
    flagged_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["suspicious"]]
    nx.draw_networkx_edges(graph, positions, edgelist=safe_edges, edge_color="#94a3b8",
                           arrows=True, arrowsize=16, width=1.5, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=flagged_edges, edge_color="#dc2626",
                           arrows=True, arrowsize=18, width=2.8, ax=ax)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Blue is the selected account. Red connections include a suspicious transaction.")


transactions = load_transactions()
if transactions.empty:
    st.error("Transaction data was not found at `data/raw/transactions.csv`.")
    st.stop()

if MODEL_PATH.exists() and MODEL_FEATURES_PATH.exists():
    model_predictions = load_model_predictions(
        MODEL_PATH.stat().st_mtime, MODEL_FEATURES_PATH.stat().st_mtime
    )
else:
    model_predictions = pd.DataFrame(
        columns=["transaction_id", "ml_prediction", "ml_suspicious_probability"]
    )

min_date = transactions["timestamp"].min().date()
max_date = transactions["timestamp"].max().date()

with st.sidebar:
    st.markdown("<div class='sidebar-note'><strong>Investigation filters</strong></div>", unsafe_allow_html=True)
    selected_dates = st.date_input("Transaction date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="date_range")
    account_search = st.text_input("Search account ID", placeholder="e.g. A01085", key="account_search")
    selected_categories = st.multiselect(
        "Risk categories", ["High", "Moderate", "Low"], default=["High", "Moderate", "Low"], key="risk_categories"
    )
    minimum_score = st.slider("Minimum risk score", 0, 100, 0, format="%d%%", key="minimum_score")
    suspicious_only = st.checkbox("Suspicious accounts only", key="suspicious_only")
    if st.button("Reset filters", use_container_width=True):
        for state_key in ("date_range", "account_search", "risk_categories", "minimum_score", "suspicious_only"):
            st.session_state.pop(state_key, None)
        st.rerun()

if len(selected_dates) != 2:
    st.sidebar.warning("Choose a start and end date.")
    st.stop()

start_date, end_date = selected_dates
filtered_transactions = transactions[
    transactions["timestamp"].dt.date.between(start_date, end_date)
].copy()
accounts = account_summary(filtered_transactions)
visible_accounts = accounts[
    accounts["risk_category"].isin(selected_categories)
    & accounts["mule_risk_score"].ge(minimum_score)
]
if suspicious_only:
    visible_accounts = visible_accounts[visible_accounts["suspicious_transactions"] > 0]
if account_search:
    visible_accounts = visible_accounts[
        visible_accounts["account"].str.contains(account_search.strip(), case=False, na=False)
    ]

if visible_accounts.empty:
    st.warning("No accounts match the current sidebar filters. Adjust or reset them to continue.")
    st.stop()

high_risk = visible_accounts[visible_accounts["risk_category"] == "High"]
suspicious_accounts = visible_accounts[visible_accounts["suspicious_transactions"] > 0]

with st.sidebar:
    selected_account = st.selectbox("Selected account", visible_accounts["account"].tolist(), key="selected_account")
    selected = visible_accounts.loc[visible_accounts["account"] == selected_account].iloc[0]
    selected_model_prediction = account_model_prediction(
        model_predictions, filtered_transactions, selected_account
    )
    st.markdown(
        f"<div class='sidebar-note'>Risk score: <strong>{selected['mule_risk_score']:.1f}%</strong><br>"
        f"Transactions: <strong>{int(selected['transactions']):,}</strong></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sidebar-note'>Risk legend: High 20%+ · Moderate 5–20% · Low below 5%</div>", unsafe_allow_html=True)

st.markdown(
    """<div class="hero"><h1>FlyGuard AI</h1><p>Financial-network risk intelligence for clear, responsible investigations.</p></div>""",
    unsafe_allow_html=True,
)

overview, account_view, network_view, analytics, how_it_works = st.tabs(
    ["Overview", "Account view", "Network view", "Analytics", "How it works"]
)

with overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total accounts", f"{len(visible_accounts):,}")
    c2.metric("Transactions analyzed", f"{len(filtered_transactions):,}")
    c3.metric("Suspicious accounts", f"{len(suspicious_accounts):,}")
    c4.metric("High-risk accounts", f"{len(high_risk):,}")
    st.subheader("Investigation queue")
    queue = visible_accounts.head(12)[["account", "mule_risk_score", "risk_category", "transactions", "total_volume"]].copy()
    queue.columns = ["Account ID", "Mule risk score", "Risk category", "Transactions", "Total volume"]
    queue["Mule risk score"] = queue["Mule risk score"].map(lambda value: f"{value:.1f}%")
    queue["Total volume"] = queue["Total volume"].map(lambda value: f"${value:,.0f}")
    st.dataframe(queue, use_container_width=True, hide_index=True)
    st.download_button("Download filtered queue (CSV)", queue.to_csv(index=False), "flyguard_investigation_queue.csv", "text/csv")

with account_view:
    background, color = risk_style(selected["risk_category"])
    st.markdown(f"## {selected_account}  <span class='risk-chip' style='background:{background};color:{color}'>{selected['risk_category']} risk</span>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Mule risk score", f"{selected['mule_risk_score']:.1f}%")
    b.metric(
        "ML prediction",
        selected_model_prediction["label"] if selected_model_prediction else "Unavailable",
    )
    c.metric("Suspicious transactions", f"{int(selected['suspicious_transactions']):,}")
    d.metric("Unique counterparties", f"{int(selected['counterparties']):,}")
    if selected_model_prediction:
        st.caption(
            "Graph-enhanced model result: "
            f"**{int(selected_model_prediction['flagged_predictions']):,} of "
            f"{int(selected_model_prediction['transactions_scored']):,} related payments** "
            "were predicted suspicious. "
            f"Highest predicted suspicious probability: "
            f"**{float(selected_model_prediction['highest_probability']):.1%}**."
        )
    else:
        st.caption(
            "The graph-enhanced model prediction is unavailable. "
            "Ensure `models/graph_model.pkl` and `data/processed/final_features.csv` are present."
        )
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Important features")
        st.bar_chart(feature_rows(selected), x="Feature", y="Relative signal", color="#115e72")
    with right:
        st.subheader("Explanation")
        st.write(
            f"This account has **{int(selected['suspicious_transactions'])} flagged transactions** across "
            f"**{int(selected['transactions'])} observed payments**. It has interacted with "
            f"**{int(selected['counterparties'])} counterparties** and moved **${selected['total_volume']:,.0f}**. "
            "This dashboard indicates investigation priority, not proof of wrongdoing."
        )

with network_view:
    st.subheader(f"Financial relationships for {selected_account}")
    draw_network(filtered_transactions, selected_account)

with analytics:
    left, right = st.columns(2)
    with left:
        st.subheader("Account risk distribution")
        distribution = visible_accounts["risk_category"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)
        st.bar_chart(distribution, color="#115e72")
    with right:
        st.subheader("Transaction volume over time")
        volume = filtered_transactions.set_index("timestamp").resample("D")["amount"].sum()
        st.line_chart(volume, color="#115e72")
    st.subheader("Flagged accounts by risk score")
    st.scatter_chart(
        visible_accounts.assign(label=visible_accounts["risk_category"])[["transactions", "mule_risk_score"]],
        x="transactions", y="mule_risk_score", color="#dc2626", size=20,
    )

with how_it_works:
    st.subheader("How FlyGuard AI works")
    st.write(
        "FlyGuard AI turns transaction activity into an account-level investigation view. "
        "It helps teams prioritize where to look first; it does not determine whether anyone has committed wrongdoing."
    )

    with st.expander("1. Transaction data is loaded and prepared", expanded=True):
        st.write(
            "The dashboard reads `data/raw/transactions.csv`. Each record includes a timestamp, "
            "sender, receiver, amount, and a suspicious-transaction flag. Amounts are converted "
            "to numeric values and timestamps are used by the date filter and time-series chart."
        )

    with st.expander("2. Activity is grouped by account"):
        st.write(
            "For every account, FlyGuard combines outgoing and incoming payments. It then calculates "
            "the payment count, total and average volume, number of unique counterparties, and the "
            "number of flagged transactions."
        )

    with st.expander("3. A risk score and category are assigned"):
        st.markdown(
            "**Mule risk score** = `(flagged transactions ÷ total transactions) × 100`\n\n"
            "- **Low:** below 5%\n"
            "- **Moderate:** 5% to below 20%\n"
            "- **High:** 20% or more"
        )
        st.caption("The score is an investigation-priority signal based on the supplied flags, not a finding of guilt.")

    with st.expander("4. Sidebar filters narrow the investigation"):
        st.write(
            "Choose a date range, search for an account ID, select risk categories, set a minimum "
            "risk score, or limit the view to accounts with at least one flagged transaction. These "
            "controls update the overview, account list, analytics, and relationship graph together."
        )

    with st.expander("5. Use each view to investigate"):
        st.markdown(
            "- **Overview:** prioritized account queue and headline counts.\n"
            "- **Account view:** selected account's metrics and the signals contributing to its profile.\n"
            "- **Network view:** the selected account's strongest relationships; red edges contain a flagged transaction.\n"
            "- **Analytics:** risk-category distribution, daily transaction volume, and risk-score patterns."
        )

    with st.expander("6. Export and review responsibly"):
        st.write(
            "Download the filtered investigation queue as CSV from the Overview tab for case review. "
            "Validate alerts against underlying evidence and follow your organisation's approval, privacy, "
            "and escalation procedures before taking action."
        )
