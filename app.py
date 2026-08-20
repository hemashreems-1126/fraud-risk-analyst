"""
app.py
------
Streamlit dashboard for the AI Risk Manager pipeline.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Risk Manager", layout="wide")

# ---- Load data ----
try:
    report_df = pd.read_csv("risk_report.csv")
except FileNotFoundError:
    st.error("risk_report.csv not found. Run score_transactions.py then risk_agent.py first.")
    st.stop()

metrics = {}
try:
    with open("metrics_summary.txt") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                metrics[key.strip()] = val.strip()
except FileNotFoundError:
    st.warning("metrics_summary.txt not found — run score_transactions.py first.")

# ---- Header ----
st.title("AI Risk Manager")
st.caption("Detects anomalous transactions, then an AI agent explains each flag and recommends an action.")

# ---- Metrics row ----
st.subheader("Model performance (held-out test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", metrics.get("Precision", "—"))
col2.metric("Recall", metrics.get("Recall", "—"))
col3.metric("False positives", metrics.get("False positives", "—"))
col4.metric("Transactions investigated", len(report_df))

st.caption(
    f"{metrics.get('Test set', '')}. "
    "Each false positive means a legitimate customer's transaction gets held for manual review — "
    "the real-world cost of this model being wrong."
)

st.divider()

# ---- Verdict breakdown ----
st.subheader("Verdicts")
verdict_counts = report_df["verdict"].value_counts()
vcol1, vcol2, vcol3 = st.columns(3)
vcol1.metric("BLOCK", int(verdict_counts.get("BLOCK", 0)))
vcol2.metric("REVIEW", int(verdict_counts.get("REVIEW", 0)))
vcol3.metric("ALLOW", int(verdict_counts.get("ALLOW", 0)))

st.divider()

# ---- Flagged transactions table ----
st.subheader("Investigated transactions")

verdict_filter = st.multiselect(
    "Filter by verdict",
    options=sorted(report_df["verdict"].unique()),
    default=sorted(report_df["verdict"].unique()),
)

filtered = report_df[report_df["verdict"].isin(verdict_filter)]

for _, row in filtered.iterrows():
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"**{row['type']}**")
            st.markdown(f"₹{row['amount']:,.2f}")
            badge_color = {"BLOCK": "🔴", "REVIEW": "🟡", "ALLOW": "🟢"}.get(row["verdict"], "⚪")
            st.markdown(f"{badge_color} **{row['verdict']}**")
        with c2:
            st.markdown(row["reason"])
            st.caption(f"Anomaly score: {row['anomaly_score']:.4f} · Actual fraud: {'Yes' if row['isFraud'] == 1 else 'No'}")

st.divider()
st.caption("Built for the Razorpay AI Buildathon — AI Risk Manager track.")