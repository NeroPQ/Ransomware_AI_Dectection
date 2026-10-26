from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Ransomware AI Early Warning", layout="wide")

st.title("Ransomware AI Early Warning & Behavior Analysis")

metrics_path = "reports/behavior_model_metrics.csv"
scored_path = "reports/behavior_scored_windows.csv"
mitre_path = "reports/mitre_alerts.csv"

missing = [path for path in [metrics_path, scored_path, mitre_path] if not Path(path).is_file()]
if missing:
    st.info("Missing demo reports. Generate safe logs, build features, train behavior models, then run mitre_mapping.py. Single-session training requires an explicit demonstration split.")
    st.stop()

metrics = pd.read_csv(metrics_path)
scored = pd.read_csv(scored_path)
mitre = pd.read_csv(mitre_path)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Early Detection",
    "MITRE Mapping",
    "Raw Tables"
])

with tab1:
    st.subheader("Model Metrics")
    st.dataframe(metrics, use_container_width=True)

    st.subheader("Label Distribution by Window")
    st.bar_chart(scored["label"].value_counts())

    st.subheader("Event Timeline")
    fig = px.line(
        scored,
        x="first_event_index",
        y=["file_write_count", "file_rename_count", "dns_query_count", "service_stop_count"],
        title="Behavior Events by Window"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption("All-window scores include training data. They are exploratory, not held-out evidence of early detection.")
    st.subheader("Scored Windows")
    cols = [
        "window_id", "first_event_index", "label",
        "rf_pred", "rf_score", "xgb_pred", "xgb_score", "if_pred", "if_score",
        "rf_detection_lead_events"
    ]
    st.dataframe(scored[cols], use_container_width=True)

    lead = scored["rf_detection_lead_events"].dropna()
    if len(lead) > 0:
        st.metric("RF Detection Lead Events", int(lead.iloc[0]))

    st.subheader("Risk Score Timeline")
    fig = px.line(
        scored,
        x="first_event_index",
        y=["rf_score", "xgb_score", "if_score"],
        title="Risk Scores over Event Index"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("MITRE ATT&CK Explanation")
    st.dataframe(mitre, use_container_width=True)

    if len(mitre) > 0:
        st.subheader("Technique Frequency")
        st.bar_chart(mitre["technique"].value_counts())

with tab4:
    st.subheader("Full Feature Table")
    st.dataframe(scored, use_container_width=True)