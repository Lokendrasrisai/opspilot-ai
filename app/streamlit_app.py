import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import (
    INCIDENTS_PATH, PREDICTIONS_PATH, SIMILAR_INCIDENTS_PATH,
    PRIORITY_QUEUE_PATH, SUMMARY_PATH, METRICS_PATH
)

st.set_page_config(page_title="OpsPilot AI", layout="wide")

st.markdown(
    '''
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1280px;}
    .hero {
      padding: 1.4rem 1.5rem;
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(2,6,23,0.98), rgba(15,23,42,0.88));
      border: 1px solid rgba(255,255,255,0.08);
      margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin: 0 0 0.3rem 0;}
    .hero p {color: #cbd5e1; line-height: 1.7; margin: 0;}
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="hero">
      <h1>OpsPilot AI</h1>
      <p>AI Copilot for Incident Triage, Root Cause Signals, and Operational Decision Support.</p>
    </div>
    ''',
    unsafe_allow_html=True
)

from src.pipeline import main as run_pipeline

if not PREDICTIONS_PATH.exists():
    st.warning("No data found. Running pipeline automatically...")
    with st.spinner("Generating incident data and training model..."):
        run_pipeline()
    st.success("Pipeline executed successfully. Reloading app...")
    st.rerun()

incidents = pd.read_csv(INCIDENTS_PATH)
preds = pd.read_csv(PREDICTIONS_PATH)
similar = pd.read_csv(SIMILAR_INCIDENTS_PATH)
queue = pd.read_csv(PRIORITY_QUEUE_PATH)
summary = json.loads(Path(SUMMARY_PATH).read_text(encoding="utf-8"))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Incidents", len(preds))
m2.metric("Predicted Escalations", int(preds["predicted_escalation"].sum()))
m3.metric("Avg Priority", f"{summary['avg_priority_score']:.3f}")
m4.metric("Services Impacted", preds["service"].nunique())

left, right = st.columns([1.2, 0.8])

with right:
    st.markdown("### Executive Summary")
    st.info(summary["executive_summary"])
    st.markdown("### Top Root Cause Signals")
    st.dataframe(pd.DataFrame(summary["top_root_causes"]), use_container_width=True, hide_index=True)
    st.markdown("### Top Services")
    st.dataframe(pd.DataFrame(summary["top_services"]), use_container_width=True, hide_index=True)

with left:
    st.markdown("### Priority Queue")
    st.dataframe(
        queue[["incident_id", "service", "category", "severity", "predicted_escalation", "escalation_risk", "priority_score"]].head(20),
        use_container_width=True,
        hide_index=True
    )

st.markdown("### Incident Explorer")
incident_id = st.selectbox("Select incident", preds["incident_id"].tolist())
row = preds[preds["incident_id"] == incident_id].iloc[0]
match_rows = similar[similar["incident_id"] == incident_id].sort_values("rank")

c1, c2 = st.columns([1.05, 0.95])
with c1:
    st.markdown("#### Incident Detail")
    st.write(f"**Title:** {row['title']}")
    st.write(f"**Description:** {row['description']}")
    st.write(f"**Service:** {row['service']} | **Category:** {row['category']} | **Severity:** {row['severity']}")
    st.write(f"**Predicted escalation:** {int(row['predicted_escalation'])} | **Risk:** {row['escalation_risk']:.3f}")
    st.write(f"**Likely signal:** {row['root_cause_signal']}")
    st.write("**Recommended next action:**")
    if row["predicted_escalation"] == 1:
        st.write("- Escalate immediately to owning service team and inspect dominant operational signals first.")
    else:
        st.write("- Keep in standard queue, compare with similar incidents, and validate service-specific metrics before escalation.")

with c2:
    st.markdown("#### Similar Historical Incidents")
    st.dataframe(match_rows, use_container_width=True, hide_index=True)

st.markdown("### Operational Trends")
a, b = st.columns(2)

with a:
    sev_counts = preds.groupby("severity").size().reindex(["P1", "P2", "P3", "P4"]).fillna(0)
    fig = plt.figure(figsize=(7, 4.3))
    plt.bar(sev_counts.index, sev_counts.values)
    plt.title("Incident Volume by Severity")
    plt.ylabel("Count")
    st.pyplot(fig)

with b:
    cat_counts = preds.groupby("category").size().sort_values(ascending=False)
    fig2 = plt.figure(figsize=(8, 4.3))
    plt.bar(cat_counts.index, cat_counts.values)
    plt.xticks(rotation=35, ha="right")
    plt.title("Recurring Incident Categories")
    plt.ylabel("Count")
    st.pyplot(fig2)

if METRICS_PATH.exists():
    st.markdown("### Model Metrics")
    st.code(Path(METRICS_PATH).read_text(encoding="utf-8"))
