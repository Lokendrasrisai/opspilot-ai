import pandas as pd

def build_operational_summary(df: pd.DataFrame):
    top_services = (
        df.groupby("service")
        .agg(
            incident_count=("incident_id", "count"),
            avg_priority=("priority_score", "mean"),
            escalation_rate=("predicted_escalation", "mean")
        )
        .sort_values(["avg_priority", "incident_count"], ascending=False)
        .head(5)
        .reset_index()
        .to_dict(orient="records")
    )

    recurring_categories = (
        df.groupby("category")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    top_root_causes = (
        df.groupby("root_cause_signal")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    return {
        "total_incidents": int(len(df)),
        "predicted_escalations": int(df["predicted_escalation"].sum()),
        "avg_priority_score": round(float(df["priority_score"].mean()), 4),
        "top_services": top_services,
        "recurring_categories": recurring_categories,
        "top_root_causes": top_root_causes,
        "executive_summary": (
            "OpsPilot AI analyzed operational incidents, estimated escalation risk, ranked the incident queue by priority, "
            "and surfaced recurring service and root-cause patterns to support faster triage and operational decision-making."
        )
    }
