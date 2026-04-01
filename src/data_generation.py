import numpy as np
import pandas as pd

from src.config import N_INCIDENTS, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)

SERVICES = ["auth", "payments", "search", "recommendations", "data-pipeline", "api-gateway", "notifications"]
REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]
TEAMS = ["platform", "sre", "ml-platform", "data", "payments", "core-services"]
CATEGORIES = ["latency", "outage", "deployment", "dependency", "data-quality", "security", "capacity"]
SEVERITIES = ["P4", "P3", "P2", "P1"]

ROOT_CAUSES = {
    "latency": ["db saturation", "cache miss storm", "network slowdown"],
    "outage": ["service crash", "dependency failure", "regional outage"],
    "deployment": ["bad rollout", "config drift", "schema mismatch"],
    "dependency": ["upstream timeout", "third-party degradation", "queue backlog"],
    "data-quality": ["null spike", "broken transform", "late partition"],
    "security": ["token validation bug", "auth anomaly", "rate-limit bypass"],
    "capacity": ["cpu saturation", "memory leak", "autoscaling lag"],
}

def generate_incidents():
    rows = []
    for i in range(1, N_INCIDENTS + 1):
        category = rng.choice(CATEGORIES)
        service = rng.choice(SERVICES)
        region = rng.choice(REGIONS)
        team = rng.choice(TEAMS)

        base_sev_probs = {
            "latency": [0.18, 0.38, 0.28, 0.16],
            "outage": [0.08, 0.24, 0.34, 0.34],
            "deployment": [0.16, 0.36, 0.30, 0.18],
            "dependency": [0.12, 0.34, 0.32, 0.22],
            "data-quality": [0.20, 0.38, 0.27, 0.15],
            "security": [0.10, 0.28, 0.33, 0.29],
            "capacity": [0.14, 0.34, 0.32, 0.20],
        }[category]

        severity = rng.choice(SEVERITIES, p=base_sev_probs)
        sev_num = {"P4":1, "P3":2, "P2":3, "P1":4}[severity]

        cpu = float(np.clip(rng.normal(45 + 10*sev_num, 15), 5, 100))
        mem = float(np.clip(rng.normal(40 + 8*sev_num, 14), 5, 100))
        latency = float(np.clip(rng.normal(140 * sev_num, 90), 10, 2000))
        error_rate = float(np.clip(rng.normal(0.4 * sev_num, 0.5), 0.0, 8.0))
        backlog = int(np.clip(rng.normal(250 * sev_num, 180), 0, 5000))

        root_cause = rng.choice(ROOT_CAUSES[category])

        title = f"{service} {category} incident in {region}"
        description = (
            f"{service} experienced {category} symptoms in {region}. "
            f"Observed latency={latency:.0f}ms, error_rate={error_rate:.2f}%, cpu={cpu:.0f}%, memory={mem:.0f}%, backlog={backlog}. "
            f"Possible root cause around {root_cause}."
        )

        escalation = int(
            (sev_num >= 3)
            or (latency > 600)
            or (error_rate > 2.5)
            or (cpu > 82 and backlog > 1000)
        )

        resolution_mins = int(np.clip(rng.normal(25 * sev_num + (20 if escalation else 0), 18), 8, 360))

        rows.append({
            "incident_id": f"INC-{i:04d}",
            "title": title,
            "description": description,
            "service": service,
            "region": region,
            "team": team,
            "category": category,
            "severity": severity,
            "cpu_pct": round(cpu, 2),
            "memory_pct": round(mem, 2),
            "latency_ms": round(latency, 2),
            "error_rate_pct": round(error_rate, 3),
            "queue_backlog": backlog,
            "root_cause_signal": root_cause,
            "escalated": escalation,
            "resolution_mins": resolution_mins
        })
    return pd.DataFrame(rows)
