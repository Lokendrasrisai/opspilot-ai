from src.config import (
    OUTPUTS_DIR, MODELS_DIR, INCIDENTS_PATH, PREDICTIONS_PATH, SIMILAR_INCIDENTS_PATH,
    PRIORITY_QUEUE_PATH, SUMMARY_PATH, METRICS_PATH
)
from src.utils import ensure_dirs, save_json
from src.data_generation import generate_incidents
from src.modeling import train_triage_model, score_incidents
from src.retrieval import build_similarity_table
from src.summary import build_operational_summary

def main():
    ensure_dirs([OUTPUTS_DIR, MODELS_DIR])

    print("Step 1/5: Generating enterprise incident data...")
    incidents = generate_incidents()
    incidents.to_csv(INCIDENTS_PATH, index=False)

    print("Step 2/5: Training triage model...")
    pipeline, metrics = train_triage_model(incidents)

    print("Step 3/5: Scoring incidents and building priority queue...")
    scored = score_incidents(incidents, pipeline)
    scored.to_csv(PREDICTIONS_PATH, index=False)
    priority = scored.sort_values("priority_score", ascending=False).copy()
    priority.to_csv(PRIORITY_QUEUE_PATH, index=False)

    print("Step 4/5: Retrieving similar historical incidents...")
    similar = build_similarity_table(scored, top_k=3)
    similar.to_csv(SIMILAR_INCIDENTS_PATH, index=False)

    print("Step 5/5: Building executive operational summary...")
    summary = build_operational_summary(scored)
    save_json(SUMMARY_PATH, summary)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n\n")
        f.write(metrics["report"])

    print(f"Saved incidents to: {INCIDENTS_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")
    print(f"Saved similar incidents to: {SIMILAR_INCIDENTS_PATH}")
    print(f"Saved priority queue to: {PRIORITY_QUEUE_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
