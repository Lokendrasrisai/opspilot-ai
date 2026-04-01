from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

INCIDENTS_PATH = OUTPUTS_DIR / "incidents.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "incident_predictions.csv"
SIMILAR_INCIDENTS_PATH = OUTPUTS_DIR / "similar_incidents.csv"
PRIORITY_QUEUE_PATH = OUTPUTS_DIR / "priority_queue.csv"
SUMMARY_PATH = OUTPUTS_DIR / "operational_summary.json"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.txt"
MODEL_PATH = MODELS_DIR / "triage_model.joblib"

RANDOM_STATE = 42
N_INCIDENTS = 500
