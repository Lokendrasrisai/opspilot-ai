import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import MODEL_PATH, RANDOM_STATE

TEXT_COL = "text_blob"
CAT_COLS = ["service", "region", "team", "category", "severity"]
NUM_COLS = ["cpu_pct", "memory_pct", "latency_ms", "error_rate_pct", "queue_backlog"]

def prepare_frame(df: pd.DataFrame):
    out = df.copy()
    out["text_blob"] = out["title"].fillna("") + " " + out["description"].fillna("") + " " + out["root_cause_signal"].fillna("")
    return out

def train_triage_model(df: pd.DataFrame):
    df = prepare_frame(df)
    X = df[[TEXT_COL] + CAT_COLS + NUM_COLS]
    y = df["escalated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=800, ngram_range=(1,2), stop_words="english"), TEXT_COL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "report": classification_report(y_test, preds)
    }

    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, metrics

def score_incidents(df: pd.DataFrame, pipeline):
    df = prepare_frame(df)
    X = df[[TEXT_COL] + CAT_COLS + NUM_COLS]
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)
    out = df.copy()
    out["predicted_escalation"] = preds
    out["escalation_risk"] = probas[:, 1] if probas.shape[1] > 1 else 0.0
    out["priority_score"] = (
        0.34 * out["escalation_risk"] +
        0.18 * (out["latency_ms"] / max(out["latency_ms"].max(), 1)) +
        0.16 * (out["error_rate_pct"] / max(out["error_rate_pct"].max(), 1)) +
        0.16 * (out["cpu_pct"] / 100.0) +
        0.16 * (out["queue_backlog"] / max(out["queue_backlog"].max(), 1))
    )
    return out
