# OpsPilot AI

OpsPilot AI is a portfolio-grade enterprise incident intelligence platform that:
- simulates enterprise incidents and resolution history
- predicts escalation risk
- retrieves similar historical incidents
- ranks incidents by action priority
- surfaces root-cause signals
- generates operational summaries in a polished Streamlit dashboard

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
streamlit run app/streamlit_app.py
```

## Outputs
- `outputs/incidents.csv`
- `outputs/incident_predictions.csv`
- `outputs/similar_incidents.csv`
- `outputs/priority_queue.csv`
- `outputs/operational_summary.json`
- `outputs/model_metrics.txt`
