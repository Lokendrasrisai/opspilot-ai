from pathlib import Path
import json

def ensure_dirs(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
