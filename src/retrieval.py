import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_table(df: pd.DataFrame, top_k=3):
    text = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    vect = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1,2))
    mat = vect.fit_transform(text)
    sim = cosine_similarity(mat)

    rows = []
    for i in range(len(df)):
        sims = [(j, sim[i, j]) for j in range(len(df)) if j != i]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
        for rank, (j, score) in enumerate(sims, start=1):
            rows.append({
                "incident_id": df.iloc[i]["incident_id"],
                "similar_incident_id": df.iloc[j]["incident_id"],
                "rank": rank,
                "similarity_score": round(float(score), 4),
                "similar_title": df.iloc[j]["title"],
                "similar_root_cause": df.iloc[j]["root_cause_signal"]
            })
    return pd.DataFrame(rows)
