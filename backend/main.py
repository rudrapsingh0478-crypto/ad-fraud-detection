"""FastAPI backend.   Run from the project folder:
    uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
import io
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import inference as inf  # noqa: E402

app = FastAPI(title="Ad Fraud Detection API", version="1.0")


def _records(df):
    """DataFrame -> JSON-safe list of dicts (no NaN, no numpy types)."""
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            df[c] = df[c].astype(str)
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    res = inf.load_results()
    if res is None:
        raise HTTPException(404, "results/results.csv not found. Run train.py first.")
    return _records(res)


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  model: str = Query("RandomForest", pattern="^(RandomForest|LightGBM)$"),
                  variant: str = Query("proposed", pattern="^(proposed|baseline)$"),
                  min_clicks: int = Query(20, ge=1),
                  top: int = Query(50, ge=1, le=1000)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        scored = inf.score(df, model, variant)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    ips = inf.ip_summary(scored)
    review = inf.ips_to_review(ips, min_clicks)
    cols = ["row_id", "ip", "app", "device", "os", "channel", "click_time", "p_install", "quality"]
    return {
        "summary": inf.headline(scored),
        "ips_to_review": _records(review.head(top)),
        "most_likely_genuine_clicks": _records(scored.nlargest(top, "p_install")[cols]),
    }
