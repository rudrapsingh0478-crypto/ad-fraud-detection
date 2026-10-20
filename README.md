# AI-Based Fraudulent Advertisement Detection System

Relationship-aware feature framework for detecting low-quality (suspicious) ad clicks on the
TalkingData AdTracking dataset. The project tests whether features that capture how IP, device,
app, channel and time relate to each other improve standard machine-learning models over raw
click fields alone.

## Approach

- **Dataset:** TalkingData AdTracking Fraud Detection (Kaggle). The first 1,000,000 clicks of
  `train.csv` are used. Positive rate: 0.17%.
- **Label:** `is_attributed = 1` means the click led to an app install. This is a *proxy*: a low
  predicted probability marks a lower-quality click and is not proof of fraud.
- **Features:** 6 raw features plus 18 relationship-aware features in five groups
  - Counts: `ip_cnt`, `ip_app_cnt`, `ip_dev_os_cnt`, `ip_app_hour_cnt`
  - Diversity: unique apps, devices, channels and OS per IP
  - Timing: seconds since the previous click from the same IP / IP+app / IP+device+OS
  - Ratios: clicks per app, clicks per channel, app share
  - Coordination: `dev_os_app_nuniq_ip` (device-to-IP spread), `ip_dev_switches` (device changes
    per IP), `ip_quick_cnt` (repeat clicks within 2 s) and `coord_score` (Cross-Entity
    Coordination Score: mean percentile rank of six signals, equal weights, no labels used)
- **Split:** chronological, 60% train / 10% validation / 30% test. No random shuffling.
- **Models:** Logistic Regression, Random Forest, LightGBM, XGBoost.
- **Ablation:** baseline, each feature group alone, all groups together.
- **Feature selection:** top 12 of 24 features by Random Forest importance (training split only).
- **Explainability:** SHAP summary plot on the proposed Random Forest.

## Results (1M clicks, one run)

PR-AUC (primary metric):

| Model | Raw | + Counts | All 24 features | Top 12 |
|---|---|---|---|---|
| XGBoost | 0.408 | 0.501 | 0.524 | 0.492 |
| Random Forest | 0.224 | 0.351 | 0.340 | 0.332 |
| LightGBM | 0.036 | 0.245 | 0.127 | 0.234 |
| Logistic Regression | 0.013 | 0.013 | 0.038 | 0.037 |

Counts is the strongest single group for XGBoost, Random Forest and LightGBM. All groups together
beat counts only for XGBoost and Logistic Regression. Accuracy is 0.99 or higher for every model,
so it cannot separate them. Random Forest and XGBoost inference is about 2.5 ms per 1,000 rows.
Training times vary between runs on a busy laptop.

## Project structure

```
backend/main.py        FastAPI API (POST /predict, GET /metrics, GET /explain)
frontend/app.py        Streamlit app (score clicks, model comparison, limits)
src/features.py        Relationship-aware feature framework (shared by training and app)
src/train.py           Trains all models, runs the ablation and feature selection
src/explain.py         SHAP summary plot
src/inference.py       Scoring logic used by the app and API
src/make_sample.py     Builds data/sample_clicks.csv for the demo
src/plot_results.py    Ablation charts and tables for the paper (seaborn)
data/                  train.csv, sample_clicks.csv
models/                Saved models
results/               results.csv, charts, tables, feature importance, SHAP plot
```

## Setup (Windows PowerShell)

```
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `train.csv` from the Kaggle competition page (join the competition first) and put it
in `data\`.

## Run

Run every command from the project folder, with the venv active.

```
python src/train.py --data data/train.csv --nrows 1000000
python src/explain.py --data data/train.csv --nrows 1000000 --n 2000
python src/plot_results.py
python src/make_sample.py
streamlit run frontend/app.py
uvicorn backend.main:app --reload
```

The last two start the app (http://localhost:8501) and the API (http://127.0.0.1:8000/docs).

## Limitations

- The label is a proxy for click quality, not confirmed fraud.
- Only the first 1M rows of one dataset were used: one time window, one run.
- Features are computed over a whole uploaded batch, so a click's counts include later clicks
  in the same file. The app does batch scoring, not real-time scoring.
- Missing previous-click gaps are encoded as -1, so a first click looks like a very short gap.
- The coordination score is a heuristic with equal, untuned weights.
- LightGBM is unstable on this data. Real-time scoring is future work.
