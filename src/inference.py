"""Shared scoring logic used by the FastAPI backend and the Streamlit frontend.

The model predicts the probability that a click leads to an app install
(is_attributed = 1). In TalkingData-style logs most clicks never convert, so a
low probability means "low quality", NOT confirmed fraud.
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["ip", "app", "device", "os", "channel", "click_time"]
LABEL = "is_attributed"
GENUINE, LOW = "Likely genuine", "Low quality"


@lru_cache(maxsize=4)
def load_bundle(model="RandomForest", variant="proposed"):
    path = ROOT / "models" / f"{model}_{variant}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run: python src/train.py --data data/train.csv --nrows 1000000")
    return joblib.load(path)


def score(df, model="RandomForest", variant="proposed"):
    """Raw click table -> table with features, install probability and quality label."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    keep = REQUIRED + ([LABEL] if LABEL in df.columns else [])
    df = df[keep].dropna(subset=REQUIRED).copy()
    if df.empty:
        raise ValueError("The file has no usable rows.")
    df["row_id"] = np.arange(len(df))
    bundle = load_bundle(model, variant)
    feats = build_features(df)
    p = bundle["model"].predict_proba(feats[bundle["features"]])[:, 1]
    feats["p_install"] = p
    feats["quality"] = np.where(p >= bundle["threshold"], GENUINE, LOW)
    return feats


def label_metrics(scored):
    """ROC-AUC / PR-AUC if the uploaded file contains the is_attributed label."""
    if LABEL not in scored.columns or scored[LABEL].nunique() < 2:
        return {}
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"roc_auc": round(float(roc_auc_score(scored[LABEL], scored["p_install"])), 4),
            "pr_auc": round(float(average_precision_score(scored[LABEL], scored["p_install"])), 4)}


def headline(scored):
    n = len(scored)
    g = int((scored["quality"] == GENUINE).sum())
    out = {"clicks": n, "unique_ips": int(scored["ip"].nunique()), "likely_genuine": g,
           "likely_genuine_pct": round(100 * g / n, 3), "low_quality": n - g}
    out.update(label_metrics(scored))
    return out


def ip_summary(scored):
    """One row per IP, heaviest first."""
    g = scored.groupby("ip")
    genuine = (scored["quality"] == GENUINE).astype(int).groupby(scored["ip"]).sum()
    out = pd.DataFrame({
        "clicks": g.size(),
        "unique_apps": g["app"].nunique(),
        "unique_devices": g["device"].nunique(),
        "unique_channels": g["channel"].nunique(),
        "likely_genuine_clicks": genuine,
        "avg_install_prob": g["p_install"].mean().round(5),
    })
    gaps = scored.loc[scored["ip_prev_gap"] >= 0].groupby("ip")["ip_prev_gap"].min()
    out["fastest_repeat_s"] = gaps
    return out.reset_index().sort_values("clicks", ascending=False).reset_index(drop=True)


def ips_to_review(ips, min_clicks=20):
    """Heavy IPs where not a single click looks genuine."""
    return ips[(ips["likely_genuine_clicks"] == 0) & (ips["clicks"] >= min_clicks)].reset_index(drop=True)


def explain(bundle, X):
    """SHAP contributions for the rows in X, or None if shap is not installed."""
    try:
        import shap
        sv = shap.TreeExplainer(bundle["model"]).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        return pd.DataFrame(sv, columns=X.columns, index=X.index)
    except Exception:
        return None


def load_results():
    path = ROOT / "results" / "results.csv"
    return pd.read_csv(path) if path.exists() else None
