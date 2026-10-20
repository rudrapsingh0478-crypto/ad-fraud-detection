"""Train 4 models x 8 feature sets (ablation + feature selection) and report metrics.

Models:       Logistic Regression, Random Forest, LightGBM, XGBoost
Feature sets: baseline (raw), +counts, +diversity, +timing, +ratios, +coordination,
              proposed (all), selected (top-K by Random Forest importance, train split only)

Usage:
    python src/train.py --data data/train_sample.csv
    python src/train.py --data data/train.csv --nrows 1000000
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # keep going without it, but say so
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from features import FEATURE_GROUPS, feature_list, load_and_build, time_split

ROOT = Path(__file__).resolve().parent.parent
LABEL = "is_attributed"
TOP_K = 12  # number of features kept by the feature-selection step


def make_models(pos_weight):
    models = {
        "LogReg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=60, max_depth=14, min_samples_leaf=20, max_samples=0.3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            scale_pos_weight=min(pos_weight, 10), n_jobs=-1, random_state=42, verbose=-1)
    else:
        print("WARNING: lightgbm not installed, skipping LightGBM (pip install lightgbm)")
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(pos_weight, 10),
            tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=42, verbosity=0)
    else:
        print("WARNING: xgboost not installed, skipping XGBoost (pip install xgboost)")
    return models


def best_threshold(y, p):
    """Threshold that maximises F1 on the validation slice."""
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()

    print("Building features ...")
    t0 = time.time()
    df = load_and_build(args.data, args.nrows)
    print(f"{len(df):,} rows, feature build {time.time() - t0:.1f}s, "
          f"positive rate {df[LABEL].mean():.4%}")

    train, val, test = time_split(df)
    pos_weight = (train[LABEL] == 0).sum() / max((train[LABEL] == 1).sum(), 1)

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows = []
    state = {"importances": None}

    def run_set(fs_name, cols, tag=None):
        Xtr, ytr = train[cols], train[LABEL]
        Xva, yva = val[cols], val[LABEL]
        Xte, yte = test[cols], test[LABEL]
        for m_name, model in make_models(pos_weight).items():
            t = time.time()
            model.fit(Xtr, ytr)  # class imbalance handled on TRAIN only
            train_s = time.time() - t

            thr = best_threshold(yva, model.predict_proba(Xva)[:, 1])

            t = time.time()
            p = model.predict_proba(Xte)[:, 1]
            infer_ms = (time.time() - t) / len(Xte) * 1000 * 1000  # ms per 1000 rows
            pred = (p >= thr).astype(int)

            rows.append({
                "feature_set": fs_name, "model": m_name, "n_features": len(cols),
                "accuracy": accuracy_score(yte, pred),
                "precision": precision_score(yte, pred, zero_division=0),
                "recall": recall_score(yte, pred, zero_division=0),
                "f1": f1_score(yte, pred, zero_division=0),
                "roc_auc": roc_auc_score(yte, p),
                "pr_auc": average_precision_score(yte, p),
                "train_s": round(train_s, 2),
                "infer_ms_per_1000": round(infer_ms, 2),
            })
            print(f"{fs_name:20s} {m_name:13s} PR-AUC={rows[-1]['pr_auc']:.4f} "
                  f"F1={rows[-1]['f1']:.4f}")

            # Save baseline + proposed models for SHAP and the app
            if tag and m_name in ("RandomForest", "LightGBM", "XGBoost"):
                joblib.dump({"model": model, "features": cols, "threshold": thr},
                            ROOT / "models" / f"{m_name}_{tag}.pkl")

            # Feature importance from the proposed Random Forest drives feature selection
            if tag == "proposed" and m_name == "RandomForest":
                state["importances"] = pd.Series(model.feature_importances_, index=cols)

    # 1) ablation: raw, each group alone, all groups
    feature_sets = {"baseline (raw)": []}
    feature_sets.update({f"+{g}": [g] for g in FEATURE_GROUPS})
    feature_sets["proposed (all)"] = list(FEATURE_GROUPS)
    for fs_name, groups in feature_sets.items():
        tag = {"baseline (raw)": "baseline", "proposed (all)": "proposed"}.get(fs_name)
        run_set(fs_name, feature_list(groups), tag)

    # 2) feature selection: keep the TOP_K features by Random Forest importance (train split only)
    imp = state["importances"].sort_values(ascending=False)
    imp.rename("importance").to_csv(ROOT / "results" / "feature_importance.csv")
    keep = set(imp.index[:TOP_K])
    sel_cols = [c for c in feature_list(list(FEATURE_GROUPS)) if c in keep]
    print(f"\nFeature selection: kept {len(sel_cols)} of {len(imp)} features:\n  " + ", ".join(sel_cols) + "\n")
    (ROOT / "results" / "selected_features.txt").write_text("\n".join(sel_cols))
    run_set(f"selected (top {TOP_K})", sel_cols, tag=None)

    res = pd.DataFrame(rows).round(4)
    res.to_csv(ROOT / "results" / "results.csv", index=False)
    print("\n", res.to_string(index=False))
    print("\nSaved results/results.csv, feature_importance.csv and models/*.pkl")


if __name__ == "__main__":
    main()