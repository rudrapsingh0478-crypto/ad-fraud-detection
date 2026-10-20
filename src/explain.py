"""SHAP summary plot for the proposed Random Forest model.

Usage:
    python src/explain.py --data data/train_sample.csv
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import load_and_build, time_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--n", type=int, default=5000, help="rows to explain")
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / "RandomForest_proposed.pkl")
    model, cols = bundle["model"], bundle["features"]

    _, _, test = time_split(load_and_build(args.data, args.nrows))
    X = test[cols].sample(min(args.n, len(test)), random_state=42)

    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # newer shap: (rows, features, classes)
        sv = sv[:, :, 1]

    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    out = ROOT / "results" / "shap_summary.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
