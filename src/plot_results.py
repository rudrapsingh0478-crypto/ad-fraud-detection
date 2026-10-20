"""Make paper-ready ablation charts and tables from results/results.csv (seaborn).

Run from the project folder:  python src/plot_results.py
Outputs (in results/): ablation_prauc.png, ablation_rocauc.png, ablation_heatmap.png,
cost_vs_gain.png, table_ablation.csv, table_cost.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
df = pd.read_csv(RES / "results.csv")

ORDER = list(dict.fromkeys(df["feature_set"]))   # keeps the order used in train.py
MODELS = list(dict.fromkeys(df["model"]))
PALETTE = dict(zip(MODELS, ["#8a8f98", "#6F1322", "#F0B120", "#2E7D32", "#1f6feb"]))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def grouped_bar(metric, title, fname):
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200)
    sns.barplot(data=df, x="feature_set", y=metric, hue="model", order=ORDER,
                hue_order=MODELS, palette=PALETTE, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=5.5, padding=1)
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", "-").upper())
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    sns.despine()
    ax.legend(frameon=False, fontsize=8, ncol=len(MODELS), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(RES / fname)
    plt.close(fig)


grouped_bar("pr_auc", "Ablation: PR-AUC by feature set (time-ordered split)", "ablation_prauc.png")
grouped_bar("roc_auc", "Ablation: ROC-AUC by feature set (time-ordered split)", "ablation_rocauc.png")

pr = df.pivot(index="feature_set", columns="model", values="pr_auc").loc[ORDER, MODELS]
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
sns.heatmap(pr, annot=True, fmt=".3f", cmap="Reds", cbar_kws={"label": "PR-AUC"}, ax=ax, linewidths=0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("PR-AUC heatmap: feature set x model", fontsize=11)
fig.tight_layout()
fig.savefig(RES / "ablation_heatmap.png")
plt.close(fig)

tab = pr.copy().round(4)
for m in MODELS:
    base = pr.loc["baseline (raw)", m]
    tab[f"{m} gain %"] = ((pr[m] / base - 1) * 100).round(0)
tab.to_csv(RES / "table_ablation.csv")

cost_sets = ["baseline (raw)", "proposed (all)"] + [s for s in ORDER if s.startswith("selected")]
cost = df[df.feature_set.isin(cost_sets)][
    ["feature_set", "model", "n_features", "accuracy", "train_s", "infer_ms_per_1000"]]
cost.to_csv(RES / "table_cost.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
sub = df[df.model != "LogReg"]
sns.scatterplot(data=sub, x="infer_ms_per_1000", y="pr_auc", hue="model", palette=PALETTE, s=40, ax=ax)
for _, r in sub.iterrows():
    ax.annotate(r.feature_set.replace("baseline (raw)", "raw").replace("proposed (all)", "all"),
                (r.infer_ms_per_1000, r.pr_auc), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Inference time (ms per 1,000 rows)")
ax.set_ylabel("PR-AUC")
ax.set_title("Accuracy gain vs inference cost", fontsize=10)
sns.despine()
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(RES / "cost_vs_gain.png")
plt.close(fig)

print(tab.to_string())
print("\nSaved charts and tables to", RES)