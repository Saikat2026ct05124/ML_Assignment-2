"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import RESULTS_DIR, TARGET

sns.set_theme(style="whitegrid")


def compute_metrics(y_true, pred, proba) -> dict:
    return {"Accuracy": accuracy_score(y_true, pred), "AUC": roc_auc_score(y_true, proba),
            "Precision": precision_score(y_true, pred), "Recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred), "MCC": matthews_corrcoef(y_true, pred)}


def plot_eda(df):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    df[TARGET].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4c72b0", "#dd8452"])
    ax[0].set_xticklabels(["Not canceled (0)", "Canceled (1)"], rotation=0); ax[0].set_title("Class distribution")
    sns.boxplot(data=df, x=TARGET, y="lead_time", ax=ax[1]); ax[1].set_title("Lead time vs cancellation")
    df.groupby("deposit_type")[TARGET].mean().plot.bar(ax=ax[2], color="#55a868")
    ax[2].set_title("Cancellation rate by deposit type"); ax[2].set_ylabel("rate"); ax[2].tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "eda_overview.png", dpi=150); plt.close()


def plot_results(res: pd.DataFrame, y_test, probas: dict, cms: dict):
    ax = res.T.plot.bar(figsize=(13, 5.5), width=0.85, colormap="tab10")
    ax.set_ylim(0, 1.05); ax.set_title("All evaluation metrics by model (test set)"); ax.set_ylabel("score")
    plt.xticks(rotation=0); plt.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "metrics_grouped_bar.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for a, m in zip(axes.ravel(), res.columns):
        s = res[m].sort_values()
        a.barh(s.index, s.values, color=sns.color_palette("viridis", len(s))); a.set_title(m); a.set_xlim(0, 1)
        for i, v in enumerate(s.values): a.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "metrics_per_metric.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p); plt.plot(fpr, tpr, label=f"{name} (AUC={res.loc[name, 'AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, len(cms), figsize=(20, 4))
    for a, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a)
        a.set_title(name, fontsize=10); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150); plt.close()
