import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Generates a single Seaborn Confusion Matrix Heatmap figure."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Tier", fontsize=10)
    ax.set_ylabel("Actual Tier", fontsize=10)
    plt.tight_layout()
    return fig

def plot_side_by_side_confusion_matrix(
    y_true, y_pred_base, y_pred_hybrid, class_names, 
    base_name="Standalone Model", hybrid_name="Hybrid Model"
):
    """Generates side-by-side comparison heatmaps between baseline and hybrid models."""
    cm_base = confusion_matrix(y_true, y_pred_base, labels=class_names)
    cm_hybrid = confusion_matrix(y_true, y_pred_hybrid, labels=class_names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Baseline Heatmap (Left)
    sns.heatmap(
        cm_base, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_title(f"{base_name}\n(Baseline)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Tier", fontsize=10)
    axes[0].set_ylabel("Actual Tier", fontsize=10)

    # Hybrid Heatmap (Right)
    sns.heatmap(
        cm_hybrid, annot=True, fmt="d", cmap="Greens", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_title(f"{hybrid_name}\n(Hybrid Enhanced)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Tier", fontsize=10)
    axes[1].set_ylabel("Actual Tier", fontsize=10)

    plt.tight_layout()
    return fig