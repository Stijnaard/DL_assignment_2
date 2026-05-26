"""
plot_training_curves()    -> loss & accuracy over epochs
plot_confusion_matrix()   -> heatmap of predicted vs true labels
plot_comparison_bar()     -> side-by-side accuracy comparison across models
plot_intra_vs_cross()     -> generalisation gap between experiments
plot_class_distribution() -> how many windows per class in train/test
print_report()            -> full classification report (precision, recall, F1)
save_results_csv()        -> save all results to a CSV for the paper table
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import LABELS, FIGURES_DIR, RESULTS_DIR

# Constants
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
LABEL_MAP = {
    "rest":                "Rest",
    "task_motor":          "Motor",
    "task_story_math":     "Math/Story",
    "task_working_memory": "Working Mem."
}
SHORT_LABELS = [LABEL_MAP[l] for l in LABELS]

# Constant styling
def get_plot_style():
    """Apply a clean representation to each plot"""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         15,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "figure.dpi":        150
    })

# Constant bar labeling
def get_bar_labels(ax, bars, fmt = "{:.1f}%", offset = 0.5):
    """Write a value label on top of each bar"""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha = "center", va = "bottom", fontsize = 12)

# Save the plots to designated file location
def save_plot(fig, name: str):
    """Save figure to outputs/figures/ as PNG's"""
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi = 180, bbox_inches = "tight")
    print(f"Figure {FIGURES_DIR / name}.png saved")
    plt.close(fig)

# 1. Training curves  (loss + accuracy over epochs)
def plot_training_curves(history: dict[str, list[float]],
        model_name: str, experiment: str):
    """
    Two-panel plot: training/validation loss (left) and accuracy (right).
    """
    get_plot_style()
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
    fig.suptitle(
        f"{model_name} - {experiment}-subject Training Curves",
        fontsize = 13, fontweight = "bold")

    # Loss plot
    ax1.plot(epochs, history["train_loss"],
        label = "Train", color = PALETTE[0], linewidth = 2)
    ax1.plot(epochs, history["val_loss"],
        label = "Validation", color = PALETTE[1], linewidth = 2, linestyle = "--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
        label = "Train", color = PALETTE[0], linewidth = 2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],
        label = "Validation", color = PALETTE[1], linewidth = 2, linestyle = "--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend()

    fig.tight_layout()
    save_plot(fig, f"{model_name}_{experiment}_curves")

# 2. Confusion matrix
def plot_confusion_matrix(
        y_true:     list[int],
        y_pred:     list[int],
        model_name: str,
        experiment: str,
        normalize:  bool = True,
    ):
    """Heatmap of true vs predicted labels"""
    get_plot_style()
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_plot      = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt          = "{:.2f}"
        vmax         = 1.0
        title_suffix = "(normalised)"
    else:
        cm_plot      = cm.astype(float)
        fmt          = "{:d}"
        vmax         = float(cm.max())
        title_suffix = "(counts)"

    n   = len(SHORT_LABELS)
    fig, ax = plt.subplots(figsize = (7, 6))

    # Colour grid
    im = ax.imshow(cm_plot, interpolation = "nearest", cmap = "Blues",
                   vmin = 0, vmax = vmax, aspect = "auto")
    fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)

    # White cell borders
    for i in range(n + 1):
        ax.axhline(i - 0.5, color = "white", linewidth = 1.5)
        ax.axvline(i - 0.5, color = "white", linewidth = 1.5)

    # Annotate each cell
    thresh = cm_plot.max() / 2.0
    for row in range(n):
        for col in range(n):
            val   = cm_plot[row, col]
            raw   = cm[row, col]
            label = fmt.format(val if normalize else raw)
            color = "white" if val > thresh else "black"
            ax.text(col, row, label,
                    ha = "center", va = "center",
                    fontsize = 13, color = color, fontweight = "bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SHORT_LABELS, rotation = 25, ha = "right", fontsize = 10)
    ax.set_yticklabels(SHORT_LABELS, fontsize = 10)
    ax.set_xlabel("Predicted label", fontsize = 12, labelpad = 8)
    ax.set_ylabel("True label",      fontsize = 12, labelpad = 8)
    ax.set_title(
        f"{model_name} - {experiment}-subject Confusion Matrix {title_suffix}",
        fontsize = 12, fontweight = "bold", pad = 12)

    fig.tight_layout()
    save_plot(fig, f"{model_name}_{experiment}_confusion")
    return cm

# 3. Model comparison bar chart
def plot_comparison_bar(results: dict[str, dict[str, float]], experiment: str):
    """
    Grouped bar chart: Accuracy and Macro F1 for every model side by side
    """
    get_plot_style()
    models = list(results.keys())
    accs   = [results[m]["accuracy"] * 100 for m in models]
    f1s    = [results[m]["f1"]       * 100 for m in models]

    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize = (10, 5))

    bars1 = ax.bar(x-w/2, accs, w, label = "Accuracy", color = PALETTE[0], alpha = 0.87)
    bars2 = ax.bar(x+w/2, f1s,  w, label = "Macro F1", color = PALETTE[1], alpha = 0.87)

    get_bar_labels(ax, bars1)
    get_bar_labels(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize = 13)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        f"Model Comparison - {experiment}-subject Classification",
        fontsize = 13, fontweight = "bold")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, f"comparison_{experiment}_bar")

# 4. Intra vs Cross generalisation gap
def plot_intra_vs_cross(
        intra_results: dict[str, float],
        cross_results: dict[str, float],
    ):
    """Grouped bars showing how much accuracy drops from Intra to Cross"""
    get_plot_style()
    models = list(intra_results.keys())
    intra  = [intra_results[m]        * 100 for m in models]
    cross  = [cross_results.get(m, 0) * 100 for m in models]

    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize = (10, 5))

    bars_i = ax.bar(x - w / 2, intra, w, label="Intra-subject",
                    color = PALETTE[0], alpha = 0.87)
    bars_c = ax.bar(x + w / 2, cross, w, label="Cross-subject",
                    color = PALETTE[2], alpha = 0.87)

    get_bar_labels(ax, bars_i)
    get_bar_labels(ax, bars_c)

    # Annotate the accuracy drop in red above each Cross bar
    for i, (ia, ca) in enumerate(zip(intra, cross)):
        drop = ia - ca
        if drop > 0.5:
            ax.text(
                i + w / 2, ca + 10,
                f"\u2212{drop:.1f}%",
                ha = "center", va = "bottom", fontsize = 10,
                color="#C44E52", fontweight = "bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize = 13)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Intra-subject vs Cross-subject Accuracy",
        fontsize = 13, fontweight = "bold")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "intra_vs_cross")

# 5. Classification report (text only)
def print_report(
        y_true:     list[int],
        y_pred:     list[int],
        model_name: str,
        experiment: str,
    ) -> dict[str, float]:
    """
    Print a full per-class classification report and return accuracy + macro F1
    """
    print(f"\n{'─' * 50}")
    print(f"  Classification Report: {model_name} ({experiment}-subject)")
    print(f"{'─' * 50}")
    print(classification_report(y_true, y_pred, target_names = SHORT_LABELS, digits = 3))

    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro"))
    print(f"  Overall Accuracy : {acc:.3f}  ({acc * 100:.1f}%)")
    print(f"  Macro F1         : {f1:.3f}")
    return {"accuracy": acc, "f1": f1}

# 7. Save results to CSV
def save_results_csv(
        all_results: dict[str, dict[str, dict[str, float]]],
        filename: str = "results_summary.csv",
    ) -> Path:
    """
    Write a summary table to outputs/results/results_summary.csv
    """
    path = RESULTS_DIR / filename
    rows: list[dict] = []
    for experiment, models in all_results.items():
        for model, metrics in models.items():
            rows.append({
                "Experiment": experiment,
                "Model":      model,
                "Accuracy":   f"{metrics.get('accuracy',   0):.3f}",
                "Macro F1":   f"{metrics.get('f1',         0):.3f}",
                "Parameters": f"{(int)(metrics.get('parameters', 0))}",
                "Training-time(s)": f"{metrics.get('training_time', 0):.3f}"
            })

    fieldnames = ["Experiment", "Model", "Accuracy", "Macro F1", "Parameters", "Training-time(s)"]
    with open(path, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print to console
    col_w  = [max(len(h), max((len(r[h]) for r in rows), default = 0)) for h in fieldnames]
    header = "  " + "  ".join(h.ljust(w) for h, w in zip(fieldnames, col_w))
    print(f"\n--Results saved to {path}")
    print(header)
    print("  " + "  ".join("─" * w for w in col_w))
    for row in rows:
        print("  " + "  ".join(row[h].ljust(w) for h, w in zip(fieldnames, col_w)))

    return path
