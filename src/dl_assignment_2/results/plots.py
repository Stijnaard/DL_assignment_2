from typing import Optional
from pathlib import Path

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch import Tensor
import numpy as np

from dl_assignment_2.Niels_models.config import LABELS

def plot_confusion_matrix(labels: Tensor, predicted_indices: Tensor, show: bool = False, save_path:Optional[Path] = None):
    cm = confusion_matrix(labels.cpu(), predicted_indices.cpu())
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.colormaps['Blues'])
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()

def plot_compare_accuracies(model_accuracies: dict[str, float], experiment: str, show: bool = False, save_path: Optional[Path] = None):
    plt.figure(figsize=(10, 6))
    #plt.bar(model_accuracies.keys().
    model_names = list(model_accuracies.keys())
    accuracies = [acc * 100 for acc in model_accuracies.values()]
    plt.bar(model_names, accuracies, color='skyblue')
    plt.title(f'Model Accuracies Comparison - {experiment}')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)  # Assuming accuracy is between 0 and 100
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()


PALETTE = ["#416AAD", "#D27B49", "#55AE6A", "#C24B4F", "#7A69B2"]
LABEL_MAP = {
    "rest":                "Rest",
    "task_motor":          "Motor",
    "task_story_math":     "Math/Story",
    "task_working_memory": "Working Mem."}
SHORT_LABELS = [LABEL_MAP[l] for l in LABELS]

# Constant styling
def get_plot_style() -> None:
    """Apply a clean representation to each plot"""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         15,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "figure.dpi":        150})


# def plot_confusion_matrix(y_true: list[int], y_pred: list[int],
#         model_name: str, experiment: str, normalize:  bool = True) -> np.ndarray:
def plot_confusion_matrix(labels: Tensor, predicted_indices: Tensor, model_name: str, experiment: str, normalize: bool = False, show: bool = False, save_path:Optional[Path] = None):
    """Heatmap of true vs predicted labels"""
    get_plot_style()
    cm = confusion_matrix(labels.cpu(), predicted_indices.cpu())

    if normalize:
        cm_plot      = cm.astype(float) / cm.sum(axis = 1, keepdims = True)
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
            ax.text(col, row, label, ha = "center", va = "center",
                fontsize = 13, color = color, fontweight = "bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SHORT_LABELS, rotation = 25, ha = "right", fontsize = 12)
    ax.set_yticklabels(SHORT_LABELS, fontsize = 12)
    ax.set_xlabel("Predicted label", fontsize = 16, labelpad = 8)
    ax.set_ylabel("True label",      fontsize = 16, labelpad = 8)
    ax.set_title(
        f"{model_name} - {experiment}-subject Confusion Matrix {title_suffix}",
        fontsize = 16, fontweight = "bold", pad = 12)

    fig.tight_layout()
    #save_plot(fig, f"{model_name}_{experiment}_confusion")

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    return cm

# 3. Model comparison bar chart
def plot_comparison_bar(results: dict[str, dict[str, float]], experiment: str, show: bool = False, save_path: Optional[Path] = None) -> None:
    """
    Grouped bar chart: Accuracy and Macro F1 for every model side by side.
    - Accuracy label sits left of bar centre, F1 label sits right, so they
      never overlap even when values are close.
    - Model names are rotated and anchored so long names stay readable.
    """
    get_plot_style()
    models = list(results.keys())
    accs   = [results[m]["accuracy"] * 100 for m in models]
    f1s    = [results[m]["f1"]       * 100 for m in models]

    n   = len(models)
    w   = 0.35
    x   = np.arange(n)
    # Extra horizontal room so rotated labels don't get clipped
    fig, ax = plt.subplots(figsize = (max(10, n * 1.4), 6))

    bars1 = ax.bar(x - w/2, accs, w, label = "Accuracy", color = PALETTE[0], alpha = 0.87)
    bars2 = ax.bar(x + w/2, f1s,  w, label = "Macro F1", color = PALETTE[1], alpha = 0.87)

    # Accuracy label: anchored to the LEFT edge of its bar
    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() * 0.25,
            val + 0.8, f"{val:.1f}%",
            ha = "center", va = "bottom", fontsize = 10, color = PALETTE[0])

    # F1 label: anchored to the RIGHT edge of its bar
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() * 0.75,
            val + 0.8, f"{val:.1f}%",
            ha = "center", va = "bottom", fontsize = 10, color = PALETTE[1])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize = 12, rotation = 30, ha = "right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(f"Model Comparison - {experiment}-subject Classification",
        fontsize = 16, fontweight = "bold")
    ax.legend(loc = "center left", bbox_to_anchor = (1.02, 0.5), borderaxespad = 0)
    fig.tight_layout(rect = (0, 0, 0.82, 1))
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()

# 4. Intra vs Cross generalisation gap
def plot_intra_vs_cross(intra_results: dict[str, float],
        cross_results: dict[str, float], show: bool = False, save_path: Optional[Path] = None) -> None:
    """
    Grouped bars: Intra (left bar) vs Cross (right bar) accuracy per model.
    - Intra percentage label is pinned to the left side of the Intra bar.
    - Cross percentage label is pinned to the right side of the Cross bar.
    - Model names are rotated so long names never overlap.
    - Drop bracket is drawn above both bars with enough vertical clearance.
    """
    get_plot_style()
    models = list(intra_results.keys())
    intra  = [intra_results[m]        * 100 for m in models]
    cross  = [cross_results.get(m, 0) * 100 for m in models]

    n     = len(models)
    width = 0.35
    x     = np.arange(n)

    fig, ax = plt.subplots(figsize = (max(10, n * 1.4), 6))

    bars_i = ax.bar(x - width / 2, intra, width,
        label = "Intra-subject", color = PALETTE[0], alpha = 0.87)
    bars_c = ax.bar(x + width / 2, cross, width,
        label = "Cross-subject", color = PALETTE[2], alpha = 0.87)

    # Intra label: left quarter of the Intra bar
    for bar, val in zip(bars_i, intra):
        ax.text(bar.get_x() + bar.get_width() * 0.25,
            val + 0.8, f"{val:.1f}%",
            ha = "center", va = "bottom", fontsize = 10, color = PALETTE[0])

    # Cross label: right quarter of the Cross bar
    for bar, val in zip(bars_c, cross):
        ax.text(bar.get_x() + bar.get_width() * 0.75,
            val + 0.8, f"{val:.1f}%",
            ha = "center", va = "bottom", fontsize = 10, color = PALETTE[2])

    # Drop bracket - only when Intra beats Cross by more than 0.5 pp
    for i, (ia, ca) in enumerate(zip(intra, cross)):
        drop = ia - ca
        if drop > 0.5:
            # Place bracket 12 pp above the taller bar so it clears the labels
            y_top = max(ia, ca) + 12
            lx = x[i] - width / 2
            rx = x[i] + width / 2
            ax.plot([lx, lx, rx, rx], [y_top - 2, y_top, y_top, y_top - 2],
                color = PALETTE[3], linewidth = 1.5)
            ax.text(x[i], y_top + 0.5, f"\u2212{drop:.1f}%",
                ha = "center", va = "bottom", fontsize = 10,
                color = PALETTE[3], fontweight = "bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize = 12, rotation = 30, ha = "right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Intra-subject vs Cross-subject Accuracy",
        fontsize = 16, fontweight = "bold")
    ax.legend(loc = "center left", bbox_to_anchor = (1.02, 0.5), borderaxespad = 0)
    fig.tight_layout(rect = (0, 0, 0.82, 1))

    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
