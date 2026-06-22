"""
utils/viz.py
------------
Lightweight visualization helpers for ML/NLP exercises.
Requires: matplotlib, numpy. seaborn is optional but recommended.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


# ---------------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------------

def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Loss Curve",
    save_path: Optional[str] = None,
):
    """Plot training (and optionally validation) loss over epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", linewidth=2)
    if val_losses:
        plt.plot(val_losses, label="Val loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# NLP-specific
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
):
    """Visualize a 2D attention matrix as a heatmap.

    Args:
        attention_weights: shape (query_len, key_len)
        row_labels:        query token labels (y-axis)
        col_labels:        key token labels (x-axis)
    """
    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False

    fig, ax = plt.subplots(figsize=(max(6, attention_weights.shape[1]), max(4, attention_weights.shape[0])))

    if use_seaborn:
        import seaborn as sns
        sns.heatmap(
            attention_weights,
            xticklabels=col_labels or "auto",
            yticklabels=row_labels or "auto",
            annot=True, fmt=".2f", cmap="Blues", ax=ax,
        )
    else:
        im = ax.imshow(attention_weights, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)
        if row_labels:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
        if col_labels:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved to {save_path}")
    plt.show()


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Embeddings (2D projection)",
    save_path: Optional[str] = None,
):
    """Scatter plot of 2D embeddings (e.g. after PCA or t-SNE).

    Args:
        embeddings: shape (n, 2)
        labels:     optional word/token labels for each point
    """
    assert embeddings.shape[1] == 2, "embeddings must be 2D — run PCA/t-SNE first"

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7, s=60)

    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings[i, 0], embeddings[i, 1]),
                         fontsize=9, ha="right", va="bottom")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# General ML
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    resolution: float = 0.05,
    save_path: Optional[str] = None,
):
    """Plot a 2D classification decision boundary.

    Args:
        model_fn: callable that takes (n, 2) array and returns (n,) predicted class labels
        X:        shape (n, 2) feature matrix
        y:        shape (n,) true labels
    """
    assert X.shape[1] == 2, "Only works for 2-feature datasets"

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_fn(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=40)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # loss curve
    plot_loss_curve([1.0, 0.8, 0.6, 0.5, 0.4], [1.1, 0.9, 0.7, 0.65, 0.6])

    # attention heatmap
    attn = np.random.dirichlet(np.ones(5), size=4)  # (4, 5)
    plot_attention_heatmap(
        attn,
        row_labels=["the", "cat", "sat", "mat"],
        col_labels=["the", "cat", "sat", "on", "mat"],
    )