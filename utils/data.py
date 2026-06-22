"""
utils/data.py
-------------
Toy dataset loaders and basic text preprocessing utilities.
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------

def make_classification_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 2,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple linearly separable classification dataset.

    Returns:
        X: shape (n_samples, n_features)
        y: shape (n_samples,) with integer class labels
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    samples_per_class = n_samples // n_classes

    for cls in range(n_classes):
        center = rng.uniform(-3, 3, size=n_features)
        points = rng.normal(loc=center, scale=noise + 0.5, size=(samples_per_class, n_features))
        X.append(points)
        y.append(np.full(samples_per_class, cls))

    return np.vstack(X), np.concatenate(y)


def make_sequence_data(
    n_samples: int = 50,
    seq_len: int = 10,
    vocab_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate random integer token sequences (useful for testing embeddings or RNNs).

    Returns:
        tokens: shape (n_samples, seq_len) with values in [0, vocab_size)
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(n_samples, seq_len))


def make_regression_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple regression dataset (y = Xw + noise).

    Returns:
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    w = rng.normal(size=(n_features,))
    y = X @ w + rng.normal(scale=noise, size=(n_samples,))
    return X, y


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """Lowercase and split text into tokens on whitespace and basic punctuation."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    return text.split()


def build_vocab(texts: List[str], max_size: int = None) -> dict:
    """Build a word -> index vocabulary from a list of raw strings.

    Special tokens:
        <PAD> = 0
        <UNK> = 1

    Args:
        texts:    list of raw text strings
        max_size: if set, keeps only the top-N most frequent words

    Returns:
        vocab: dict mapping word -> int index
    """
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    most_common = counter.most_common(max_size)
    for word, _ in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(tokens: List[str], vocab: dict) -> List[int]:
    """Convert a list of tokens to integer indices using vocab."""
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]


def pad_sequences(sequences: List[List[int]], pad_value: int = 0) -> np.ndarray:
    """Pad a list of variable-length sequences to the same length.

    Returns:
        array of shape (n_sequences, max_len)
    """
    max_len = max(len(s) for s in sequences)
    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple train/test split without sklearn.

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = make_classification_data()
    print(f"Classification data: X={X.shape}, y={y.shape}")

    tokens_batch = make_sequence_data()
    print(f"Sequence data: {tokens_batch.shape}")

    texts = ["The cat sat on the mat", "The dog sat on the log"]
    vocab = build_vocab(texts)
    print(f"Vocab: {vocab}")

    encoded = [encode(simple_tokenize(t), vocab) for t in texts]
    padded = pad_sequences(encoded)
    print(f"Padded sequences:\n{padded}")