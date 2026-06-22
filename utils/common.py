"""
utils/common.py
---------------
General-purpose helpers: reproducibility, device setup, and timing.
"""

import os
import random
import time
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass  # torch not installed, skip

    print(f"[seed] All seeds set to {seed}")


def get_device():
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"[device] Using: {device}")
        return device
    except ImportError:
        print("[device] torch not installed, returning 'cpu' string")
        return "cpu"


class Timer:
    """Simple context manager for timing code blocks.

    Usage:
        with Timer("matrix multiply"):
            result = A @ B
        # prints: [timer] matrix multiply: 0.0032s
    """

    def __init__(self, label: str = "block"):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[timer] {self.label}: {elapsed:.4f}s")