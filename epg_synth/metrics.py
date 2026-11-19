"""Small metric helpers shared by evaluation scripts."""

from __future__ import annotations

import numpy as np


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for degenerate inputs."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size != second.size:
        raise ValueError("Metric arrays must have identical size")
    if first.size < 2 or first.std() <= 1e-12 or second.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])
