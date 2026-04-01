"""
metrics.py — Authoritative accuracy computation for LongBench diagnostic project.

This is the single source of truth for all metrics.
No other module should compute accuracy independently.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def accuracy(results: list[dict]) -> float:
    """
    Compute overall accuracy across a list of result records.

    Args:
        results: List of result dicts, each containing a boolean "correct" field.

    Returns:
        Fraction of correct predictions in [0.0, 1.0].

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("Cannot compute accuracy on an empty results list.")
    return sum(r["correct"] for r in results) / len(results)


def per_category_accuracy(results: list[dict]) -> dict[str, float]:
    """
    Compute accuracy broken down by the "category" field.

    Args:
        results: List of result dicts with "category" and "correct" fields.

    Returns:
        Dict mapping category name → accuracy float.

    Raises:
        ValueError: (propagated from accuracy()) if any category bucket is empty
                    — which cannot happen in practice since buckets are non-empty
                    by construction.
    """
    # Group by category
    buckets: dict[str, list[dict]] = {}
    for r in results:
        cat = r["category"]
        buckets.setdefault(cat, []).append(r)

    return {cat: accuracy(bucket) for cat, bucket in buckets.items()}


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    results: list[dict],
    n_resamples: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Estimate a confidence interval for overall accuracy via bootstrap resampling.

    Uses fully vectorised NumPy resampling — no Python-level loop over resamples.

    Args:
        results:     List of result dicts containing a boolean "correct" field.
        n_resamples: Number of bootstrap samples (default 10 000).
        ci:          Confidence level, e.g. 0.95 for a 95 % CI (default).

    Returns:
        (lower, upper) percentile bounds as floats in [0.0, 1.0].

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("Cannot compute bootstrap CI on an empty results list.")

    n = len(results)
    # Build a 1-D boolean array: True where the prediction was correct
    correct = np.array([r["correct"] for r in results], dtype=bool)

    # Draw n_resamples bootstrap samples in one vectorised call.
    # Shape: (n_resamples, n) — each row is one resample with replacement.
    rng = np.random.default_rng(seed=42)
    indices = rng.integers(0, n, size=(n_resamples, n))      # shape (R, n)
    resampled = correct[indices]                              # fancy indexing
    resample_accuracies = resampled.mean(axis=1)              # shape (R,)

    # Symmetric percentile bounds
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(resample_accuracies, 100 * alpha))
    upper = float(np.percentile(resample_accuracies, 100 * (1.0 - alpha)))

    return lower, upper
