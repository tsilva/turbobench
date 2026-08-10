"""Paired benchmark statistics with deterministic uncertainty."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0x54555242


def invocation_median(repetitions: Sequence[float]) -> float:
    if len(repetitions) != 3:
        raise ValueError("each measured invocation must contain exactly three repetitions")
    if not all(math.isfinite(value) and value > 0 for value in repetitions):
        raise ValueError("SPS repetitions must be finite and positive")
    return float(statistics.median(repetitions))


def bootstrap_median_ci(
    ratios: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired ratios must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("paired ratios must be finite and positive")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(resamples, len(values)))]
    medians = np.median(samples, axis=1)
    lower, upper = np.quantile(medians, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def paired_statistics(
    pairs: Sequence[dict[str, object]], *, require_official_design: bool = True
) -> dict[str, object]:
    if require_official_design and len(pairs) != 7:
        raise ValueError("official measurements require exactly seven paired invocations")
    if not pairs:
        raise ValueError("paired measurements must not be empty")
    ratios: list[float] = []
    left_medians: list[float] = []
    right_medians: list[float] = []
    for pair in pairs:
        left = invocation_median(pair["left_sps"])
        right = invocation_median(pair["right_sps"])
        left_medians.append(left)
        right_medians.append(right)
        ratios.append(left / right)
    median_ratio = float(statistics.median(ratios))
    lower, upper = bootstrap_median_ci(ratios)
    if median_ratio >= 1.03 and lower > 1.0:
        outcome = "left_faster"
    elif median_ratio <= 1.0 / 1.03 and upper < 1.0:
        outcome = "right_faster"
    else:
        outcome = "inconclusive"
    return {
        "left_invocation_median_sps": left_medians,
        "right_invocation_median_sps": right_medians,
        "paired_ratios_left_over_right": ratios,
        "median_paired_ratio_left_over_right": median_ratio,
        "bootstrap": {
            "method": "paired median, deterministic percentile bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
            "ci": [lower, upper],
        },
        "median_left_sps": float(statistics.median(left_medians)),
        "median_right_sps": float(statistics.median(right_medians)),
        "outcome": outcome,
    }


def reciprocal_statistics(stats: dict[str, object]) -> dict[str, object]:
    ratios = [1.0 / float(value) for value in stats["paired_ratios_left_over_right"]]
    lower, upper = bootstrap_median_ci(ratios)
    median_ratio = float(statistics.median(ratios))
    outcome = stats["outcome"]
    reversed_outcome = {
        "left_faster": "right_faster",
        "right_faster": "left_faster",
        "inconclusive": "inconclusive",
    }[outcome]
    return {
        "left_invocation_median_sps": list(stats["right_invocation_median_sps"]),
        "right_invocation_median_sps": list(stats["left_invocation_median_sps"]),
        "paired_ratios_left_over_right": ratios,
        "median_paired_ratio_left_over_right": median_ratio,
        "bootstrap": {
            "method": "paired median, deterministic percentile bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
            "ci": [lower, upper],
        },
        "median_left_sps": float(stats["median_right_sps"]),
        "median_right_sps": float(stats["median_left_sps"]),
        "outcome": reversed_outcome,
    }
