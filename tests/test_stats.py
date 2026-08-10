from __future__ import annotations

import math

import pytest

from turbobench.stats import (
    BOOTSTRAP_RESAMPLES,
    bootstrap_median_ci,
    invocation_median,
    paired_statistics,
    reciprocal_statistics,
)


def _pairs(ratios: list[float]) -> list[dict[str, object]]:
    return [
        {
            "left_sps": [ratio * 999, ratio * 1000, ratio * 1001],
            "right_sps": [999, 1000, 1001],
        }
        for ratio in ratios
    ]


def test_paired_statistics_declares_only_supported_winners() -> None:
    left = paired_statistics(_pairs([1.20, 1.18, 1.21, 1.19, 1.22, 1.17, 1.20]))
    assert left["outcome"] == "left_faster"
    assert left["bootstrap"]["resamples"] == BOOTSTRAP_RESAMPLES
    right = paired_statistics(_pairs([0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.80]))
    assert right["outcome"] == "right_faster"
    close = paired_statistics(_pairs([1.01, 1.02, 0.99, 1.00, 1.01, 1.00, 1.02]))
    assert close["outcome"] == "inconclusive"


def test_bootstrap_is_deterministic_and_reversal_is_reciprocal() -> None:
    ratios = [1.4, 1.5, 1.45, 1.42, 1.51, 1.39, 1.47]
    assert bootstrap_median_ci(ratios) == bootstrap_median_ci(ratios)
    result = paired_statistics(_pairs(ratios))
    reversed_result = reciprocal_statistics(result)
    assert reversed_result["outcome"] == "right_faster"
    assert math.isclose(
        reversed_result["median_paired_ratio_left_over_right"],
        1 / result["median_paired_ratio_left_over_right"],
    )


def test_statistics_rejects_invalid_design_and_samples() -> None:
    with pytest.raises(ValueError, match="seven"):
        paired_statistics(_pairs([1.2]))
    assert paired_statistics(_pairs([1.2]), require_official_design=False)["outcome"] == "left_faster"
    with pytest.raises(ValueError, match="three"):
        invocation_median([1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        invocation_median([1.0, 0.0, 2.0])

