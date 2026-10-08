from copy import deepcopy
from dataclasses import replace

import pytest

from turbobench.profiles import get_profile
from turbobench.scaling import scaling_progress, verify_scaling
from turbobench.workloads import _resource


@pytest.fixture
def profile():
    protocol = _resource("comparison_protocols", "paired-policy/v2")
    return replace(
        get_profile("breakout/start-v1"),
        measurement_shapes=tuple(protocol["run"]["measurement_shapes"]),
        resolved_workload={"protocol": protocol},
    )


def shapes(left, right):
    return {
        str(2**i): {"statistics": {"median_left_sps": a, "median_right_sps": b}}
        for i, (a, b) in enumerate(zip(left, right, strict=True))
    }


def test_does_not_stop_at_upstream_plateau_or_one_noisy_doubling(profile):
    assert (
        scaling_progress(profile, shapes([100, 101, 102], [100, 200, 400]))["stop_reason"] is None
    )
    one_flat = scaling_progress(profile, shapes([100, 101], [100, 102]))
    assert one_flat["stop_reason"] is None
    recovered = scaling_progress(profile, shapes([100, 101, 150], [100, 102, 180]))
    assert all(p["low_gain_streak"] == 0 for p in recovered["history"][-1]["providers"].values())


def test_plateau_requires_two_consecutive_low_gains_for_both_providers(profile):
    decision = scaling_progress(profile, shapes([100, 101, 102], [200, 202, 204]))
    assert decision["stop_reason"] == "both_providers_saturated"
    assert decision["complete"]
    assert all(p["status"] == "plateau" for p in decision["history"][-1]["providers"].values())


def test_downturn_and_later_recovery_are_compared_with_best_throughput(profile):
    assert scaling_progress(profile, shapes([100, 90], [200, 300]))["stop_reason"] is None
    recovered = scaling_progress(profile, shapes([100, 90, 110], [200, 300, 450]))
    assert recovered["history"][-1]["providers"]["left"]["status"] == "improving"
    decision = scaling_progress(profile, shapes([100, 90], [200, 180]))
    assert decision["complete"]
    assert all(p["status"] == "downgrade" for p in decision["history"][-1]["providers"].values())


def test_safety_cap_does_not_claim_plateau(profile):
    counts = profile.measurement_shapes
    result = {
        "schema": "turbobench.result/v4",
        "claim": {"status": "diagnostic"},
        "comparison": {"shapes": shapes(counts, counts)},
    }
    result["scaling"] = scaling_progress(profile, result["comparison"]["shapes"])
    assert result["scaling"]["stop_reason"] == "safety_cap"
    assert not result["scaling"]["complete"]
    assert verify_scaling(profile, result) == counts
    result["claim"]["status"] = "official"
    with pytest.raises(ValueError, match="safety cap"):
        verify_scaling(profile, result)


def test_stop_record_is_recomputed_and_rejects_truncation_or_continuation(profile):
    result = {
        "schema": "turbobench.result/v4",
        "claim": {"status": "official"},
        "comparison": {"shapes": shapes([100, 101, 102], [200, 201, 202])},
    }
    result["scaling"] = scaling_progress(profile, result["comparison"]["shapes"])
    assert verify_scaling(profile, result) == (1, 2, 4)
    changed = deepcopy(result)
    changed["scaling"]["history"][-1]["providers"]["right"]["status"] = "improving"
    with pytest.raises(ValueError, match="inconsistent"):
        verify_scaling(profile, changed)
    with pytest.raises(ValueError, match="continues"):
        scaling_progress(profile, shapes([100, 101, 102, 103], [200, 201, 202, 203]))
    with pytest.raises(ValueError, match="prefix"):
        scaling_progress(
            profile,
            {"1": result["comparison"]["shapes"]["1"], "4": result["comparison"]["shapes"]["4"]},
        )
    changed = deepcopy(result)
    del changed["comparison"]["shapes"]["4"]
    changed["scaling"] = scaling_progress(profile, changed["comparison"]["shapes"])
    with pytest.raises(ValueError, match="incomplete"):
        verify_scaling(profile, changed)


@pytest.mark.parametrize("sps", [0, -1, float("inf"), float("nan")])
def test_invalid_throughput_cannot_drive_stopping(profile, sps):
    with pytest.raises(ValueError, match="finite and positive"):
        scaling_progress(profile, shapes([sps], [1]))
