from copy import deepcopy

import pytest

from turbobench.policy import require_policy_frame_skip


@pytest.fixture
def capture():
    return {
        "frame_skip": 2,
        "contract": {"matches_training": True, "frame_skip": {"training": 2, "playback": 2}},
        "initial": {"reset_info": {"noop_reset_count": 1}},
        "action_contract": {"requested": {"table": [["BUTTON"], ["RIGHT"], ["LEFT"]]}},
        "transitions": [{"native_action": 0}, {"native_action": 1}],
    }


@pytest.fixture
def actions():
    return {
        "policy_frame_skip": 2,
        "reset_noop_prefix": 1,
        "actions": [[], ["BUTTON"], ["BUTTON"], ["RIGHT"], ["RIGHT"]],
    }


def test_matching_raw_frame_expansion_is_accepted(capture, actions):
    assert require_policy_frame_skip(capture, 2, benchmark_frame_skip=2, action_stream=actions) == 2


@pytest.mark.parametrize("field", ["training", "playback"])
def test_contract_mismatch_is_rejected_even_if_match_flag_is_true(capture, field):
    capture["contract"]["frame_skip"][field] = 4
    with pytest.raises(ValueError, match="frame skip mismatch"):
        require_policy_frame_skip(capture, 2)


@pytest.mark.parametrize("field", ["frame_skip", "contract"])
def test_missing_policy_metadata_is_rejected(capture, field):
    del capture[field]
    with pytest.raises(ValueError):
        require_policy_frame_skip(capture, 2)


def test_benchmark_with_other_frame_skip_is_rejected(capture):
    with pytest.raises(ValueError, match="benchmark=4"):
        require_policy_frame_skip(capture, 2, benchmark_frame_skip=4)


def test_raw_action_cadence_is_checked_beyond_metadata(capture, actions):
    actions["actions"][2] = ["RIGHT"]
    with pytest.raises(ValueError, match="trained action cadence"):
        require_policy_frame_skip(capture, 2, action_stream=actions)


def test_partial_decision_is_rejected(capture, actions):
    actions["actions"].pop()
    with pytest.raises(ValueError, match="complete policy decision"):
        require_policy_frame_skip(capture, 2, action_stream=actions)


def test_missing_stream_cadence_is_rejected(capture, actions):
    del actions["policy_frame_skip"]
    with pytest.raises(ValueError, match="rendered action stream"):
        require_policy_frame_skip(capture, 2, action_stream=actions)


def test_mutually_consistent_capture_cannot_override_training_metadata(capture):
    capture = deepcopy(capture)
    capture["frame_skip"] = 4
    capture["contract"]["frame_skip"] = {"training": 4, "playback": 4}
    with pytest.raises(ValueError, match="training=2"):
        require_policy_frame_skip(capture, 2)
