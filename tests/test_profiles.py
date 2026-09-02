from __future__ import annotations

import hashlib
import tomllib
from copy import deepcopy

import numpy as np
import pytest

from turbobench.profile_config import _parse_document
from turbobench.profiles import (
    BREAKOUT_RGB_TRANSPORT_CONVERSION,
    MARIO_BUTTON_ORDER,
    MARIO_PROMO_ACTION_COUNT,
    MARIO_PROMO_RAW_ACTION_SHA256,
    PROFILES,
    action_stream_hash,
    allowed_representation_conversion,
    canonical_actions,
    mario_promo_actions,
    profile_hash,
    profile_payload,
    profile_toml,
    promo_action_hash,
    promo_actions,
)


def test_workload_profiles_are_unified_and_complete() -> None:
    assert set(PROFILES) == {
        "supermario/world1-v1",
        "breakout/start-v1",
        "vizdoom/basic-v1",
    }
    for profile in PROFILES.values():
        assert profile.measurement_shapes == (1, 16, 32)
        assert profile.parity_shapes == (1, 4)
        assert profile.frame_skip == 4
        assert profile.frame_stack == 4
        assert profile.resize == (84, 84)
        assert profile.grayscale
        assert profile.layout == "chw"
        assert profile.resize_algorithm == "area"
        assert not profile.maxpool_last_two
        assert profile.quick_parity_steps == 32
        assert profile.full_parity_steps == 4_096
        assert profile.measurement_steps == 256
        assert profile.warmup_pairs == 1
        assert profile.light_pairs == 2
        assert profile.full_pairs == 7
        assert profile.snapshot_prefix_steps == 128
        assert profile.snapshot_suffix_steps == 128
        assert len(profile_hash(profile)) == 64
        serialized = profile_toml(profile)
        parsed = tomllib.loads(serialized)
        assert parsed["schema"] == "turbobench.workload-profile/v2"
        assert parsed["id"] == profile.id
        assert "[action_table]" in serialized
        assert "[run]" in serialized
        assert "[parity]" in serialized
        assert profile_payload(profile)["semantic_actions"]
        assert profile_payload(profile)["action_table"]


def test_current_breakout_benchmark_supports_upstream_and_turbo_references() -> None:
    profile = PROFILES["breakout/start-v1"]
    assert profile.providers == (
        "env-breakoutatari2600-turbo-native",
        "env-stableretro-turbo",
        "stable-retro",
    )
    assert profile.measurement_shapes == (1, 16, 32)
    assert profile.states == ("Start",)
    assert profile.info_integer == ("score", "lives", "ball_y")
    assert profile.native_transition_exact


def test_parity_sections_have_declarative_commitments() -> None:
    for profile in PROFILES.values():
        assert profile.native_transition_exact
        assert profile.measurement_shapes == (1, 16, 32)
        assert profile.parity_shapes == (1, 4)
        assert profile.full_parity_steps == 4_096
    assert PROFILES["supermario/world1-v1"].states == (
        "Level1-1",
        "Level1-2",
        "Level1-3",
        "Level1-4",
    )
    assert PROFILES["supermario/world1-v1"].quick_parity_shapes == (4,)
    assert PROFILES["breakout/start-v1"].quick_parity_shapes == (1,)
    assert PROFILES["vizdoom/basic-v1"].quick_parity_shapes == (1,)
    breakout = PROFILES["breakout/start-v1"]
    assert (
        allowed_representation_conversion(breakout)
        == BREAKOUT_RGB_TRANSPORT_CONVERSION
    )
    assert (
        profile_payload(breakout)["exact"]["allowed_representation_conversion"]
        == BREAKOUT_RGB_TRANSPORT_CONVERSION
    )


def test_canonical_actions_are_seeded_prefix_compatible_and_directed() -> None:
    profile = PROFILES["breakout/start-v1"]
    short = canonical_actions(profile, 4, 32)
    medium = canonical_actions(profile, 4, 256)
    long = canonical_actions(profile, 4, 4_096)
    np.testing.assert_array_equal(short, medium[:32])
    np.testing.assert_array_equal(medium, long[:256])
    np.testing.assert_array_equal(short, canonical_actions(profile, 4, 32))
    assert not np.array_equal(short, canonical_actions(profile, 4, 32, seed=124))
    directed = short[: len(profile.semantic_actions)]
    for lane in range(4):
        assert set(directed[:, lane]) == set(range(len(profile.semantic_actions)))


def test_serialized_profile_binds_the_semantic_action_table() -> None:
    profile = PROFILES["vizdoom/basic-v1"]
    serialized = profile_toml(profile)
    assert 'attack = ["ATTACK"]' in serialized
    assert profile_payload(profile)["action_table"]["right"] == ["MOVE_RIGHT"]


def test_canonical_actions_are_varied_and_hashed() -> None:
    profile = PROFILES["supermario/world1-v1"]
    first = canonical_actions(profile, 32, 200)
    second = canonical_actions(profile, 32, 200)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (200, 32)
    assert len(np.unique(first[:, 0])) > 1
    assert any(not np.array_equal(first[step], np.repeat(first[step, 0], 32)) for step in range(200))
    assert action_stream_hash(profile, first) == action_stream_hash(profile, second)
    assert action_stream_hash(profile, first) != action_stream_hash(profile, first[:, :16])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["parity"].update(authority="missing"), "parity providers"),
        (lambda raw: raw["parity"].update(checks=["unknown"]), "check list"),
        (lambda raw: raw["run"].update(measurement_shapes=[16, 32]), "begin with shape 1"),
    ],
)
def test_malformed_authority_check_and_shape_declarations_are_rejected(
    mutate, message: str
) -> None:
    raw = deepcopy(profile_payload(PROFILES["breakout/start-v1"]))
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        _parse_document(raw, "breakout--start-v1.toml", "")


def test_imported_mario_promo_reproduces_verified_action_hash() -> None:
    actions = mario_promo_actions()
    assert len(actions) == MARIO_PROMO_ACTION_COUNT
    raw = bytearray()
    for labels in actions:
        pressed = set(labels)
        raw.extend(int(button in pressed) if button else 0 for button in MARIO_BUTTON_ORDER)
    assert hashlib.sha256(raw).hexdigest() == MARIO_PROMO_RAW_ACTION_SHA256
    profile = PROFILES["supermario/world1-v1"]
    assert promo_action_hash(profile, actions) == MARIO_PROMO_RAW_ACTION_SHA256


def test_other_promo_trajectories_are_semantic_and_deterministic() -> None:
    breakout = PROFILES["breakout/start-v1"]
    first = promo_actions(breakout)
    assert first == promo_actions(breakout)
    assert len(first) == breakout.promo_steps
    assert ("BUTTON",) in first
    assert ("LEFT",) in first and ("RIGHT",) in first
    doom = PROFILES["vizdoom/basic-v1"]
    doom_actions = promo_actions(doom)
    assert len(doom_actions) == doom.promo_steps
    assert () in doom_actions
    assert ("MOVE_LEFT",) in doom_actions and ("MOVE_RIGHT",) in doom_actions
    assert doom.completion["kind"] == "terminal-or-info-at-least"
