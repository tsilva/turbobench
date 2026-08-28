from __future__ import annotations

import hashlib
import tomllib

import numpy as np

from turbobench.profiles import (
    BREAKOUT_RGB_TRANSPORT_CONVERSION,
    MARIO_BUTTON_ORDER,
    MARIO_PROMO_ACTION_COUNT,
    MARIO_PROMO_RAW_ACTION_SHA256,
    PROFILES,
    action_stream_hash,
    allowed_representation_conversion,
    benchmark_actions,
    mario_promo_actions,
    oracle_actions,
    profile_hash,
    profile_payload,
    profile_toml,
    promo_action_hash,
    promo_actions,
)


def test_v1_profiles_are_immutable_and_complete() -> None:
    assert tuple(profile_id for profile_id in PROFILES if profile_id.endswith("-v1")) == (
        "supermario/canonical-v1",
        "breakout/start-v1",
        "vizdoom/basic-v1",
    )
    for profile in (profile for profile in PROFILES.values() if profile.id.endswith("-v1")):
        assert profile.shapes == (1, 16, 32)
        assert profile.frame_skip == 4
        assert profile.frame_stack == 4
        assert profile.resize == (84, 84)
        assert profile.grayscale
        assert profile.layout == "chw"
        assert profile.resize_algorithm == "area"
        assert not profile.maxpool_last_two
        assert profile.benchmark_steps == 250
        assert len(profile_hash(profile)) == 64
        serialized = profile_toml(profile)
        assert tomllib.loads(serialized)["id"] == profile.id
        assert "[action_table]" in serialized
        assert profile_payload(profile)["benchmark"]["semantic_actions"]
        assert profile_payload(profile)["benchmark"]["action_table"]


def test_current_breakout_benchmark_supports_upstream_and_turbo_references() -> None:
    profile = PROFILES["breakout/start-v3"]
    assert profile.providers == (
        "env-breakoutatari2600-turbo-native",
        "env-stableretro-turbo",
        "stable-retro",
    )
    assert profile.shapes == (1, 16, 32)
    assert profile.states == ("Start",)
    assert profile.info_integer == ("score", "lives")
    assert not profile.native_transition_exact


def test_v2_profiles_are_exact_stable_retro_oracles() -> None:
    assert {profile_id for profile_id in PROFILES if profile_id.endswith("-v2")} == {
        "supermario/canonical-v2",
        "breakout/start-v2",
    }
    for profile_id in ("supermario/canonical-v2", "breakout/start-v2"):
        profile = PROFILES[profile_id]
        assert profile.semantic_authority == "stable-retro"
        assert profile.native_transition_exact
        assert profile.shapes == (1, 4)
        assert profile.oracle_shapes == (1, 4)
        assert profile.oracle_steps == 4_096
        payload = profile_payload(profile)
        assert payload["oracle"]["native_transition_exact"]
        assert tomllib.loads(profile_toml(profile))["schema"] == "turbobench.profile/v2"
    breakout = PROFILES["breakout/start-v2"]
    assert (
        allowed_representation_conversion(breakout)
        == BREAKOUT_RGB_TRANSPORT_CONVERSION
    )
    assert (
        profile_payload(breakout)["oracle"]["allowed_representation_conversion"]
        == BREAKOUT_RGB_TRANSPORT_CONVERSION
    )


def test_oracle_actions_are_seeded_random_with_directed_prefix() -> None:
    profile = PROFILES["breakout/start-v2"]
    first = oracle_actions(profile, 4, 64, seed=123)
    second = oracle_actions(profile, 4, 64, seed=123)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, oracle_actions(profile, 4, 64, seed=124))
    assert first.shape == (64, 4)
    assert set(first[:, 0]) == set(range(len(profile.semantic_actions)))


def test_serialized_profile_binds_the_semantic_action_table() -> None:
    profile = PROFILES["vizdoom/basic-v1"]
    serialized = profile_toml(profile)
    assert 'attack = ["ATTACK"]' in serialized
    assert profile_payload(profile)["benchmark"]["action_table"]["right"] == ["MOVE_RIGHT"]


def test_benchmark_actions_are_deterministic_varied_and_hashed() -> None:
    profile = PROFILES["supermario/canonical-v1"]
    first = benchmark_actions(profile, 32, 200)
    second = benchmark_actions(profile, 32, 200)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (200, 32)
    assert len(np.unique(first[:, 0])) > 1
    assert any(not np.array_equal(first[step], np.repeat(first[step, 0], 32)) for step in range(200))
    assert action_stream_hash(profile, first) == action_stream_hash(profile, second)
    assert action_stream_hash(profile, first) != action_stream_hash(profile, first[:, :16])


def test_imported_mario_promo_reproduces_verified_action_hash() -> None:
    actions = mario_promo_actions()
    assert len(actions) == MARIO_PROMO_ACTION_COUNT
    raw = bytearray()
    for labels in actions:
        pressed = set(labels)
        raw.extend(int(button in pressed) if button else 0 for button in MARIO_BUTTON_ORDER)
    assert hashlib.sha256(raw).hexdigest() == MARIO_PROMO_RAW_ACTION_SHA256
    profile = PROFILES["supermario/canonical-v1"]
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
