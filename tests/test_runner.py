from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from turbobench.correctness import compare_replays, compare_traces
from turbobench.profiles import (
    action_stream_hash,
    benchmark_actions,
    get_profile,
    promo_action_hash,
    promo_actions,
)
from turbobench.runner import (
    Adapter,
    BreakoutPaddleNormalizer,
    FakeAdapter,
    ScalarPreprocessingEnv,
    ScalarWorkerConfig,
    _button_masks,
    _canonical_raw_rgb,
    _create_retro_overlay,
    _normalize_scalar_rgb,
    _semantic_raw_rgb,
    _turbo_v2_options,
    execute,
    fractional_area_resize,
    integer_area_resize,
    preprocess_frame,
)


def test_turbo_provider_options_spell_out_every_benchmark_semantic() -> None:
    profile = get_profile("breakout/start-v2")
    options = _turbo_v2_options(profile, 16, profile.frame_skip)
    assert tuple(options) == (
        "state",
        "scenario",
        "info",
        "record",
        "players",
        "inttype",
        "obs_type",
        "num_envs",
        "num_threads",
        "rom_path",
        "transport",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_crop_mode",
        "obs_crop_fill",
        "obs_grayscale",
        "obs_resize_algorithm",
        "obs_layout",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "sticky_action_prob",
        "noop_reset_max",
        "use_fire_reset",
        "reward_clip",
        "info_filter",
        "info_frame_stack_keys",
        "use_restricted_actions",
        "state_catalog",
        "render_mode",
    )
    assert options["transport"] == "numpy"
    assert options["num_envs"] == options["num_threads"] == 16
    assert options["state"] is None
    assert options["state_catalog"] == list(profile.states)
    assert options["use_restricted_actions"] == list(profile.action_table.values())


def test_button_mapping_translates_semantics_through_advertised_order() -> None:
    buttons = ("B", None, "START", "LEFT", "RIGHT", "A")
    masks = _button_masks([("RIGHT", "A"), (), ("B",)], buttons)
    assert masks.tolist() == [
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]


def test_cartridge_raw_rgb_normalization_preserves_native_rgb565_values() -> None:
    direct = np.asarray([[[88, 148, 248]]], dtype=np.uint8)
    expanded = np.asarray([[[90, 149, 255]]], dtype=np.uint8)
    for profile_id in ("supermario/canonical-v1", "breakout/start-v1"):
        profile = get_profile(profile_id)
        np.testing.assert_array_equal(
            _canonical_raw_rgb(expanded, profile),
            _canonical_raw_rgb(direct, profile),
        )


def test_exact_mario_raw_frames_decode_to_lossless_native_rgb565_codes() -> None:
    profile = get_profile("supermario/canonical-v2")
    native = np.asarray([[[88, 148, 248]]], dtype=np.uint8)
    expanded = np.asarray([[[90, 149, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        _semantic_raw_rgb(native, profile),
        _semantic_raw_rgb(expanded, profile),
    )


def test_upstream_atari_scalar_pixels_are_normalized_from_bgr() -> None:
    config = ScalarWorkerConfig(
        provider="stable-retro",
        game="Breakout-Atari2600-v0",
        state="Start",
        integration_path=None,
        frame_skip=4,
        frame_stack=4,
        crop_top=0,
        crop_bottom=0,
        crop_mode="remove",
        grayscale=True,
        resize=(84, 84),
    )
    bgr = np.asarray([[[72, 72, 205], [139, 141, 139]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        _normalize_scalar_rgb(bgr, config),
        np.asarray([[[200, 72, 72], [136, 136, 136]]], dtype=np.uint8),
    )


def test_exact_upstream_atari_preserves_public_rgb_bytes() -> None:
    config = ScalarWorkerConfig(
        provider="stable-retro",
        game="Breakout-Atari2600-v0",
        state="Start",
        integration_path=None,
        frame_skip=4,
        frame_stack=4,
        crop_top=0,
        crop_bottom=0,
        crop_mode="remove",
        grayscale=True,
        resize=(84, 84),
        native_transition_exact=True,
    )
    rgb = np.asarray([[[205, 72, 72], [139, 141, 139]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        _normalize_scalar_rgb(rgb, config),
        rgb,
    )


def test_scalar_retro_overlay_exposes_every_canonical_state(tmp_path: Path) -> None:
    profile = get_profile("supermario/canonical-v1")
    rom = tmp_path / "game.nes"
    rom.write_bytes(b"rom")
    info = tmp_path / "data.json"
    info.write_text("{}")
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}")
    states = {}
    for name in profile.states:
        state = tmp_path / f"{name}.state"
        state.write_bytes(name.encode())
        states[name] = str(state)
    overlay = _create_retro_overlay(
        profile,
        {
            "rom_path": str(rom),
            "state_paths": states,
            "info_schema_path": str(info),
            "scenario_path": str(scenario),
        },
    )
    try:
        game = Path(overlay.name) / profile.game
        assert (game / "rom.nes").resolve() == rom
        assert all(
            (game / f"{name}.state").resolve() == Path(path) for name, path in states.items()
        )
    finally:
        overlay.cleanup()


def test_scalar_breakout_overlay_uses_runtime_compatible_packaged_state(tmp_path: Path) -> None:
    profile = get_profile("breakout/start-v1")
    rom = tmp_path / "game.a26"
    rom.write_bytes(b"rom")
    state = tmp_path / "Start.state"
    state.write_bytes(b"provider-specific-state")
    overlay = _create_retro_overlay(
        profile,
        {"rom_path": str(rom), "state_paths": {"Start": str(state)}},
    )
    try:
        game = Path(overlay.name) / profile.game
        assert (game / "rom.a26").resolve() == rom
        assert not (game / "Start.state").exists()
    finally:
        overlay.cleanup()


def test_upstream_breakout_paddle_shim_matches_corrected_stella_repeat_sequence() -> None:
    normalizer = BreakoutPaddleNormalizer()

    def action(label: str) -> np.ndarray:
        value = np.zeros(8, dtype=np.int8)
        if label == "left":
            value[6] = 1
        elif label == "right":
            value[7] = 1
        return value

    positions = []
    for label in ("noop", "noop", "right", "left", "left"):
        for _ in range(4):
            normalizer.step(action(label))
        positions.append(normalizer.x)
    assert positions == [31, 26, 26, 25, 17]

    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[189:193, 8:24] = [200, 72, 72]
    corrected = normalizer.normalize_frame(frame)
    assert not corrected[189:193, 8:17].any()
    assert np.all(corrected[189:193, 17:33] == [200, 72, 72])


def test_upstream_breakout_reset_advances_blank_tia_frame(monkeypatch) -> None:
    class Box:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules, "gymnasium", SimpleNamespace(spaces=SimpleNamespace(Box=Box))
    )

    class TransientBlankEnv:
        buttons = ("BUTTON",)
        action_space = SimpleNamespace(shape=(1,))

        def __init__(self) -> None:
            self.calls = 0
            self.steps = 0
            self.unwrapped = self
            self.metadata = {}

        def reset(self, *, seed=None, options=None):
            self.calls += 1
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
            return frame, {}

        def step(self, action):
            self.steps += 1
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
            # Upstream Atari video is exposed in BGR order.
            frame[189:193, 8:24] = [72, 72, 200]
            return frame, 0.0, False, False, {}

        def close(self) -> None:
            pass

    config = ScalarWorkerConfig(
        provider="stable-retro",
        game="Breakout-Atari2600-v0",
        state="Start",
        integration_path=None,
        frame_skip=4,
        frame_stack=4,
        crop_top=0,
        crop_bottom=0,
        crop_mode="remove",
        grayscale=True,
        resize=(84, 84),
    )
    env = TransientBlankEnv()
    wrapped = ScalarPreprocessingEnv(env, config)
    observation, _info = wrapped.reset(seed=123)
    assert env.calls == 1
    assert env.steps == 1
    assert observation.shape == (4, 84, 84)
    assert np.all(wrapped.render()[189:193, 115:131] == [200, 72, 72])


def test_integer_area_preprocessing_matches_manual_bins_and_masks_hud() -> None:
    image = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    expected = np.asarray(
        [
            [image[y : y + 2, x : x + 2].astype(np.uint64).sum() // 4 for x in range(0, 8, 2)]
            for y in range(0, 8, 2)
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(integer_area_resize(image, 4, 4), expected)
    frame = np.full((8, 8, 3), 255, dtype=np.uint8)
    config = ScalarWorkerConfig(
        provider="stable-retro",
        game="test",
        state="start",
        integration_path=None,
        frame_skip=4,
        frame_stack=4,
        crop_top=2,
        crop_bottom=0,
        crop_mode="mask",
        grayscale=True,
        resize=(8, 8),
    )
    processed = preprocess_frame(frame, config)
    assert processed.shape == (1, 8, 8)
    assert not processed[:, :2].any()
    assert np.all(processed[:, 2:] == 255)


def test_vizdoom_fractional_area_resizes_rgb_before_grayscale() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[..., 0] = [[0, 10, 20], [30, 40, 50]]
    frame[..., 1] = [[50, 40, 30], [20, 10, 0]]
    resized = fractional_area_resize(frame, 1, 2)
    # Each destination covers 3/2 source columns. Integer overlap weights are
    # [2, 1] and [1, 2], with positive half-up per-channel rounding.
    np.testing.assert_array_equal(
        resized,
        np.asarray([[[18, 32, 0], [32, 18, 0]]], dtype=np.uint8),
    )
    config = ScalarWorkerConfig(
        provider="vizdoom",
        game="VizdoomBasic-v1",
        state="default",
        integration_path=None,
        frame_skip=4,
        frame_stack=4,
        crop_top=0,
        crop_bottom=0,
        crop_mode="remove",
        grayscale=True,
        resize=(1, 2),
    )
    expected_gray = (
        (
            77 * resized[..., 0].astype(np.uint32)
            + 150 * resized[..., 1].astype(np.uint32)
            + 29 * resized[..., 2].astype(np.uint32)
            + 128
        )
        >> 8
    ).astype(np.uint8)
    np.testing.assert_array_equal(preprocess_frame(frame, config), expected_gray[None])


def test_fake_adapter_disabled_autoreset_and_masked_reset_leave_other_lanes_unchanged() -> None:
    profile = get_profile("breakout/start-v1")
    adapter = FakeAdapter(profile, "fake", 4, 1.0)
    adapter.initial_reset(123)
    adapter.step(np.asarray([0, 1, 2, 3]))
    before = adapter._state.copy()
    mask = np.asarray([False, True, False, True])
    adapter.selective_reset(mask)
    np.testing.assert_array_equal(adapter._state[~mask], before[~mask])
    assert adapter.metadata()["autoreset_mode"] == "disabled"


def test_native_initial_reset_assigns_profile_states_round_robin() -> None:
    class StateCatalogEnv:
        num_envs = 6
        state_catalog = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
        buttons = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")

        def __init__(self) -> None:
            self.options = None

        def active_state_indices(self) -> np.ndarray:
            # Native providers expose their default state before the first reset.
            return np.zeros(self.num_envs, dtype=np.int32)

        def reset(self, *, seed=None, options=None):
            self.options = options
            return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8), {}

        def close(self) -> None:
            pass

    env = StateCatalogEnv()
    adapter = Adapter(
        env,
        get_profile("supermario/canonical-v1"),
        "supermariobrosnes-turbo",
        native_discrete=True,
    )
    adapter.initial_reset(123)
    np.testing.assert_array_equal(env.options["state_indices"], [0, 1, 2, 3, 0, 1])


def _trace_request(profile_id: str, shape: int = 3) -> dict:
    profile = get_profile(profile_id)
    actions = benchmark_actions(profile, shape, profile.correctness_steps)
    return {
        "operation": "trace",
        "provider": "fake",
        "adapter": "fake",
        "distribution": "turbobench",
        "profile": profile.id,
        "shape": shape,
        "assets": {},
        "fake_speed": 1.0,
        "seed": 123,
        "actions": actions.tolist(),
        "action_stream_sha256": action_stream_hash(profile, actions),
    }


def test_direct_trace_correctness_detects_every_contract_class() -> None:
    profile = get_profile("breakout/start-v1")
    left = execute(_trace_request(profile.id))
    right = execute(_trace_request(profile.id))
    assert compare_traces(left, right, profile)["passed"]
    cases = (
        ("observation", lambda item: item["steps"][0]["observation_sha256"].__setitem__(0, "bad")),
        ("raw", lambda item: item["steps"][0]["raw_frame_sha256"].__setitem__(0, "bad")),
        ("reward", lambda item: item["steps"][0]["rewards"].__setitem__(0, 1e-4)),
        ("termination", lambda item: item["steps"][0]["terminations"].__setitem__(0, True)),
        ("reset", lambda item: item["steps"][0]["reset_lanes"].append(0)),
        ("info", lambda item: item["steps"][0]["infos"][0].pop("score")),
    )
    for _name, mutate in cases:
        changed = deepcopy(right)
        mutate(changed)
        assert not compare_traces(left, changed, profile)["passed"]


def test_exact_profile_rejects_reward_delta_inside_v1_tolerance() -> None:
    profile = get_profile("breakout/start-v2")
    left = execute(_trace_request(profile.id, shape=1))
    right = deepcopy(left)
    right["steps"][0]["rewards"][0] += 5e-7
    result = compare_traces(left, right, profile)
    assert not result["passed"]
    assert result["first_mismatches"][0]["field"] == "steps[1].rewards[0]"


def test_fake_mario_promo_replay_completes_at_verified_step(tmp_path: Path) -> None:
    profile = get_profile("supermario/canonical-v1")
    actions = promo_actions(profile)
    base = {
        "operation": "promo",
        "provider": "fake",
        "adapter": "fake",
        "distribution": "turbobench",
        "profile": profile.id,
        "shape": 1,
        "assets": {},
        "fake_speed": 1.0,
        "seed": 123,
        "promo_actions": actions,
        "promo_action_sha256": promo_action_hash(profile, actions),
    }
    left = execute({**base, "output_frames": str(tmp_path / "left.rgb")})
    right = execute({**base, "provider": "fake-2", "output_frames": str(tmp_path / "right.rgb")})
    gate = compare_replays(left, right, profile)
    assert gate["passed"]
    assert gate["completion_step"] == 1_986
    assert gate["frame_count"] == 1_987
