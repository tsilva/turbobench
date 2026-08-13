from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from turbobench.runner import _construct_turbo_environment
from turbobench.turbo_api import (
    CAPABILITY_KEYS,
    COMMON_CONSTRUCTOR_DEFAULTS,
    TurboContractError,
    legacy_report,
    validate_constructor,
    validate_environment,
)


class ConformingFakeV2:
    metadata: ClassVar[dict[str, object]] = {
        "autoreset_mode": "disabled",
        "render_modes": ["rgb_array"],
        "turbo_api_version": 2,
        "transition_transport": "numpy",
    }

    def __init__(
        self,
        game,
        state=None,
        scenario=None,
        info=None,
        use_restricted_actions="default",
        record=False,
        players=1,
        inttype="stable",
        obs_type="image",
        render_mode=None,
        *,
        num_envs=1,
        num_threads=None,
        rom_path=None,
        transport="default",
        obs_copy="safe_view",
        obs_resize=(84, 84),
        obs_crop=None,
        obs_crop_mode="remove",
        obs_crop_fill=0,
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=False,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="all",
        info_frame_stack_keys=None,
        state_catalog=None,
    ):
        del (
            game,
            state,
            scenario,
            info,
            use_restricted_actions,
            record,
            players,
            inttype,
            obs_type,
            num_threads,
            rom_path,
            obs_copy,
            obs_resize,
            obs_crop,
            obs_crop_mode,
            obs_crop_fill,
            obs_grayscale,
            obs_resize_algorithm,
            obs_layout,
            frame_skip,
            frame_stack,
            maxpool_last_two,
            noop_reset_max,
            use_fire_reset,
            sticky_action_prob,
            reward_clip,
            info_filter,
            info_frame_stack_keys,
        )
        if transport not in {"default", "numpy"}:
            raise ValueError("NumPy only")
        if state_catalog is not None and tuple(state_catalog) != ("default",):
            raise ValueError("one fake state")
        self.num_envs = num_envs
        self.transport = "numpy"
        self.render_mode = render_mode
        self.state_catalog = ("default",)
        self.observation_ownership = "safe_view"
        self.observation_buffer_depth = 2
        self.single_observation_space = _Space((4, 84, 84), np.uint8)
        self.observation_space = _Space((num_envs, 4, 84, 84), np.uint8)
        self.action_space = _Space((num_envs,), np.int64)
        values = {
            "supported_action_modes": ("custom_discrete",),
            "supported_observation_layouts": ("chw",),
            "supported_observation_color_modes": ("grayscale",),
            "supported_resize_algorithms": ("area",),
            "supported_crop_modes": ("remove",),
            "supported_observation_copy_modes": ("copy", "safe_view", "unsafe_view"),
            "supported_transition_transports": ("numpy",),
            "supports_async_step": False,
            "supports_branching": False,
            "supports_device_api": False,
            "supports_emulator_ram": False,
            "supports_enemy_variants": False,
            "supports_fire_reset": False,
            "supports_info_frame_stack": False,
            "supports_live_snapshots": False,
            "supports_maxpool_last_two": False,
            "supports_noop_reset": False,
            "supports_per_lane_rgb": render_mode == "rgb_array",
            "supports_reward_clipping": False,
            "supports_snapshot_codec": False,
            "supports_state_catalog": True,
            "supports_sticky_action_prob": False,
            "supports_surface_variants": False,
        }
        assert tuple(values) == CAPABILITY_KEYS
        self.capabilities = MappingProxyType(values)
        self.signal_schema = MappingProxyType({})
        self._obs = np.zeros((num_envs, 4, 84, 84), dtype=np.uint8)
        self._indices = np.zeros(num_envs, dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        del seed
        options = dict(options or {})
        mask = options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_))
        self._obs[mask] = 0
        return self._obs.copy(), {
            "state_index": self._indices.copy(),
            "_state_index": mask.copy(),
            "start_source": np.zeros(self.num_envs, dtype=np.int8),
            "_start_source": mask.copy(),
            "noop_reset_count": np.zeros(self.num_envs, dtype=np.int64),
            "_noop_reset_count": mask.copy(),
        }

    def step(self, actions):
        self._obs += (np.asarray(actions, dtype=np.uint8) + 1)[:, None, None, None]
        return (
            self._obs.copy(),
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            {},
        )

    def active_state_indices(self):
        return self._indices

    def render(self):
        return self.render_lane(0)

    def render_lane(self, lane):
        if self.render_mode != "rgb_array":
            return None
        return np.zeros((8, 8, 3), dtype=np.uint8) + lane

    def get_images(self):
        return [self.render_lane(lane) for lane in range(self.num_envs)]

    def close(self):
        pass


class _Space:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def sample(self):
        return np.zeros(self.shape, dtype=self.dtype)


def test_normative_constructor_and_runtime_validator_pass() -> None:
    constructor = validate_constructor(ConformingFakeV2, "fake")
    assert constructor["passed"]
    env = ConformingFakeV2(
        "Fake-v0", num_envs=2, state_catalog=("default",), render_mode="rgb_array"
    )
    report = validate_environment(ConformingFakeV2, env, "fake")
    assert report["passed"]
    assert report["promotable"]
    assert len(report["report_sha256"]) == 64


def test_malformed_v2_constructor_fails() -> None:
    class Bad(ConformingFakeV2):
        def __init__(self, game, **kwargs):
            super().__init__(game, **kwargs)

    report = validate_constructor(Bad, "bad")
    assert not report["passed"]
    assert any("common order" in error or "catch-all" in error for error in report["errors"])

    class WrongDefault(ConformingFakeV2):
        pass

    signature = inspect.signature(ConformingFakeV2)
    parameters = list(signature.parameters.values())
    players = list(signature.parameters).index("players")
    parameters[players] = parameters[players].replace(default=True)
    WrongDefault.__signature__ = signature.replace(parameters=parameters)
    strict_report = validate_constructor(WrongDefault, "wrong-default")
    assert not strict_report["passed"]
    assert any("constructor default players" in error for error in strict_report["errors"])


def test_malformed_v2_runtime_rejects_object_transition_arrays() -> None:
    class BadRuntime(ConformingFakeV2):
        def step(self, actions):
            observations, rewards, terminated, truncated, _infos = super().step(actions)
            infos = {
                "label": np.asarray(["bad"] * self.num_envs, dtype=object),
            }
            return observations, rewards, terminated, truncated, infos

    env = BadRuntime(
        "Fake-v0",
        num_envs=2,
        state_catalog=("default",),
        render_mode="rgb_array",
    )
    report = validate_environment(BadRuntime, env, "bad-runtime")
    assert not report["passed"]
    assert any("object" in error or "step infos transport" in error for error in report["errors"])


def test_malformed_v2_is_rejected_before_construction() -> None:
    constructions = 0

    class Bad:
        metadata: ClassVar[dict[str, object]] = {
            "turbo_api_version": 2,
            "transition_transport": "numpy",
        }

        def __init__(self, game, **kwargs):
            nonlocal constructions
            constructions += 1

    with np.testing.assert_raises(TurboContractError):
        _construct_turbo_environment(Bad, "bad", "Bad-v0", {})
    assert constructions == 0


def test_v1_is_runnable_but_not_promotable() -> None:
    report = legacy_report("legacy", 1)
    assert report["passed"]
    assert not report["promotable"]

    class Legacy:
        metadata: ClassVar[dict[str, object]] = {"turbo_api_version": 1}

        def __init__(self, game, num_envs=1):
            self.game = game
            self.num_envs = num_envs

    env, constructed_report = _construct_turbo_environment(
        Legacy,
        "legacy",
        "Legacy-v0",
        {"num_envs": 3, "transport": "numpy", "state_catalog": ["Start"]},
    )
    assert env.game == "Legacy-v0"
    assert env.num_envs == 3
    assert constructed_report["passed"]
    assert not constructed_report["promotable"]


def test_torch_is_imported_lazily_for_numpy_validation(monkeypatch) -> None:
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("NumPy validator imported Torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    env = ConformingFakeV2(
        "Fake-v0", num_envs=2, state_catalog=("default",), render_mode="rgb_array"
    )
    assert validate_environment(ConformingFakeV2, env, "fake")["passed"]


def test_schema_constant_matches_requested_common_defaults() -> None:
    assert tuple(name for name, _default in COMMON_CONSTRUCTOR_DEFAULTS)[-3:] == (
        "info_filter",
        "info_frame_stack_keys",
        "state_catalog",
    )
