"""Provider-runtime subprocess runner.

This module is executed by each content-addressed provider Python. Construction,
action generation, correctness, rendering, and encoding remain outside timed
regions; timed rollouts include preprocessing, IPC, infos, terminal detection,
and selective resets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from turbobench.lifecycle import (
    attest,
    evidence_binding,
    require_attestation,
    require_request_matches_spec,
)
from turbobench.model import Profile
from turbobench.profiles import (
    BREAKOUT_RGB_TRANSPORT_CONVERSION,
    allowed_representation_conversion,
    get_profile,
)
from turbobench.turbo_api import (
    TurboContractError,
    declared_api_version,
    legacy_report,
    validate_constructor,
    validate_environment,
)
from turbobench.util import canonical_json_hash, read_json, sha256_file, write_json


@dataclass(frozen=True)
class ScalarWorkerConfig:
    provider: str
    game: str
    state: str
    integration_path: str | None
    frame_skip: int
    frame_stack: int
    crop_top: int
    crop_bottom: int
    crop_mode: str
    grayscale: bool
    resize: tuple[int, int]
    profile_id: str = ""
    info_keys: tuple[str, ...] = ()
    worker_index: int = 0
    native_transition_exact: bool = False
    rom_path: str | None = None
    state_paths: tuple[tuple[str, str], ...] = ()
    noop_reset_max: int = 0


class ScalarPreprocessingEnv:
    """Gymnasium-compatible preprocessing kept inside each scalar worker."""

    def __init__(
        self,
        env: Any,
        config: ScalarWorkerConfig,
        worker_tempdir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        import gymnasium as gym

        self.env = env
        self.config = config
        self._worker_tempdir = worker_tempdir
        self.buttons = tuple(getattr(env.unwrapped, "buttons", ()))
        channels = 1 if config.grayscale else 3
        self._channels = channels
        self._stack = np.empty((channels * config.frame_stack, *config.resize), dtype=np.uint8)
        self._raw_frame: np.ndarray | None = None
        self._parity_action_history: list[np.ndarray] = []
        self._parity_reset_seed: int | None = None
        self._restoring_snapshot = False
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self._stack.shape, dtype=np.uint8
        )
        self.action_space = env.action_space
        self.metadata = dict(getattr(env, "metadata", {}))
        self.render_mode = "rgb_array"

    @property
    def unwrapped(self) -> Any:
        return self

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        return self.env.get_wrapper_attr(name)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if not self._restoring_snapshot:
            self._parity_action_history.clear()
            self._parity_reset_seed = seed
        observation, info = self.env.reset(seed=seed, options=options)
        raw = _screen(observation)
        self._raw_frame = _normalize_scalar_rgb(raw, self.config)
        noop_count = 0
        if self.config.noop_reset_max:
            if seed is None:
                raise ValueError("seeded reset distribution requires an explicit seed")
            noop_count = int(
                np.random.default_rng(seed).integers(
                    1, self.config.noop_reset_max + 1, dtype=np.uint64
                )
            )
            neutral = np.zeros(self.action_space.shape, dtype=np.int8)
            for _ in range(noop_count):
                observation, _reward, terminated, truncated, step_info = self.env.step(neutral)
                if terminated or truncated:
                    observation, step_info = self.env.reset(seed=seed)
                if step_info:
                    info = step_info
            self._raw_frame = _normalize_scalar_rgb(_screen(observation), self.config)
        if not info:
            data = getattr(self.env.unwrapped, "data", None)
            if data is not None and hasattr(data, "lookup_all"):
                data.update_ram()
                info = dict(data.lookup_all())
        info = self._semantic_info(info)
        if self.config.noop_reset_max:
            info["noop_reset_count"] = noop_count
        assert self._raw_frame is not None
        frame = preprocess_frame(self._raw_frame, self.config)
        for offset in range(0, self._stack.shape[0], self._channels):
            self._stack[offset : offset + self._channels] = frame
        return self._stack.copy(), info

    def step(self, action: Any):
        if not self._restoring_snapshot:
            self._parity_action_history.append(np.asarray(action).copy())
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        observation: Any = None
        for _ in range(self.config.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("scalar provider returned no observation")
        info = self._semantic_info(info)
        if self.config.provider == "vizdoom" and (terminated or truncated):
            # ViZDoom's public Gymnasium wrapper has no terminal GameState and
            # returns a synthetic zero screen. The Turbo API deliberately
            # repeats the last valid policy frame. Apply that provider-neutral
            # terminal convention before vectorization and retain the last
            # valid raw render for the trace contract.
            last_frame = self._stack[-self._channels :].copy()
            if self._stack.shape[0] > self._channels:
                self._stack[: -self._channels] = self._stack[self._channels :]
            self._stack[-self._channels :] = last_frame
            return self._stack.copy(), total_reward, terminated, truncated, info
        self._raw_frame = _normalize_scalar_rgb(_screen(observation), self.config)
        frame = preprocess_frame(self._raw_frame, self.config)
        if self._stack.shape[0] > self._channels:
            self._stack[: -self._channels] = self._stack[self._channels :]
        self._stack[-self._channels :] = frame
        return self._stack.copy(), total_reward, terminated, truncated, info

    def _semantic_info(self, info: Any) -> dict[str, Any]:
        result = dict(info) if isinstance(info, Mapping) else {}
        if self.config.provider != "vizdoom":
            return result
        vzd = importlib.import_module("vizdoom")
        game = self.env.unwrapped.game
        for name in self.config.info_keys:
            if name.casefold() == "episode_time":
                result[name] = game.get_episode_time()
            else:
                variable = getattr(vzd.GameVariable, name.upper())
                result[name] = game.get_game_variable(variable)
        return result

    def render(self) -> np.ndarray:
        if self._raw_frame is None:
            raise RuntimeError("render requested before reset")
        return self._raw_frame.copy()

    def ram(self) -> np.ndarray:
        getter = getattr(self.env.unwrapped, "get_ram", None)
        if getter is None:
            raise NotImplementedError("scalar provider does not expose emulator RAM")
        return np.asarray(getter(), dtype=np.uint8).copy()

    def capture_parity_snapshot(
        self,
    ) -> tuple[int | None, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        if self._raw_frame is None:
            raise RuntimeError("cannot capture a snapshot before reset")
        return (
            self._parity_reset_seed,
            tuple(action.copy() for action in self._parity_action_history),
            self._stack.copy(),
            self._raw_frame.copy(),
        )

    def restore_parity_snapshots(
        self,
        snapshots: Sequence[
            tuple[int | None, tuple[np.ndarray, ...], np.ndarray, np.ndarray]
        ],
    ) -> bool:
        reset_seed, history, expected_stack, expected_raw = snapshots[
            self.config.worker_index
        ]
        self._restoring_snapshot = True
        try:
            self.reset(seed=reset_seed)
            for action in history:
                self.step(action)
        finally:
            self._restoring_snapshot = False
        self._parity_action_history = [action.copy() for action in history]
        if not np.array_equal(self._stack, expected_stack) or not np.array_equal(
            self._raw_frame, expected_raw
        ):
            raise RuntimeError(
                "authority reset-and-replay failed to reconstruct the snapshot point"
            )
        return True

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            if self._worker_tempdir is not None:
                self._worker_tempdir.cleanup()


def _construct_scalar_env(
    config: ScalarWorkerConfig,
) -> tuple[Any, tempfile.TemporaryDirectory[str] | None]:
    worker_tempdir: tempfile.TemporaryDirectory[str] | None = None
    if config.provider == "stable-retro":
        retro = importlib.import_module("retro")
        if config.native_transition_exact:
            worker_tempdir = _create_original_retro_integration(config, retro)
            inttype = retro.data.Integrations.CUSTOM
        elif config.integration_path:
            retro.data.add_custom_integration(config.integration_path)
            inttype = retro.data.Integrations.ALL
        else:
            inttype = retro.data.Integrations.ALL
        env = retro.RetroEnv(
            game=config.game,
            state=config.state,
            use_restricted_actions=retro.Actions.ALL,
            inttype=inttype,
            render_mode="rgb_array",
        )
    elif config.provider == "vizdoom":
        import gymnasium as gym

        vzd = importlib.import_module("vizdoom")
        importlib.import_module("vizdoom.gymnasium_wrapper")
        # Upstream ViZDoom derives its native IPC instance ID from a 32-bit
        # PID/random/clock XOR. Simultaneous process construction can collide,
        # leaving both engines blocked on the same queues. Stagger only the
        # excluded construction region so every worker receives a unique ID.
        time.sleep(config.worker_index * 0.02)
        env = gym.make(
            config.game,
            frame_skip=1,
            max_buttons_pressed=0,
            render_mode="rgb_array",
            treat_episode_timeout_as_truncation=True,
        )
        worker_tempdir = tempfile.TemporaryDirectory(prefix="turbobench-vizdoom-worker-")
        env.unwrapped.game.set_doom_config_path(str(Path(worker_tempdir.name) / "engine.ini"))
        for name in config.info_keys:
            if name.casefold() != "episode_time":
                env.unwrapped.game.add_available_game_variable(
                    getattr(vzd.GameVariable, name.upper())
                )
    else:
        raise ValueError(f"unsupported scalar provider {config.provider!r}")
    return env, worker_tempdir


def _create_original_retro_integration(
    config: ScalarWorkerConfig,
    retro: Any,
) -> tempfile.TemporaryDirectory[str]:
    """Build a worker-local integration owned by the pinned Stable Retro wheel.

    Only the verified ROM and public state fixtures are overlaid. data.json and
    scenario.json always come from the installed authority package so ambient
    RETRO_DATA_PATH content cannot alter parity evidence.
    """

    package_root = Path(retro.__file__).resolve().parent
    source = package_root / "data" / "stable" / config.game
    if not source.is_dir():
        raise FileNotFoundError(
            f"Stable Retro authority integration is missing {config.game!r}: {source}"
        )
    worker_tempdir = tempfile.TemporaryDirectory(prefix="turbobench-retro-authority-")
    target_root = Path(worker_tempdir.name)
    target = target_root / config.game
    shutil.copytree(source, target)
    if config.game.startswith("Breakout-Atari2600") and "ball_y" in config.info_keys:
        info_path = target / "data.json"
        payload = read_json(info_path)
        payload.setdefault("info", {})["ball_y"] = {"address": 229, "type": "|u1"}
        write_json(info_path, payload)
    if config.rom_path:
        rom_path = Path(config.rom_path).resolve()
        for packaged_rom in target.glob("rom.*"):
            packaged_rom.unlink()
        (target / f"rom{rom_path.suffix}").symlink_to(rom_path)
    for state, raw_path in config.state_paths:
        destination = target / f"{state}.state"
        destination.unlink(missing_ok=True)
        destination.symlink_to(Path(raw_path).resolve())
    retro.data.add_custom_integration(str(target_root))
    return worker_tempdir


def _make_scalar_worker(config: ScalarWorkerConfig) -> ScalarPreprocessingEnv:
    env, worker_tempdir = _construct_scalar_env(config)
    return ScalarPreprocessingEnv(env, config, worker_tempdir)


class Adapter:
    def __init__(
        self,
        env: Any,
        profile: Profile,
        provider: str,
        *,
        native_discrete: bool,
        contract_report: dict[str, Any] | None = None,
        attestation_sha256: str | None = None,
        overlay: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.env = env
        self.profile = profile
        self.provider = provider
        self.native_discrete = native_discrete
        self.contract_report = contract_report or legacy_report(provider, None)
        self.attestation_sha256 = attestation_sha256
        self.instance_id = uuid.uuid4().hex
        self.closed = False
        self.overlay = overlay
        self.num_envs = int(env.num_envs)
        self._terminal_mask = np.zeros(self.num_envs, dtype=np.bool_)
        self._initial_seed: int | None = None
        self._reset_generations = np.zeros(self.num_envs, dtype=np.uint64)
        self._render_cache: list[np.ndarray] | None = None
        self._table = tuple(profile.action_table.values())
        self._table_index = {_labels_key(labels): index for index, labels in enumerate(self._table)}
        self._semantic_indices = np.asarray(
            [
                self._table_index[_labels_key(profile.action_table[name])]
                for name in profile.semantic_actions
            ],
            dtype=np.int64,
        )
        if native_discrete:
            self.buttons = tuple(getattr(env, "buttons", ()))
        else:
            self.buttons = self._scalar_buttons()

    def _scalar_buttons(self) -> tuple[str | None, ...]:
        if self.provider == "vizdoom":
            return ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
        try:
            values = self.env.get_attr("buttons")
            if values:
                return tuple(values[0])
        except (AttributeError, TypeError):
            pass
        if self.profile.logical_environment == "supermario":
            return ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
        return ("BUTTON",)

    def initial_reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        options = self._reset_options(np.ones(self.num_envs, dtype=np.bool_), initial=True)
        self._initial_seed = int(seed)
        self._reset_generations.fill(0)
        self._terminal_mask.fill(False)
        self._render_cache = None
        return self.env.reset(seed=seed, options=options)

    def selective_reset(self, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._initial_seed is None:
            raise RuntimeError("selective reset requested before the initial seeded reset")
        self._reset_generations[mask] += 1
        seeds = [
            (
                self._initial_seed
                + int(self._reset_generations[lane]) * self.num_envs
                + lane
            )
            % (2**32)
            if mask[lane]
            else None
            for lane in range(self.num_envs)
        ]
        result = self.env.reset(seed=seeds, options=self._reset_options(mask))
        self._terminal_mask[mask] = False
        return result

    def _reset_options(self, mask: np.ndarray, *, initial: bool = False) -> dict[str, Any]:
        options: dict[str, Any] = {"reset_mask": mask}
        if self.native_discrete and getattr(self.env, "state_catalog", ()):
            current = np.asarray(self.env.active_state_indices(), dtype=np.int32)
            state_indices = np.full(self.num_envs, -1, dtype=np.int32)
            assigned = np.arange(self.num_envs, dtype=np.int32) % len(self.profile.states)
            state_indices[mask] = (
                assigned[mask]
                if initial
                else np.where(current[mask] >= 0, current[mask], assigned[mask])
            )
            options["state_indices"] = state_indices
        return options

    def benchmark_action(self, semantic_indices: np.ndarray) -> np.ndarray:
        table_indices = self._semantic_indices[np.asarray(semantic_indices, dtype=np.int64)]
        if self.native_discrete:
            return table_indices
        labels = [self._table[int(index)] for index in table_indices]
        return _button_masks(labels, self.buttons)

    def promo_action(self, labels: Sequence[str]) -> np.ndarray:
        repeated = [tuple(labels)] * self.num_envs
        if self.native_discrete:
            index = self._table_index.get(_labels_key(labels))
            if index is None:
                raise ValueError(
                    f"promo action {tuple(labels)!r} is absent from the profile action table"
                )
            return np.full(self.num_envs, index, dtype=np.int64)
        return _button_masks(repeated, self.buttons)

    def step(self, action: np.ndarray):
        result = self.env.step(action)
        self._terminal_mask = np.logical_or(result[2], result[3])
        return result

    def render_frames(self) -> list[np.ndarray]:
        if self.native_discrete:
            if hasattr(self.env, "get_images"):
                raw = self.env.get_images()
            else:
                raw = [self.env.render_lane(index) for index in range(self.num_envs)]
        else:
            raw = self.env.call("render")
        frames = [
            _semantic_raw_rgb(frame, self.profile, self.provider)
            if self.profile.native_transition_exact
            else _comparison_raw_rgb(frame, self.profile, self.provider)
            for frame in raw
        ]
        if self.profile.logical_environment == "vizdoom-basic":
            if self._render_cache is not None:
                frames = [
                    self._render_cache[lane].copy() if self._terminal_mask[lane] else frame
                    for lane, frame in enumerate(frames)
                ]
            if self._render_cache is None:
                self._render_cache = [frame.copy() for frame in frames]
            else:
                for lane, frame in enumerate(frames):
                    if not self._terminal_mask[lane]:
                        self._render_cache[lane] = frame.copy()
        return frames

    def rams(self) -> list[np.ndarray]:
        if self.native_discrete:
            getter = getattr(self.env, "ram", None)
            if getter is None:
                raise NotImplementedError(
                    f"{self.provider} does not expose lane-aligned emulator RAM"
                )
            values = np.asarray(getter(), dtype=np.uint8)
        else:
            values = self.env.call("ram")
        if (
            self.profile.native_transition_exact
            and self.profile.logical_environment == "supermario"
        ):
            # Stable Retro exposes registered memory blocks in address order:
            # the canonical 2 KiB CPU RAM followed by mapper work RAM.  The
            # Mario fidelity contract is deliberately scoped to CPU RAM, which
            # is also the public Turbo provider contract.
            values = np.asarray(values, dtype=np.uint8)[:, :2048]
        return [np.ascontiguousarray(value, dtype=np.uint8) for value in values]

    def capture_snapshots(self) -> Any:
        mask = np.ones(self.num_envs, dtype=np.bool_)
        if self.native_discrete:
            return self.env.capture_snapshots(mask)
        return tuple(self.env.call("capture_parity_snapshot"))

    def restore_snapshots(self, snapshots: Any) -> None:
        self._terminal_mask.fill(False)
        self._render_cache = None
        if self.native_discrete:
            self.env.reset(
                options={
                    "reset_mask": np.ones(self.num_envs, dtype=np.bool_),
                    "state_indices": np.full(self.num_envs, -1, dtype=np.int32),
                    "snapshots": list(snapshots),
                }
            )
        else:
            restored = self.env.call("restore_parity_snapshots", snapshots)
            if not all(restored):
                raise RuntimeError("scalar snapshot restoration was incomplete")

    def metadata(self) -> dict[str, Any]:
        observation_space = getattr(self.env, "single_observation_space", None)
        return {
            "provider": self.provider,
            "adapter": "native-vector" if self.native_discrete else "scalar-async-vector",
            "autoreset_mode": str(getattr(self.env, "autoreset_mode", "disabled")),
            "turbo_api_version": getattr(self.env, "metadata", {}).get("turbo_api_version"),
            "turbo_contract_report": self.contract_report,
            "action_table": [list(labels) for labels in self._table],
            "action_meanings": list(getattr(self.env, "action_meanings", ()) or ()),
            "buttons": list(self.buttons),
            "capabilities": _jsonable(dict(getattr(self.env, "capabilities", {}))),
            "snapshot_strategy": (
                "native-live-snapshot"
                if self.native_discrete
                else "authority-reset-and-action-replay"
            ),
            "raw_render_normalization": (
                "rgb565-native-code"
                if self.profile.native_transition_exact
                and self.profile.logical_environment == "supermario"
                else BREAKOUT_RGB_TRANSPORT_CONVERSION
                if self.profile.logical_environment == "breakout"
                and self.provider in {"stable-retro", "env-stableretro-turbo"}
                else "identity"
                if self.profile.native_transition_exact
                else "rgb565-high-bits"
                if self.profile.logical_environment in {"supermario", "breakout"}
                else "rgb8"
            ),
            "source_channel_order": (
                "bgr"
                if self.provider in {"stable-retro", "env-stableretro-turbo"}
                and self.profile.logical_environment == "breakout"
                else "rgb"
            ),
            "palette_normalization": (
                BREAKOUT_RGB_TRANSPORT_CONVERSION
                if self.provider == "stable-retro"
                and self.profile.logical_environment == "breakout"
                and self.profile.native_transition_exact
                else "stella-legacy-to-canonical-v1"
                if self.provider in {"stable-retro", "env-stableretro-turbo"}
                and self.profile.logical_environment == "breakout"
                and not self.profile.native_transition_exact
                else "identity"
            ),
            "compatibility_normalization": (
                "vizdoom-last-valid-terminal-frame-v1"
                if self.profile.logical_environment == "vizdoom-basic"
                else "identity"
            ),
            "native_transition_exact": self.profile.native_transition_exact,
            "allowed_representation_conversion": allowed_representation_conversion(
                self.profile
            ),
            "ram": {
                "representation": (
                    "nes-cpu-ram-0x0000-0x07ff"
                    if self.profile.logical_environment == "supermario"
                    and self.profile.native_transition_exact
                    else "stable-retro-registered-block-order"
                ),
                "conversion": (
                    "select-canonical-cpu-ram-address-range"
                    if self.profile.logical_environment == "supermario"
                    and self.profile.native_transition_exact
                    else "identity"
                ),
            },
            "observation": {
                "shape": list(getattr(observation_space, "shape", ())),
                "dtype": str(getattr(observation_space, "dtype", "unknown")),
                "layout": self.profile.layout,
                "ownership": getattr(self.env, "observation_ownership", "owned"),
            },
        }

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.closed = True
            if self.overlay is not None:
                self.overlay.cleanup()


class _InMemorySpace:
    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)


_FAKE_PROCESS_POISONED = False
_WORKLOAD_OPERATIONS = frozenset({"trace", "benchmark", "reset-distribution", "promo"})


class _InMemoryFakeV2Env:
    """Small real v2 provider used to exercise contract gates in harness tests."""

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
    ) -> None:
        poison_enabled = game == "Poison-v0"
        del (
            game,
            scenario,
            info,
            use_restricted_actions,
            num_threads,
            rom_path,
            info_filter,
        )
        if state is not None and state_catalog is not None:
            raise ValueError("state and state_catalog are mutually exclusive")
        catalog = ("default",) if state_catalog is None else tuple(state_catalog)
        if not catalog or len(set(catalog)) != len(catalog):
            raise ValueError("state_catalog must be non-empty and duplicate-free")
        if state not in (None, "default") or catalog != ("default",):
            raise ValueError("the fake provider supports only its default state")
        if transport == "default":
            transport = "numpy"
        if transport != "numpy":
            raise ValueError("the fake provider uses NumPy transport")
        neutral = (
            record is False
            and players == 1
            and inttype == "stable"
            and obs_type == "image"
            and obs_copy in {"copy", "safe_view", "unsafe_view"}
            and obs_resize == (84, 84)
            and obs_crop is None
            and obs_crop_mode == "remove"
            and obs_crop_fill == 0
            and obs_grayscale is True
            and obs_resize_algorithm == "area"
            and obs_layout == "chw"
            and frame_skip == 4
            and frame_stack == 4
            and maxpool_last_two is False
            and noop_reset_max == 0
            and use_fire_reset is False
            and sticky_action_prob == 0.0
            and reward_clip is False
            and info_frame_stack_keys is None
        )
        if not neutral:
            raise ValueError("unsupported non-neutral fake provider option")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.num_envs = int(num_envs)
        self.poison_enabled = poison_enabled
        self.instance_poisoned = False
        self.transport = transport
        self.render_mode = render_mode
        self.state_catalog = catalog
        self.observation_ownership = "owned" if obs_copy == "copy" else obs_copy
        self.observation_buffer_depth = (
            None if obs_copy == "copy" else 2 if obs_copy == "safe_view" else 1
        )
        self.single_observation_space = _InMemorySpace((4, 84, 84), np.uint8)
        self.observation_space = _InMemorySpace((self.num_envs, 4, 84, 84), np.uint8)
        self.action_space = _InMemorySpace((self.num_envs,), np.int64)
        self._obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._states = np.zeros(self.num_envs, dtype=np.int32)
        self.signal_schema = MappingProxyType({})
        self.capabilities = MappingProxyType(
            {
                "supported_action_modes": ("custom_discrete",),
                "supported_observation_layouts": ("chw",),
                "supported_observation_color_modes": ("grayscale",),
                "supported_resize_algorithms": ("area",),
                "supported_crop_modes": ("remove",),
                "supported_observation_copy_modes": (
                    "copy",
                    "safe_view",
                    "unsafe_view",
                ),
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
        )

    def reset(self, *, seed=None, options=None):
        del seed
        options = dict(options or {})
        mask = options.pop("reset_mask", np.ones(self.num_envs, dtype=np.bool_))
        indices = options.pop("state_indices", self._states)
        if options:
            raise ValueError(f"unsupported reset options: {sorted(options)}")
        self._states[mask] = indices[mask]
        self._obs[mask] = 0
        return self._obs.copy(), {
            "state_index": self._states.copy(),
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
        return self._states

    def render_lane(self, lane):
        global _FAKE_PROCESS_POISONED
        if self.poison_enabled:
            self.instance_poisoned = True
            _FAKE_PROCESS_POISONED = True
        if self.render_mode != "rgb_array":
            return None
        return np.full((8, 8, 3), lane, dtype=np.uint8)

    def render(self):
        return self.render_lane(0)

    def get_images(self):
        return [self.render_lane(lane) for lane in range(self.num_envs)]

    def close(self):
        pass


class FakeAdapter:
    def __init__(
        self,
        profile: Profile,
        provider: str,
        shape: int,
        speed: float,
        *,
        contract_report: dict[str, Any] | None = None,
        attestation_sha256: str | None = None,
    ) -> None:
        self.profile = profile
        self.provider = provider
        self.num_envs = shape
        self.speed = speed
        self.step_index = 0
        self._state = np.arange(shape, dtype=np.int64)
        self.in_promo = False
        self.contract_report = contract_report or legacy_report(provider, None)
        self.attestation_sha256 = attestation_sha256
        self.instance_id = uuid.uuid4().hex
        self.closed = False
        self.process_poisoned_at_construction = _FAKE_PROCESS_POISONED
        self.instance_poisoned = False
        self.render_calls = 0

    def initial_reset(self, seed: int):
        self.step_index = 0
        self.in_promo = False
        self._state = np.arange(self.num_envs, dtype=np.int64) + seed % 17
        infos = {key: np.zeros(self.num_envs, dtype=np.int64) for key in self.profile.info_integer}
        return self._obs(), infos

    def selective_reset(self, mask: np.ndarray):
        self._state[mask] = np.flatnonzero(mask)
        return self._obs(), {}

    def benchmark_action(self, semantic_indices: np.ndarray) -> np.ndarray:
        return np.asarray(semantic_indices, dtype=np.int64)

    def promo_action(self, labels: Sequence[str]) -> np.ndarray:
        self.in_promo = True
        value = sum(sum(map(ord, label)) for label in labels) % 7
        return np.full(self.num_envs, value, dtype=np.int64)

    def step(self, action: np.ndarray):
        self.step_index += 1
        self._state = (self._state * 33 + np.asarray(action) + self.step_index) % 251
        reward = (np.asarray(action) % 3).astype(np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        if not self.in_promo and self.step_index % 97 == 0:
            terminated[(self.step_index // 97) % self.num_envs] = True
        if self.in_promo:
            infos = {
                key: np.zeros(self.num_envs, dtype=np.int64) for key in self.profile.info_integer
            }
            if self.profile.logical_environment == "supermario" and self.step_index >= 1_986:
                infos["levelLo"][:] = 1
            if self.profile.logical_environment == "vizdoom-basic" and self.step_index >= 300:
                infos["killcount"][:] = 1
        else:
            infos = {
                key: (self._state + index).astype(np.int64)
                for index, key in enumerate(self.profile.info_integer)
            }
        return self._obs(), reward, terminated, truncated, infos

    def _obs(self) -> np.ndarray:
        obs = np.empty((self.num_envs, 4, 84, 84), dtype=np.uint8)
        for lane in range(self.num_envs):
            obs[lane].fill(int(self._state[lane]))
        return obs

    def render_frames(self) -> list[np.ndarray]:
        self.render_calls += 1
        if "poison" in self.provider:
            self.instance_poisoned = True
        frames = []
        for lane in range(self.num_envs):
            value = int(self._state[lane])
            frame = np.empty((96, 128, 3), dtype=np.uint8)
            frame[..., 0] = value
            frame[..., 1] = np.arange(128, dtype=np.uint8)
            frame[..., 2] = np.arange(96, dtype=np.uint8)[:, None]
            frames.append(frame)
        return frames

    def rams(self) -> list[np.ndarray]:
        return [np.full(128, int(value), dtype=np.uint8) for value in self._state]

    def capture_snapshots(self) -> tuple[np.ndarray, int, bool]:
        return self._state.copy(), self.step_index, self.in_promo

    def restore_snapshots(self, snapshots: tuple[np.ndarray, int, bool]) -> None:
        state, step_index, in_promo = snapshots
        self._state = state.copy()
        self.step_index = step_index
        self.in_promo = in_promo

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "adapter": "fake",
            "autoreset_mode": "disabled",
            "turbo_api_version": 2,
            "turbo_contract_report": self.contract_report,
            "action_table": [list(value) for value in self.profile.action_table.values()],
            "action_meanings": list(self.profile.action_table),
            "buttons": sorted(
                {button for labels in self.profile.action_table.values() for button in labels}
            ),
            "capabilities": {"supports_per_lane_rgb": True, "supports_state_catalog": True},
            "observation": {
                "shape": [4, 84, 84],
                "dtype": "uint8",
                "layout": "chw",
                "ownership": "owned",
            },
        }

    def close(self) -> None:
        self.closed = True


def integer_area_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"expected HW or HWC image, got shape {image.shape}")
    source_height, source_width = image.shape[:2]
    y0 = np.arange(height, dtype=np.int64) * source_height // height
    y1 = np.maximum(
        np.arange(1, height + 1, dtype=np.int64) * source_height // height, y0 + 1
    ).clip(max=source_height)
    x0 = np.arange(width, dtype=np.int64) * source_width // width
    x1 = np.maximum(np.arange(1, width + 1, dtype=np.int64) * source_width // width, x0 + 1).clip(
        max=source_width
    )
    integral = np.asarray(image, dtype=np.uint64).cumsum(axis=0).cumsum(axis=1)
    padding = ((1, 0), (1, 0)) + (((0, 0),) if image.ndim == 3 else ())
    integral = np.pad(integral, padding, mode="constant")
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    counts = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    if image.ndim == 3:
        counts = counts[:, :, None]
    return (sums // counts).astype(np.uint8)


@lru_cache(maxsize=16)
def _fractional_area_plan(
    source_height: int,
    source_width: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the exact integer-overlap area plan used by ViZDoom Turbo."""

    samples: list[list[tuple[int, int]]] = []
    maximum = 0
    for out_y in range(height):
        y_start = out_y * source_height
        y_end = (out_y + 1) * source_height
        y_samples = []
        for source_y in range(y_start // height, min(source_height, -(-y_end // height))):
            overlap = min(y_end, (source_y + 1) * height) - max(y_start, source_y * height)
            y_samples.append((source_y, overlap))
        for out_x in range(width):
            x_start = out_x * source_width
            x_end = (out_x + 1) * source_width
            pixel_samples = []
            for source_x in range(x_start // width, min(source_width, -(-x_end // width))):
                x_overlap = min(x_end, (source_x + 1) * width) - max(x_start, source_x * width)
                pixel_samples.extend(
                    (source_y * source_width + source_x, y_overlap * x_overlap)
                    for source_y, y_overlap in y_samples
                )
            maximum = max(maximum, len(pixel_samples))
            samples.append(pixel_samples)
    offsets = np.zeros((height * width, maximum), dtype=np.intp)
    weights = np.zeros((height * width, maximum), dtype=np.uint64)
    for pixel, pixel_samples in enumerate(samples):
        for sample, (offset, weight) in enumerate(pixel_samples):
            offsets[pixel, sample] = offset
            weights[pixel, sample] = weight
    offsets.setflags(write=False)
    weights.setflags(write=False)
    return offsets, weights, source_height * source_width


def fractional_area_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Fractional box resize with exact positive half-up channel rounding."""

    if image.ndim not in (2, 3):
        raise ValueError(f"expected HW or HWC image, got shape {image.shape}")
    source_height, source_width = image.shape[:2]
    offsets, weights, divisor = _fractional_area_plan(source_height, source_width, height, width)
    flat = np.asarray(image, dtype=np.uint64).reshape(source_height * source_width, -1)
    sums = (flat[offsets] * weights[..., None]).sum(axis=1, dtype=np.uint64)
    resized = ((sums + divisor // 2) // divisor).astype(np.uint8)
    # ViZDoom Turbo deliberately recomputes exact half cases through its
    # floating reference path. Reproduce its operation order because binary
    # rounding can place a mathematical half just below the boundary.
    half_pixels = np.flatnonzero(np.any((sums % divisor) * 2 == divisor, axis=1))
    source = np.asarray(image, dtype=np.uint8).reshape(source_height, source_width, -1)
    for pixel in half_pixels:
        out_y, out_x = divmod(int(pixel), width)
        y_start = float(out_y) * float(source_height) / float(height)
        y_end = float(out_y + 1) * float(source_height) / float(height)
        x_start = float(out_x) * float(source_width) / float(width)
        x_end = float(out_x + 1) * float(source_width) / float(width)
        floating_sums = [0.0] * source.shape[2]
        weight_sum = 0.0
        for source_y in range(int(y_start // 1), min(source_height, int(-(-y_end // 1)))):
            y_weight = max(0.0, min(y_end, float(source_y + 1)) - max(y_start, float(source_y)))
            for source_x in range(int(x_start // 1), min(source_width, int(-(-x_end // 1)))):
                x_weight = max(
                    0.0,
                    min(x_end, float(source_x + 1)) - max(x_start, float(source_x)),
                )
                weight = y_weight * x_weight
                for channel in range(source.shape[2]):
                    floating_sums[channel] += float(source[source_y, source_x, channel]) * weight
                weight_sum += weight
        resized[pixel] = np.asarray(
            [int(value / weight_sum + 0.5) for value in floating_sums], dtype=np.uint8
        )
    shaped = resized.reshape(height, width, -1)
    return shaped[..., 0] if image.ndim == 2 else shaped


def preprocess_frame(frame: np.ndarray, config: ScalarWorkerConfig) -> np.ndarray:
    value = normalize_rgb(frame)
    if config.crop_mode == "mask":
        value = value.copy()
        if config.crop_top:
            value[: config.crop_top] = 0
        if config.crop_bottom:
            value[-config.crop_bottom :] = 0
    else:
        end = value.shape[0] - config.crop_bottom if config.crop_bottom else value.shape[0]
        value = value[config.crop_top : end]
    if config.provider == "vizdoom":
        # ViZDoom Turbo area-resizes RGB channels with fractional box weights,
        # rounds each channel, and only then applies integer grayscale.
        value = fractional_area_resize(value, *config.resize)
        if config.grayscale:
            rgb = value.astype(np.uint32)
            value = ((77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2] + 128) >> 8).astype(
                np.uint8
            )
        return value[None, ...] if value.ndim == 2 else np.moveaxis(value, -1, 0)
    if config.grayscale:
        rgb = value.astype(np.uint32)
        value = ((77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2] + 128) >> 8).astype(
            np.uint8
        )
    resized = integer_area_resize(value, *config.resize)
    return resized[None, ...] if resized.ndim == 2 else np.moveaxis(resized, -1, 0)


def normalize_rgb(frame: Any) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    elif value.ndim == 3 and value.shape[0] in (1, 3, 4) and value.shape[-1] not in (3, 4):
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"cannot normalize raw render with shape {value.shape}")
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=2)
    if value.shape[-1] == 4:
        value = value[..., :3]
    return np.ascontiguousarray(value, dtype=np.uint8)


def _canonical_raw_rgb(frame: Any, profile: Profile) -> np.ndarray:
    value = normalize_rgb(frame)
    if profile.logical_environment not in {"supermario", "breakout"}:
        return value
    # Upstream NES video is RGB565. Some adapters expose its high bits directly
    # while others expand each channel across 0..255. Clear only the replicated
    # low bits so both losslessly represent the same native pixel value.
    return np.bitwise_and(value, np.asarray([0xF8, 0xFC, 0xF8], dtype=np.uint8))


def _comparison_raw_rgb(frame: Any, profile: Profile, provider: str) -> np.ndarray:
    value = frame
    if (
        profile.logical_environment == "breakout"
        and provider in {"stable-retro", "env-stableretro-turbo"}
    ):
        value = _canonical_stella_rgb(value)
    return _canonical_raw_rgb(value, profile)


def _semantic_raw_rgb(frame: Any, profile: Profile, provider: str) -> np.ndarray:
    """Decode public renders to the emulator's lossless native pixel code.

    The NES core is RGB565. Stable Retro expands those bits to RGB888 while
    native vector providers may expose the bits in their high positions. The
    discarded low bits are deterministic copies of native bits, not additional
    image information. Atari public frames are already canonical RGB888.
    """

    value = normalize_rgb(frame)
    if profile.logical_environment == "breakout" and provider in {
        "stable-retro",
        "env-stableretro-turbo",
    }:
        return _canonical_stella_rgb(value)
    if profile.logical_environment == "supermario":
        return np.bitwise_and(value, np.asarray([0xF8, 0xFC, 0xF8], dtype=np.uint8))
    return value


def _normalize_scalar_rgb(frame: Any, config: ScalarWorkerConfig) -> np.ndarray:
    value = normalize_rgb(frame)
    if config.provider == "stable-retro" and config.game.startswith("Breakout-Atari2600"):
        # Stable Retro derives policy observations from these raw bytes.
        # Human rendering normalizes the separate comparison boundary.
        return value
    return value


def _canonical_stella_rgb(frame: Any) -> np.ndarray:
    """Decode Stable Retro's BGR/RGB565 transport to canonical Stella RGB."""
    value = np.ascontiguousarray(normalize_rgb(frame)[..., ::-1])
    value = np.bitwise_and(value, np.asarray([0xF8, 0xFC, 0xF8], dtype=np.uint8))
    for legacy, canonical in (
        ((136, 140, 136), (136, 136, 136)),
        ((192, 108, 56), (192, 104, 56)),
        ((64, 156, 128), (64, 152, 128)),
    ):
        value[np.all(value == legacy, axis=-1)] = canonical
    return value


def _screen(observation: Any) -> np.ndarray:
    if isinstance(observation, Mapping):
        if "screen" not in observation:
            raise ValueError(f"scalar observation has no screen key: {sorted(observation)}")
        return np.asarray(observation["screen"])
    return np.asarray(observation)


def _labels_key(labels: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(str(label).upper() for label in labels))


def _button_masks(
    labels_per_lane: Sequence[Sequence[str]], buttons: Sequence[str | None]
) -> np.ndarray:
    index = {str(button).upper(): offset for offset, button in enumerate(buttons) if button}
    masks = np.zeros((len(labels_per_lane), len(buttons)), dtype=np.int8)
    for lane, labels in enumerate(labels_per_lane):
        for label in labels:
            try:
                masks[lane, index[str(label).upper()]] = 1
            except KeyError as exc:
                raise ValueError(f"provider does not advertise semantic button {label!r}") from exc
    return masks


def _turbo_construction(
    request: dict[str, Any], profile: Profile, frame_skip: int
) -> tuple[type[Any], str, dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
    provider = str(request["provider"])
    shape = int(request["shape"])
    assets = request.get("assets", {})
    noop_reset_max = int(request.get("noop_reset_max", 0))
    common = _turbo_v2_options(profile, shape, frame_skip, noop_reset_max=noop_reset_max)
    rom_path = assets.get("rom_path")
    module = importlib.import_module(str(request["import_name"]))
    overlay: tempfile.TemporaryDirectory[str] | None = None
    if provider in {"env-supermariobrosnes-turbo-emu", "env-stableretro-turbo"}:
        common["rom_path"] = rom_path
    if provider == "env-stableretro-turbo" and profile.native_transition_exact:
        common["state_catalog"] = [
            assets.get("state_paths", {}).get(state, state) for state in profile.states
        ]
    if provider == "env-stableretro-turbo":
        if profile.native_transition_exact and profile.logical_environment == "breakout":
            overlay, common["info"] = _augmented_breakout_info(module, profile)
        else:
            common["info"] = (
                None if profile.native_transition_exact else assets.get("info_schema_path")
            )
        common["scenario"] = (
            None if profile.native_transition_exact else assets.get("scenario_path")
        )
    if provider == "env-vizdoom-turbo":
        common["game_variables"] = [
            key.upper() for key in profile.info_integer if key.casefold() != "episode_time"
        ]
    environment_type = getattr(module, str(request["environment_class"]))
    return environment_type, profile.game, common, overlay


def _create_workload_adapter(request: dict[str, Any], profile: Profile) -> Adapter | FakeAdapter:
    """Construct a fresh workload environment after fail-closed attestation checks."""

    attestation = request.get("contract_attestation")
    require_request_matches_spec(request, request.get("execution_spec", {}))
    attestation_sha256 = require_attestation(request.get("execution_spec", {}), attestation)
    contract_report = dict(attestation["contract_report"])
    provider = str(request["provider"])
    shape = int(request["shape"])
    frame_skip = int(request.get("frame_skip", profile.frame_skip))
    if request.get("adapter") == "fake":
        return FakeAdapter(
            profile,
            provider,
            shape,
            float(request.get("fake_speed", 1.0)),
            contract_report=contract_report,
            attestation_sha256=attestation_sha256,
        )
    if request.get("adapter") == "turbo-vector-v2":
        environment_type, game, options, overlay = _turbo_construction(
            request, profile, frame_skip
        )
        try:
            env = _construct_turbo_workload_environment(environment_type, provider, game, options)
        except BaseException:
            if overlay is not None:
                overlay.cleanup()
            raise
        return Adapter(
            env,
            profile,
            provider,
            native_discrete=True,
            contract_report=contract_report,
            attestation_sha256=attestation_sha256,
            overlay=overlay,
        )
    if provider in {"stable-retro", "vizdoom"}:
        adapter = _create_scalar_adapter(request, profile, frame_skip)
        adapter.contract_report = contract_report
        adapter.attestation_sha256 = attestation_sha256
        return adapter
    raise ValueError(f"no built-in adapter for {provider!r}")


def _turbo_v2_options(
    profile: Profile, shape: int, frame_skip: int, *, noop_reset_max: int = 0
) -> dict[str, Any]:
    """Spell out every shared benchmark semantic independently of API defaults."""

    return {
        "state": None,
        "scenario": None,
        "info": None,
        "record": False,
        "players": 1,
        "inttype": "stable",
        "obs_type": "image",
        "num_envs": shape,
        "num_threads": shape,
        "rom_path": None,
        "transport": "numpy",
        "obs_copy": "copy",
        "obs_resize": profile.resize,
        "obs_crop": (profile.crop_top, profile.crop_bottom, 0, 0),
        "obs_crop_mode": profile.crop_mode,
        "obs_crop_fill": 0,
        "obs_grayscale": profile.grayscale,
        "obs_resize_algorithm": profile.resize_algorithm,
        "obs_layout": profile.layout,
        "frame_skip": frame_skip,
        "frame_stack": profile.frame_stack,
        "maxpool_last_two": profile.maxpool_last_two,
        "sticky_action_prob": 0.0,
        "noop_reset_max": noop_reset_max,
        "use_fire_reset": False,
        "reward_clip": False,
        "info_filter": {"mode": "all", "keys": list(profile.info_integer + profile.info_float)},
        "info_frame_stack_keys": None,
        "use_restricted_actions": list(profile.action_table.values()),
        "state_catalog": list(profile.states),
        "render_mode": "rgb_array",
    }


def _augmented_breakout_info(
    module: Any, profile: Profile
) -> tuple[tempfile.TemporaryDirectory[str], str]:
    source = (
        Path(module.__file__).resolve().parent
        / "data"
        / "stable"
        / profile.game
        / "data.json"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Stable Retro Turbo data schema is missing: {source.name}")
    temporary = tempfile.TemporaryDirectory(prefix="turbobench-breakout-info-")
    target = Path(temporary.name) / "data.json"
    payload = read_json(source)
    payload.setdefault("info", {})["ball_y"] = {"address": 229, "type": "|u1"}
    write_json(target, payload)
    return temporary, str(target)


def _construct_turbo_workload_environment(
    environment_type: type[Any],
    provider: str,
    game: str,
    options: Mapping[str, Any],
) -> Any:
    """Construct only; dynamic validation belongs exclusively to probe processes."""

    api_version = declared_api_version(environment_type)
    kwargs = dict(options)
    kwargs.pop("state", None)  # state_catalog is the sole benchmark start selector
    if api_version == 1:
        parameters = inspect.signature(environment_type).parameters
        kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    elif api_version != 2:
        raise TurboContractError(legacy_report(provider, api_version))
    return environment_type(game=game, **kwargs)


def _probe_contract(
    request: dict[str, Any], profile: Profile
) -> tuple[dict[str, Any], str, bool]:
    """Consume one environment while exercising its complete runtime contract."""

    provider = str(request["provider"])
    shape = int(request["shape"])
    frame_skip = int(request.get("frame_skip", profile.frame_skip))
    instance_id = uuid.uuid4().hex
    closed = False
    if request.get("adapter") == "fake":
        env = _InMemoryFakeV2Env(
            "Poison-v0" if "poison" in provider else "Fake-v0",
            num_envs=shape,
            obs_copy="copy",
            render_mode="rgb_array",
        )
        try:
            instance_id = uuid.uuid4().hex
            report = validate_environment(_InMemoryFakeV2Env, env, provider)
            if "contract-failure" in provider:
                report = {
                    **report,
                    "passed": False,
                    "promotable": False,
                    "errors": [*report.get("errors", []), "injected fake contract failure"],
                }
                report["report_sha256"] = canonical_json_hash(
                    {key: value for key, value in report.items() if key != "report_sha256"}
                )
        finally:
            env.close()
            closed = True
        return report, instance_id, closed
    if request.get("adapter") == "turbo-vector-v2":
        environment_type, game, options, overlay = _turbo_construction(
            request, profile, frame_skip
        )
        env: Any | None = None
        try:
            api_version = declared_api_version(environment_type)
            if api_version == 2:
                preflight = validate_constructor(environment_type, provider)
                if not preflight["passed"]:
                    return preflight, instance_id, True
            elif api_version != 1:
                return legacy_report(provider, api_version), instance_id, True
            env = _construct_turbo_workload_environment(
                environment_type, provider, game, options
            )
            instance_id = uuid.uuid4().hex
            report = (
                validate_environment(environment_type, env, provider)
                if api_version == 2
                else legacy_report(provider, 1)
            )
        finally:
            if env is not None:
                env.close()
            if overlay is not None:
                overlay.cleanup()
            closed = True
        return report, instance_id, closed
    if provider in {"stable-retro", "vizdoom"}:
        adapter = _create_scalar_adapter(request, profile, frame_skip)
        instance_id = adapter.instance_id
        try:
            adapter.initial_reset(int(request.get("seed", 123)))
            action = adapter.benchmark_action(np.zeros(shape, dtype=np.int64))
            _observations, _rewards, terminated, truncated, _infos = adapter.step(action)
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                adapter.selective_reset(done)
            adapter.render_frames()
            report = adapter.contract_report
        finally:
            adapter.close()
            closed = adapter.closed
        return report, instance_id, closed
    raise ValueError(f"no built-in adapter for {provider!r}")


def _create_scalar_adapter(request: dict[str, Any], profile: Profile, frame_skip: int) -> Adapter:
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode

    shape = int(request["shape"])
    provider = str(request["provider"])
    assets = request.get("assets", {})
    overlay: tempfile.TemporaryDirectory[str] | None = None
    integration_path: str | None = None
    if provider == "stable-retro" and not profile.native_transition_exact:
        overlay = _create_retro_overlay(profile, assets)
        integration_path = overlay.name
    configs = [
        ScalarWorkerConfig(
            provider=provider,
            profile_id=profile.id,
            game=profile.game,
            state=profile.states[lane % len(profile.states)],
            integration_path=integration_path,
            frame_skip=frame_skip,
            frame_stack=profile.frame_stack,
            crop_top=profile.crop_top,
            crop_bottom=profile.crop_bottom,
            crop_mode=profile.crop_mode,
            grayscale=profile.grayscale,
            resize=profile.resize,
            info_keys=profile.info_integer + profile.info_float,
            worker_index=lane,
            native_transition_exact=profile.native_transition_exact,
            rom_path=assets.get("rom_path"),
            state_paths=tuple(
                (str(state), str(path))
                for state, path in dict(assets.get("state_paths", {})).items()
            ),
            noop_reset_max=int(request.get("noop_reset_max", 0)),
        )
        for lane in range(shape)
    ]
    try:
        env = AsyncVectorEnv(
            [partial(_make_scalar_worker, config) for config in configs],
            shared_memory=True,
            copy=True,
            context="spawn",
            autoreset_mode=AutoresetMode.DISABLED,
        )
    except BaseException:
        if overlay:
            overlay.cleanup()
        raise
    return Adapter(env, profile, provider, native_discrete=False, overlay=overlay)


def _create_retro_overlay(
    profile: Profile, assets: Mapping[str, Any]
) -> tempfile.TemporaryDirectory[str]:
    overlay = tempfile.TemporaryDirectory(prefix="turbobench-retro-")
    game_dir = Path(overlay.name) / profile.game
    game_dir.mkdir(parents=True)
    rom_path = Path(str(assets["rom_path"])).resolve()
    (game_dir / f"rom{rom_path.suffix}").symlink_to(rom_path)
    if profile.logical_environment != "breakout":
        for state, raw_path in dict(assets.get("state_paths", {})).items():
            (game_dir / f"{state}.state").symlink_to(Path(str(raw_path)).resolve())
    for target, key in (("data.json", "info_schema_path"), ("scenario.json", "scenario_path")):
        if raw_path := assets.get(key):
            (game_dir / target).symlink_to(Path(str(raw_path)).resolve())
    return overlay


def _array_hashes(array: Any) -> list[str]:
    values = np.asarray(array)
    return [hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for value in values]


def _frame_hashes(frames: Sequence[np.ndarray]) -> list[str]:
    return [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]


def _ram_hashes(rams: Sequence[np.ndarray]) -> list[str]:
    return [hashlib.sha256(ram.tobytes()).hexdigest() for ram in rams]


def _selected_infos(infos: Any, profile: Profile, shape: int) -> list[dict[str, int | float]]:
    result: list[dict[str, int | float]] = [{} for _ in range(shape)]
    if not isinstance(infos, Mapping):
        return result
    lookup = {str(key).casefold(): key for key in infos if not str(key).startswith("_")}
    for name in profile.info_integer + profile.info_float:
        source_key = lookup.get(name.casefold())
        if source_key is None:
            continue
        values = np.asarray(infos[source_key])
        mask_key = f"_{source_key}"
        mask = np.asarray(infos.get(mask_key, np.ones(shape, dtype=np.bool_)), dtype=np.bool_)
        for lane in range(shape):
            if not mask[lane]:
                continue
            value = values[lane]
            if name in profile.info_integer:
                result[lane][name] = int(value)
            else:
                result[lane][name] = float(value)
    return result


def _trace_transition(
    adapter: Adapter | FakeAdapter,
    profile: Profile,
    action: np.ndarray,
    step: int,
    *,
    trace_ram: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    observations, rewards, terminated, truncated, infos = adapter.step(action)
    frames = adapter.render_frames()
    done = np.logical_or(terminated, truncated)
    record: dict[str, Any] = {
        "step": step,
        "observation_sha256": _array_hashes(observations),
        "raw_frame_sha256": _frame_hashes(frames),
        "rewards": [float(value) for value in np.asarray(rewards)],
        "terminations": [bool(value) for value in np.asarray(terminated)],
        "truncations": [bool(value) for value in np.asarray(truncated)],
        "infos": _selected_infos(infos, profile, adapter.num_envs),
        "reset_lanes": np.flatnonzero(done).astype(int).tolist(),
    }
    if trace_ram:
        rams = adapter.rams()
        record["ram_sha256"] = _ram_hashes(rams)
        record["ram_shapes"] = [list(ram.shape) for ram in rams]
    return record, done


def _snapshot_mismatches(
    expected: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(expected, replayed, strict=False), start=1):
        for field in sorted(set(left) | set(right)):
            if left.get(field) != right.get(field):
                mismatches.append(
                    {
                        "suffix_step": index,
                        "field": field,
                        "uninterrupted": left.get(field),
                        "replayed": right.get(field),
                    }
                )
                if len(mismatches) == 20:
                    return mismatches
    return mismatches


def _snapshot_episode_window(
    trace: Sequence[Mapping[str, Any]], prefix_steps: int, suffix_steps: int
) -> list[Mapping[str, Any]]:
    """Return the requested continuation through its first lifecycle boundary."""
    expected = list(trace[prefix_steps : prefix_steps + suffix_steps])
    for index, record in enumerate(expected):
        if record.get("reset_lanes"):
            return expected[: index + 1]
    return expected


def _workload_lifecycle(adapter: Adapter | FakeAdapter) -> dict[str, Any]:
    if adapter.attestation_sha256 is None:
        raise RuntimeError("workload adapter has no contract attestation binding")
    record = {
        **evidence_binding(adapter.attestation_sha256),
        "environment_instance_id": adapter.instance_id,
        "dynamic_contract_validation_calls": 0,
    }
    if isinstance(adapter, FakeAdapter):
        record.update(
            {
                "process_poisoned_at_construction": adapter.process_poisoned_at_construction,
                "instance_poisoned": adapter.instance_poisoned,
                "render_calls": adapter.render_calls,
            }
        )
    return record


def run_trace(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_workload_adapter(request, profile)
    result: dict[str, Any] | None = None
    try:
        observations, reset_infos = adapter.initial_reset(int(request.get("seed", 123)))
        frames = adapter.render_frames()
        actions = np.asarray(request["actions"], dtype=np.int64)
        prepared = [adapter.benchmark_action(row) for row in actions]
        trace: list[dict[str, Any]] = []
        reset_points: list[list[int]] = []
        trace_ram = bool(request.get("trace_ram"))
        snapshot_prefix = int(request.get("snapshot_prefix_steps", 0))
        snapshot_suffix = int(request.get("snapshot_suffix_steps", 0))
        if snapshot_prefix or snapshot_suffix:
            if snapshot_prefix <= 0 or snapshot_suffix <= 0:
                raise ValueError("snapshot prefix and suffix must both be positive")
            if snapshot_prefix + snapshot_suffix > len(prepared):
                raise ValueError("snapshot continuation exceeds the action trace")
        snapshots: Any = None
        initial = {
            "observation_sha256": _array_hashes(observations),
            "raw_frame_sha256": _frame_hashes(frames),
            "raw_frame_shapes": [list(frame.shape) for frame in frames],
            "infos": _selected_infos(reset_infos, profile, adapter.num_envs),
        }
        if trace_ram:
            rams = adapter.rams()
            initial["ram_sha256"] = _ram_hashes(rams)
            initial["ram_shapes"] = [list(ram.shape) for ram in rams]
        for step, action in enumerate(prepared, start=1):
            record, done = _trace_transition(
                adapter,
                profile,
                action,
                step,
                trace_ram=trace_ram,
            )
            trace.append(record)
            reset_lanes = record["reset_lanes"]
            if reset_lanes:
                adapter.selective_reset(done)
                # Refresh the correctness-only raw-frame cache after a lane
                # reset. Benchmark rollouts never render or encode frames.
                adapter.render_frames()
                reset_points.append([step, *reset_lanes])
            if step == snapshot_prefix:
                snapshots = adapter.capture_snapshots()

        snapshot_continuation = None
        if snapshots is not None:
            expected = _snapshot_episode_window(trace, snapshot_prefix, snapshot_suffix)
            adapter.restore_snapshots(snapshots)
            replayed: list[dict[str, Any]] = []
            for offset, action in enumerate(
                prepared[snapshot_prefix : snapshot_prefix + len(expected)],
                start=snapshot_prefix + 1,
            ):
                record, done = _trace_transition(
                    adapter,
                    profile,
                    action,
                    offset,
                    trace_ram=trace_ram,
                )
                replayed.append(record)
                if record["reset_lanes"]:
                    adapter.selective_reset(done)
                    adapter.render_frames()
            snapshot_continuation = {
                "prefix_steps": snapshot_prefix,
                "requested_suffix_steps": snapshot_suffix,
                "suffix_steps": len(expected),
                "uninterrupted_sha256": canonical_json_hash(expected),
                "replayed_sha256": canonical_json_hash(replayed),
                "replay_exact": expected == replayed,
                "first_mismatches": _snapshot_mismatches(expected, replayed),
            }

        result = {
            "schema": (
                "turbobench.semantic-trace/v3"
                if profile.native_transition_exact
                else "turbobench.trace/v2"
            ),
            "provider": request["provider"],
            "profile": profile.id,
            "shape": adapter.num_envs,
            "action_stream_sha256": request["action_stream_sha256"],
            "initial": initial,
            "steps": trace,
            "reset_points": reset_points,
            "completion_step": _completion_step(trace, profile.completion),
            "environment": adapter.metadata(),
            "lifecycle": _workload_lifecycle(adapter),
        }
        if snapshot_continuation is not None:
            result["snapshot_continuation"] = snapshot_continuation
        return result
    finally:
        adapter.close()
        if result is not None:
            result["lifecycle"]["environment_closed"] = adapter.closed


def run_benchmark(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_workload_adapter(request, profile)
    result: dict[str, Any] | None = None
    try:
        actions = np.asarray(request["actions"], dtype=np.int64)
        prepared = [adapter.benchmark_action(row) for row in actions]
        warmup_count = int(request.get("warmup_steps", min(500, len(prepared))))
        adapter.initial_reset(int(request.get("seed", 123)))
        _rollout(adapter, prepared[:warmup_count])
        if isinstance(adapter, FakeAdapter):
            base = 10_000.0 * adapter.speed * adapter.num_envs**0.2
            if adapter.process_poisoned_at_construction or adapter.instance_poisoned:
                base *= 0.1
            repetitions = [base * factor for factor in (0.999, 1.0, 1.001)]
        else:
            repetitions = []
            for repetition in range(3):
                adapter.initial_reset(int(request.get("seed", 123)) + repetition)
                started = time.perf_counter_ns()
                _rollout(adapter, prepared)
                elapsed_ns = time.perf_counter_ns() - started
                repetitions.append(len(prepared) * adapter.num_envs * 1e9 / elapsed_ns)
        result = {
            "schema": "turbobench.invocation/v2",
            "provider": request["provider"],
            "profile": profile.id,
            "shape": adapter.num_envs,
            "steps": len(prepared),
            "repetitions": 3,
            "sps": repetitions,
            "action_stream_sha256": request["action_stream_sha256"],
            "timed_includes": [
                "step",
                "preprocessing",
                "ipc",
                "infos",
                "terminal_detection",
                "selective_reset",
            ],
            "timed_excludes": [
                "construction",
                "initial_reset",
                "action_generation",
                "warmup",
                "correctness",
                "rendering",
                "encoding",
            ],
            "turbo_contract_report": adapter.contract_report,
            "lifecycle": _workload_lifecycle(adapter),
        }
        return result
    finally:
        adapter.close()
        if result is not None:
            result["lifecycle"]["environment_closed"] = adapter.closed


def run_reset_distribution(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    maximum = int(request.get("noop_reset_max", 30))
    seeds = tuple(map(int, request.get("seeds", range(256))))
    if profile.logical_environment != "breakout" or maximum <= 0 or len(seeds) < 32:
        raise ValueError("reset distribution requires Breakout, a positive maximum, and 32 seeds")
    adapter = _create_workload_adapter({**request, "noop_reset_max": maximum}, profile)
    result: dict[str, Any] | None = None
    try:
        if adapter.num_envs != 1:
            raise ValueError("reset distribution uses one lane")
        samples: list[dict[str, Any]] = []
        for seed in seeds:
            observation, infos = adapter.initial_reset(seed)
            selected = _selected_infos(infos, profile, 1)[0]
            raw_count = infos.get("noop_reset_count")
            count = int(np.asarray(raw_count).reshape(-1)[0])
            frame = adapter.render_frames()[0]
            samples.append(
                {
                    "seed": seed,
                    "count": count,
                    "observation_sha256": _array_hashes(observation)[0],
                    "raw_frame_sha256": _frame_hashes([frame])[0],
                    "infos": selected,
                }
            )
        result = {
            "schema": "turbobench.reset-distribution/v2",
            "provider": request["provider"],
            "profile": profile.id,
            "maximum": maximum,
            "samples": samples,
            "lifecycle": _workload_lifecycle(adapter),
        }
        return result
    finally:
        adapter.close()
        if result is not None:
            result["lifecycle"]["environment_closed"] = adapter.closed


def run_contract(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    require_request_matches_spec(request, request["execution_spec"])
    report, instance_id, closed = _probe_contract(request, profile)
    contract_attestation = attest(request["execution_spec"], report)
    return {
        "schema": "turbobench.contract-preflight/v2",
        "provider": request["provider"],
        "profile": profile.id,
        "shape": int(request["shape"]),
        "workload_executed": False,
        "turbo_contract_report": report,
        "execution_spec": request["execution_spec"],
        "contract_attestation": contract_attestation,
        "lifecycle": {
            "execution_protocol": contract_attestation["protocol"],
            "environment_instance_id": instance_id,
            "environment_closed": closed,
            "process_global_poisoned": (
                _FAKE_PROCESS_POISONED if request.get("adapter") == "fake" else None
            ),
            "instance_poisoned": (
                "poison" in str(request.get("provider"))
                if request.get("adapter") == "fake"
                else None
            ),
        },
    }


def _rollout(adapter: Adapter | FakeAdapter, prepared: Sequence[np.ndarray]) -> None:
    for action in prepared:
        _observations, _rewards, terminated, truncated, _infos = adapter.step(action)
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            adapter.selective_reset(done)


def run_promo_replay(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_workload_adapter(request, profile)
    result: dict[str, Any] | None = None
    output = Path(request["output_frames"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_hashes: list[str] = []
    transitions: list[dict[str, Any]] = []
    try:
        _observation, reset_infos = adapter.initial_reset(int(request.get("seed", 123)))
        initial_infos = _selected_infos(reset_infos, profile, adapter.num_envs)[0]
        initial_frame = adapter.render_frames()[0]
        height, width = initial_frame.shape[:2]
        completion_step: int | None = None
        with output.open("wb") as handle:
            handle.write(initial_frame.tobytes())
            frame_hashes.append(hashlib.sha256(initial_frame.tobytes()).hexdigest())
            for step, labels in enumerate(request["promo_actions"], start=1):
                action = adapter.promo_action(labels)
                observations, rewards, terminated, truncated, infos = adapter.step(action)
                frame = adapter.render_frames()[0]
                if frame.shape[:2] != (height, width):
                    raise RuntimeError("raw render dimensions changed during promo replay")
                handle.write(frame.tobytes())
                frame_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                selected = _selected_infos(infos, profile, adapter.num_envs)[0]
                transition = {
                    "step": step,
                    "observation_sha256": _array_hashes(observations)[0],
                    "raw_frame_sha256": frame_hashes[-1],
                    "reward": float(np.asarray(rewards)[0]),
                    "terminated": bool(np.asarray(terminated)[0]),
                    "truncated": bool(np.asarray(truncated)[0]),
                    "infos": selected,
                }
                transitions.append(transition)
                if completion_step is None and _is_complete(
                    profile.completion, initial_infos, transition
                ):
                    completion_step = step
                    if profile.completion.get("kind") != "trajectory-end":
                        break
                if transition["terminated"] or transition["truncated"]:
                    if completion_step is None:
                        completion_step = step
                    break
        result = {
            "schema": "turbobench.replay/v2",
            "provider": request["provider"],
            "profile": profile.id,
            "action_stream_sha256": request["promo_action_sha256"],
            "frame_count": len(frame_hashes),
            "frame_width": width,
            "frame_height": height,
            "frame_sha256": frame_hashes,
            "transitions": transitions,
            "completion_step": completion_step,
            "raw_file_sha256": sha256_file(output),
            "turbo_contract_report": adapter.contract_report,
            "lifecycle": _workload_lifecycle(adapter),
        }
        return result
    finally:
        adapter.close()
        if result is not None:
            result["lifecycle"]["environment_closed"] = adapter.closed


def _completion_step(trace: Sequence[dict[str, Any]], completion: dict[str, Any]) -> int | None:
    if not trace:
        return None
    initial_infos = trace[0].get("infos", [{}])[0]
    for transition in trace:
        lane = {
            "step": transition["step"],
            "infos": transition.get("infos", [{}])[0],
            "terminated": transition.get("terminations", [False])[0],
            "truncated": transition.get("truncations", [False])[0],
        }
        if _is_complete(completion, initial_infos, lane):
            return int(transition["step"])
    return None


def _is_complete(
    completion: dict[str, Any], initial: dict[str, Any], transition: dict[str, Any]
) -> bool:
    kind = completion.get("kind")
    if kind == "trajectory-end":
        return int(transition["step"]) >= int(completion["step"])
    if kind == "info-change":
        keys = completion.get("keys", [])
        return (
            bool(keys)
            and all(key in transition["infos"] for key in keys)
            and any(transition["infos"].get(key) != initial.get(key) for key in keys)
        )
    if kind == "terminal-or-info-at-least":
        return bool(transition.get("terminated") or transition.get("truncated")) or float(
            transition["infos"].get(completion.get("key"), float("-inf"))
        ) >= float(completion.get("value", 0))
    return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def execute(request: dict[str, Any]) -> dict[str, Any]:
    profile = get_profile(str(request["profile"]))
    operation = request["operation"]
    if operation in _WORKLOAD_OPERATIONS:
        require_request_matches_spec(request, request.get("execution_spec", {}))
        require_attestation(
            request.get("execution_spec", {}), request.get("contract_attestation")
        )
    started = time.time_ns()
    try:
        if operation == "contract":
            payload = run_contract(request, profile)
        elif operation == "trace":
            payload = run_trace(request, profile)
        elif operation == "benchmark":
            payload = run_benchmark(request, profile)
        elif operation == "reset-distribution":
            payload = run_reset_distribution(request, profile)
        elif operation == "promo":
            payload = run_promo_replay(request, profile)
        elif operation == "probe":
            payload = {"schema": "turbobench.runner-probe/v1"}
        else:
            raise ValueError(f"unknown runner operation {operation!r}")
    except TurboContractError as exc:
        payload = {
            "schema": "turbobench.contract-failure/v1",
            "provider": request.get("provider"),
            "profile": profile.id,
            "workload_executed": False,
            "turbo_contract_report": exc.report,
        }
    payload["runner"] = {
        "pid": os.getpid(),
        "operation": operation,
        "python": os.sys.version.split()[0],
        "provider_distribution": request.get("distribution"),
        "provider_version": _distribution_version(request.get("distribution")),
        "elapsed_ns_including_untimed_work": time.time_ns() - started,
    }
    return payload


def _distribution_version(distribution: Any) -> str | None:
    if not distribution:
        return None
    try:
        return importlib.metadata.version(str(distribution))
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv)
    request = read_json(args.request)
    response = execute(request)
    write_json(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
