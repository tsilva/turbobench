"""Provider-runtime subprocess runner.

This module is executed by each content-addressed provider Python. Construction,
action generation, correctness, rendering, and encoding remain outside timed
regions; timed rollouts include preprocessing, IPC, infos, terminal detection,
and selective resets.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from turbobench.model import Profile
from turbobench.profiles import (
    BREAKOUT_RGB_TRANSPORT_CONVERSION,
    allowed_representation_conversion,
    get_profile,
)
from turbobench.provider_imports import PROVIDER_IMPORT_NAMES
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


_PADDLE_MEASURE_LOWER_BOUNDS = (
    1,
    272,
    295,
    318,
    341,
    366,
    388,
    411,
    447,
    470,
    493,
    516,
    539,
    563,
    586,
    609,
    633,
    657,
    680,
    703,
    733,
    757,
    780,
    803,
    828,
    851,
    874,
    897,
    920,
    943,
    966,
    991,
    1013,
    1036,
    1060,
    1083,
    1107,
    1130,
    1157,
    1180,
    1203,
    1228,
    1251,
    1274,
    1297,
    1320,
    1343,
    1366,
    1391,
    1413,
    1436,
    1460,
    1483,
    1507,
    1530,
    1553,
    1576,
    1599,
    1624,
    1647,
    1670,
    1693,
    1716,
    1739,
    1762,
    1786,
    1809,
    1832,
    1857,
    1880,
    1903,
    1926,
    1949,
    1972,
    1995,
    2020,
    2039,
    2066,
    2088,
    2112,
    2135,
    2158,
    2182,
    2205,
    2228,
    2253,
    2276,
    2299,
    2322,
)


def _needs_legacy_breakout_paddle_normalization(config: ScalarWorkerConfig) -> bool:
    """Retain the historical v1 workload without changing current comparisons."""

    return (
        config.provider == "stable-retro"
        and config.game.startswith("Breakout-Atari2600")
        and config.profile_id == "breakout/start-v1"
        and not config.native_transition_exact
    )


class BreakoutPaddleNormalizer:
    """Canonical corrected-Stella paddle state for upstream 1.0.1 frames."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.x = 115
        self.charge = 2048
        self.repeat = 0
        self.held = False
        self.measurement = 162

    def step(self, action: Any) -> None:
        buttons = np.asarray(action)
        direction = 2 if bool(buttons[7]) else 3 if bool(buttons[6]) else 0
        raw_x = self.x + 47
        target = 235 - self.measurement
        self.x = min(191, max(55, (raw_x + target) // 2)) - 47
        if self.held:
            self.repeat += 1
            if self.repeat > 5:
                self.repeat = 25
        if direction == 2 and self.charge > self.repeat:
            self.charge -= self.repeat
        elif direction == 3 and self.charge + self.repeat < 3856:
            self.charge += self.repeat
        self.held = direction in {2, 3}
        index = max(0, bisect.bisect_right(_PADDLE_MEASURE_LOWER_BOUNDS, self.charge) - 1)
        self.measurement = 0 if index == 0 else 12 + 2 * (index - 1)

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (210, 160):
            raise ValueError(f"canonical Breakout frame must be 210x160, got {frame.shape}")
        candidates: list[tuple[int, int]] = []
        red = np.asarray([200, 72, 72], dtype=np.uint8)
        runs: list[tuple[int, int]] = []
        for candidate_red in (
            red,
            np.asarray([72, 72, 200], dtype=np.uint8),
            np.asarray([72, 72, 205], dtype=np.uint8),
        ):
            mask = np.all(frame[190] == candidate_red, axis=1)
            runs = []
            start: int | None = None
            for offset, enabled in enumerate((*mask.tolist(), False)):
                if enabled and start is None:
                    start = offset
                elif not enabled and start is not None:
                    runs.append((start, offset - start))
                    start = None
            candidates = [run for run in runs if run[1] in {12, 16}]
            if candidates:
                red = candidate_red
                break
        if not candidates:
            colors, counts = np.unique(frame[190], axis=0, return_counts=True)
            dominant = sorted(
                (
                    (int(count), tuple(int(channel) for channel in color))
                    for color, count in zip(colors, counts, strict=True)
                ),
                reverse=True,
            )[:8]
            red_rows = [
                (row, int(count))
                for row, count in enumerate(np.all(frame == red, axis=2).sum(axis=1))
                if count
            ]
            raise RuntimeError(
                "could not locate canonical Breakout paddle raster; "
                f"red_runs={runs!r}; dominant_row_colors={dominant!r}; "
                f"red_rows={red_rows!r}"
            )
        old_x, width = max(candidates, key=lambda run: run[1])
        new_x = min(144, max(8, self.x))
        if old_x == new_x:
            return frame
        value = frame.copy()
        value[189:193, old_x : old_x + width] = 0
        value[189:193, new_x : new_x + width] = red
        return value


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
        self._oracle_action_history: list[np.ndarray] = []
        self._restoring_snapshot = False
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self._stack.shape, dtype=np.uint8
        )
        self.action_space = env.action_space
        self.metadata = dict(getattr(env, "metadata", {}))
        self.render_mode = "rgb_array"
        self._paddle_normalizer = (
            BreakoutPaddleNormalizer()
            if _needs_legacy_breakout_paddle_normalization(config)
            else None
        )

    @property
    def unwrapped(self) -> Any:
        return self

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        return self.env.get_wrapper_attr(name)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if not self._restoring_snapshot:
            self._oracle_action_history.clear()
        if self._paddle_normalizer is not None:
            self._paddle_normalizer.reset()
        observation, info = self.env.reset(seed=seed, options=options)
        raw = _screen(observation)
        self._raw_frame = _normalize_scalar_rgb(raw, self.config)
        if self._paddle_normalizer is not None:
            try:
                self._raw_frame = self._paddle_normalizer.normalize_frame(self._raw_frame)
            except RuntimeError:
                if np.any(self._raw_frame):
                    raise
                # Upstream Stable Retro's Atari reset returns the blank TIA
                # frame immediately before the canonical post-restore frame.
                # Advance that neutral frame inside reset; selective resets
                # remain timed by the benchmark contract.
                neutral = np.zeros(self.action_space.shape, dtype=np.int8)
                observation, _reward, terminated, truncated, step_info = self.env.step(neutral)
                if terminated or truncated:
                    raise RuntimeError(
                        "upstream Breakout terminated during reset bootstrap"
                    ) from None
                if step_info:
                    info = step_info
                self._raw_frame = _normalize_scalar_rgb(_screen(observation), self.config)
                self._raw_frame = self._paddle_normalizer.normalize_frame(self._raw_frame)
        if not info:
            data = getattr(self.env.unwrapped, "data", None)
            if data is not None and hasattr(data, "lookup_all"):
                data.update_ram()
                info = dict(data.lookup_all())
        info = self._semantic_info(info)
        assert self._raw_frame is not None
        frame = preprocess_frame(self._raw_frame, self.config)
        for offset in range(0, self._stack.shape[0], self._channels):
            self._stack[offset : offset + self._channels] = frame
        return self._stack.copy(), info

    def step(self, action: Any):
        if not self._restoring_snapshot:
            self._oracle_action_history.append(np.asarray(action).copy())
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        observation: Any = None
        for _ in range(self.config.frame_skip):
            if self._paddle_normalizer is not None:
                self._paddle_normalizer.step(action)
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
        if self._paddle_normalizer is not None:
            self._raw_frame = self._paddle_normalizer.normalize_frame(self._raw_frame)
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
        if (
            self.config.native_transition_exact
            and self.config.provider == "stable-retro"
            and self.config.game.startswith("Breakout-Atari2600")
        ):
            return _canonical_stella_rgb(self._raw_frame)
        return self._raw_frame.copy()

    def ram(self) -> np.ndarray:
        getter = getattr(self.env.unwrapped, "get_ram", None)
        if getter is None:
            raise NotImplementedError("scalar provider does not expose emulator RAM")
        return np.asarray(getter(), dtype=np.uint8).copy()

    def capture_oracle_snapshot(
        self,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        if self._raw_frame is None:
            raise RuntimeError("cannot capture a snapshot before reset")
        return (
            tuple(action.copy() for action in self._oracle_action_history),
            self._stack.copy(),
            self._raw_frame.copy(),
        )

    def restore_oracle_snapshots(
        self,
        snapshots: Sequence[tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]],
    ) -> bool:
        history, expected_stack, expected_raw = snapshots[self.config.worker_index]
        self._restoring_snapshot = True
        try:
            self.reset()
            for action in history:
                self.step(action)
        finally:
            self._restoring_snapshot = False
        self._oracle_action_history = [action.copy() for action in history]
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
    RETRO_DATA_PATH content cannot alter the semantic oracle.
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
        overlay: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.env = env
        self.profile = profile
        self.provider = provider
        self.native_discrete = native_discrete
        self.contract_report = contract_report or legacy_report(provider, None)
        self.overlay = overlay
        self.num_envs = int(env.num_envs)
        self._terminal_mask = np.zeros(self.num_envs, dtype=np.bool_)
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
        self._terminal_mask.fill(False)
        self._render_cache = None
        return self.env.reset(seed=seed, options=options)

    def selective_reset(self, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        result = self.env.reset(options=self._reset_options(mask))
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
            _semantic_raw_rgb(frame, self.profile)
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
        return tuple(self.env.call("capture_oracle_snapshot"))

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
            restored = self.env.call("restore_oracle_snapshots", snapshots)
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
                "upstream-stella-reset-and-paddle-v1"
                if self.provider == "stable-retro"
                and self.profile.id == "breakout/start-v1"
                else "vizdoom-last-valid-terminal-frame-v1"
                if self.profile.logical_environment == "vizdoom-basic"
                else "identity"
            ),
            "semantic_authority": self.profile.semantic_authority,
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
            if self.overlay is not None:
                self.overlay.cleanup()


class _InMemorySpace:
    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)


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
    def __init__(self, profile: Profile, provider: str, shape: int, speed: float) -> None:
        self.profile = profile
        self.provider = provider
        self.num_envs = shape
        self.speed = speed
        self.step_index = 0
        self._state = np.arange(shape, dtype=np.int64)
        self.in_promo = False
        contract_env = _InMemoryFakeV2Env(
            "Fake-v0",
            num_envs=shape,
            obs_copy="copy",
            render_mode="rgb_array",
        )
        try:
            self.contract_report = validate_environment(_InMemoryFakeV2Env, contract_env, provider)
        finally:
            contract_env.close()
        if not self.contract_report["passed"]:
            raise TurboContractError(self.contract_report)

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
        pass


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


def _semantic_raw_rgb(frame: Any, profile: Profile) -> np.ndarray:
    """Decode public renders to the emulator's lossless native pixel code.

    The NES core is RGB565. Stable Retro expands those bits to RGB888 while
    native vector providers may expose the bits in their high positions. The
    discarded low bits are deterministic copies of native bits, not additional
    image information. Atari public frames are already canonical RGB888.
    """

    value = normalize_rgb(frame)
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


def _create_adapter(request: dict[str, Any], profile: Profile) -> Adapter | FakeAdapter:
    provider = str(request["provider"])
    shape = int(request["shape"])
    frame_skip = int(request.get("frame_skip", profile.frame_skip))
    assets = request.get("assets", {})
    if request.get("adapter") == "fake":
        return FakeAdapter(profile, provider, shape, float(request.get("fake_speed", 1.0)))
    common = _turbo_v2_options(profile, shape, frame_skip)
    rom_path = assets.get("rom_path")
    if provider == "env-supermariobrosnes-turbo-emu":
        module = importlib.import_module(PROVIDER_IMPORT_NAMES[provider])
        common["rom_path"] = rom_path
        env, report = _construct_turbo_environment(
            module.SuperMarioBrosNesTurboVecEnv, provider, profile.game, common
        )
        return Adapter(env, profile, provider, native_discrete=True, contract_report=report)
    if provider == "env-breakoutatari2600-turbo-native":
        module = importlib.import_module(PROVIDER_IMPORT_NAMES[provider])
        env, report = _construct_turbo_environment(
            module.BreakoutVecEnv, provider, profile.game, common
        )
        return Adapter(env, profile, provider, native_discrete=True, contract_report=report)
    if provider == "env-stableretro-turbo":
        module = importlib.import_module(PROVIDER_IMPORT_NAMES[provider])
        if profile.native_transition_exact:
            common["state_catalog"] = [
                assets.get("state_paths", {}).get(state, state) for state in profile.states
            ]
        common["rom_path"] = rom_path
        common["info"] = None if profile.native_transition_exact else assets.get("info_schema_path")
        common["scenario"] = (
            None if profile.native_transition_exact else assets.get("scenario_path")
        )
        env, report = _construct_turbo_environment(
            module.RetroVecEnv, provider, profile.game, common
        )
        return Adapter(env, profile, provider, native_discrete=True, contract_report=report)
    if provider == "env-vizdoom-turbo":
        module = importlib.import_module(PROVIDER_IMPORT_NAMES[provider])
        common["game_variables"] = [
            key.upper() for key in profile.info_integer if key.casefold() != "episode_time"
        ]
        env, report = _construct_turbo_environment(
            module.VizdoomTurboVecEnv, provider, profile.game, common
        )
        return Adapter(env, profile, provider, native_discrete=True, contract_report=report)
    if provider in {"stable-retro", "vizdoom"}:
        return _create_scalar_adapter(request, profile, frame_skip)
    raise ValueError(f"no built-in adapter for {provider!r}")


def _turbo_v2_options(profile: Profile, shape: int, frame_skip: int) -> dict[str, Any]:
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
        "noop_reset_max": 0,
        "use_fire_reset": False,
        "reward_clip": False,
        "info_filter": {"mode": "all", "keys": list(profile.info_integer + profile.info_float)},
        "info_frame_stack_keys": None,
        "use_restricted_actions": list(profile.action_table.values()),
        "state_catalog": list(profile.states),
        "render_mode": "rgb_array",
    }


def _construct_turbo_environment(
    environment_type: type[Any],
    provider: str,
    game: str,
    options: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Detect the declaration before construction and enforce v2 before use."""

    api_version = declared_api_version(environment_type)
    kwargs = dict(options)
    kwargs.pop("state", None)  # state_catalog is the sole benchmark start selector
    if api_version == 2:
        preflight = validate_constructor(environment_type, provider)
        if not preflight["passed"]:
            raise TurboContractError(preflight)
    elif api_version == 1:
        parameters = inspect.signature(environment_type).parameters
        kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    else:
        raise TurboContractError(legacy_report(provider, api_version))
    env = environment_type(game=game, **kwargs)
    if api_version == 1:
        return env, legacy_report(provider, 1)
    report = validate_environment(environment_type, env, provider)
    if not report["passed"]:
        env.close()
        raise TurboContractError(report)
    return env, report


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


def run_trace(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_adapter(request, profile)
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
            expected = trace[snapshot_prefix : snapshot_prefix + snapshot_suffix]
            adapter.restore_snapshots(snapshots)
            replayed: list[dict[str, Any]] = []
            for offset, action in enumerate(
                prepared[snapshot_prefix : snapshot_prefix + snapshot_suffix],
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
                "suffix_steps": snapshot_suffix,
                "uninterrupted_sha256": canonical_json_hash(expected),
                "replayed_sha256": canonical_json_hash(replayed),
                "replay_exact": expected == replayed,
                "first_mismatches": _snapshot_mismatches(expected, replayed),
            }

        result = {
            "schema": (
                "turbobench.semantic-trace/v2"
                if profile.native_transition_exact
                else "turbobench.trace/v1"
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
        }
        if snapshot_continuation is not None:
            result["snapshot_continuation"] = snapshot_continuation
        return result
    finally:
        adapter.close()


def run_benchmark(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_adapter(request, profile)
    try:
        actions = np.asarray(request["actions"], dtype=np.int64)
        prepared = [adapter.benchmark_action(row) for row in actions]
        warmup_count = int(request.get("warmup_steps", min(500, len(prepared))))
        adapter.initial_reset(int(request.get("seed", 123)))
        _rollout(adapter, prepared[:warmup_count])
        if isinstance(adapter, FakeAdapter):
            base = 10_000.0 * adapter.speed * adapter.num_envs**0.2
            repetitions = [base * factor for factor in (0.999, 1.0, 1.001)]
        else:
            repetitions = []
            for repetition in range(3):
                adapter.initial_reset(int(request.get("seed", 123)) + repetition)
                started = time.perf_counter_ns()
                _rollout(adapter, prepared)
                elapsed_ns = time.perf_counter_ns() - started
                repetitions.append(len(prepared) * adapter.num_envs * 1e9 / elapsed_ns)
        return {
            "schema": "turbobench.invocation/v1",
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
        }
    finally:
        adapter.close()


def run_contract(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_adapter(request, profile)
    try:
        return {
            "schema": "turbobench.contract-preflight/v1",
            "provider": request["provider"],
            "profile": profile.id,
            "workload_executed": False,
            "turbo_contract_report": adapter.contract_report,
        }
    finally:
        adapter.close()


def _rollout(adapter: Adapter | FakeAdapter, prepared: Sequence[np.ndarray]) -> None:
    for action in prepared:
        _observations, _rewards, terminated, truncated, _infos = adapter.step(action)
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            adapter.selective_reset(done)


def run_promo_replay(request: dict[str, Any], profile: Profile) -> dict[str, Any]:
    adapter = _create_adapter(request, profile)
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
        return {
            "schema": "turbobench.replay/v1",
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
        }
    finally:
        adapter.close()


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
    started = time.time_ns()
    try:
        if operation == "contract":
            payload = run_contract(request, profile)
        elif operation == "trace":
            payload = run_trace(request, profile)
        elif operation == "benchmark":
            payload = run_benchmark(request, profile)
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
