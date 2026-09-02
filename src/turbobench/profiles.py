"""Immutable v1 workload profiles and deterministic semantic action streams."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import numpy as np

from turbobench.model import Profile
from turbobench.util import canonical_json_hash

MARIO_ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"
MARIO_PROMO_RAW_ACTION_SHA256 = "544c08230ff5104f90d8cfca930bb0f4c961ac7ddfce02e34df93735b788f805"
MARIO_PROMO_PACKED_SHA256 = "5a3f60229d88d8d0cbd72c89c149a4304bea463688fe749dab6aa34a87a791e5"
MARIO_PROMO_ACTION_COUNT = 1_986
MARIO_BUTTON_ORDER = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BREAKOUT_RGB_TRANSPORT_CONVERSION = (
    "stable-retro-bgr-rgb565-to-canonical-stella-rgb"
)

# Imported once from the verified isolated GymRec replay. The unpacked 1,986x9
# bytes hash to MARIO_PROMO_RAW_ACTION_SHA256; no GymRec/Hugging Face access is
# required to replay it.
_MARIO_PROMO_PACKED = (
    'gcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAAAAAAgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCAYDAYDAYDAYDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCAAAAAAgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAAAAAAAQCAQCgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgcDgcDgUCgUCAgEAgEAgEAgEAgEAgEgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAAAAAAgUCgUCgUCgUCgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCAYDAYDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAQCAQCAQCAQCgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAYDAYDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCAIBAIBgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCAYDAYDAYDAYDAYDAYDAYDAYDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAQCAQCAgEAgEAgEAgEgcDgcDAQCAQCAQCAQCAYDAYDAYDAYDAYDAYDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAYDAYDAQCAQCAQCAQCAgEAgEgcDgcDAQCAQCAQCAQCAYDAYDAYDAYDAYDAYDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCAIBAIBAYDAYDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDAAAAAAAIBAIBAgEAgEAgEAgEgcDgcDgcDgcDAYDAYDgcDgcDAAAAAAAAAAAAAAAAAAAYDAYDgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgcDgUCgUCgUCgUCgUCgUCgcDgcDgcDgcDAgEAgEAgEAgEgUCgUCgUCgUCgUCgUCgUCgUCAgEAgEgUCgUCgUCgUCgUCgUCgUCgUCAgEAgEAgEAgEgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCgUCAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAQCAQCgcDgcDgcDgcDgcDgcDAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAQCAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEAgEA'
)


_MARIO_ACTION_TABLE = {
    "noop": (),
    "right": ("RIGHT",),
    "right_b": ("RIGHT", "B"),
    "right_a": ("RIGHT", "A"),
    "right_a_b": ("RIGHT", "A", "B"),
    "a": ("A",),
    "left": ("LEFT",),
    "start": ("START",),
}
_BREAKOUT_ACTION_TABLE = {
    "noop": (),
    "fire": ("BUTTON",),
    "right": ("RIGHT",),
    "left": ("LEFT",),
}
_VIZDOOM_ACTION_TABLE = {
    "noop": (),
    "left": ("MOVE_LEFT",),
    "right": ("MOVE_RIGHT",),
    "attack": ("ATTACK",),
    "left_attack": ("MOVE_LEFT", "ATTACK"),
    "right_attack": ("MOVE_RIGHT", "ATTACK"),
}


PROFILES: dict[str, Profile] = {
    "supermario/world1-v1": Profile(
        id="supermario/world1-v1",
        logical_environment="supermario",
        game="SuperMarioBros-Nes-v0",
        providers=("env-supermariobrosnes-turbo-emu", "env-stableretro-turbo", "stable-retro"),
        shapes=(1, 16, 32),
        states=("Level1-1", "Level1-2", "Level1-3", "Level1-4"),
        semantic_actions=("noop", "right", "right_b", "right_a"),
        action_table=_MARIO_ACTION_TABLE,
        info_integer=("levelHi", "levelLo", "lives", "score", "time", "scrolling", "xscrollHi", "xscrollLo"),
        crop_top=32,
        crop_mode="mask",
        promo_kind="mario-imported-v1",
        promo_steps=MARIO_PROMO_ACTION_COUNT,
        completion={"kind": "info-change", "keys": ["levelHi", "levelLo"], "step": 1986},
        asset_sha256=MARIO_ROM_SHA256,
    ),
    "breakout/start-v1": Profile(
        id="breakout/start-v1",
        logical_environment="breakout",
        game="Breakout-Atari2600-v0",
        providers=("env-breakoutatari2600-turbo-native", "env-stableretro-turbo", "stable-retro"),
        shapes=(1, 16, 32),
        states=("Start",),
        semantic_actions=("noop", "fire", "right", "left"),
        action_table=_BREAKOUT_ACTION_TABLE,
        info_integer=("score", "lives", "ball_y"),
        promo_kind="breakout-deterministic-v1",
        promo_steps=1_800,
        completion={"kind": "trajectory-end", "step": 1800},
    ),
    "supermario/world1-v2": Profile(
        id="supermario/world1-v2",
        logical_environment="supermario",
        game="SuperMarioBros-Nes-v0",
        providers=("env-supermariobrosnes-turbo-emu", "env-stableretro-turbo", "stable-retro"),
        shapes=(1, 4),
        states=("Level1-1", "Level1-2", "Level1-3", "Level1-4"),
        semantic_actions=("noop", "right", "right_b", "right_a"),
        action_table=_MARIO_ACTION_TABLE,
        info_integer=("levelHi", "levelLo", "lives", "score", "time", "scrolling", "xscrollHi", "xscrollLo"),
        crop_top=32,
        crop_mode="mask",
        correctness_steps=256,
        promo_kind="mario-imported-v1",
        promo_steps=MARIO_PROMO_ACTION_COUNT,
        completion={"kind": "info-change", "keys": ["levelHi", "levelLo"], "step": 1986},
        asset_sha256=MARIO_ROM_SHA256,
        native_transition_exact=True,
    ),
    "breakout/start-v2": Profile(
        id="breakout/start-v2",
        logical_environment="breakout",
        game="Breakout-Atari2600-v0",
        providers=("env-breakoutatari2600-turbo-native", "env-stableretro-turbo", "stable-retro"),
        shapes=(1, 4),
        states=("Start",),
        semantic_actions=("noop", "fire", "right", "left"),
        action_table=_BREAKOUT_ACTION_TABLE,
        info_integer=("score", "lives", "ball_y"),
        correctness_steps=256,
        promo_kind="breakout-deterministic-v1",
        promo_steps=1_800,
        completion={"kind": "trajectory-end", "step": 1800},
        native_transition_exact=True,
    ),
    "breakout/start-v3": Profile(
        id="breakout/start-v3",
        logical_environment="breakout",
        game="Breakout-Atari2600-v0",
        providers=("env-breakoutatari2600-turbo-native", "env-stableretro-turbo", "stable-retro"),
        shapes=(1, 16, 32),
        states=("Start",),
        semantic_actions=("noop", "fire", "right", "left"),
        action_table=_BREAKOUT_ACTION_TABLE,
        info_integer=("score", "lives"),
        promo_kind="breakout-deterministic-v1",
        promo_steps=1_800,
        completion={"kind": "trajectory-end", "step": 1800},
    ),
    "vizdoom/basic-v1": Profile(
        id="vizdoom/basic-v1",
        logical_environment="vizdoom-basic",
        game="VizdoomBasic-v1",
        providers=("env-vizdoom-turbo", "vizdoom"),
        shapes=(1, 16, 32),
        states=("default",),
        semantic_actions=("noop", "left", "right"),
        action_table=_VIZDOOM_ACTION_TABLE,
        info_integer=("killcount", "health", "ammo2", "episode_time"),
        promo_kind="vizdoom-basic-deterministic-v1",
        promo_steps=2_100,
        completion={"kind": "terminal-or-info-at-least", "key": "killcount", "value": 1},
    ),
    "vizdoom/basic-v2": Profile(
        id="vizdoom/basic-v2",
        logical_environment="vizdoom-basic",
        game="VizdoomBasic-v1",
        providers=("env-vizdoom-turbo", "vizdoom"),
        shapes=(1, 4),
        states=("default",),
        semantic_actions=("noop", "left", "right"),
        action_table=_VIZDOOM_ACTION_TABLE,
        info_integer=("killcount", "health", "ammo2", "episode_time"),
        correctness_steps=256,
        promo_kind="vizdoom-basic-deterministic-v1",
        promo_steps=2_100,
        completion={"kind": "terminal-or-info-at-least", "key": "killcount", "value": 1},
        native_transition_exact=True,
    ),
}


def allowed_representation_conversion(profile: Profile) -> str:
    if profile.logical_environment == "supermario":
        return "rgb888-expanded-to-rgb565-native-code"
    if profile.logical_environment == "breakout" and profile.native_transition_exact:
        return BREAKOUT_RGB_TRANSPORT_CONVERSION
    return "identity"


def get_profile(profile_id: str) -> Profile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile_id!r}") from exc


def benchmark_actions(profile: Profile, shape: int, steps: int | None = None) -> np.ndarray:
    if shape <= 0:
        raise ValueError("shape must be positive")
    count = profile.benchmark_steps if steps is None else steps
    if count <= 0:
        raise ValueError("steps must be positive")
    step_index = np.arange(count, dtype=np.int64)[:, None]
    lane_index = np.arange(shape, dtype=np.int64)[None, :]
    if profile.logical_environment == "vizdoom-basic":
        # Movement/noop is the common scalar/native contract. Weapon-sprite
        # presentation differs between the upstream RGB renderer and Turbo's
        # indexed renderer, so firing is retained in the advertised action
        # table but excluded from the canonical raw-frame workload.
        return ((step_index // 8 + lane_index) % len(profile.semantic_actions)).astype(np.int16)
    action_count = len(profile.semantic_actions)
    # Varied by both lane and step, deterministic, and never generated in a timed section.
    return ((step_index * 17 + lane_index * 7 + (step_index // 4) * 3) % action_count).astype(np.int16)


def parity_actions(profile: Profile, shape: int, steps: int, *, seed: int = 123) -> np.ndarray:
    """Return a reproducible random semantic-action trace for fidelity testing."""

    if shape <= 0:
        raise ValueError("shape must be positive")
    count = steps
    if count <= 0:
        raise ValueError("steps must be positive")
    generator = np.random.default_rng(seed)
    actions = generator.integers(
        0,
        len(profile.semantic_actions),
        size=(count, shape),
        dtype=np.int16,
    )
    # Exercise every shared action immediately in every lane before the random
    # suffix. This makes short diagnostic runs useful without changing the
    # seeded long-trace contract.
    directed = min(count, len(profile.semantic_actions))
    for step in range(directed):
        actions[step] = (step + np.arange(shape, dtype=np.int16)) % len(profile.semantic_actions)
    return actions


def action_stream_hash(profile: Profile, actions: np.ndarray) -> str:
    payload = {
        "profile": profile.id,
        "semantic_actions": list(profile.semantic_actions),
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "bytes_sha256": hashlib.sha256(actions.tobytes(order="C")).hexdigest(),
    }
    return canonical_json_hash(payload)


def mario_promo_actions() -> tuple[tuple[str, ...], ...]:
    packed = base64.b64decode(_MARIO_PROMO_PACKED)
    if hashlib.sha256(packed).hexdigest() != MARIO_PROMO_PACKED_SHA256:
        raise RuntimeError("embedded Mario promo action payload is corrupt")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[: MARIO_PROMO_ACTION_COUNT * 9]
    raw = bits.reshape(MARIO_PROMO_ACTION_COUNT, 9).astype(np.uint8)
    if hashlib.sha256(raw.tobytes()).hexdigest() != MARIO_PROMO_RAW_ACTION_SHA256:
        raise RuntimeError("embedded Mario promo action stream failed provenance hash")
    return tuple(
        tuple(button for button, pressed in zip(MARIO_BUTTON_ORDER, row, strict=True) if button and pressed)
        for row in raw
    )


def promo_actions(profile: Profile) -> tuple[tuple[str, ...], ...]:
    if profile.promo_kind == "mario-imported-v1":
        return mario_promo_actions()
    if profile.promo_kind == "breakout-deterministic-v1":
        actions: list[tuple[str, ...]] = []
        for step in range(profile.promo_steps):
            if step in {0, 1, 90, 91, 600, 601, 1200, 1201}:
                actions.append(("BUTTON",))
            elif (step // 75) % 2:
                actions.append(("LEFT",))
            else:
                actions.append(("RIGHT",))
        return tuple(actions)
    actions = []
    for step in range(profile.promo_steps):
        phase = (step // 35) % 3
        labels = ((), ("MOVE_LEFT",), ("MOVE_RIGHT",))[phase]
        actions.append(labels)
    return tuple(actions)


def promo_action_hash(profile: Profile, actions: tuple[tuple[str, ...], ...]) -> str:
    if profile.promo_kind == "mario-imported-v1":
        return MARIO_PROMO_RAW_ACTION_SHA256
    return canonical_json_hash({"profile": profile.id, "actions": actions})


def profile_payload(profile: Profile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "logical_environment": profile.logical_environment,
        "game": profile.game,
        "providers": list(profile.providers),
        "shapes": list(profile.shapes),
        "states": list(profile.states),
        "benchmark": {
            "frame_skip": profile.frame_skip,
            "frame_stack": profile.frame_stack,
            "grayscale": profile.grayscale,
            "resize": list(profile.resize),
            "layout": profile.layout,
            "crop_top": profile.crop_top,
            "crop_bottom": profile.crop_bottom,
            "crop_mode": profile.crop_mode,
            "resize_algorithm": profile.resize_algorithm,
            "maxpool_last_two": profile.maxpool_last_two,
            "steps": profile.benchmark_steps,
            "correctness_steps": profile.correctness_steps,
            "semantic_actions": list(profile.semantic_actions),
            "action_table": {
                name: list(labels) for name, labels in profile.action_table.items()
            },
        },
        "promo": {
            "kind": profile.promo_kind,
            "steps": profile.promo_steps,
            "completion": profile.completion,
        },
        "signals": {"integer": list(profile.info_integer), "float": list(profile.info_float)},
        "asset_sha256": profile.asset_sha256,
        "exact": {
            "native_transition_exact": profile.native_transition_exact,
            "allowed_representation_conversion": allowed_representation_conversion(profile),
        },
    }


def profile_hash(profile: Profile) -> str:
    return canonical_json_hash(profile_payload(profile))


def profile_toml(profile: Profile) -> str:
    def quoted(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def strings(values: tuple[str, ...]) -> str:
        return "[" + ", ".join(quoted(value) for value in values) + "]"

    lines = [
        f"schema = {quoted('turbobench.profile/v2' if profile.native_transition_exact else 'turbobench.profile/v1')}",
        f"id = {quoted(profile.id)}",
        f"logical_environment = {quoted(profile.logical_environment)}",
        f"game = {quoted(profile.game)}",
        f"providers = {strings(profile.providers)}",
        f"shapes = [{', '.join(map(str, profile.shapes))}]",
        f"states = {strings(profile.states)}",
        "",
        "[benchmark]",
        f"frame_skip = {profile.frame_skip}",
        f"frame_stack = {profile.frame_stack}",
        f"grayscale = {str(profile.grayscale).lower()}",
        f"resize = [{profile.resize[0]}, {profile.resize[1]}]",
        f"layout = {quoted(profile.layout)}",
        f"crop_top = {profile.crop_top}",
        f"crop_bottom = {profile.crop_bottom}",
        f"crop_mode = {quoted(profile.crop_mode)}",
        f"resize_algorithm = {quoted(profile.resize_algorithm)}",
        f"maxpool_last_two = {str(profile.maxpool_last_two).lower()}",
        f"steps = {profile.benchmark_steps}",
        f"correctness_steps = {profile.correctness_steps}",
        f"semantic_actions = {strings(profile.semantic_actions)}",
        "",
        "[action_table]",
        *(f"{name} = {strings(labels)}" for name, labels in profile.action_table.items()),
        "",
        "[promo]",
        f"kind = {quoted(profile.promo_kind)}",
        f"steps = {profile.promo_steps}",
        f"completion_json = {quoted(__import__('json').dumps(profile.completion, sort_keys=True, separators=(',', ':')))}",
        "",
        "[signals]",
        f"integer = {strings(profile.info_integer)}",
        f"float = {strings(profile.info_float)}",
    ]
    if profile.native_transition_exact:
        lines.extend(
            (
                "",
                "[exact]",
                "native_transition_exact = true",
                "allowed_representation_conversion = "
                + quoted(allowed_representation_conversion(profile)),
            )
        )
    if profile.asset_sha256:
        lines.extend(("", "[asset]", f"sha256 = {quoted(profile.asset_sha256)}"))
    return "\n".join(lines) + "\n"
