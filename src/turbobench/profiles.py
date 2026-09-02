"""Immutable v1 workload profiles and deterministic semantic action streams."""

from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
from typing import Any

import numpy as np

from turbobench.model import Profile
from turbobench.profile_config import get_profile_document, load_profile_documents
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


_PROFILE_DOCUMENTS = load_profile_documents()
PROFILES: dict[str, Profile] = {
    profile_id: document.profile for profile_id, document in _PROFILE_DOCUMENTS.items()
}


def allowed_representation_conversion(profile: Profile) -> str:
    return profile.allowed_representation_conversion


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
    return deepcopy(get_profile_document(profile.id).payload)


def profile_hash(profile: Profile) -> str:
    return canonical_json_hash(profile_payload(profile))


def profile_toml(profile: Profile) -> str:
    return get_profile_document(profile.id).toml
