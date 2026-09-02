"""Load unified, declarative workload profiles."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from turbobench.model import ParityProfile, Profile

WORKLOAD_PROFILE_SCHEMA = "turbobench.workload-profile/v1"
PARITY_PROFILE_SCHEMA = "turbobench.parity-profile/v1"
STANDARD_CHECKS = frozenset(
    {
        "action-identity",
        "transition-exact",
        "reset-exact",
        "snapshot-continuation",
        "required-info",
        "reset-distribution",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "id",
        "logical_environment",
        "game",
        "providers",
        "states",
        "semantic_actions",
        "info_integer",
        "info_float",
        "action_table",
        "observation",
        "benchmark",
        "parity",
        "promo",
        "exact",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "frame_skip",
        "frame_stack",
        "crop_top",
        "crop_bottom",
        "crop_mode",
        "resize",
        "grayscale",
        "layout",
        "resize_algorithm",
        "maxpool_last_two",
    }
)
_BENCHMARK_KEYS = frozenset({"shapes", "steps", "correctness_steps"})
_PARITY_KEYS = frozenset(
    {
        "authority",
        "authority_version",
        "candidates",
        "checks",
        "shapes",
        "steps",
        "quick_shapes",
        "quick_steps",
        "seed",
        "snapshot_prefix_steps",
        "snapshot_suffix_steps",
    }
)
_PROMO_KEYS = frozenset({"kind", "steps", "completion_json"})
_EXACT_KEYS = frozenset(
    {"native_transition_exact", "allowed_representation_conversion"}
)


@dataclass(frozen=True)
class ProfileDocument:
    profile: Profile
    parity: ParityProfile
    payload: dict[str, Any]
    toml: str


@lru_cache(maxsize=1)
def load_profile_documents() -> dict[str, ProfileDocument]:
    loaded: dict[str, ProfileDocument] = {}
    root = files("turbobench").joinpath("workload_profiles")
    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".toml"):
            continue
        text = resource.read_text(encoding="utf-8")
        document = _parse_document(tomllib.loads(text), resource.name, text)
        profile_id = document.profile.id
        if profile_id in loaded:
            raise ValueError(f"duplicate workload profile {profile_id!r}")
        loaded[profile_id] = document
    return loaded


def get_profile_document(profile_id: str) -> ProfileDocument:
    try:
        return load_profile_documents()[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile_id!r}") from exc


def _parse_document(raw: dict[str, Any], source: str, text: str) -> ProfileDocument:
    _require_keys(raw, _TOP_LEVEL_KEYS, source, optional=frozenset({"asset"}))
    observation = _mapping(raw["observation"], source, "observation")
    benchmark = _mapping(raw["benchmark"], source, "benchmark")
    parity = _mapping(raw["parity"], source, "parity")
    promo = _mapping(raw["promo"], source, "promo")
    exact = _mapping(raw["exact"], source, "exact")
    _require_keys(observation, _OBSERVATION_KEYS, source)
    _require_keys(benchmark, _BENCHMARK_KEYS, source)
    _require_keys(parity, _PARITY_KEYS, source)
    _require_keys(promo, _PROMO_KEYS, source)
    _require_keys(exact, _EXACT_KEYS, source)

    profile_id = str(raw["id"])
    if source != _filename(profile_id):
        raise ValueError(f"workload profile filename does not match id in {source}")
    if str(raw["schema"]) != WORKLOAD_PROFILE_SCHEMA:
        raise ValueError(f"unsupported workload profile schema in {source}")

    providers = tuple(map(str, raw["providers"]))
    states = tuple(map(str, raw["states"]))
    semantic_actions = tuple(map(str, raw["semantic_actions"]))
    action_table = {
        str(name): tuple(map(str, labels))
        for name, labels in _mapping(raw["action_table"], source, "action_table").items()
    }
    benchmark_shapes = tuple(map(int, benchmark["shapes"]))
    parity_shapes = tuple(map(int, parity["shapes"]))
    quick_shapes = tuple(map(int, parity["quick_shapes"]))
    completion = json.loads(str(promo["completion_json"]))
    if not isinstance(completion, dict):
        raise ValueError(f"promo completion must be an object in {source}")
    resize = tuple(map(int, observation["resize"]))
    if len(resize) != 2:
        raise ValueError(f"observation resize must have two dimensions in {source}")

    asset = raw.get("asset")
    asset_sha256 = None
    if asset is not None:
        asset_mapping = _mapping(asset, source, "asset")
        _require_keys(asset_mapping, frozenset({"sha256"}), source)
        asset_sha256 = str(asset_mapping["sha256"])

    profile = Profile(
        id=profile_id,
        logical_environment=str(raw["logical_environment"]),
        game=str(raw["game"]),
        providers=providers,
        shapes=benchmark_shapes,
        states=states,
        semantic_actions=semantic_actions,
        action_table=action_table,
        info_integer=tuple(map(str, raw["info_integer"])),
        info_float=tuple(map(str, raw["info_float"])),
        frame_skip=int(observation["frame_skip"]),
        frame_stack=int(observation["frame_stack"]),
        crop_top=int(observation["crop_top"]),
        crop_bottom=int(observation["crop_bottom"]),
        crop_mode=str(observation["crop_mode"]),
        resize=(resize[0], resize[1]),
        grayscale=bool(observation["grayscale"]),
        layout=str(observation["layout"]),
        resize_algorithm=str(observation["resize_algorithm"]),
        maxpool_last_two=bool(observation["maxpool_last_two"]),
        benchmark_steps=int(benchmark["steps"]),
        correctness_steps=int(benchmark["correctness_steps"]),
        promo_kind=str(promo["kind"]),
        promo_steps=int(promo["steps"]),
        completion=completion,
        asset_sha256=asset_sha256,
        native_transition_exact=bool(exact["native_transition_exact"]),
        allowed_representation_conversion=str(
            exact["allowed_representation_conversion"]
        ),
    )
    parity_profile = ParityProfile(
        schema=PARITY_PROFILE_SCHEMA,
        id=profile_id,
        base_profile=profile_id,
        authority=str(parity["authority"]),
        authority_version=str(parity["authority_version"]),
        candidates=tuple(map(str, parity["candidates"])),
        checks=tuple(map(str, parity["checks"])),
        shapes=parity_shapes,
        steps=int(parity["steps"]),
        quick_shapes=quick_shapes,
        quick_steps=int(parity["quick_steps"]),
        seed=int(parity["seed"]),
        snapshot_prefix_steps=int(parity["snapshot_prefix_steps"]),
        snapshot_suffix_steps=int(parity["snapshot_suffix_steps"]),
        allowed_representation_conversion=profile.allowed_representation_conversion,
    )
    _validate(profile, parity_profile, source)
    return ProfileDocument(profile=profile, parity=parity_profile, payload=raw, toml=text)


def _validate(profile: Profile, parity: ParityProfile, source: str) -> None:
    if not profile.providers or len(set(profile.providers)) != len(profile.providers):
        raise ValueError(f"workload providers must be non-empty and unique in {source}")
    if not profile.states or not profile.semantic_actions:
        raise ValueError(f"workload states and semantic actions must be non-empty in {source}")
    if any(action not in profile.action_table for action in profile.semantic_actions):
        raise ValueError(f"semantic action is absent from the action table in {source}")
    if any(shape <= 0 for shape in (*profile.shapes, *parity.shapes, *parity.quick_shapes)):
        raise ValueError(f"workload shapes must be positive in {source}")
    if profile.benchmark_steps <= 0 or profile.correctness_steps <= 0:
        raise ValueError(f"benchmark step counts must be positive in {source}")
    if parity.authority not in profile.providers or not parity.candidates:
        raise ValueError(f"parity providers are incompatible in {source}")
    if any(not parity.accepts(item) or item not in profile.providers for item in parity.candidates):
        raise ValueError(f"invalid parity candidate in {source}")
    if not parity.checks or any(item not in STANDARD_CHECKS for item in parity.checks):
        raise ValueError(f"unknown or empty parity check list in {source}")
    if parity.steps < 2 or not 2 <= parity.quick_steps <= parity.steps:
        raise ValueError(f"invalid parity step counts in {source}")
    if not profile.native_transition_exact:
        raise ValueError(f"unified workload must require exact native transitions in {source}")


def _mapping(value: Any, source: str, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a table in {source}")
    return value


def _require_keys(
    raw: dict[str, Any],
    required: frozenset[str],
    source: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    keys = frozenset(raw)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        raise ValueError(f"invalid workload profile {source}: missing={missing}, unknown={unknown}")


def _filename(profile_id: str) -> str:
    return profile_id.replace("/", "--") + ".toml"
