"""Load unified, declarative workload profiles."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from turbobench.model import Profile

WORKLOAD_PROFILE_SCHEMA = "turbobench.workload-profile/v2"
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
        "run",
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
_RUN_KEYS = frozenset(
    {
        "action_stream_version",
        "seed",
        "measurement_shapes",
        "quick_parity_steps",
        "full_parity_steps",
        "measurement_steps",
        "warmup_pairs",
        "light_pairs",
        "full_pairs",
    }
)
_PARITY_KEYS = frozenset(
    {
        "authority",
        "authority_version",
        "candidates",
        "checks",
        "shapes",
        "quick_shapes",
        "snapshot_prefix_steps",
        "snapshot_suffix_steps",
        "reset_samples",
    }
)
_PROMO_KEYS = frozenset({"kind", "steps", "completion_json"})
_EXACT_KEYS = frozenset(
    {"native_transition_exact", "allowed_representation_conversion"}
)


@dataclass(frozen=True)
class ProfileDocument:
    profile: Profile
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
    run = _mapping(raw["run"], source, "run")
    parity = _mapping(raw["parity"], source, "parity")
    promo = _mapping(raw["promo"], source, "promo")
    exact = _mapping(raw["exact"], source, "exact")
    _require_keys(observation, _OBSERVATION_KEYS, source)
    _require_keys(run, _RUN_KEYS, source)
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
    measurement_shapes = tuple(map(int, run["measurement_shapes"]))
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
        measurement_shapes=measurement_shapes,
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
        action_stream_version=str(run["action_stream_version"]),
        run_seed=int(run["seed"]),
        quick_parity_steps=int(run["quick_parity_steps"]),
        full_parity_steps=int(run["full_parity_steps"]),
        measurement_steps=int(run["measurement_steps"]),
        warmup_pairs=int(run["warmup_pairs"]),
        light_pairs=int(run["light_pairs"]),
        full_pairs=int(run["full_pairs"]),
        authority=str(parity["authority"]),
        authority_version=str(parity["authority_version"]),
        candidates=tuple(map(str, parity["candidates"])),
        checks=tuple(map(str, parity["checks"])),
        parity_shapes=parity_shapes,
        quick_parity_shapes=quick_shapes,
        snapshot_prefix_steps=int(parity["snapshot_prefix_steps"]),
        snapshot_suffix_steps=int(parity["snapshot_suffix_steps"]),
        reset_samples=int(parity["reset_samples"]),
        promo_kind=str(promo["kind"]),
        promo_steps=int(promo["steps"]),
        completion=completion,
        asset_sha256=asset_sha256,
        native_transition_exact=bool(exact["native_transition_exact"]),
        allowed_representation_conversion=str(
            exact["allowed_representation_conversion"]
        ),
    )
    _validate(profile, source)
    return ProfileDocument(profile=profile, payload=raw, toml=text)


def _validate(profile: Profile, source: str) -> None:
    if not profile.providers or len(set(profile.providers)) != len(profile.providers):
        raise ValueError(f"workload providers must be non-empty and unique in {source}")
    if not profile.states or not profile.semantic_actions:
        raise ValueError(f"workload states and semantic actions must be non-empty in {source}")
    if any(action not in profile.action_table for action in profile.semantic_actions):
        raise ValueError(f"semantic action is absent from the action table in {source}")
    if any(
        shape <= 0
        for shape in (
            *profile.measurement_shapes,
            *profile.parity_shapes,
            *profile.quick_parity_shapes,
        )
    ):
        raise ValueError(f"workload shapes must be positive in {source}")
    if len(set(profile.measurement_shapes)) != len(profile.measurement_shapes):
        raise ValueError(f"measurement shapes must be unique in {source}")
    if profile.measurement_shapes[:1] != (1,):
        raise ValueError(f"measurement shapes must begin with shape 1 in {source}")
    if any(
        count <= 0
        for count in (
            profile.quick_parity_steps,
            profile.full_parity_steps,
            profile.measurement_steps,
            profile.warmup_pairs,
            profile.light_pairs,
            profile.full_pairs,
            profile.snapshot_prefix_steps,
            profile.snapshot_suffix_steps,
            profile.reset_samples,
        )
    ):
        raise ValueError(f"run budgets must be positive in {source}")
    if profile.quick_parity_steps > profile.full_parity_steps:
        raise ValueError(f"quick parity cannot exceed full parity in {source}")
    if profile.light_pairs > profile.full_pairs:
        raise ValueError(f"light pairs cannot exceed full pairs in {source}")
    if profile.action_stream_version != "seeded-random-with-directed-prefix/v1":
        raise ValueError(f"unsupported action stream in {source}")
    if profile.authority not in profile.providers or not profile.candidates:
        raise ValueError(f"parity providers are incompatible in {source}")
    if any(
        not profile.accepts(item) or item not in profile.providers
        for item in profile.candidates
    ):
        raise ValueError(f"invalid parity candidate in {source}")
    if not profile.checks or any(item not in STANDARD_CHECKS for item in profile.checks):
        raise ValueError(f"unknown or empty parity check list in {source}")
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
