"""Strict loader for immutable, declarative parity profiles."""

from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any

from turbobench.model import ParityProfile
from turbobench.profiles import get_profile
from turbobench.util import canonical_json_hash

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
_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "id",
        "base_profile",
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
        "allowed_representation_conversion",
    }
)


def load_parity_profiles() -> dict[str, ParityProfile]:
    loaded: dict[str, ParityProfile] = {}
    root = files("turbobench").joinpath("parity_profiles")
    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".toml"):
            continue
        raw = tomllib.loads(resource.read_text(encoding="utf-8"))
        profile = _parse_profile(raw, resource.name)
        if profile.id in loaded:
            raise ValueError(f"duplicate parity profile {profile.id!r}")
        loaded[profile.id] = profile
    return loaded


def get_parity_profile(profile_id: str) -> ParityProfile:
    try:
        return load_parity_profiles()[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown parity profile {profile_id!r}") from exc


def parity_profile_payload(profile: ParityProfile) -> dict[str, Any]:
    return {
        "schema": profile.schema,
        "id": profile.id,
        "base_profile": profile.base_profile,
        "authority": profile.authority,
        "authority_version": profile.authority_version,
        "candidates": list(profile.candidates),
        "checks": list(profile.checks),
        "shapes": list(profile.shapes),
        "steps": profile.steps,
        "quick_shapes": list(profile.quick_shapes),
        "quick_steps": profile.quick_steps,
        "seed": profile.seed,
        "snapshot_prefix_steps": profile.snapshot_prefix_steps,
        "snapshot_suffix_steps": profile.snapshot_suffix_steps,
        "allowed_representation_conversion": profile.allowed_representation_conversion,
    }


def parity_profile_hash(profile: ParityProfile) -> str:
    return canonical_json_hash(parity_profile_payload(profile))


def parity_profile_toml(profile: ParityProfile) -> str:
    resource = files("turbobench").joinpath("parity_profiles", _filename(profile.id))
    return resource.read_text(encoding="utf-8")


def _filename(profile_id: str) -> str:
    return profile_id.replace("/", "--") + ".toml"


def _parse_profile(raw: dict[str, Any], source: str) -> ParityProfile:
    keys = frozenset(raw)
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        unknown = sorted(keys - _REQUIRED_KEYS)
        raise ValueError(f"invalid parity profile {source}: missing={missing}, unknown={unknown}")
    profile = ParityProfile(
        schema=str(raw["schema"]),
        id=str(raw["id"]),
        base_profile=str(raw["base_profile"]),
        authority=str(raw["authority"]),
        authority_version=str(raw["authority_version"]),
        candidates=tuple(map(str, raw["candidates"])),
        checks=tuple(map(str, raw["checks"])),
        shapes=tuple(map(int, raw["shapes"])),
        steps=int(raw["steps"]),
        quick_shapes=tuple(map(int, raw["quick_shapes"])),
        quick_steps=int(raw["quick_steps"]),
        seed=int(raw["seed"]),
        snapshot_prefix_steps=int(raw["snapshot_prefix_steps"]),
        snapshot_suffix_steps=int(raw["snapshot_suffix_steps"]),
        allowed_representation_conversion=str(raw["allowed_representation_conversion"]),
    )
    base = get_profile(profile.base_profile)
    if profile.schema != PARITY_PROFILE_SCHEMA:
        raise ValueError(f"unsupported parity profile schema in {source}")
    if profile.id != profile.base_profile:
        raise ValueError(f"parity profile id and base_profile must match in {source}")
    if profile.authority not in base.providers or not profile.candidates:
        raise ValueError(f"parity providers are incompatible with {profile.base_profile}")
    if any(not profile.accepts(item) or item not in base.providers for item in profile.candidates):
        raise ValueError(f"invalid parity candidate in {source}")
    if not profile.checks or any(item not in STANDARD_CHECKS for item in profile.checks):
        raise ValueError(f"unknown or empty standard check list in {source}")
    if not profile.shapes or not profile.quick_shapes or any(
        item <= 0 for item in (*profile.shapes, *profile.quick_shapes)
    ):
        raise ValueError(f"parity shapes must be positive in {source}")
    if profile.steps < 2 or not 2 <= profile.quick_steps <= profile.steps:
        raise ValueError(f"invalid parity step counts in {source}")
    return profile
