"""Parity projections from unified workload profiles."""

from __future__ import annotations

from typing import Any

from turbobench.model import ParityProfile
from turbobench.profile_config import get_profile_document, load_profile_documents
from turbobench.util import canonical_json_hash


def load_parity_profiles() -> dict[str, ParityProfile]:
    return {
        profile_id: document.parity
        for profile_id, document in load_profile_documents().items()
    }


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
    return get_profile_document(profile.id).toml
