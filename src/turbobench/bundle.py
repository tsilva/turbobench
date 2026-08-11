"""Portable result-bundle finalization and self-verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from turbobench import DISTRIBUTION_NAME, RESULT_SCHEMA, __version__
from turbobench.profiles import get_profile, profile_hash, profile_toml
from turbobench.stats import paired_statistics, reciprocal_statistics
from turbobench.util import (
    canonical_json_hash,
    find_portability_violations,
    read_json,
    relative_files,
    sha256_file,
    write_json,
)

REQUIRED_ENTRIES = {
    "result.json",
    "profile.toml",
    "resolved-lock.json",
    "report.md",
    "chart.svg",
}
REQUIRED_DIRECTORIES = {"raw", "verification", "media"}


def _media_manifest_hash(path: Path) -> str:
    payload = read_json(path)
    payload["bundle_id"] = ""
    return canonical_json_hash(payload)


def artifact_record(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if relative.as_posix() == "media/media-manifest.json":
        return {
            "path": relative.as_posix(),
            "size": None,
            "sha256": _media_manifest_hash(path),
            "hash_mode": "canonical-json-excluding-bundle-id",
        }
    return {
        "path": relative.as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "hash_mode": "bytes",
    }


def finalize_manifest(bundle: Path) -> dict[str, Any]:
    for directory in REQUIRED_DIRECTORIES:
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    missing = sorted(name for name in REQUIRED_ENTRIES if not (bundle / name).is_file())
    if missing:
        raise ValueError(f"bundle is missing required files: {missing}")
    artifacts = [artifact_record(bundle, relative) for relative in relative_files(bundle)]
    manifest = {
        "schema": "turbobench.manifest/v1",
        "bundle_id": "",
        "tool": {"distribution": DISTRIBUTION_NAME, "version": __version__},
        "artifacts": artifacts,
    }
    manifest["bundle_id"] = canonical_json_hash({**manifest, "bundle_id": ""})
    write_json(bundle / "manifest.json", manifest)
    media_manifest = bundle / "media" / "media-manifest.json"
    if media_manifest.is_file():
        payload = read_json(media_manifest)
        payload["bundle_id"] = manifest["bundle_id"]
        write_json(media_manifest, payload)
        # Canonical media hashing excludes only this circular reference.
        expected = next(
            item for item in artifacts if item["path"] == "media/media-manifest.json"
        )["sha256"]
        if _media_manifest_hash(media_manifest) != expected:
            raise RuntimeError("media manifest changed outside its bundle_id binding")
    return manifest


def verify_bundle(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = ["integrity verification does not authenticate the result author"]
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "errors": ["manifest.json is missing"], "warnings": warnings}
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"manifest.json is unreadable: {exc}"], "warnings": warnings}
    if manifest.get("schema") != "turbobench.manifest/v1":
        errors.append("unsupported manifest schema")
    expected_id = canonical_json_hash({**manifest, "bundle_id": ""})
    if manifest.get("bundle_id") != expected_id:
        errors.append("manifest bundle_id does not match its canonical content")
    recorded_paths: set[str] = set()
    for record in manifest.get("artifacts", []):
        relative = str(record.get("path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            errors.append(f"unsafe artifact path {relative!r}")
            continue
        if relative in recorded_paths:
            errors.append(f"duplicate artifact record {relative!r}")
            continue
        recorded_paths.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"artifact is missing: {relative}")
            continue
        if record.get("size") is not None and path.stat().st_size != record.get("size"):
            errors.append(f"artifact size mismatch: {relative}")
        actual = (
            _media_manifest_hash(path)
            if record.get("hash_mode") == "canonical-json-excluding-bundle-id"
            else sha256_file(path)
        )
        if actual != record.get("sha256"):
            errors.append(f"artifact hash mismatch: {relative}")
    actual_paths = {path.as_posix() for path in relative_files(root)}
    if actual_paths != recorded_paths:
        for relative in sorted(actual_paths - recorded_paths):
            errors.append(f"unrecorded artifact: {relative}")
        for relative in sorted(recorded_paths - actual_paths):
            errors.append(f"recorded artifact absent: {relative}")
    for name in REQUIRED_ENTRIES:
        if name not in recorded_paths:
            errors.append(f"required artifact is not recorded: {name}")
    for directory in REQUIRED_DIRECTORIES:
        if not (root / directory).is_dir():
            errors.append(f"required directory is missing: {directory}/")
    if not errors:
        _verify_consistency(root, manifest, errors)
    return {
        "passed": not errors,
        "bundle_id": manifest.get("bundle_id"),
        "artifact_count": len(manifest.get("artifacts", [])),
        "errors": errors,
        "warnings": warnings,
    }


def _verify_consistency(root: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    try:
        result = read_json(root / "result.json")
        lock = read_json(root / "resolved-lock.json")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"result or lock is unreadable: {exc}")
        return
    if result.get("schema") != RESULT_SCHEMA:
        errors.append("unsupported result schema")
        return
    try:
        profile = get_profile(result["profile"]["id"])
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))
        return
    if (root / "profile.toml").read_text(encoding="utf-8") != profile_toml(profile):
        errors.append("profile.toml does not match the immutable built-in profile")
    if result["profile"].get("sha256") != profile_hash(profile):
        errors.append("result profile hash is inconsistent")
    if result.get("lock_sha256") != canonical_json_hash(lock):
        errors.append("resolved-lock.json hash is inconsistent")
    validity = bool(result.get("validity", {}).get("passed"))
    claim = result.get("claim", {}).get("status")
    if validity and claim not in {"official", "diagnostic"}:
        errors.append("valid result has invalid claim status")
    if not validity and claim != "diagnostic":
        errors.append("failed validity must force diagnostic claim status")
    if result.get("comparison", {}).get("outcome") not in {
        "left_faster",
        "right_faster",
        "inconclusive",
    }:
        errors.append("invalid comparison outcome")
    reversal_path = root / "verification" / "order-reversal.json"
    reversal = read_json(reversal_path) if reversal_path.is_file() else None
    if reversal is None:
        errors.append("order-reversal verification is missing")
    elif reversal.get("schema") != "turbobench.order-reversal/v1":
        errors.append("unsupported order-reversal schema")
    else:
        expected_providers = {
            "left": result["comparison"]["right"]["provider"],
            "right": result["comparison"]["left"]["provider"],
        }
        if reversal.get("providers") != expected_providers:
            errors.append("order-reversal providers are not swapped")
    for shape, shape_result in result.get("comparison", {}).get("shapes", {}).items():
        raw_path = root / "raw" / f"shape-{shape}" / "pairs.json"
        if not raw_path.is_file():
            errors.append(f"shape {shape} raw pairs are missing")
            continue
        pairs = read_json(raw_path)["pairs"]
        expected = paired_statistics(
            pairs,
            require_official_design=len(pairs) == 7,
        )
        actual = shape_result.get("statistics", {})
        for field in (
            "paired_ratios_left_over_right",
            "median_paired_ratio_left_over_right",
            "outcome",
        ):
            if actual.get(field) != expected.get(field):
                errors.append(f"shape {shape} statistics mismatch: {field}")
        if reversal is not None:
            reversed_shape = reversal.get("shapes", {}).get(shape, {})
            if not reversed_shape.get("raw_evidence_reused"):
                errors.append(f"shape {shape} reversal does not reuse raw evidence")
            if reversed_shape.get("source_pairs_sha256") != canonical_json_hash(
                read_json(raw_path)
            ):
                errors.append(f"shape {shape} reversal source hash mismatch")
            if reversed_shape.get("statistics") != reciprocal_statistics(expected):
                errors.append(f"shape {shape} reciprocal statistics mismatch")
    promo = result.get("promo", {})
    media_manifest_path = root / "media" / "media-manifest.json"
    if promo.get("eligible") and result["comparison"]["outcome"] == "inconclusive":
        errors.append("inconclusive result cannot be promo eligible")
    if media_manifest_path.is_file():
        media = read_json(media_manifest_path)
        if media.get("bundle_id") != manifest.get("bundle_id"):
            errors.append("media manifest is not bound to this bundle")
        if not promo.get("eligible") and not media.get("diagnostic_watermark"):
            errors.append("ineligible media must carry a diagnostic watermark")
    portability = find_portability_violations(result) + find_portability_violations(lock)
    for path in root.rglob("*.json"):
        if path.name == "manifest.json":
            continue
        try:
            portability.extend(find_portability_violations(read_json(path)))
        except json.JSONDecodeError:
            errors.append(f"JSON artifact is unreadable: {path.relative_to(root)}")
    errors.extend(f"portable output violation: {item}" for item in portability)


def update_result_and_refinalize(bundle: Path, result: dict[str, Any]) -> dict[str, Any]:
    (bundle / "manifest.json").unlink(missing_ok=True)
    write_json(bundle / "result.json", result)
    return finalize_manifest(bundle)
