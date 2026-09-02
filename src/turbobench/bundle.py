"""Portable result-bundle finalization and self-verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from turbobench import DISTRIBUTION_NAME, RESULT_SCHEMA, __version__
from turbobench.lifecycle import EXECUTION_PROTOCOL, AttestationError, require_attestation
from turbobench.profiles import (
    action_stream_hash,
    canonical_actions,
    get_profile,
    profile_hash,
    profile_toml,
)
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
REQUIRED_DIRECTORIES = {"raw", "verification"}
FINALIZED_DIRECTORIES = REQUIRED_DIRECTORIES | {"media"}


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
    for directory in FINALIZED_DIRECTORIES:
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
        expected = next(item for item in artifacts if item["path"] == "media/media-manifest.json")[
            "sha256"
        ]
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
        return {
            "passed": False,
            "errors": [f"manifest.json is unreadable: {exc}"],
            "warnings": warnings,
        }
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
        _verify_consistency(root, manifest, errors, warnings)
    return {
        "passed": not errors,
        "bundle_id": manifest.get("bundle_id"),
        "artifact_count": len(manifest.get("artifacts", [])),
        "errors": errors,
        "warnings": warnings,
    }


def _verify_consistency(
    root: Path, manifest: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    try:
        result = read_json(root / "result.json")
        lock = read_json(root / "resolved-lock.json")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"result or lock is unreadable: {exc}")
        return
    result_schema = result.get("schema")
    if result_schema not in {RESULT_SCHEMA, "turbobench.result/v2"}:
        errors.append("unsupported result schema")
        return
    if result_schema == "turbobench.result/v2":
        warnings.append(
            "legacy 1.0.3-2.0.6 bundle integrity is verifiable, but performance evidence "
            "may be lifecycle-contaminated and should be rerun before supporting claims"
        )
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
    if lock.get("profile") != result.get("profile"):
        errors.append("resolved lock profile differs from result.json")
    for side in ("left", "right"):
        provider = lock.get("providers", {}).get(side, {})
        expected_summary = {
            key: provider.get(key)
            for key in (
                "provider",
                "version",
                "adapter",
                "artifact_sha256",
                "source_identity",
                "runtime_id",
                "compatibility_lineage",
            )
        }
        if result.get("comparison", {}).get(side) != expected_summary:
            errors.append(f"comparison {side} summary differs from the resolved lock")
    validity = bool(result.get("validity", {}).get("passed"))
    claim = result.get("claim", {}).get("status")
    if validity and claim not in {"official", "diagnostic"}:
        errors.append("valid result has invalid claim status")
    if not validity and claim != "diagnostic":
        errors.append("failed validity must force diagnostic claim status")
    contract = result.get("turbo_contract")
    contract_path = root / "verification" / "turbo-contract.json"
    if not isinstance(contract, dict) or not contract_path.is_file():
        errors.append("Turbo API contract reports are missing")
    else:
        recorded = read_json(contract_path)
        if result_schema == RESULT_SCHEMA:
            _verify_phase_isolated_contracts(root, result, lock, recorded, errors)
        else:
            if recorded.get("schema") != "turbobench.turbo-contract-reports/v1":
                errors.append("unsupported Turbo API contract report collection")
            if recorded.get("reports") != contract:
                errors.append("Turbo API contract reports differ from result.json")
            for side, report in contract.items():
                if report.get("schema") != "turbobench.turbo-contract-report/v1":
                    errors.append(f"{side} has an unsupported Turbo contract report")
                    continue
                expected_report_hash = canonical_json_hash(
                    {key: value for key, value in report.items() if key != "report_sha256"}
                )
                if report.get("report_sha256") != expected_report_hash:
                    errors.append(f"{side} Turbo contract report hash mismatch")
        turbo_gate = next(
            (
                gate
                for gate in result.get("validity", {}).get("gates", [])
                if gate.get("name") == "Turbo API validity"
            ),
            None,
        )
        expected_gate = all(
            report.get("promotable")
            for value in contract.values()
            for report in (value.values() if result_schema == RESULT_SCHEMA else (value,))
        )
        if turbo_gate is None or bool(turbo_gate.get("passed")) != expected_gate:
            errors.append("Turbo API validity gate is missing or inconsistent")
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
    correctness_path = root / "verification" / "correctness.json"
    correctness_record = read_json(correctness_path) if correctness_path.is_file() else {}
    if correctness_record.get("schema") != "turbobench.correctness/v2":
        errors.append("unsupported or missing correctness record")
    parity_gate_path = root / "verification" / "parity-gate.json"
    parity_gate = read_json(parity_gate_path) if parity_gate_path.is_file() else None
    selected_identities = {
        (
            side["provider"],
            side["version"],
            side["artifact_sha256"],
        )
        for side in (
            result["comparison"]["left"],
            result["comparison"]["right"],
        )
    }
    for shape, shape_result in result.get("comparison", {}).get("shapes", {}).items():
        raw_path = root / "raw" / f"shape-{shape}" / "pairs.json"
        if not raw_path.is_file():
            errors.append(f"shape {shape} raw pairs are missing")
            continue
        pairs = read_json(raw_path)["pairs"]
        action_record = result.get("actions", {}).get(shape, {})
        try:
            validation_steps = int(action_record.get("validation_steps", 0))
            measurement_steps = int(action_record.get("measurement_steps", 0))
            seed = int(action_record.get("seed"))
            validation_actions = canonical_actions(
                profile, int(shape), validation_steps, seed=seed
            )
            measurement_actions = canonical_actions(
                profile, int(shape), measurement_steps, seed=seed
            )
        except (TypeError, ValueError):
            errors.append(f"shape {shape} action record is malformed")
        else:
            raw_pairs = read_json(raw_path)
            if (
                action_record.get("version") != profile.action_stream_version
                or seed != profile.run_seed
                or validation_steps != profile.measurement_steps
                or action_record.get("validation_sha256")
                != action_stream_hash(profile, validation_actions)
                or action_record.get("measurement_sha256")
                != action_stream_hash(profile, measurement_actions)
                or raw_pairs.get("action_stream_sha256")
                != action_record.get("measurement_sha256")
            ):
                errors.append(f"shape {shape} action record is inconsistent")
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
        light = shape_result.get("light_statistics")
        if shape == "1" and len(pairs) == 7:
            expected_light = paired_statistics(
                pairs[:2], require_official_design=False
            )
            if light != expected_light:
                errors.append("shape 1 light statistics do not match its first two pairs")
        elif light is not None:
            errors.append(f"shape {shape} unexpectedly contains light statistics")
        correctness = shape_result.get("correctness", {})
        if correctness_record.get("shapes", {}).get(shape) != correctness:
            errors.append(f"shape {shape} correctness record differs from result.json")
        source = correctness.get("source")
        if source == "executed pair":
            for side in ("left", "right"):
                if not (root / "raw" / f"shape-{shape}" / f"trace-{side}.json").is_file():
                    errors.append(f"shape {shape} executed correctness trace is missing for {side}")
        elif source in {"direct receipt", "transitive receipts"}:
            _verify_reused_correctness(
                shape,
                source,
                correctness,
                parity_gate,
                selected_identities,
                result["profile"]["id"],
                errors,
            )
        else:
            errors.append(f"shape {shape} has an invalid correctness source")
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


def _verify_phase_isolated_contracts(
    root: Path,
    result: dict[str, Any],
    lock: dict[str, Any],
    recorded: dict[str, Any],
    errors: list[str],
) -> None:
    attestations = result.get("contract_attestations")
    if (
        result.get("execution_protocol") != EXECUTION_PROTOCOL
        or lock.get("execution_protocol") != EXECUTION_PROTOCOL
        or recorded.get("protocol") != EXECUTION_PROTOCOL
    ):
        errors.append("phase-isolated execution protocol identity is missing or inconsistent")
    if recorded.get("schema") != "turbobench.contract-attestations/v1":
        errors.append("unsupported contract attestation collection")
    if not isinstance(attestations, dict) or recorded.get("attestations") != attestations:
        errors.append("contract attestations differ from result.json")
        return
    response_by_hash: dict[str, dict[str, Any]] = {}
    for path in (root / "verification" / "attestations").glob("*.json"):
        response = read_json(path)
        attestation = response.get("contract_attestation", {})
        digest = attestation.get("attestation_sha256")
        if isinstance(digest, str):
            response_by_hash[digest] = response
    expected_by_side_shape: dict[tuple[str, str], str] = {}
    for side, records in attestations.items():
        if not isinstance(records, dict):
            errors.append(f"{side} contract attestations are malformed")
            continue
        for shape, attestation in records.items():
            digest = attestation.get("attestation_sha256")
            response = response_by_hash.get(digest)
            if response is None:
                errors.append(f"{side} shape {shape} attestation evidence is missing")
                continue
            try:
                spec = response.get("execution_spec", {})
                require_attestation(spec, attestation)
            except AttestationError as exc:
                errors.append(f"{side} shape {shape} attestation is invalid: {exc}")
                spec = {}
            report = attestation.get("contract_report", {})
            expected_report_hash = canonical_json_hash(
                {key: value for key, value in report.items() if key != "report_sha256"}
            )
            if (
                report != result.get("turbo_contract", {}).get(side, {}).get(shape)
                or report.get("schema") != "turbobench.turbo-contract-report/v1"
                or report.get("report_sha256") != expected_report_hash
                or bool(attestation.get("passed")) != bool(report.get("passed"))
                or bool(attestation.get("promotable")) != bool(report.get("promotable"))
            ):
                errors.append(f"{side} shape {shape} contract report binding is invalid")
            if (
                spec.get("provider") != lock.get("providers", {}).get(side)
                or spec.get("profile") != lock.get("profile")
                or spec.get("assets") != lock.get("assets")
                or spec.get("python_minor") != lock.get("python_minor")
                or str(spec.get("constructor", {}).get("shape")) != shape
            ):
                errors.append(f"{side} shape {shape} execution spec differs from the lock")
            if not response.get("lifecycle", {}).get("environment_closed"):
                errors.append(f"{side} shape {shape} contract environment was not closed")
            expected_by_side_shape[(side, str(shape))] = str(digest)
    promo_expected: dict[str, str] = {}
    for side, attestation in result.get("promo", {}).get(
        "contract_attestations", {}
    ).items():
        digest = attestation.get("attestation_sha256")
        response = response_by_hash.get(digest)
        if response is None:
            errors.append(f"promo {side} contract attestation evidence is missing")
            continue
        try:
            spec = response.get("execution_spec", {})
            require_attestation(spec, attestation)
        except AttestationError as exc:
            errors.append(f"promo {side} contract attestation is invalid: {exc}")
            spec = {}
        if (
            spec.get("provider") != lock.get("providers", {}).get(side)
            or spec.get("constructor", {}).get("frame_skip") != 1
            or spec.get("constructor", {}).get("shape") != 1
        ):
            errors.append(f"promo {side} execution spec differs from the lock")
        promo_expected[side] = str(digest)
    for shape in result.get("comparison", {}).get("shapes", {}):
        shape_dir = root / "raw" / f"shape-{shape}"
        for path in shape_dir.glob("*.json"):
            if path.name == "pairs.json":
                continue
            side = "left" if "left" in path.stem else "right" if "right" in path.stem else None
            if side is None:
                continue
            lifecycle = read_json(path).get("lifecycle", {})
            expected = expected_by_side_shape.get((side, shape))
            if (
                lifecycle.get("execution_protocol") != EXECUTION_PROTOCOL
                or lifecycle.get("contract_attestation_sha256") != expected
                or lifecycle.get("dynamic_contract_validation_calls") != 0
                or lifecycle.get("environment_closed") is not True
            ):
                errors.append(f"{path.relative_to(root)} has a mismatched contract attestation")
    replay_path = root / "verification" / "promo-replay.json"
    if replay_path.is_file():
        replay = read_json(replay_path)
        for side in ("left", "right"):
            lifecycle = replay.get(side, {}).get("lifecycle", {})
            if (
                lifecycle.get("execution_protocol") != EXECUTION_PROTOCOL
                or lifecycle.get("contract_attestation_sha256") != promo_expected.get(side)
                or lifecycle.get("dynamic_contract_validation_calls") != 0
                or lifecycle.get("environment_closed") is not True
            ):
                errors.append(f"promo {side} replay attestation is inconsistent")
    gate = next(
        (
            item
            for item in result.get("validity", {}).get("gates", [])
            if item.get("name") == "phase-isolated execution protocol"
        ),
        None,
    )
    if gate is None or not gate.get("passed"):
        errors.append("phase-isolated execution validity gate is missing or failed")


def _verify_reused_correctness(
    shape: str,
    source: str,
    correctness: dict[str, Any],
    parity_gate: dict[str, Any] | None,
    selected_identities: set[tuple[str, str, str]],
    profile_id: str,
    errors: list[str],
) -> None:
    if (
        parity_gate is None
        or parity_gate.get("schema") != "turbobench.parity-gate/v1"
        or not parity_gate.get("passed")
        or parity_gate.get("errors")
        or parity_gate.get("profile") != profile_id
    ):
        errors.append(f"shape {shape} has no valid parity reuse decision")
        return
    decision = parity_gate.get("shapes", {}).get(shape, {})
    receipt_ids = correctness.get("receipt_ids", [])
    if decision != {"source": source, "receipt_ids": receipt_ids}:
        errors.append(f"shape {shape} parity reuse decision is inconsistent")
        return
    evidence = {
        item.get("receipt_id"): item for item in parity_gate.get("evidence", [])
    }
    selected = [evidence.get(receipt_id) for receipt_id in receipt_ids]
    if any(item is None for item in selected):
        errors.append(f"shape {shape} parity receipt evidence is missing")
        return
    profile = get_profile(profile_id)
    for item in selected:
        record = item.get("actions", {}).get(shape, {})
        try:
            steps = int(record.get("steps", 0))
            seed = int(record.get("seed"))
            actions = canonical_actions(profile, int(shape), steps, seed=seed)
        except (TypeError, ValueError):
            errors.append(f"shape {shape} parity action evidence is malformed")
            return
        if (
            steps < profile.measurement_steps
            or seed != profile.run_seed
            or record.get("version") != profile.action_stream_version
            or record.get("sha256") != action_stream_hash(profile, actions)
        ):
            errors.append(f"shape {shape} parity action evidence is incompatible")
            return
    if source == "direct receipt":
        if len(selected) != 1 or {
            tuple(selected[0][role]) for role in ("authority", "candidate")
        } != selected_identities:
            errors.append(f"shape {shape} direct receipt does not bind the selected pair")
        return
    if len(selected) != 2:
        errors.append(f"shape {shape} transitive reuse must bind two receipts")
        return
    if selected[0]["authority"] != selected[1]["authority"]:
        errors.append(f"shape {shape} transitive receipts use different authorities")
    if {tuple(item["candidate"]) for item in selected} != selected_identities:
        errors.append(f"shape {shape} transitive receipts do not bind the selected pair")


def update_result_and_refinalize(bundle: Path, result: dict[str, Any]) -> dict[str, Any]:
    (bundle / "manifest.json").unlink(missing_ok=True)
    write_json(bundle / "result.json", result)
    return finalize_manifest(bundle)
