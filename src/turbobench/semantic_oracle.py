"""ROM-backed, exact semantic-oracle receipts.

The semantic oracle is deliberately separate from performance comparison.
Compatibility normalization remains available to immutable v1 benchmark
profiles, but it can never make an exact v2 receipt pass.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from turbobench import DISTRIBUTION_NAME, __version__
from turbobench.assets import discover_assets
from turbobench.correctness import compare_traces
from turbobench.engine import _provider_summary, _trace
from turbobench.model import Profile, ProviderRef, ResolvedProvider
from turbobench.profiles import (
    action_stream_hash,
    allowed_representation_conversion,
    get_profile,
    oracle_actions,
    profile_hash,
    profile_toml,
)
from turbobench.providers import load_providers
from turbobench.resolution import resolve_pair
from turbobench.runtime import HARNESS_REQUIREMENTS, harness_source_hash, prepare_runtime
from turbobench.util import (
    canonical_json_hash,
    find_portability_violations,
    read_json,
    relative_files,
    sha256_file,
    write_json,
)

ORACLE_RESULT_SCHEMA = "turbobench.semantic-oracle-result/v2"
ORACLE_MANIFEST_SCHEMA = "turbobench.semantic-oracle-manifest/v2"


@dataclass(frozen=True)
class OracleOptions:
    allow_dirty: bool = False
    python_minor: str = "3.14"
    steps: int | None = None
    shapes: tuple[int, ...] | None = None
    seed: int = 123
    command: tuple[str, ...] = ()
    progress: Callable[[str], None] | None = field(default=None, compare=False, repr=False)

    def report_progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


def run_oracle(
    profile_id: str,
    left_ref: ProviderRef,
    right_ref: ProviderRef,
    output: Path,
    options: OracleOptions,
) -> tuple[Path, dict[str, Any]]:
    profile = get_profile(profile_id)
    _require_oracle_profile(profile)
    definitions = load_providers()
    options.report_progress(f"Resolving exact providers for {profile.id}")
    resolution = resolve_pair(
        profile,
        left_ref,
        right_ref,
        definitions,
        python_minor=options.python_minor,
        allow_dirty=options.allow_dirty,
    )
    options.report_progress(f"Preparing isolated runtime for {resolution.left.provider}")
    left = prepare_runtime(
        resolution.left,
        checkout_path=Path(left_ref.value) if left_ref.selector == "checkout" else None,
        progress=options.progress,
    )
    options.report_progress(f"Preparing isolated runtime for {resolution.right.provider}")
    right = prepare_runtime(
        resolution.right,
        checkout_path=Path(right_ref.value) if right_ref.selector == "checkout" else None,
        progress=options.progress,
    )
    return run_oracle_resolved(
        profile,
        left,
        right,
        output,
        options,
        excluded=resolution.excluded,
    )


def run_oracle_resolved(
    profile: Profile,
    left: ResolvedProvider,
    right: ResolvedProvider,
    output: Path,
    options: OracleOptions,
    *,
    excluded: dict[str, tuple[dict[str, str], ...]] | None = None,
    private_assets: dict[str, Any] | None = None,
    portable_assets: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    _require_oracle_profile(profile)
    if not profile.compatible(left.provider, right.provider) and not (
        left.adapter == right.adapter == "fake"
    ):
        raise ValueError(
            f"{left.provider!r} and {right.provider!r} are not a compatible pair for {profile.id}"
        )
    final = output.expanduser().resolve()
    partial = final.with_name(final.name + ".partial")
    if final.exists():
        raise FileExistsError(f"oracle receipt already exists: {final}")
    if partial.exists():
        raise FileExistsError(f"partial oracle receipt already exists: {partial}")
    partial.mkdir(parents=True)
    (partial / "raw").mkdir()
    (partial / "verification").mkdir()
    try:
        discovered_private, discovered_portable = discover_assets(profile)
        assets = private_assets if private_assets is not None else discovered_private
        asset_record = portable_assets if portable_assets is not None else discovered_portable
        if asset_record.get("required") and not asset_record.get("available"):
            raise FileNotFoundError(str(asset_record.get("detail", "canonical assets unavailable")))

        shapes = _oracle_shapes(profile, options)
        steps = _oracle_steps(profile, options)
        lock = {
            "schema": "turbobench.semantic-oracle-lock/v2",
            "python_minor": options.python_minor,
            "harness_dependencies": list(HARNESS_REQUIREMENTS),
            "harness_source_sha256": harness_source_hash(),
            "profile": {"id": profile.id, "sha256": profile_hash(profile)},
            "providers": {"left": left.portable(), "right": right.portable()},
            "excluded_releases": {
                key: list(value) for key, value in (excluded or {}).items()
            },
            "assets": asset_record,
            "network_disabled_after_resolution": True,
        }
        (partial / "profile.toml").write_text(profile_toml(profile), encoding="utf-8")
        write_json(partial / "resolved-lock.json", lock)

        checks: dict[str, Any] = {}
        action_records: dict[str, Any] = {}
        require_ram = _requires_ram(profile, left, right)
        snapshot_prefix, snapshot_suffix = _snapshot_window(profile, steps)
        for shape in shapes:
            options.report_progress(
                f"Exact semantic trace for shape {shape}: {steps} seeded transitions"
            )
            actions = oracle_actions(profile, shape, steps, seed=options.seed)
            stream_hash = action_stream_hash(profile, actions)
            action_records[str(shape)] = {
                "kind": "seeded-random-with-directed-prefix",
                "seed": options.seed,
                "steps": steps,
                "sha256": stream_hash,
            }
            left_trace = _trace(
                partial,
                left,
                profile,
                shape,
                actions,
                stream_hash,
                assets,
                side="left",
                trace_ram=require_ram,
                snapshot_prefix_steps=snapshot_prefix,
                snapshot_suffix_steps=snapshot_suffix,
            )
            right_trace = _trace(
                partial,
                right,
                profile,
                shape,
                actions,
                stream_hash,
                assets,
                side="right",
                trace_ram=require_ram,
                snapshot_prefix_steps=snapshot_prefix,
                snapshot_suffix_steps=snapshot_suffix,
            )
            checks[str(shape)] = compare_traces(left_trace, right_trace, profile)
            options.report_progress(
                f"Shape {shape}: {'passed' if checks[str(shape)]['passed'] else 'failed'}"
            )

        passed = all(check["passed"] for check in checks.values())
        result = {
            "schema": ORACLE_RESULT_SCHEMA,
            "passed": passed,
            "profile": {"id": profile.id, "sha256": profile_hash(profile)},
            "semantic_authority": profile.semantic_authority,
            "authority_present": profile.semantic_authority
            in {left.provider, right.provider},
            "allowed_representation_conversion": allowed_representation_conversion(profile),
            "compatibility_normalization_permitted": False,
            "ram_required": require_ram,
            "snapshot_continuation_required": True,
            "providers": {
                "left": _provider_summary(left),
                "right": _provider_summary(right),
            },
            "checks": checks,
            "actions": action_records,
            "assets": asset_record,
            "lock_sha256": canonical_json_hash(lock),
            "tool": {
                "distribution": DISTRIBUTION_NAME,
                "version": __version__,
                "source_sha256": harness_source_hash(),
            },
        }
        write_json(partial / "verification" / "semantic-exact.json", result)
        write_json(partial / "result.json", result)
        _finalize_oracle_manifest(partial)
        verification = verify_oracle_receipt(partial)
        if not verification["passed"]:
            raise RuntimeError(
                "oracle receipt failed self-verification: "
                + "; ".join(verification["errors"])
            )
        os.replace(partial, final)
        options.report_progress(f"Semantic oracle complete: {final}")
        return final, result
    except BaseException:
        # Keep failed semantic evidence out of the final receipt path. The
        # partial directory is intentionally retained for diagnosis.
        raise


def verify_oracle_receipt(
    receipt: Path,
    *,
    require_canonical: bool = False,
    require_provider: str | None = None,
) -> dict[str, Any]:
    root = receipt.expanduser().resolve()
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "errors": ["manifest.json is missing"]}
    manifest = read_json(manifest_path)
    if manifest.get("schema") != ORACLE_MANIFEST_SCHEMA:
        errors.append("unsupported semantic-oracle manifest schema")
    expected_id = canonical_json_hash({**manifest, "receipt_id": ""})
    if manifest.get("receipt_id") != expected_id:
        errors.append("semantic-oracle receipt_id does not match its content")
    recorded: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        relative = str(artifact.get("path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            errors.append(f"unsafe artifact path {relative!r}")
            continue
        recorded.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"artifact is missing: {relative}")
        elif path.stat().st_size != artifact.get("size"):
            errors.append(f"artifact size mismatch: {relative}")
        elif sha256_file(path) != artifact.get("sha256"):
            errors.append(f"artifact hash mismatch: {relative}")
    actual = {path.as_posix() for path in relative_files(root)}
    if actual != recorded:
        errors.extend(f"unrecorded artifact: {path}" for path in sorted(actual - recorded))
        errors.extend(f"recorded artifact absent: {path}" for path in sorted(recorded - actual))
    try:
        result = read_json(root / "result.json")
        lock = read_json(root / "resolved-lock.json")
        profile = get_profile(result["profile"]["id"])
        if result.get("schema") != ORACLE_RESULT_SCHEMA:
            errors.append("unsupported semantic-oracle result schema")
        if not profile.native_transition_exact:
            errors.append("receipt profile is not an exact semantic profile")
        if result["profile"].get("sha256") != profile_hash(profile):
            errors.append("receipt profile hash is inconsistent")
        if (root / "profile.toml").read_text(encoding="utf-8") != profile_toml(profile):
            errors.append("profile.toml does not match the built-in profile")
        if result.get("lock_sha256") != canonical_json_hash(lock):
            errors.append("resolved lock hash is inconsistent")
        checks_passed = bool(result.get("checks")) and all(
            check.get("passed") for check in result.get("checks", {}).values()
        )
        if bool(result.get("passed")) != checks_passed:
            errors.append("result pass flag is inconsistent with exact checks")
        if not checks_passed:
            errors.append("one or more exact semantic checks failed")
        authority = profile.semantic_authority
        authority_version = profile.semantic_authority_version
        provider_records = tuple(lock.get("providers", {}).values())
        if require_provider and not any(
            provider.get("provider") == require_provider for provider in provider_records
        ):
            errors.append(f"required provider is absent: {require_provider}")
        authority_records = tuple(
            provider for provider in provider_records if provider.get("provider") == authority
        )
        if not authority or not authority_records or result.get("authority_present") is not True:
            errors.append("the profile's semantic authority is absent")
        if authority_version and not any(
            provider.get("version") == authority_version for provider in authority_records
        ):
            errors.append(f"semantic authority version must be {authority_version}")
        if result.get("compatibility_normalization_permitted") is not False:
            errors.append("semantic receipt permits compatibility normalization")
        if require_canonical:
            expected_shapes = {str(shape) for shape in profile.oracle_shapes}
            if set(result.get("checks", {})) != expected_shapes:
                errors.append("receipt does not contain every canonical oracle shape")
            action_records = result.get("actions", {})
            if set(action_records) != expected_shapes or any(
                record.get("steps") != profile.oracle_steps or record.get("seed") != 123
                for record in action_records.values()
            ):
                errors.append("receipt does not use the canonical action workload")
            if any(provider.get("diagnostic_reasons") for provider in provider_records):
                errors.append("canonical receipt contains a diagnostic provider override")
            if any(
                provider.get("provider") != authority
                and provider.get("source_kind") != "pypi"
                for provider in provider_records
            ):
                errors.append("canonical receipt candidate must be an installed PyPI release")
            if not any(
                provider.get("source_kind") == "pypi"
                and provider.get("source_identity")
                == f"pypi:{authority}=={authority_version}"
                for provider in authority_records
            ):
                errors.append("canonical receipt does not pin the PyPI semantic authority")
            assets = lock.get("assets", {})
            if assets.get("required") and not assets.get("available"):
                errors.append("canonical receipt is missing required private assets")
        for payload_path in root.rglob("*.json"):
            if payload_path.name == "manifest.json":
                continue
            errors.extend(
                f"portable output violation: {item}"
                for item in find_portability_violations(read_json(payload_path))
            )
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"semantic-oracle receipt is inconsistent: {exc}")
    return {
        "passed": not errors,
        "receipt_id": manifest.get("receipt_id"),
        "artifact_count": len(manifest.get("artifacts", [])),
        "errors": errors,
    }


def _finalize_oracle_manifest(receipt: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": relative.as_posix(),
            "size": (receipt / relative).stat().st_size,
            "sha256": sha256_file(receipt / relative),
        }
        for relative in relative_files(receipt)
    ]
    manifest = {
        "schema": ORACLE_MANIFEST_SCHEMA,
        "receipt_id": "",
        "tool": {"distribution": DISTRIBUTION_NAME, "version": __version__},
        "artifacts": artifacts,
    }
    manifest["receipt_id"] = canonical_json_hash(manifest)
    write_json(receipt / "manifest.json", manifest)
    return manifest


def _require_oracle_profile(profile: Profile) -> None:
    if not profile.native_transition_exact or not profile.semantic_authority:
        raise ValueError(
            f"profile {profile.id!r} is not a semantic-oracle v2 profile"
        )


def _oracle_shapes(profile: Profile, options: OracleOptions) -> tuple[int, ...]:
    shapes = profile.oracle_shapes if options.shapes is None else options.shapes
    if not shapes or any(shape <= 0 for shape in shapes):
        raise ValueError("oracle shapes must contain positive values")
    return shapes


def _oracle_steps(profile: Profile, options: OracleOptions) -> int:
    steps = profile.oracle_steps if options.steps is None else options.steps
    if steps <= 0:
        raise ValueError("oracle steps must be positive")
    return steps


def _requires_ram(
    profile: Profile, left: ResolvedProvider, right: ResolvedProvider
) -> bool:
    if profile.logical_environment == "supermario":
        return True
    return "breakout-turbo-env" not in {left.provider, right.provider}


def _snapshot_window(profile: Profile, steps: int) -> tuple[int, int]:
    prefix = min(profile.snapshot_prefix_steps, max(1, steps // 2))
    suffix = min(profile.snapshot_suffix_steps, steps - prefix)
    if suffix <= 0:
        raise ValueError("semantic oracle needs at least two steps for snapshot replay")
    return prefix, suffix
