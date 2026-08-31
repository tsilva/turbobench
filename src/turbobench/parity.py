"""ROM-backed, exact cross-provider parity receipts.

Parity is deliberately separate from performance comparison.
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
from turbobench.correctness import compare_reset_distributions, compare_traces
from turbobench.engine import _provider_summary, _reset_distribution, _trace
from turbobench.model import ParityProfile, Profile, ProviderRef, ResolvedProvider
from turbobench.parity_profiles import (
    get_parity_profile,
    parity_profile_hash,
    parity_profile_toml,
)
from turbobench.profiles import (
    action_stream_hash,
    benchmark_actions,
    get_profile,
    parity_actions,
    profile_hash,
)
from turbobench.providers import load_providers
from turbobench.resolution import resolve_pair
from turbobench.runtime import HARNESS_REQUIREMENTS, harness_source_hash, prepare_runtime
from turbobench.system import host_record
from turbobench.util import (
    canonical_json_hash,
    find_portability_violations,
    read_json,
    relative_files,
    sha256_file,
    write_json,
)

PARITY_RESULT_SCHEMA = "turbobench.parity-result/v1"
PARITY_MANIFEST_SCHEMA = "turbobench.parity-manifest/v1"


@dataclass(frozen=True)
class ParityOptions:
    allow_dirty: bool = False
    quick: bool = False
    python_minor: str = "3.14"
    steps: int | None = None
    shapes: tuple[int, ...] | None = None
    seed: int | None = None
    command: tuple[str, ...] = ()
    progress: Callable[[str], None] | None = field(default=None, compare=False, repr=False)

    def report_progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


def run_parity(
    profile_id: str,
    candidate_ref: ProviderRef,
    output: Path,
    options: ParityOptions,
    *,
    authority_ref: ProviderRef | None = None,
) -> tuple[Path, dict[str, Any]]:
    parity_profile = get_parity_profile(profile_id)
    profile = get_profile(parity_profile.base_profile)
    if not parity_profile.accepts(candidate_ref.provider):
        raise ValueError(
            f"{candidate_ref.provider!r} is not a candidate for parity profile {profile_id!r}"
        )
    definitions = load_providers()
    pinned_authority = ProviderRef(
        parity_profile.authority, "version", parity_profile.authority_version
    )
    authority = authority_ref or pinned_authority
    if authority.provider != parity_profile.authority:
        raise ValueError(
            f"authority override must use {parity_profile.authority!r} for {profile_id!r}"
        )
    options.report_progress(f"Resolving exact providers for {profile.id}")
    resolution = resolve_pair(
        profile,
        authority,
        candidate_ref,
        definitions,
        python_minor=options.python_minor,
        allow_dirty=options.allow_dirty,
    )
    options.report_progress(f"Preparing isolated authority runtime for {resolution.left.provider}")
    resolved_authority = prepare_runtime(
        resolution.left,
        checkout_path=Path(authority.value) if authority.selector == "checkout" else None,
        artifact_path=Path(authority.value) if authority.selector == "artifact" else None,
        cache_context=parity_profile_hash(parity_profile),
        progress=options.progress,
    )
    options.report_progress(f"Preparing isolated candidate runtime for {resolution.right.provider}")
    candidate = prepare_runtime(
        resolution.right,
        checkout_path=Path(candidate_ref.value) if candidate_ref.selector == "checkout" else None,
        artifact_path=Path(candidate_ref.value) if candidate_ref.selector == "artifact" else None,
        cache_context=parity_profile_hash(parity_profile),
        progress=options.progress,
    )
    if authority_ref is not None:
        resolved_authority = _add_diagnostic_reason(
            resolved_authority, "authority override"
        )
    return run_parity_resolved(
        parity_profile,
        profile,
        resolved_authority,
        candidate,
        output,
        options,
        excluded=resolution.excluded,
    )


def run_parity_resolved(
    parity_profile: ParityProfile,
    profile: Profile,
    authority: ResolvedProvider,
    candidate: ResolvedProvider,
    output: Path,
    options: ParityOptions,
    *,
    excluded: dict[str, tuple[dict[str, str], ...]] | None = None,
    private_assets: dict[str, Any] | None = None,
    portable_assets: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    _require_parity_profile(parity_profile, profile)
    if not profile.compatible(authority.provider, candidate.provider) and not (
        authority.adapter == candidate.adapter == "fake"
    ):
        raise ValueError(
            f"{authority.provider!r} and {candidate.provider!r} are not a compatible pair for {profile.id}"
        )
    final = output.expanduser().resolve()
    partial = final.with_name(final.name + ".partial")
    if final.exists():
        raise FileExistsError(f"parity receipt already exists: {final}")
    if partial.exists():
        raise FileExistsError(f"partial parity receipt already exists: {partial}")
    partial.mkdir(parents=True)
    (partial / "raw").mkdir()
    (partial / "verification").mkdir()
    try:
        discovered_private, discovered_portable = discover_assets(profile)
        assets = private_assets if private_assets is not None else discovered_private
        asset_record = portable_assets if portable_assets is not None else discovered_portable
        if asset_record.get("required") and not asset_record.get("available"):
            raise FileNotFoundError(str(asset_record.get("detail", "canonical assets unavailable")))

        shapes = _parity_shapes(parity_profile, options)
        steps = _parity_steps(parity_profile, options)
        seed = parity_profile.seed if options.seed is None else options.seed
        lock = {
            "schema": "turbobench.parity-lock/v1",
            "python_minor": options.python_minor,
            "harness_dependencies": list(HARNESS_REQUIREMENTS),
            "harness_source_sha256": harness_source_hash(),
            "profile": {
                "id": parity_profile.id,
                "sha256": parity_profile_hash(parity_profile),
            },
            "providers": {
                "authority": authority.portable(),
                "candidate": candidate.portable(),
            },
            "excluded_releases": {
                key: list(value) for key, value in (excluded or {}).items()
            },
            "assets": asset_record,
            "network_disabled_after_resolution": True,
            "host": host_record(),
        }
        (partial / "profile.toml").write_text(
            parity_profile_toml(parity_profile), encoding="utf-8"
        )
        write_json(partial / "resolved-lock.json", lock)

        checks: dict[str, Any] = {}
        benchmark_gate_checks: dict[str, Any] = {}
        benchmark_gate_actions: dict[str, Any] = {}
        action_records: dict[str, Any] = {}
        require_ram = _requires_ram(profile, authority, candidate)
        snapshot_prefix, snapshot_suffix = _snapshot_window(parity_profile, steps)
        for shape in shapes:
            options.report_progress(
                f"Exact parity trace for shape {shape}: {steps} seeded transitions"
            )
            actions = parity_actions(profile, shape, steps, seed=seed)
            stream_hash = action_stream_hash(profile, actions)
            action_records[str(shape)] = {
                "kind": "seeded-random-with-directed-prefix",
                "seed": seed,
                "steps": steps,
                "sha256": stream_hash,
            }
            left_trace = _trace(
                partial,
                authority,
                profile,
                shape,
                actions,
                stream_hash,
                assets,
                side="authority",
                trace_ram=require_ram,
                snapshot_prefix_steps=snapshot_prefix,
                snapshot_suffix_steps=snapshot_suffix,
            )
            right_trace = _trace(
                partial,
                candidate,
                profile,
                shape,
                actions,
                stream_hash,
                assets,
                side="candidate",
                trace_ram=require_ram,
                snapshot_prefix_steps=snapshot_prefix,
                snapshot_suffix_steps=snapshot_suffix,
            )
            checks[str(shape)] = compare_traces(
                left_trace, right_trace, profile, require_snapshot=True
            )
            options.report_progress(
                f"Shape {shape}: {'passed' if checks[str(shape)]['passed'] else 'failed'}"
            )

            gate_actions = benchmark_actions(profile, shape, profile.correctness_steps)
            gate_hash = action_stream_hash(profile, gate_actions)
            benchmark_gate_actions[str(shape)] = {
                "steps": profile.correctness_steps,
                "sha256": gate_hash,
            }
            gate_authority = _trace(
                partial,
                authority,
                profile,
                shape,
                gate_actions,
                gate_hash,
                assets,
                side="benchmark-gate-authority",
                trace_ram=require_ram,
            )
            gate_candidate = _trace(
                partial,
                candidate,
                profile,
                shape,
                gate_actions,
                gate_hash,
                assets,
                side="benchmark-gate-candidate",
                trace_ram=require_ram,
            )
            benchmark_gate_checks[str(shape)] = compare_traces(
                gate_authority, gate_candidate, profile
            )

        distribution_check = None
        if "reset-distribution" in parity_profile.checks:
            seed_count = 256
            options.report_progress(
                f"Seeded reset distribution: {seed_count} one-lane reset samples per provider"
            )
            authority_resets = _reset_distribution(
                partial,
                authority,
                profile,
                assets,
                side="authority",
                seed_count=seed_count,
            )
            candidate_resets = _reset_distribution(
                partial,
                candidate,
                profile,
                assets,
                side="candidate",
                seed_count=seed_count,
            )
            distribution_check = compare_reset_distributions(
                authority_resets, candidate_resets
            )

        passed = (
            all(check["passed"] for check in checks.values())
            and all(check["passed"] for check in benchmark_gate_checks.values())
            and (distribution_check is None or distribution_check["passed"])
        )
        canonical_workload = (
            not options.quick
            and options.steps is None
            and options.shapes is None
            and options.seed is None
        )
        official = (
            passed
            and canonical_workload
            and authority.source_kind == "pypi"
            and authority.source_identity
            == f"pypi:{parity_profile.authority}=={parity_profile.authority_version}"
            and candidate.source_kind == "artifact"
            and not authority.diagnostic_reasons
            and not candidate.diagnostic_reasons
        )
        result = {
            "schema": PARITY_RESULT_SCHEMA,
            "passed": passed,
            "claim": {"status": "official" if official else "diagnostic"},
            "profile": {
                "id": parity_profile.id,
                "sha256": parity_profile_hash(parity_profile),
            },
            "authority": parity_profile.authority,
            "authority_present": parity_profile.authority
            in {authority.provider, candidate.provider},
            "allowed_representation_conversion": parity_profile.allowed_representation_conversion,
            "compatibility_normalization_permitted": False,
            "ram_required": require_ram,
            "snapshot_continuation_required": True,
            "providers": {
                "authority": _provider_summary(authority),
                "candidate": _provider_summary(candidate),
            },
            "checks": checks,
            "benchmark_gate": {
                "profile": profile.id,
                "profile_sha256": profile_hash(profile),
                "checks": benchmark_gate_checks,
                "actions": benchmark_gate_actions,
            },
            "reset_distribution": distribution_check,
            "actions": action_records,
            "assets": asset_record,
            "lock_sha256": canonical_json_hash(lock),
            "tool": {
                "distribution": DISTRIBUTION_NAME,
                "version": __version__,
                "source_sha256": harness_source_hash(),
            },
        }
        write_json(partial / "verification" / "parity-exact.json", result)
        write_json(partial / "result.json", result)
        _finalize_parity_manifest(partial)
        verification = verify_parity_receipt(partial)
        if not verification["passed"]:
            raise RuntimeError(
                "parity receipt failed self-verification: "
                + "; ".join(verification["errors"])
            )
        os.replace(partial, final)
        options.report_progress(f"Parity check complete: {final}")
        return final, result
    except BaseException:
        # Keep failed semantic evidence out of the final receipt path. The
        # partial directory is intentionally retained for diagnosis.
        raise


def verify_parity_receipt(
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
    if manifest.get("schema") != PARITY_MANIFEST_SCHEMA:
        errors.append("unsupported parity manifest schema")
    expected_id = canonical_json_hash({**manifest, "receipt_id": ""})
    if manifest.get("receipt_id") != expected_id:
        errors.append("parity receipt_id does not match its content")
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
        parity_profile = get_parity_profile(result["profile"]["id"])
        if result.get("schema") != PARITY_RESULT_SCHEMA:
            errors.append("unsupported parity result schema")
        if result["profile"].get("sha256") != parity_profile_hash(parity_profile):
            errors.append("receipt profile hash is inconsistent")
        if (root / "profile.toml").read_text(encoding="utf-8") != parity_profile_toml(parity_profile):
            errors.append("profile.toml does not match the built-in profile")
        if result.get("lock_sha256") != canonical_json_hash(lock):
            errors.append("resolved lock hash is inconsistent")
        checks_passed = bool(result.get("checks")) and all(
            check.get("passed") for check in result.get("checks", {}).values()
        )
        benchmark_gate = result.get("benchmark_gate", {})
        gate_checks = benchmark_gate.get("checks", {})
        gate_actions = benchmark_gate.get("actions", {})
        gate_passed = bool(gate_checks) and all(
            check.get("passed") for check in gate_checks.values()
        )
        if benchmark_gate.get("profile") != parity_profile.base_profile:
            errors.append("benchmark gate profile is inconsistent")
        base_profile = get_profile(parity_profile.base_profile)
        if benchmark_gate.get("profile_sha256") != profile_hash(base_profile):
            errors.append("benchmark gate profile hash is inconsistent")
        if set(gate_checks) != set(gate_actions):
            errors.append("benchmark gate action and check shapes differ")
        for shape_text, record in gate_actions.items():
            try:
                shape = int(shape_text)
                steps = int(record.get("steps", 0))
                expected_hash = action_stream_hash(
                    base_profile, benchmark_actions(base_profile, shape, steps)
                )
                if record.get("sha256") != expected_hash:
                    errors.append(f"benchmark gate action stream is inconsistent for shape {shape}")
                for side in ("authority", "candidate"):
                    trace = read_json(
                        root
                        / "raw"
                        / f"shape-{shape}"
                        / f"trace-benchmark-gate-{side}.json"
                    )
                    if trace.get("action_stream_sha256") != expected_hash:
                        errors.append(
                            f"benchmark gate trace action stream is inconsistent for {side} shape {shape}"
                        )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"invalid benchmark gate workload for shape {shape_text}: {exc}")
        checks_passed = checks_passed and gate_passed
        if "reset-distribution" in parity_profile.checks:
            checks_passed = checks_passed and bool(
                result.get("reset_distribution", {}).get("passed")
            )
        if bool(result.get("passed")) != checks_passed:
            errors.append("result pass flag is inconsistent with exact checks")
        if not checks_passed:
            errors.append("one or more exact semantic checks failed")
        authority = parity_profile.authority
        authority_version = parity_profile.authority_version
        provider_records = tuple(lock.get("providers", {}).values())
        authority_record = lock.get("providers", {}).get("authority", {})
        candidate_record = lock.get("providers", {}).get("candidate", {})
        if set(lock.get("providers", {})) != {"authority", "candidate"}:
            errors.append("receipt provider roles are incomplete")
        if authority_record.get("provider") != authority:
            errors.append("receipt authority role does not match the parity profile")
        if not parity_profile.accepts(str(candidate_record.get("provider", ""))):
            errors.append("receipt candidate role is not allowed by the parity profile")
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
            if result.get("claim", {}).get("status") != "official":
                errors.append("canonical receipt is not an official parity claim")
            expected_shapes = {str(shape) for shape in parity_profile.shapes}
            if set(result.get("checks", {})) != expected_shapes:
                errors.append("receipt does not contain every canonical parity shape")
            if set(gate_checks) != expected_shapes:
                errors.append("receipt does not contain every canonical benchmark-gate shape")
            action_records = result.get("actions", {})
            if set(action_records) != expected_shapes or any(
                record.get("steps") != parity_profile.steps
                or record.get("seed") != parity_profile.seed
                for record in action_records.values()
            ):
                errors.append("receipt does not use the canonical action workload")
            if any(provider.get("diagnostic_reasons") for provider in provider_records):
                errors.append("canonical receipt contains a diagnostic provider override")
            if any(
                provider.get("provider") != authority
                and provider.get("source_kind") != "artifact"
                for provider in provider_records
            ):
                errors.append("canonical receipt candidate must be an exact local distribution artifact")
            selected = candidate_record.get("selected_artifact") or {}
            if (
                selected.get("sha256") != candidate_record.get("artifact_sha256")
                or selected.get("packagetype") != "bdist_wheel"
            ):
                errors.append("canonical receipt does not bind the exact candidate wheel")
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
        errors.append(f"parity receipt is inconsistent: {exc}")
    return {
        "passed": not errors,
        "receipt_id": manifest.get("receipt_id"),
        "artifact_count": len(manifest.get("artifacts", [])),
        "errors": errors,
    }


def parity_gate_for_benchmark(
    receipt: Path,
    benchmark_profile: Profile,
    providers: tuple[ResolvedProvider, ResolvedProvider],
    shapes: tuple[int, ...],
) -> dict[str, Any]:
    """Validate whether an existing receipt can replace benchmark correctness traces."""

    verification = verify_parity_receipt(receipt)
    errors = list(verification["errors"])
    if errors:
        return {"passed": False, "errors": errors}
    root = receipt.expanduser().resolve()
    result = read_json(root / "result.json")
    lock = read_json(root / "resolved-lock.json")
    parity_profile = get_parity_profile(result["profile"]["id"])
    parity_base = get_profile(parity_profile.base_profile)
    comparable_fields = (
        "logical_environment",
        "game",
        "states",
        "semantic_actions",
        "action_table",
        "info_integer",
        "info_float",
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
    )
    if any(
        getattr(parity_base, field) != getattr(benchmark_profile, field)
        for field in comparable_fields
    ):
        errors.append("parity and benchmark profiles do not have compatible semantics")
    benchmark_gate = result.get("benchmark_gate", {})
    gate_checks = benchmark_gate.get("checks", {})
    gate_actions = benchmark_gate.get("actions", {})
    selected_shapes = {str(shape) for shape in shapes}
    if not selected_shapes.issubset(gate_checks):
        errors.append("parity receipt does not cover every benchmark lane shape")
    if benchmark_gate.get("profile_sha256") != profile_hash(parity_base):
        errors.append("parity receipt benchmark-gate profile is inconsistent")
    for shape in shapes:
        record = gate_actions.get(str(shape), {})
        recorded_steps = int(record.get("steps", 0))
        if recorded_steps < benchmark_profile.correctness_steps:
            errors.append(f"parity receipt benchmark workload is too short for shape {shape}")
            continue
        recorded_actions = benchmark_actions(parity_base, shape, recorded_steps)
        required_actions = benchmark_actions(
            benchmark_profile, shape, benchmark_profile.correctness_steps
        )
        if (
            recorded_actions[: benchmark_profile.correctness_steps].shape
            != required_actions.shape
            or recorded_actions[: benchmark_profile.correctness_steps].tobytes()
            != required_actions.tobytes()
        ):
            errors.append(f"parity and benchmark action workloads differ for shape {shape}")

    receipt_providers = tuple(lock["providers"].values())
    expected = sorted(_artifact_identity(item) for item in providers)
    recorded = sorted(_artifact_identity_payload(item) for item in receipt_providers)
    if expected != recorded:
        errors.append("parity receipt binds different provider artifacts")
    return {
        "passed": not errors,
        "errors": errors,
        "receipt_id": verification.get("receipt_id"),
        "profile": parity_profile.id,
        "checks": gate_checks,
    }


def _artifact_identity(provider: ResolvedProvider) -> tuple[str, str, str]:
    selected = provider.selected_artifact or {}
    digest = str(selected.get("sha256") or provider.artifact_sha256)
    return provider.provider, provider.version, digest


def _artifact_identity_payload(provider: dict[str, Any]) -> tuple[str, str, str]:
    selected = provider.get("selected_artifact") or {}
    digest = str(selected.get("sha256") or provider.get("artifact_sha256", ""))
    return str(provider.get("provider", "")), str(provider.get("version", "")), digest


def _finalize_parity_manifest(receipt: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": relative.as_posix(),
            "size": (receipt / relative).stat().st_size,
            "sha256": sha256_file(receipt / relative),
        }
        for relative in relative_files(receipt)
    ]
    manifest = {
        "schema": PARITY_MANIFEST_SCHEMA,
        "receipt_id": "",
        "tool": {"distribution": DISTRIBUTION_NAME, "version": __version__},
        "artifacts": artifacts,
    }
    manifest["receipt_id"] = canonical_json_hash(manifest)
    write_json(receipt / "manifest.json", manifest)
    return manifest


def _require_parity_profile(parity_profile: ParityProfile, profile: Profile) -> None:
    if parity_profile.base_profile != profile.id or not profile.native_transition_exact:
        raise ValueError(f"profile {profile.id!r} is not an exact parity profile")


def _parity_shapes(profile: ParityProfile, options: ParityOptions) -> tuple[int, ...]:
    default = profile.quick_shapes if options.quick else profile.shapes
    shapes = default if options.shapes is None else options.shapes
    if not shapes or any(shape <= 0 for shape in shapes):
        raise ValueError("parity shapes must contain positive values")
    return shapes


def _parity_steps(profile: ParityProfile, options: ParityOptions) -> int:
    default = profile.quick_steps if options.quick else profile.steps
    steps = default if options.steps is None else options.steps
    if steps <= 0:
        raise ValueError("parity steps must be positive")
    return steps


def _requires_ram(
    profile: Profile, left: ResolvedProvider, right: ResolvedProvider
) -> bool:
    if profile.logical_environment == "supermario":
        return True
    if profile.logical_environment == "breakout":
        return "env-breakoutatari2600-turbo-native" not in {
            left.provider,
            right.provider,
        }
    return False


def _snapshot_window(profile: ParityProfile, steps: int) -> tuple[int, int]:
    prefix = min(profile.snapshot_prefix_steps, max(1, steps // 2))
    suffix = min(profile.snapshot_suffix_steps, steps - prefix)
    if suffix <= 0:
        raise ValueError("parity needs at least two steps for snapshot replay")
    return prefix, suffix


def _add_diagnostic_reason(provider: ResolvedProvider, reason: str) -> ResolvedProvider:
    from dataclasses import replace

    return replace(provider, diagnostic_reasons=(*provider.diagnostic_reasons, reason))
