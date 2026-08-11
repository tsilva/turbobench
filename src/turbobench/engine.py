"""Correctness-gated alternating paired benchmark orchestration."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from turbobench import DISTRIBUTION_NAME, RESULT_SCHEMA, __version__
from turbobench.assets import discover_assets
from turbobench.bundle import finalize_manifest, verify_bundle
from turbobench.correctness import compare_replays, compare_traces
from turbobench.model import Gate, Profile, ProviderRef, ResolvedProvider
from turbobench.profiles import (
    action_stream_hash,
    benchmark_actions,
    get_profile,
    profile_hash,
    profile_toml,
    promo_action_hash,
    promo_actions,
)
from turbobench.promo import generate_media, promo_is_eligible
from turbobench.providers import load_providers
from turbobench.reporting import write_views
from turbobench.resolution import resolve_pair
from turbobench.runner_client import invoke_runner
from turbobench.runtime import (
    HARNESS_REQUIREMENTS,
    cache_root,
    harness_source_hash,
    prepare_runtime,
    runtimes_are_isolated,
)
from turbobench.stats import paired_statistics, reciprocal_statistics
from turbobench.system import host_record, wait_for_load
from turbobench.util import canonical_json_hash, read_json, redact, write_json


@dataclass(frozen=True)
class ComparisonOptions:
    promo: bool = False
    quick: bool = False
    force_busy: bool = False
    allow_dirty: bool = False
    python_minor: str = "3.14"
    steps: int | None = None
    shapes: tuple[int, ...] | None = None
    command: tuple[str, ...] = ()
    progress: Callable[[str], None] | None = dataclass_field(
        default=None, compare=False, repr=False
    )

    @property
    def diagnostic_overrides(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.quick:
            reasons.append("quick workload override")
        if self.force_busy:
            reasons.append("force-busy override")
        if self.allow_dirty:
            reasons.append("allow-dirty override")
        if self.steps is not None:
            reasons.append("benchmark step-count override")
        if self.shapes is not None:
            reasons.append("shape override")
        return tuple(reasons)

    def report_progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


def run_comparison(
    profile_id: str,
    left_ref: ProviderRef,
    right_ref: ProviderRef,
    output: Path,
    options: ComparisonOptions,
) -> tuple[Path, dict[str, Any]]:
    profile = get_profile(profile_id)
    definitions = load_providers()
    options.report_progress(f"Resolving providers for {profile.id}")
    resolution = resolve_pair(
        profile,
        left_ref,
        right_ref,
        definitions,
        python_minor=options.python_minor,
        allow_dirty=options.allow_dirty,
    )
    options.report_progress(
        f"Resolved left {resolution.left.provider} {resolution.left.version} and "
        f"right {resolution.right.provider} {resolution.right.version}"
    )
    options.report_progress(f"Preparing isolated left runtime for {resolution.left.provider}")
    left = prepare_runtime(
        resolution.left,
        checkout_path=Path(left_ref.value) if left_ref.selector == "checkout" else None,
        progress=options.progress,
    )
    options.report_progress("Left runtime ready")
    options.report_progress(f"Preparing isolated right runtime for {resolution.right.provider}")
    right = prepare_runtime(
        resolution.right,
        checkout_path=Path(right_ref.value) if right_ref.selector == "checkout" else None,
        progress=options.progress,
    )
    options.report_progress("Right runtime ready")
    return run_comparison_resolved(
        profile,
        left,
        right,
        output,
        options,
        excluded=resolution.excluded,
    )


def run_comparison_resolved(
    profile: Profile,
    left: ResolvedProvider,
    right: ResolvedProvider,
    output: Path,
    options: ComparisonOptions,
    *,
    excluded: dict[str, tuple[dict[str, str], ...]] | None = None,
    private_assets: dict[str, Any] | None = None,
    portable_assets: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    final = output.expanduser().resolve()
    partial = final.with_name(final.name + ".partial")
    if final.exists():
        raise FileExistsError(f"output bundle already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)
    for directory in ("raw", "verification", "media"):
        (partial / directory).mkdir(exist_ok=True)
    discovered_private, discovered_portable = discover_assets(profile)
    assets = private_assets if private_assets is not None else discovered_private
    asset_record = portable_assets if portable_assets is not None else discovered_portable
    lock = _resolved_lock(profile, left, right, excluded or {}, asset_record, options)
    run_key = canonical_json_hash(
        {
            "lock": lock,
            "options": {
                "quick": options.quick,
                "steps": options.steps,
                "shapes": options.shapes,
            },
        }
    )
    journal_path = partial / "raw" / "journal.json"
    if journal_path.is_file():
        journal = read_json(journal_path)
        if journal.get("run_key") != run_key:
            raise RuntimeError(f"existing partial bundle belongs to another run: {partial}")
    else:
        journal = {"schema": "turbobench.journal/v1", "run_key": run_key, "completed": []}
        write_json(journal_path, journal)
    (partial / "profile.toml").write_text(profile_toml(profile), encoding="utf-8")
    write_json(partial / "resolved-lock.json", lock)

    shapes = _selected_shapes(profile, options)
    step_count = _selected_steps(profile, options)
    options.report_progress(
        f"Starting {profile.id}: shapes {', '.join(map(str, shapes))}, "
        f"{step_count} benchmark steps"
    )
    correctness: dict[str, Any] = {}
    action_records: dict[str, Any] = {}
    for shape in shapes:
        trace_actions = benchmark_actions(profile, shape, profile.correctness_steps)
        trace_hash = action_stream_hash(profile, trace_actions)
        action_records[str(shape)] = {
            "correctness_sha256": trace_hash,
            "correctness_steps": profile.correctness_steps,
        }
        options.report_progress(f"Correctness trace for shape {shape}: left provider")
        left_trace = _trace(
            partial, left, profile, shape, trace_actions, trace_hash, assets, side="left"
        )
        options.report_progress(f"Correctness trace for shape {shape}: right provider")
        right_trace = _trace(
            partial, right, profile, shape, trace_actions, trace_hash, assets, side="right"
        )
        correctness[str(shape)] = compare_traces(left_trace, right_trace, profile)
        status = "passed" if correctness[str(shape)]["passed"] else "failed"
        options.report_progress(f"Correctness for shape {shape}: {status}")
    write_json(
        partial / "verification" / "correctness.json",
        {"schema": "turbobench.correctness/v1", "shapes": correctness},
    )

    options.report_progress("Checking system load")
    load = wait_for_load(
        timeout_seconds=0 if options.quick else 900,
        force_busy=options.force_busy or options.quick,
        progress=options.progress,
    )
    options.report_progress(f"System-load gate: {'passed' if load.get('passed') else 'failed'}")
    comparison_shapes: dict[str, Any] = {}
    pair_count = 2 if options.quick else 7
    for shape in shapes:
        actions = benchmark_actions(profile, shape, step_count)
        stream_hash = action_stream_hash(profile, actions)
        action_records[str(shape)].update(
            {"benchmark_sha256": stream_hash, "benchmark_steps": step_count}
        )
        shape_dir = partial / "raw" / f"shape-{shape}"
        shape_dir.mkdir(parents=True, exist_ok=True)
        options.report_progress(f"Benchmarking shape {shape}: warmup pair")
        _warmup_pair(
            shape_dir,
            left,
            right,
            profile,
            shape,
            actions,
            stream_hash,
            assets,
            progress=options.progress,
        )
        pairs: list[dict[str, Any]] = []
        for pair_index in range(pair_count):
            order = ("left", "right") if pair_index % 2 == 0 else ("right", "left")
            order_label = "AB" if order == ("left", "right") else "BA"
            responses: dict[str, dict[str, Any]] = {}
            for side in order:
                provider = left if side == "left" else right
                response_path = shape_dir / f"pair-{pair_index + 1:02d}-{side}.json"
                if response_path.is_file():
                    options.report_progress(
                        f"Shape {shape}, pair {pair_index + 1}/{pair_count} "
                        f"({order_label}): reusing {side} evidence"
                    )
                    response = read_json(response_path)
                else:
                    options.report_progress(
                        f"Shape {shape}, pair {pair_index + 1}/{pair_count} "
                        f"({order_label}): running {side} provider"
                    )
                    response = _benchmark_invocation(
                        shape_dir,
                        provider,
                        profile,
                        shape,
                        actions,
                        stream_hash,
                        assets,
                        label=f"pair-{pair_index + 1:02d}-{side}",
                    )
                    write_json(response_path, response)
                responses[side] = response
            pairs.append(
                {
                    "pair": pair_index + 1,
                    "order": order_label,
                    "left_sps": responses["left"]["sps"],
                    "right_sps": responses["right"]["sps"],
                    "left_invocation": f"pair-{pair_index + 1:02d}-left.json",
                    "right_invocation": f"pair-{pair_index + 1:02d}-right.json",
                }
            )
            journal["completed"].append(f"shape-{shape}/pair-{pair_index + 1:02d}")
            write_json(journal_path, journal)
        write_json(
            shape_dir / "pairs.json",
            {
                "schema": "turbobench.pairs/v1",
                "shape": shape,
                "action_stream_sha256": stream_hash,
                "pairs": pairs,
            },
        )
        comparison_shapes[str(shape)] = {
            "correctness": correctness[str(shape)],
            "statistics": paired_statistics(
                pairs, require_official_design=not options.quick
            ),
        }
        options.report_progress(
            f"Shape {shape} complete: "
            f"{comparison_shapes[str(shape)]['statistics']['outcome']}"
        )

    write_json(
        partial / "verification" / "order-reversal.json",
        {
            "schema": "turbobench.order-reversal/v1",
            "providers": {"left": right.provider, "right": left.provider},
            "shapes": {
                str(shape): {
                    "raw_evidence_reused": True,
                    "source_pairs_sha256": canonical_json_hash(
                        read_json(partial / "raw" / f"shape-{shape}" / "pairs.json")
                    ),
                    "statistics": reciprocal_statistics(
                        comparison_shapes[str(shape)]["statistics"]
                    ),
                }
                for shape in shapes
            },
        },
    )

    gates = _validity_gates(
        profile,
        shapes,
        pair_count,
        left,
        right,
        correctness,
        load,
        asset_record,
        options,
    )
    validity_passed = all(gate.passed for gate in gates)
    diagnostic_reasons = [gate.detail for gate in gates if not gate.passed]
    diagnostic_reasons.extend(left.diagnostic_reasons)
    diagnostic_reasons.extend(right.diagnostic_reasons)
    diagnostic_reasons.extend(options.diagnostic_overrides)
    claim_status = "official" if validity_passed and not diagnostic_reasons else "diagnostic"
    shape_one = comparison_shapes.get("1")
    headline_outcome = (
        shape_one["statistics"]["outcome"] if shape_one else "inconclusive"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "profile": {"id": profile.id, "sha256": profile_hash(profile)},
        "lock_sha256": canonical_json_hash(lock),
        "validity": {
            "passed": validity_passed,
            "gates": [gate.__dict__ for gate in gates],
        },
        "claim": {
            "status": claim_status,
            "diagnostic_reasons": sorted(set(diagnostic_reasons)),
        },
        "comparison": {
            "left": _provider_summary(left),
            "right": _provider_summary(right),
            "headline_shape": 1,
            "outcome": headline_outcome,
            "shapes": comparison_shapes,
        },
        "promo": {"requested": options.promo, "eligible": False, "generated": False},
        "actions": action_records,
        "system": {"host": host_record(), "load": load},
        "assets": asset_record,
        "commands": [" ".join(redact(list(options.command)))] if options.command else [],
        "tool": {
            "distribution": DISTRIBUTION_NAME,
            "version": __version__,
            "source_sha256": harness_source_hash(),
        },
    }

    replay_temp = partial / ".replay-frames"
    if options.promo and validity_passed and claim_status == "official" and headline_outcome != "inconclusive":
        replay_temp.mkdir(exist_ok=True)
        replay_actions = promo_actions(profile)
        replay_hash = promo_action_hash(profile, replay_actions)
        options.report_progress("Replaying promotional trajectory: left provider")
        left_replay, left_frames = _promo_replay(
            partial, replay_temp, left, profile, replay_actions, replay_hash, assets, "left"
        )
        options.report_progress("Replaying promotional trajectory: right provider")
        right_replay, right_frames = _promo_replay(
            partial, replay_temp, right, profile, replay_actions, replay_hash, assets, "right"
        )
        replay_gate = compare_replays(left_replay, right_replay, profile)
        write_json(
            partial / "verification" / "promo-replay.json",
            {"schema": "turbobench.promo-verification/v1", "gate": replay_gate, "left": left_replay, "right": right_replay},
        )
        result["promo"]["replay_gate"] = replay_gate
        result["promo"]["eligible"] = promo_is_eligible(result, replay_gate)
        if result["promo"]["eligible"]:
            options.report_progress("Generating promotional MP4 and GIF")
            generate_media(
                partial,
                result,
                lock,
                left_replay,
                right_replay,
                left_frames,
                right_frames,
                diagnostic=False,
            )
            result["promo"]["generated"] = True
    elif options.promo:
        result["promo"]["refused_reason"] = "benchmark is invalid, diagnostic, or inconclusive"
        options.report_progress("Promotional media skipped: benchmark is not eligible")
    shutil.rmtree(replay_temp, ignore_errors=True)

    options.report_progress("Writing reports and manifest")
    write_json(partial / "result.json", result)
    write_views(partial, result)
    finalize_manifest(partial)
    options.report_progress("Self-verifying result bundle")
    verification = verify_bundle(partial)
    if not verification["passed"]:
        raise RuntimeError("final bundle failed self-verification: " + "; ".join(verification["errors"]))
    os.replace(partial, final)
    options.report_progress(f"Comparison complete: {final}")
    return final, result


def _resolved_lock(
    profile: Profile,
    left: ResolvedProvider,
    right: ResolvedProvider,
    excluded: dict[str, tuple[dict[str, str], ...]],
    assets: dict[str, Any],
    options: ComparisonOptions,
) -> dict[str, Any]:
    return {
        "schema": "turbobench.resolved-lock/v1",
        "python_minor": options.python_minor,
        "harness_dependencies": list(HARNESS_REQUIREMENTS),
        "harness_source_sha256": harness_source_hash(),
        "profile": {"id": profile.id, "sha256": profile_hash(profile)},
        "providers": {"left": left.portable(), "right": right.portable()},
        "excluded_releases": {key: list(value) for key, value in excluded.items()},
        "assets": assets,
        "network_disabled_after_resolution": True,
    }


def _selected_shapes(profile: Profile, options: ComparisonOptions) -> tuple[int, ...]:
    if options.shapes is not None:
        if not options.shapes or any(shape <= 0 for shape in options.shapes):
            raise ValueError("shapes must contain positive values")
        return options.shapes
    return (1,) if options.quick else profile.shapes


def _selected_steps(profile: Profile, options: ComparisonOptions) -> int:
    value = options.steps if options.steps is not None else (100 if options.quick else profile.benchmark_steps)
    if value <= 0:
        raise ValueError("benchmark steps must be positive")
    return value


def _base_request(
    provider: ResolvedProvider,
    profile: Profile,
    shape: int,
    assets: dict[str, Any],
) -> dict[str, Any]:
    speed = 1.0
    if provider.adapter == "fake":
        with suppress(IndexError, ValueError):
            speed = float(provider.source_identity.rsplit(":", 1)[1])
    return {
        "provider": provider.provider,
        "adapter": provider.adapter,
        "distribution": provider.distribution,
        "profile": profile.id,
        "shape": shape,
        "assets": assets,
        "fake_speed": speed,
        "seed": 123,
    }


def _trace(
    bundle: Path,
    provider: ResolvedProvider,
    profile: Profile,
    shape: int,
    actions: Any,
    stream_hash: str,
    assets: dict[str, Any],
    *,
    side: str,
) -> dict[str, Any]:
    path = bundle / "raw" / f"shape-{shape}" / f"trace-{side}.json"
    if path.is_file():
        return read_json(path)
    request = {
        **_base_request(provider, profile, shape, assets),
        "operation": "trace",
        "actions": actions.tolist(),
        "action_stream_sha256": stream_hash,
    }
    response = invoke_runner(
        provider,
        request,
        log_path=bundle / "raw" / f"shape-{shape}" / f"trace-{side}.log",
    )
    write_json(path, response)
    return response


def _benchmark_invocation(
    shape_dir: Path,
    provider: ResolvedProvider,
    profile: Profile,
    shape: int,
    actions: Any,
    stream_hash: str,
    assets: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    request = {
        **_base_request(provider, profile, shape, assets),
        "operation": "benchmark",
        "actions": actions.tolist(),
        "action_stream_sha256": stream_hash,
        "warmup_steps": min(500, len(actions)),
    }
    return invoke_runner(provider, request, log_path=shape_dir / f"{label}.log")


def _warmup_pair(
    shape_dir: Path,
    left: ResolvedProvider,
    right: ResolvedProvider,
    profile: Profile,
    shape: int,
    actions: Any,
    stream_hash: str,
    assets: dict[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    for side, provider in (("left", left), ("right", right)):
        path = shape_dir / f"warmup-{side}.json"
        if path.is_file():
            if progress is not None:
                progress(f"Shape {shape} warmup: reusing {side} evidence")
            continue
        if progress is not None:
            progress(f"Shape {shape} warmup: running {side} provider")
        response = _benchmark_invocation(
            shape_dir,
            provider,
            profile,
            shape,
            actions,
            stream_hash,
            assets,
            label=f"warmup-{side}",
        )
        write_json(path, response)


def _validity_gates(
    profile: Profile,
    shapes: tuple[int, ...],
    pair_count: int,
    left: ResolvedProvider,
    right: ResolvedProvider,
    correctness: dict[str, Any],
    load: dict[str, Any],
    assets: dict[str, Any],
    options: ComparisonOptions,
) -> list[Gate]:
    harness_left = {line for line in left.installed_lock if line.casefold().startswith(("gymnasium==", "numpy=="))}
    harness_right = {line for line in right.installed_lock if line.casefold().startswith(("gymnasium==", "numpy=="))}
    return [
        Gate("compatible profile pair", profile.compatible(left.provider, right.provider) or left.adapter == right.adapter == "fake", f"{left.provider} versus {right.provider}"),
        Gate("canonical assets", not assets.get("required") or bool(assets.get("available")), assets.get("detail", "canonical digests recorded")),
        Gate("isolated runtimes", runtimes_are_isolated(left, right) or left.adapter == right.adapter == "fake", "separate content-addressed Python environments"),
        Gate("common Python minor", left.python_minor == right.python_minor == options.python_minor, options.python_minor),
        Gate("common harness lock", harness_left == harness_right, ", ".join(sorted(harness_left or harness_right)) or "fake harness"),
        Gate("eligible exact artifacts", not left.diagnostic_reasons and not right.diagnostic_reasons, "no dirty/quarantined/relaxed provider artifacts"),
        Gate("correctness at every shape", all(item["passed"] for item in correctness.values()), ", ".join(f"{shape}={'pass' if item['passed'] else 'fail'}" for shape, item in correctness.items())),
        Gate("official sample design", shapes == profile.shapes and pair_count == 7, "shapes 1/16/32; one warmup pair; seven alternating pairs; three repetitions"),
        Gate("system load", bool(load.get("passed")), f"one-minute load below {load.get('threshold')}; forced={load.get('forced')}"),
        Gate("official host platform", bool(host_record()["official_v1_platform"]), "Apple-silicon macOS or x86-64 Linux"),
        Gate("offline measurement", True, "network disabled in correctness, timing, and replay subprocesses"),
        Gate("no diagnostic overrides", not options.diagnostic_overrides, ", ".join(options.diagnostic_overrides) or "none"),
    ]


def _provider_summary(provider: ResolvedProvider) -> dict[str, Any]:
    return {
        "provider": provider.provider,
        "version": provider.version,
        "adapter": provider.adapter,
        "artifact_sha256": provider.artifact_sha256,
        "source_identity": provider.source_identity,
        "runtime_id": provider.runtime_id,
        "compatibility_lineage": provider.compatibility_lineage,
    }


def _promo_replay(
    bundle: Path,
    temporary: Path,
    provider: ResolvedProvider,
    profile: Profile,
    actions: tuple[tuple[str, ...], ...],
    stream_hash: str,
    assets: dict[str, Any],
    side: str,
) -> tuple[dict[str, Any], Path]:
    frames = temporary / f"{side}.rgb"
    request = {
        **_base_request(provider, profile, 1, assets),
        "operation": "promo",
        "frame_skip": 1,
        "promo_actions": actions,
        "promo_action_sha256": stream_hash,
        "output_frames": str(frames),
    }
    response = invoke_runner(
        provider,
        request,
        log_path=bundle / "raw" / f"promo-{side}.log",
    )
    return response, frames


def generate_promo_for_bundle(
    bundle: Path,
    *,
    diagnostic: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    source = bundle.expanduser().resolve()
    report("Verifying source bundle")
    integrity = verify_bundle(source)
    if not integrity["passed"]:
        raise ValueError("bundle integrity verification failed: " + "; ".join(integrity["errors"]))
    result = read_json(source / "result.json")
    lock = read_json(source / "resolved-lock.json")
    profile = get_profile(result["profile"]["id"])
    left = _rehydrate_provider(lock["providers"]["left"])
    right = _rehydrate_provider(lock["providers"]["right"])
    private_assets, _portable_assets = discover_assets(profile)
    staging = source.with_name(source.name + ".promo.partial")
    backup = source.with_name(source.name + ".pre-promo-backup")
    if staging.exists() or backup.exists():
        raise FileExistsError("promo staging or backup path already exists")
    shutil.copytree(source, staging)
    try:
        with __import__("tempfile").TemporaryDirectory(prefix="turbobench-promo-") as raw_temp:
            temporary = Path(raw_temp)
            actions = promo_actions(profile)
            stream_hash = promo_action_hash(profile, actions)
            report("Replaying promotional trajectory: left provider")
            left_replay, left_frames = _promo_replay(
                staging, temporary, left, profile, actions, stream_hash, private_assets, "left"
            )
            report("Replaying promotional trajectory: right provider")
            right_replay, right_frames = _promo_replay(
                staging, temporary, right, profile, actions, stream_hash, private_assets, "right"
            )
            replay_gate = compare_replays(left_replay, right_replay, profile)
            result["promo"]["requested"] = True
            result["promo"]["replay_gate"] = replay_gate
            result["promo"]["eligible"] = promo_is_eligible(result, replay_gate)
            if not diagnostic and not result["promo"]["eligible"]:
                raise ValueError("bundle is invalid, inconclusive, or replay-incompatible; use --diagnostic for watermarked media")
            write_json(
                staging / "verification" / "promo-replay.json",
                {"schema": "turbobench.promo-verification/v1", "gate": replay_gate, "left": left_replay, "right": right_replay},
            )
            report("Generating promotional MP4 and GIF")
            generate_media(
                staging,
                result,
                lock,
                left_replay,
                right_replay,
                left_frames,
                right_frames,
                diagnostic=diagnostic,
            )
            result["promo"]["generated"] = True
            result["promo"]["diagnostic_watermark"] = diagnostic
        write_json(staging / "result.json", result)
        write_views(staging, result)
        (staging / "manifest.json").unlink(missing_ok=True)
        finalize_manifest(staging)
        report("Self-verifying updated bundle")
        staged_integrity = verify_bundle(staging)
        if not staged_integrity["passed"]:
            raise RuntimeError("promoted bundle failed verification: " + "; ".join(staged_integrity["errors"]))
        os.replace(source, backup)
        try:
            os.replace(staging, source)
        except BaseException:
            os.replace(backup, source)
            raise
        shutil.rmtree(backup)
        report(f"Promotional media complete: {source}")
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _rehydrate_provider(payload: dict[str, Any]) -> ResolvedProvider:
    values = dict(payload)
    for field in ("release_files", "diagnostic_reasons", "installed_lock"):
        values[field] = tuple(values.get(field, ()))
    identifier = values.get("runtime_id")
    if values.get("source_kind") == "fake":
        values["runtime_python"] = sys.executable
    elif identifier:
        root = cache_root() / "runtimes" / identifier
        values["runtime_python"] = str(root / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    provider = ResolvedProvider(**values)
    if not provider.runtime_python or not Path(provider.runtime_python).is_file():
        raise FileNotFoundError(f"locked runtime is unavailable for {provider.provider}; rerun compare to resolve it")
    return provider
