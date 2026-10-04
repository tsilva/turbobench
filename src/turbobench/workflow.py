"""Two-host comparison coordinator: immutable measurement first, local media second."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from turbobench.assets import discover_assets
from turbobench.correctness import compare_replays
from turbobench.profiles import profile_hash, promo_action_hash
from turbobench.proofs import finalize_proof, require_proof, validate_document
from turbobench.runtime import harness_source_hash, prepare_runtime
from turbobench.system import host_record
from turbobench.util import canonical_json_hash, read_json, write_json
from turbobench.workloads import request_profile


def machine_identity() -> dict[str, Any]:
    """Hash a hardware/OS installation identifier, never an SSH alias or hostname."""
    path = Path("/etc/machine-id")
    if path.is_file():
        identity = path.read_text().strip()
    elif __import__("platform").system() == "Darwin":
        raw = subprocess.check_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True)
        found = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', raw)
        identity = found.group(1) if found else ""
    else:
        identity = ""
    if not identity:
        raise RuntimeError("cannot verify a stable machine identity for host separation")
    return {
        "machine_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "hardware": host_record(),
    }


def validate_request(request: dict[str, Any]) -> None:
    validate_document(request)
    if request["request_id"] != canonical_json_hash({**request, "request_id": ""}):
        raise ValueError("request ID mismatch")
    profile = request_profile(request)
    if request["schema"] in {
        "turbobench.comparison-request/v2",
        "turbobench.comparison-request/v3",
    }:
        workload = request["resolved_workload"]
        if (
            set(workload["declarations"]) != {"left", "right"}
            or workload["policy_contract"] != request["policy_contract"]
        ):
            raise ValueError("request is missing locked provider declarations or policy contract")
    if request["schema"] == "turbobench.comparison-request/v3":
        if (
            profile.action_stream_version != "captured-policy/v1"
            or request["actions"] != request["resolved_workload"]["policy_actions"]
        ):
            raise ValueError("policy measurement actions differ from locked workload")
    elif profile.action_stream_version == "captured-policy/v1":
        raise ValueError("policy timing requires comparison-request/v3")
    contract = request["policy_contract"]
    if contract["frame_skip"] != profile.frame_skip or contract[
        "action_sha256"
    ] != canonical_json_hash(request["actions"]):
        raise ValueError("request policy cadence/actions mismatch")
    if not request["actions"] or not request["render_machine"].get("machine_sha256"):
        raise ValueError("request is missing actions or a rendering machine identity")
    from turbobench.providers import load_providers, parse_provider_ref

    definitions = load_providers()
    refs = [parse_provider_ref(request[side], definitions) for side in ("left", "right")]
    if any(ref.selector not in {"latest", "version"} for ref in refs):
        raise ValueError(
            "two-host showcases require package releases; host-local checkout/artifact refs are unsupported"
        )
    if refs[0].provider != profile.authority or not profile.compatible(
        refs[0].provider, refs[1].provider
    ):
        raise ValueError(
            "showcase requires the upstream authority on the left and a compatible candidate on the right"
        )
    if "resolved_workload" in request:
        for side, ref in zip(("left", "right"), refs, strict=True):
            provider = request["resolved_workload"]["declarations"][side]["provider"]
            if ref.selector != "version" or (ref.provider, ref.value) != (
                provider["provider"],
                provider["version"],
            ):
                raise ValueError(
                    "resolved workload declaration differs from pinned provider request"
                )


def replay_pair(
    root: Path,
    temporary: Path,
    profile: Any,
    providers: dict[str, Any],
    actions: list[Any],
    *,
    record_frames: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    from turbobench.engine import _contract_attestation, _promo_replay

    private, portable = discover_assets(profile)
    stream = tuple(tuple(action) for action in actions)
    digest = promo_action_hash(profile, stream)
    records, paths = {}, {}
    for side, provider in providers.items():
        attestation = _contract_attestation(
            root, provider, profile, 1, private, portable, side=f"showcase-{side}", frame_skip=1
        )
        record, path = _promo_replay(
            root,
            temporary,
            provider,
            profile,
            stream,
            digest,
            private,
            portable,
            attestation,
            side,
            record_frames=record_frames,
        )
        if record["frame_count"] != len(actions) + 1 or len(record["transitions"]) != len(actions):
            raise ValueError(f"{side} replay ended before the locked excerpt finished")
        records[side], paths[side] = record, path
    gate = compare_replays(records["left"], records["right"], profile)
    if not gate["passed"]:
        raise ValueError(f"policy excerpt replay mismatch: {gate['first_mismatches'][:3]}")
    records["gate"] = gate
    write_json(root / "verification" / "showcase-replay.json", records)
    return records, paths


def finalize_measurement(
    root: Path,
    result: dict[str, Any],
    lock: dict[str, Any],
    left: Any,
    right: Any,
    profile: Any,
    assets: Any,
    portable: Any,
    options: Any,
) -> None:
    request = options.workflow_request
    if request is None:
        raise ValueError("measurement-only runs require a bound workflow request")
    validate_request(request)
    identity = machine_identity()
    if identity["machine_sha256"] == request["render_machine"]["machine_sha256"]:
        raise ValueError("benchmark and rendering must run on different machines")
    if harness_source_hash() != request["harness_sha256"]:
        raise ValueError("remote harness differs from the staged source")
    write_json(root / "request.json", request)
    write_json(root / "benchmark-machine.json", identity)
    with tempfile.TemporaryDirectory(prefix="turbobench-replay-") as temporary:
        replay_pair(
            root,
            Path(temporary),
            profile,
            {"left": left, "right": right},
            request["actions"],
            record_frames=False,
        )
    bindings = {
        "request_id": request["request_id"],
        "policy_id": request["policy_id"],
        "benchmark_machine": identity,
        "mode": "smoke" if options.smoke else "full",
    }
    version = 2 if "resolved_workload" in request else 1
    if request["schema"] == "turbobench.comparison-request/v3":
        version = 3
    if version >= 2:
        bindings["workload_sha256"] = canonical_json_hash(request["resolved_workload"])
    finalize_proof(
        root,
        f"turbobench.benchmark-proof/v{version}",
        bindings,
    )


def replay_preflight(root: Path, side: str, record: Any) -> dict[str, Any]:
    digest = record["lifecycle"]["contract_attestation_sha256"]
    matches = [
        read_json(path)
        for path in (root / "verification" / "attestations").glob(f"showcase-{side}-*.json")
    ]
    selected = [
        item for item in matches if item["contract_attestation"]["attestation_sha256"] == digest
    ]
    if len(selected) != 1:
        raise ValueError("replay successful preflight evidence is missing or ambiguous")
    return selected[0]


def verify_sampling(root: Path, request: dict[str, Any], result: Any) -> None:
    profile = request_profile(request)
    smoke = request["smoke"]
    from turbobench.scaling import scaling_rule, verify_scaling

    shapes = (
        (1, 2)
        if smoke
        else (
            verify_scaling(profile, result) if scaling_rule(profile) else profile.measurement_shapes
        )
    )
    if request["schema"] == "turbobench.comparison-request/v3":
        from turbobench.profiles import action_stream_hash, canonical_actions
        from turbobench.workloads import policy_benchmark_settings

        for shape in shapes:
            digest = action_stream_hash(profile, canonical_actions(profile, shape))
            for path in (root / "raw" / f"shape-{shape}").glob("*.json"):
                if path.name == "pairs.json":
                    continue
                record = read_json(path)
                if (
                    record.get("policy_reset") != policy_benchmark_settings(profile)
                    or record.get("action_stream_sha256") != digest
                ):
                    raise ValueError("policy timing/trace reset/action commitment mismatch")
    pair_count = 1 if smoke else profile.full_pairs
    repetitions = 1 if smoke else 3
    warmups = 0 if smoke else profile.warmup_pairs
    expected = {
        "mode": "smoke" if smoke else "full",
        "pairs": pair_count,
        "repetitions": repetitions,
        "warmup_pairs": warmups,
    }
    if result.get("sampling") != expected or set(result["comparison"]["shapes"]) != set(
        map(str, shapes)
    ):
        raise ValueError("measurement sampling differs from the requested mode/profile")
    for shape in shapes:
        directory = root / "raw" / f"shape-{shape}"
        pairs = read_json(directory / "pairs.json")["pairs"]
        if (
            len(pairs) != pair_count
            or len(list(directory.glob("pair-*.json"))) != pair_count * 2
            or len(list(directory.glob("warmup*.json"))) != warmups * 2
        ):
            raise ValueError("measurement invocation count mismatch")
        for i, pair in enumerate(pairs, 1):
            if pair["pair"] != i or pair["order"] != ("AB" if i % 2 else "BA"):
                raise ValueError("measurement pairing/order mismatch")
            for side in ("left", "right"):
                name = f"pair-{i:02d}-{side}.json"
                invocation = read_json(directory / name)
                if (
                    pair[f"{side}_invocation"] != name
                    or pair[f"{side}_sps"] != invocation["sps"]
                    or invocation["repetitions"] != repetitions
                    or len(invocation["sps"]) != repetitions
                    or invocation["steps"] != profile.measurement_steps
                    or invocation["warmup_steps"]
                    != (0 if smoke else min(500, profile.measurement_steps))
                ):
                    raise ValueError("paired statistics are not bound to raw timing invocations")


def verify_measurement_replay(root: Path, request: dict[str, Any]) -> None:
    from turbobench.engine import _require_evidence_binding

    profile = request_profile(request)
    lock = read_json(root / "resolved-lock.json")
    if profile.resolved_workload is not None:
        from turbobench.workloads import check_declaration

        for side in ("left", "right"):
            declaration = read_json(
                root / "verification" / "provider-declarations" / f"{side}.json"
            )
            check_declaration(
                declaration,
                profile.resolved_workload["configuration"],
                profile.resolved_workload["definition"]["assets"],
            )
            if declaration["provider"] != lock["providers"][side]:
                raise ValueError("measurement declaration differs from actual artifact lock")
            expected = profile.resolved_workload["declarations"][side]["provider"]
            if any(
                declaration["provider"][k] != expected[k]
                for k in ("provider", "version", "source_identity")
            ):
                raise ValueError("measurement uses a different release from resolved workload")
    records = read_json(root / "verification" / "showcase-replay.json")
    digest = promo_action_hash(profile, tuple(tuple(a) for a in request["actions"]))
    for side in ("left", "right"):
        record = records[side]
        if (
            record["action_stream_sha256"] != digest
            or record["frame_count"] != len(request["actions"]) + 1
            or len(record["frame_sha256"]) != record["frame_count"]
            or len(record["transitions"]) != len(request["actions"])
        ):
            raise ValueError("remote replay action/frame commitment mismatch")
        # Replay records bind the successful raw-cadence preflight saved with them.
        preflight = replay_preflight(root, side, record)
        _require_evidence_binding(
            record, preflight["execution_spec"], preflight["contract_attestation"]
        )
    actual = compare_replays(records["left"], records["right"], profile)
    if not actual["passed"] or actual != records["gate"]:
        raise ValueError("remote replay parity gate is invalid")
    if (
        read_json(root / "benchmark-machine.json")["machine_sha256"]
        == request["render_machine"]["machine_sha256"]
    ):
        raise ValueError("benchmark and rendering host identities are the same")
    for side in ("left", "right"):
        provider = lock["providers"][side]
        from turbobench.providers import load_providers, parse_provider_ref

        ref = parse_provider_ref(request[side], load_providers())
        if provider["provider"] != ref.provider or (
            ref.selector == "version" and provider["version"] != ref.value
        ):
            raise ValueError("benchmark provider differs from requested release")


def measurement_worker(request_path: Path, output: Path) -> None:
    from turbobench.engine import ComparisonOptions, run_comparison
    from turbobench.providers import load_providers, parse_provider_ref

    request = read_json(request_path)
    validate_request(request)
    if machine_identity()["machine_sha256"] == request["render_machine"]["machine_sha256"]:
        raise ValueError("benchmark and rendering must run on different machines")
    definitions = load_providers()
    run_comparison(
        request_profile(request),
        parse_provider_ref(request["left"], definitions),
        parse_provider_ref(request["right"], definitions),
        output,
        ComparisonOptions(
            smoke=request["smoke"],
            measurement_only=True,
            workflow_request=request,
            python_minor=request["python_minor"],
            progress=lambda message: print(message, flush=True),
        ),
    )


def pipeline_gates(result: dict[str, Any], smoke: bool) -> None:
    expected = (
        {
            "official sample design",
            "no diagnostic overrides",
            "system load",
            "eligible exact artifacts",
        }
        if smoke
        else set()
    )
    failed = [
        g["name"]
        for g in result["validity"]["gates"]
        if not g["passed"] and g["name"] not in expected
    ]
    if failed:
        raise ValueError("benchmark gates failed: " + ", ".join(failed))
    if not smoke and (
        result["claim"]["status"] != "official" or result["comparison"]["outcome"] == "inconclusive"
    ):
        raise ValueError("full showcase requires official, conclusive benchmark evidence")


def render_showcase(root: Path, progress: Any = print) -> dict[str, Any]:
    from turbobench.providers import load_providers, parse_provider_ref
    from turbobench.reporting import write_views
    from turbobench.resolution import resolve_pair
    from turbobench.showcase import COMPARISON_STYLE, generate_showcase_assets, scaling_chart

    benchmark = require_proof(root / "benchmark")
    policy = require_proof(root / "policy")
    request = read_json(root / "benchmark" / "request.json")
    identity = machine_identity()
    if identity != request["render_machine"]:
        raise ValueError("rendering host differs from the locked request")
    result = read_json(root / "benchmark" / "result.json")
    write_views(root, result)
    chart_diagnostic = (
        request["smoke"]
        or result["claim"]["status"] != "official"
        or result["comparison"]["outcome"] == "inconclusive"
    )
    (root / "chart.svg").write_text(scaling_chart(result, diagnostic=chart_diagnostic))
    pipeline_gates(result, request["smoke"])
    if policy["proof_id"] != benchmark["bindings"]["policy_id"]:
        raise ValueError("measurement/policy binding mismatch")
    profile = request_profile(request)
    lock = read_json(root / "benchmark" / "resolved-lock.json")
    definitions = load_providers()
    refs = [
        parse_provider_ref(
            f"{lock['providers'][s]['provider']}@{lock['providers'][s]['version']}", definitions
        )
        for s in ("left", "right")
    ]
    resolution = resolve_pair(profile, *refs, definitions, python_minor=request["python_minor"])
    providers = {
        s: prepare_runtime(p, cache_context=profile_hash(profile), progress=progress)
        for s, p in zip(("left", "right"), (resolution.left, resolution.right), strict=True)
    }
    for side, provider in providers.items():
        remote = lock["providers"][side]
        if (provider.provider, provider.version, provider.source_identity) != (
            remote["provider"],
            remote["version"],
            remote["source_identity"],
        ):
            raise ValueError("render runtime release/source identity differs from benchmark")
    write_json(root / "render-lock.json", {s: p.portable() for s, p in providers.items()})
    with tempfile.TemporaryDirectory(prefix="turbobench-render-") as temporary:
        progress("Replaying the policy excerpt on the rendering host")
        records, paths = replay_pair(
            root, Path(temporary), profile, providers, request["actions"], record_frames=True
        )
        remote_records = read_json(root / "benchmark" / "verification" / "showcase-replay.json")
        for side in ("left", "right"):
            if not compare_replays(records[side], remote_records[side], profile)["passed"]:
                raise ValueError(f"{side}: cross-host replay commitments differ")
        progress("Rendering full-resolution MP4, lossless WebP and scaling chart")
        assets = generate_showcase_assets(
            root, result, profile, records, paths, diagnostic=request["smoke"]
        )
    from copy import deepcopy

    view = deepcopy(result)
    view["promo"] = {"requested": True, "eligible": not request["smoke"], "generated": True}
    write_views(root, view)
    (root / "chart.svg").write_text(scaling_chart(result, diagnostic=request["smoke"]))
    contract = policy["bindings"]["contract"]
    with (root / "report.md").open("a") as report:
        report.write(
            f"\n## Policy and showcase\n\nCheckpoint: `{contract['checkpoint_sha256']}`; step {contract['checkpoint_step']}.\n\nTracking: {contract['mlflow_url']}\n\nLimitations: {contract['limitations']}\n\nExcerpt: {contract['decisions']} / {contract['total_captured_decisions']} captured decisions.\n\n"
            + "\n".join(_showcase_benchmark_differences(contract, request))
            + "\n\nPlayback illustrates shape-1 throughput with common 4x time compression; it is not a wall-clock recording.\n"
        )
    bindings = {
        "benchmark_id": benchmark["proof_id"],
        "policy_id": policy["proof_id"],
        "render_machine": identity,
        "render_harness_sha256": harness_source_hash(),
        "mode": "smoke" if request["smoke"] else "full",
        "style": COMPARISON_STYLE,
        "assets": assets,
        "pipeline_passed": True,
    }
    snippet = '<p align="center"><a href="media/comparison.mp4"><img src="media/comparison.webp" width="800" alt="Same policy, same actions: environment throughput comparison"></a></p>\n\n![Provider throughput and speedup by environment count](chart.svg)\n\n'
    timing_controls = (
        "captured policy actions"
        if request["schema"] == "turbobench.comparison-request/v3"
        else "seeded controls"
    )
    snippet += f"{'SMOKE / DIAGNOSTIC; no validated performance claim. ' if request['smoke'] else ''}Measured on {benchmark['bindings']['benchmark_machine']['hardware'].get('cpu', 'the benchmark host')}; rendered on a separate host. Frame skip={profile.frame_skip}, stack={profile.frame_stack}; n_threads=n_envs, obs_copy=copy. Timing uses {timing_controls} and excludes inference and task/context wrappers. Policy: {contract['mlflow_url']}. {contract['limitations']} See [method and shape-local SPS](report.md) and verify the archived proof with `turbobench verify`.\n"
    (root / "README-snippet.md").write_text(snippet)
    version = (
        3
        if request["schema"] == "turbobench.comparison-request/v3"
        else 2
        if profile.resolved_workload is not None
        else 1
    )
    if version >= 2:
        bindings["workload_sha256"] = canonical_json_hash(profile.resolved_workload)
    manifest = finalize_proof(root, f"turbobench.showcase-proof/v{version}", bindings)
    require_proof(root)
    return manifest


def _showcase_benchmark_differences(contract: dict, request: dict) -> list[str]:
    differences = list(contract["benchmark_differences"])
    if request["schema"] == "turbobench.comparison-request/v3":
        differences[0] = "captured policy controls for both benchmark timing and showcase"
    return differences


def verify_showcase(root: Path, bindings: dict[str, Any]) -> None:
    from turbobench.showcase import COMPARISON_STYLES

    benchmark = require_proof(root / "benchmark")
    policy = require_proof(root / "policy")
    request = read_json(root / "benchmark" / "request.json")
    if (read_json(root / "manifest.json")["schema"] == "turbobench.showcase-proof/v3") != (
        benchmark["schema"] == "turbobench.benchmark-proof/v3"
    ):
        raise ValueError("showcase proof version differs from benchmark action protocol")
    if (
        bindings["benchmark_id"] != benchmark["proof_id"]
        or bindings["policy_id"] != policy["proof_id"]
        or request["policy_id"] != policy["proof_id"]
        or request["policy_contract"] != policy["bindings"]["contract"]
        or request["actions"] != read_json(root / "policy" / "actions.json")["actions"]
    ):
        raise ValueError("showcase child/request/policy binding mismatch")
    if (
        bindings["render_machine"] != request["render_machine"]
        or bindings["render_machine"]["machine_sha256"]
        == benchmark["bindings"]["benchmark_machine"]["machine_sha256"]
    ):
        raise ValueError("showcase host role mismatch")
    smoke = request["smoke"]
    result = read_json(root / "benchmark" / "result.json")
    pipeline_gates(result, smoke)
    if (
        bindings["mode"] != ("smoke" if smoke else "full")
        or bindings["pipeline_passed"] is not True
        or bindings["style"] not in COMPARISON_STYLES
    ):
        raise ValueError("showcase mode/style/status mismatch")
    profile = request_profile(request)
    if profile.resolved_workload is not None and bindings.get(
        "workload_sha256"
    ) != canonical_json_hash(profile.resolved_workload):
        raise ValueError("showcase workload binding mismatch")
    records = read_json(root / "verification" / "showcase-replay.json")
    remote = read_json(root / "benchmark" / "verification" / "showcase-replay.json")
    if not compare_replays(records["left"], records["right"], profile)["passed"]:
        raise ValueError("render provider replay mismatch")
    from turbobench.engine import _require_evidence_binding

    render_lock = read_json(root / "render-lock.json")
    measured_lock = read_json(root / "benchmark" / "resolved-lock.json")
    for side in ("left", "right"):
        if not compare_replays(records[side], remote[side], profile)["passed"]:
            raise ValueError("render/measurement replay mismatch")
        lhs, rhs = render_lock[side], measured_lock["providers"][side]
        if any(lhs[k] != rhs[k] for k in ("provider", "version", "source_identity")):
            raise ValueError("cross-platform release identity mismatch")
        preflight = replay_preflight(root, side, records[side])
        _require_evidence_binding(
            records[side], preflight["execution_spec"], preflight["contract_attestation"]
        )
    from turbobench.showcase import scaling_chart, verify_assets

    verify_assets(root, bindings["assets"], result, smoke)
    if (root / "chart.svg").read_text() != scaling_chart(
        result, diagnostic=smoke, style=bindings["style"]
    ):
        raise ValueError("scaling chart is not derived from benchmark statistics")


def _run(argv: list[str]) -> str:
    process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=None, check=False)
    if process.returncode:
        raise RuntimeError(
            f"{argv[0]} failed with exit status {process.returncode}; partial workflow data preserved"
        )
    return process.stdout


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = destination / member.name
            if (
                member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
                or not target.resolve().is_relative_to(destination.resolve())
            ):
                raise ValueError("unsafe remote archive entry")
        tar.extractall(destination, filter="data")


def _harness_layout(package: Path | None = None) -> tuple[Path, Path, bool]:
    package = package or Path(__file__).resolve().parent
    bundled = package / "workflow_runtime"
    installed = bundled.is_dir()
    metadata = bundled if installed else package.parent.parent
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        if not (metadata / name).is_file():
            raise ValueError(f"missing workflow runtime metadata: {name}")
    return package, metadata, installed


def _measurement_command(remote: str, python_minor: str, installed: bool) -> str:
    from turbobench import __version__

    base = f'cd {shlex.quote(remote)}/harness && export PATH="$HOME/.local/bin:$PATH" && '
    if installed:
        # Only the explicitly selected TurboBench release bypasses age limits;
        # dependencies come from the bundled frozen lock, without new resolution.
        base += (
            f"uv sync --frozen --no-dev --no-install-project --python {shlex.quote(python_minor)} && "
            f"uv --no-config pip install --python .venv/bin/python --no-deps turbobench-cli=={__version__} && "
        )
        executable = ".venv/bin/turbobench"
    else:
        base += f"uv sync --frozen --no-dev --python {shlex.quote(python_minor)} && "
        executable = "uv run --frozen --no-dev turbobench"
    return (
        base
        + f"if test ! -f ../benchmark/manifest.json; then {executable} measure-request ../request.json --output ../benchmark; fi"
    )


def run_workflow(args: Any, output: Path, progress: Any = print) -> Path:
    if (
        args.render_host != "local"
        or not args.benchmark_host
        or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.benchmark_host)
    ):
        raise ValueError("use --render-host local and a configured SSH --benchmark-host")
    if (
        args.quick
        or args.promo
        or args.steps is not None
        or args.shapes is not None
        or args.force_busy
        or args.allow_dirty
        or args.parity_receipt
    ):
        raise ValueError(
            "showcases use immutable profiles; quick/promo/steps/shapes/force-busy/dirty/parity overrides are unsupported"
        )
    from turbobench.proofs import migrate_policy
    from turbobench.providers import load_providers, parse_provider_ref
    from turbobench.resolution import resolve_pair
    from turbobench.workloads import (
        preliminary_workload,
        probe_declarations,
        profile_from_workload,
        resolve_workload,
    )

    original = require_proof(args.policy)
    args.policy = migrate_policy(args.policy)
    policy = require_proof(args.policy)
    definition = policy["bindings"]["definition"]
    if args.profile not in {definition, original["bindings"].get("profile")}:
        raise ValueError("locked policy package does not match the selected comparison definition")
    policy_timing = bool(getattr(args, "policy_benchmark", False))
    selected_actions = read_json(args.policy / "actions.json")["actions"]
    preliminary = preliminary_workload(
        policy["bindings"]["contract"],
        definition,
        policy_actions=selected_actions if policy_timing else None,
    )
    profile = profile_from_workload(preliminary)
    progress(f"Resolving {definition} from the saved policy contract")
    definitions = load_providers()
    resolution = resolve_pair(
        profile,
        parse_provider_ref(args.left, definitions),
        parse_provider_ref(args.right, definitions),
        definitions,
        python_minor=args.python_minor,
    )
    providers = {
        side: prepare_runtime(provider, cache_context=profile_hash(profile), progress=progress)
        for side, provider in zip(
            ("left", "right"), (resolution.left, resolution.right), strict=True
        )
    }
    declarations = probe_declarations(profile, providers, progress)
    workload = resolve_workload(
        preliminary["definition"],
        preliminary["protocol"],
        preliminary["policy_contract"],
        declarations,
        selected_actions if policy_timing else None,
    )
    profile = profile_from_workload(workload)
    progress(
        f"Locked workload {profile.id}; frameskip={profile.frame_skip}, framestack={profile.frame_stack}"
    )
    request = {
        "schema": "turbobench.comparison-request/v3"
        if policy_timing
        else "turbobench.comparison-request/v2",
        "request_id": "",
        "profile": profile.id,
        "left": f"{providers['left'].provider}@{providers['left'].version}",
        "right": f"{providers['right'].provider}@{providers['right'].version}",
        "python_minor": args.python_minor,
        "smoke": args.smoke,
        "policy_id": policy["proof_id"],
        "actions": read_json(args.policy / "actions.json")["actions"],
        "policy_contract": policy["bindings"]["contract"],
        "render_machine": machine_identity(),
        "harness_sha256": harness_source_hash(),
        "resolved_workload": workload,
    }
    request["request_id"] = canonical_json_hash(request)
    validate_request(request)
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    saved_request = staging / "request.json"
    if saved_request.exists() and read_json(saved_request) != request:
        raise ValueError("existing partial workflow belongs to another request")
    write_json(saved_request, request)
    if not (staging / "policy").exists():
        shutil.copytree(args.policy, staging / "policy")
    host = args.benchmark_host
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]
    remote = f".cache/turbobench/workflows/{request['request_id']}"
    remote_home = _run([*ssh, 'printf %s "$HOME"']).strip()
    remote_absolute = remote_home + "/" + remote
    progress("Staging the exact harness and request on the benchmark host")
    _run([*ssh, f"mkdir -p {shlex.quote(remote_absolute)}"])
    package, metadata, installed = _harness_layout()
    with tempfile.TemporaryDirectory(prefix="turbobench-transfer-") as temporary:
        archive = Path(temporary) / "harness.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
                tar.add(metadata / name, arcname="harness/" + name)
            for path in package.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    tar.add(
                        path,
                        arcname="harness/src/turbobench/" + path.relative_to(package).as_posix(),
                    )
            tar.add(saved_request, arcname="request.json")
        _run(["scp", "-q", str(archive), f"{host}:{remote_absolute}/harness.tar.gz"])
        _run([*ssh, f"cd {shlex.quote(remote_absolute)} && tar -xzf harness.tar.gz"])
        if not (staging / "benchmark").exists():
            progress("Running isolated measurements and hash-only replay on the benchmark host")
            command = _measurement_command(remote_absolute, args.python_minor, installed)
            process = subprocess.Popen([*ssh, command])
            if process.wait() != 0:
                raise RuntimeError("remote measurement failed; rerun the same command to resume")
            remote_archive = remote_absolute + "/benchmark.tar.gz"
            _run(
                [*ssh, f"cd {shlex.quote(remote_absolute)} && tar -czf benchmark.tar.gz benchmark"]
            )
            local_archive = Path(temporary) / "benchmark.tar.gz"
            _run(["scp", "-q", f"{host}:{remote_archive}", str(local_archive)])
            _extract(local_archive, staging)
    benchmark = require_proof(staging / "benchmark")
    if benchmark["bindings"]["request_id"] != request["request_id"]:
        raise ValueError("downloaded proof belongs to another request")
    progress(
        "Remote evidence verified; finalizing policy benchmark"
        if policy_timing and not args.showcase
        else "Remote evidence verified; generating assets locally"
    )
    for name in ("request.json",):
        (staging / name).unlink()
    if policy_timing and not args.showcase:
        finalize_policy_benchmark(staging)
    else:
        render_showcase(staging, progress)
    require_proof(staging)
    staging.rename(output)
    return output


def finalize_policy_benchmark(root: Path) -> dict[str, Any]:
    from turbobench.reporting import write_views
    from turbobench.showcase import COMPARISON_STYLE, scaling_chart

    benchmark, policy = require_proof(root / "benchmark"), require_proof(root / "policy")
    request = read_json(root / "benchmark" / "request.json")
    result = read_json(root / "benchmark" / "result.json")
    pipeline_gates(result, request["smoke"])
    write_views(root, result)
    (root / "chart.svg").write_text(scaling_chart(result, diagnostic=request["smoke"]))
    contract = policy["bindings"]["contract"]
    with (root / "report.md").open("a") as report:
        report.write(
            f"\n## Locked policy workload\n\nCheckpoint: `{contract['checkpoint_sha256']}`; "
            f"training step {contract['checkpoint_step']}.\n\nTracking: {contract['mlflow_url']}\n\n"
            f"Measured all {contract['decisions']} captured decisions of {contract['total_captured_decisions']}; "
            "every lane replays the same effective controls with the captured reset seed. "
            "Repetitions restart from that same seed. Initial seeded no-op reset is outside timing. "
            "Training frame skip and observation preprocessing are preserved. "
            "Policy inference, trajectory recording, correctness hashing and rendering are excluded. "
            "Timing includes environment stepping, preprocessing, IPC, infos and required selective resets. "
            "No video was generated.\n\n"
            f"Limitations: {contract['limitations']}\n\n"
            "Task/context/reward wrappers remain excluded; obs_copy=copy and num_threads=n_envs. "
            "This workload has correlated lanes and does not estimate policy success rate.\n"
        )
    bindings = {
        "benchmark_id": benchmark["proof_id"],
        "policy_id": policy["proof_id"],
        "style": COMPARISON_STYLE,
    }
    return finalize_proof(root, "turbobench.policy-benchmark-proof/v1", bindings)


def verify_policy_benchmark(root: Path, bindings: dict[str, Any]) -> None:
    from turbobench.showcase import COMPARISON_STYLES, scaling_chart

    benchmark, policy = require_proof(root / "benchmark"), require_proof(root / "policy")
    request = read_json(root / "benchmark" / "request.json")
    if (
        benchmark["schema"] != "turbobench.benchmark-proof/v3"
        or request["schema"] != "turbobench.comparison-request/v3"
        or bindings["benchmark_id"] != benchmark["proof_id"]
        or bindings["policy_id"] != policy["proof_id"]
        or request["policy_id"] != policy["proof_id"]
        or request["policy_contract"] != policy["bindings"]["contract"]
        or request["actions"] != read_json(root / "policy" / "actions.json")["actions"]
    ):
        raise ValueError("policy benchmark child/contract/action binding mismatch")
    result = read_json(root / "benchmark" / "result.json")
    pipeline_gates(result, request["smoke"])
    if bindings["style"] not in COMPARISON_STYLES or (
        root / "chart.svg"
    ).read_text() != scaling_chart(result, diagnostic=request["smoke"], style=bindings["style"]):
        raise ValueError("policy benchmark chart differs from bound evidence")
    if list((root / "media").glob("*")):
        raise ValueError("benchmark-only proof must not contain videos")
