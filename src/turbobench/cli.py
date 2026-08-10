"""The turbobench v1 command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from turbobench import __version__
from turbobench.assets import discover_assets
from turbobench.bundle import verify_bundle
from turbobench.engine import ComparisonOptions, generate_promo_for_bundle, run_comparison
from turbobench.profiles import PROFILES, get_profile, profile_hash
from turbobench.providers import load_providers, parse_provider_ref
from turbobench.system import host_record, prerequisites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turbobench",
        description="Correctness-gated provider-neutral environment benchmarking",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check prerequisites, host, and profile assets")
    doctor.add_argument("profile", nargs="?")

    providers = commands.add_parser("providers", help="inspect provider adapters")
    providers_commands = providers.add_subparsers(dest="providers_command", required=True)
    providers_commands.add_parser("list")

    profiles = commands.add_parser("profiles", help="inspect immutable profiles")
    profiles_commands = profiles.add_subparsers(dest="profiles_command", required=True)
    profiles_commands.add_parser("list")

    compare = commands.add_parser("compare", help="run a correctness-gated paired comparison")
    compare.add_argument("profile")
    compare.add_argument("--left", required=True, metavar="PROVIDER_REF")
    compare.add_argument("--right", required=True, metavar="PROVIDER_REF")
    compare.add_argument("--promo", action="store_true")
    compare.add_argument("--output", type=Path)
    compare.add_argument("--quick", action="store_true", help="diagnostic short workload")
    compare.add_argument("--force-busy", action="store_true", help="diagnostic load override")
    compare.add_argument("--allow-dirty", action="store_true", help="diagnostic dirty-checkout override")
    compare.add_argument("--python", default="3.14", dest="python_minor")
    compare.add_argument("--steps", type=int, help="diagnostic workload step override")
    compare.add_argument("--shapes", type=_shapes, help="diagnostic comma-separated shape override")

    verify = commands.add_parser("verify", help="verify bundle integrity and consistency")
    verify.add_argument("bundle", type=Path)

    report = commands.add_parser("report", help="display the bundle's generated report")
    report.add_argument("bundle", type=Path)

    promo = commands.add_parser("promo", help="replay locked providers and generate bound media")
    promo.add_argument("bundle", type=Path)
    promo.add_argument("--diagnostic", action="store_true")
    return parser


def _shapes(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shapes must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("shapes must be unique positive integers")
    return values


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "doctor":
            return _doctor(args.profile)
        if args.command == "providers":
            _providers_list()
            return 0
        if args.command == "profiles":
            _profiles_list()
            return 0
        if args.command == "compare":
            return _compare(args, arguments)
        if args.command == "verify":
            result = verify_bundle(args.bundle)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 1
        if args.command == "report":
            integrity = verify_bundle(args.bundle)
            if not integrity["passed"]:
                raise ValueError("bundle verification failed: " + "; ".join(integrity["errors"]))
            print((args.bundle / "report.md").read_text(encoding="utf-8"), end="")
            return 0
        if args.command == "promo":
            result = generate_promo_for_bundle(
                args.bundle,
                diagnostic=args.diagnostic,
                progress=_print_progress,
            )
            print(json.dumps({"bundle": str(args.bundle.resolve()), "promo": result["promo"]}, indent=2, sort_keys=True))
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"turbobench: error: {exc}\n")
    return 2


def _doctor(profile_id: str | None) -> int:
    checked_profiles = [get_profile(profile_id)] if profile_id else list(PROFILES.values())
    payload: dict[str, Any] = {
        "schema": "turbobench.doctor/v1",
        "prerequisites": prerequisites(),
        "host": host_record(),
        "profiles": {},
    }
    for profile in checked_profiles:
        _private, portable = discover_assets(profile)
        payload["profiles"][profile.id] = {
            "profile_sha256": profile_hash(profile),
            "assets": portable,
            "providers": list(profile.providers),
        }
    payload["passed"] = bool(
        payload["prerequisites"]["passed"]
        and payload["host"]["official_v1_platform"]
        and all(
            not item["assets"].get("required") or item["assets"].get("available")
            for item in payload["profiles"].values()
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def _providers_list() -> None:
    providers = load_providers()
    for provider_id in sorted(providers):
        provider = providers[provider_id]
        print(f"{provider.id}\t{provider.adapter}\t{provider.distribution}\t{provider.import_name}")


def _profiles_list() -> None:
    for profile in PROFILES.values():
        print(f"{profile.id}\t{profile.game}\tshapes={','.join(map(str, profile.shapes))}\tproviders={','.join(profile.providers)}")


def _compare(args: argparse.Namespace, command: list[str]) -> int:
    definitions = load_providers()
    left = parse_provider_ref(args.left, definitions)
    right = parse_provider_ref(args.right, definitions)
    output = args.output or _default_output(args.profile)
    options = ComparisonOptions(
        promo=args.promo,
        quick=args.quick,
        force_busy=args.force_busy,
        allow_dirty=args.allow_dirty,
        python_minor=args.python_minor,
        steps=args.steps,
        shapes=args.shapes,
        command=("turbobench", *command),
        progress=_print_progress,
    )
    bundle, result = run_comparison(args.profile, left, right, output, options)
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "validity": result["validity"]["passed"],
                "claim": result["claim"]["status"],
                "outcome": result["comparison"]["outcome"],
                "promo": result["promo"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if args.promo and not result["promo"]["generated"] else 0


def _default_output(profile_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = profile_id.replace("/", "-")
    return Path("turbobench-results") / f"{stamp}-{slug}"


def _print_progress(message: str) -> None:
    print(f"turbobench: {message}", file=sys.stderr, flush=True)
