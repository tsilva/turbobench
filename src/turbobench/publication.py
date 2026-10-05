"""Verified, concise repository exports shared by every comparison provider pair."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from turbobench import __version__
from turbobench.bundle import verify_bundle
from turbobench.readme_chart import PUBLICATION_NAMES, peak_rows, write_publication_chart
from turbobench.runtime import harness_source_hash
from turbobench.stats import reciprocal_statistics
from turbobench.util import read_json, sha256_file, write_json
from turbobench.workloads import bundle_profile


def publication_path(proof: Path) -> Path:
    return proof.with_name(proof.name + "-publication")


def _verifier() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    revision = None
    repository = package.parent.parent
    if (repository / ".git").exists():
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--",
                "src",
                "pyproject.toml",
                "uv.lock",
                "README.md",
            ],
            text=True,
        )
        if not dirty:
            revision = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()
    return {
        "version": __version__,
        "source_sha256": harness_source_hash(),
        "source_revision": revision,
        "installed_release": (package / "workflow_runtime").is_dir(),
    }


def _provider(provider: dict[str, Any]) -> str:
    return (
        f"{PUBLICATION_NAMES.get(provider['provider'], provider['provider'])} {provider['version']}"
    )


def render_benchmarks(
    result: dict[str, Any],
    profile: Any,
    publication: dict[str, Any],
    contract: dict[str, Any] | None,
) -> str:
    comparison = result["comparison"]
    rows, omitted = peak_rows(result)
    peak_shape, peak_row = rows[-1]
    diagnostic = publication["diagnostic"]
    host = result.get("system", {}).get("host", {})
    verifier = publication["verifier"]
    lines = [
        "# Latest benchmark",
        "",
        f"**{_provider(comparison['right'])} vs {_provider(comparison['left'])}** · "
        f"TurboBench {result['tool']['version']} · `{profile.game}`",
        "",
        f"Measured on **{host.get('cpu', 'processor not recorded')}** "
        f"({host.get('os', 'OS not recorded')}, {host.get('architecture', 'architecture not recorded')}).",
    ]
    if diagnostic:
        lines += ["", "**DIAGNOSTIC — no validated performance claim.**"]
        reasons = result.get("claim", {}).get("diagnostic_reasons", [])
        if reasons:
            lines.append("Reasons: " + "; ".join(reasons) + ".")
        lines.append(f"Shape-1 outcome: `{comparison['outcome']}`.")
    else:
        headline = comparison["shapes"]["1"]["statistics"]
        ratio = 1 / headline["median_paired_ratio_left_over_right"]
        lines += ["", f"**{ratio:.2f}\u00d7 candidate/reference throughput at one environment.**"]
    lines += [
        f"Reference: **{_provider(comparison['left'])}**; candidate: **{_provider(comparison['right'])}**.",
        f"Highest measured candidate throughput: **{peak_row['statistics']['median_right_sps']:,.0f} "
        f"steps/s at {peak_shape} environments**.",
        "",
        "## Results and scaling",
        "",
        "| n_envs | Reference median SPS | Candidate median SPS | Candidate/reference | Paired 95% CI |",
        "| ---: | ---: | ---: | ---: | :---: |",
    ]
    for shape, row in sorted(comparison["shapes"].items(), key=lambda item: int(item[0])):
        stats = row["statistics"]
        ratio = 1 / stats["median_paired_ratio_left_over_right"]
        interval = stats["bootstrap"]
        ci = (
            f"[{1 / interval['ci'][1]:.2f}, {1 / interval['ci'][0]:.2f}]"
            if interval
            else "unavailable"
        )
        lines.append(
            f"| {shape} | {stats['median_left_sps']:,.1f} | {stats['median_right_sps']:,.1f} | {ratio:.2f}\u00d7 | {ci} |"
        )
    lines += ["", "SPS counts environment transitions across all lanes, not raw game frames."]
    lines.append(
        "The [README chart](benchmark-readme.svg) stops at the first measured candidate maximum; "
        "the [complete chart](benchmark.svg) and table retain every count."
    )
    if omitted:
        lines.append("Omitted from the README chart: " + ", ".join(map(str, omitted)) + ".")
    scaling = result.get("scaling")
    if scaling:
        rule = scaling["rule"]
        lines.append(
            f"Counts double until each provider plateaus ({rule['plateau_confirmations']} successive gains "
            f"below {rule['minimum_gain']:.0%} against its previous best) or downgrades "
            f"(a decline of at least {rule['downgrade_fraction']:.0%}). "
            f"Stop: `{scaling['stop_reason']}`; cap: {rule['max_n_envs']}. "
            "This heuristic does not estimate uncertainty in the peak's location."
        )
    sampling = result["sampling"]
    actions = result.get("actions", {})
    action_versions = sorted({record["version"] for record in actions.values()})
    steps = sorted({record["measurement_steps"] for record in actions.values()})
    step_label = (
        ", ".join(f"{value:,}" for value in steps) if steps else f"{profile.measurement_steps:,}"
    )
    lines += [
        "",
        "## Method and policy",
        "",
        f"- **Controls:** `{', '.join(action_versions) or profile.action_stream_version}`; seed {profile.run_seed}; "
        f"{step_label} decisions per lane and repetition.",
        f"- **Configuration:** frame skip {profile.frame_skip}, stack {profile.frame_stack}, "
        f"{profile.layout.upper()} {profile.resize[0]}\u00d7{profile.resize[1]}, "
        f"grayscale={str(profile.grayscale).lower()}, resize={profile.resize_algorithm}, "
        f"crop={profile.crop_mode}({profile.crop_top}, {profile.crop_bottom}), "
        f"maxpool={str(profile.maxpool_last_two).lower()}.",
        "- **Timing:** includes stepping, preprocessing, IPC, infos, terminal detection, and selective resets. "
        "Excludes construction, initial reset, warmup, inference, correctness replay, recording, and rendering. "
        "Correctness and timing use separate processes and fresh instances.",
        f"- **Sampling:** {sampling['warmup_pairs']} warmup pairs, {sampling['pairs']} alternating AB/BA pairs, "
        f"{sampling['repetitions']} repetitions per invocation. "
        + (
            "No confidence interval or significance claim."
            if sampling["mode"] == "smoke"
            else "Invocation medians form paired ratios; deterministic 20,000-resample bootstrap CIs. Counts are analyzed separately."
        ),
    ]
    if contract:
        lines += [
            f"- **Policy:** [training run]({contract['mlflow_url']}); "
            f"{contract['decisions']:,} of {contract['total_captured_decisions']:,} captured decisions; "
            f"{contract['reset_noop_prefix']} raw reset noops. Checkpoint, saved recipe, and effective actions are locked in `policy/`. "
            "Every lane uses the same controls; this does not estimate a policy success rate.",
        ]
        differences = list(contract.get("benchmark_differences", []))
        if differences and "captured-policy/v1" in action_versions:
            differences[0] = "captured policy controls for both timing and replay"
        if differences:
            lines.append("- **Training differences:** " + "; ".join(differences) + ".")
    lines += [
        "",
        "## Parity and media limits",
        "",
        "Correctness evidence applies only to the checked workload and selected replay. "
        "It does not establish full-episode parity or a policy success rate.",
    ]
    if contract and contract.get("limitations"):
        lines += ["", contract["limitations"]]
    if publication["media"]:
        lines += [
            "",
            "The [animation](demo.webp) and [MP4](demo.mp4) use the measured ratio and common "
            "**4\u00d7 time compression**, not wall-clock playback. Exact replay is verified on the separate render host.",
        ]
    lines += [
        "",
        "## Proof and verification",
        "",
        f"[Proof files]({publication['proof_url']}) · [Export identities and hashes](publication.json)",
        "",
    ]
    revision = verifier["source_revision"]
    if revision:
        lines += [
            f"Verify with pinned [TurboBench source {revision[:7]}](https://github.com/tsilva/turbobench/commit/{revision}):",
            "",
            "```bash",
            f"curl -fL https://github.com/tsilva/turbobench/archive/{revision}.tar.gz -o verifier.tar.gz",
            "tar -xzf verifier.tar.gz",
            f"uv run --frozen --project turbobench-{revision} turbobench verify proof",
            "```",
        ]
    elif verifier["installed_release"]:
        lines += [
            f"Verify with published **TurboBench {verifier['version']}**:",
            "",
            "```bash",
            f"uvx --from turbobench-cli=={verifier['version']} turbobench verify proof",
            "```",
        ]
    else:
        lines += [
            "Run `turbobench verify proof` with the exporter source hash recorded in `publication.json`. "
            "Pin a clean source commit or published verifier before publishing these exports."
        ]
    lines += [
        "",
        "Extract the downloaded archive as `proof/` before verification. FFprobe is required for media; "
        "verification needs no ROM. Keep models and proof archives outside the repository.",
    ]
    if publication.get("earlier_proofs"):
        lines += ["", "## Earlier proof references", ""]
        lines += [
            f"- [Proof {record['proof_id'][:12]}]({record['proof_url']})"
            for record in publication["earlier_proofs"]
        ]
    return "\n".join(lines) + "\n"


def export_publication(
    proof: Path,
    output: Path | None = None,
    *,
    proof_url: str | None = None,
    previous: Path | None = None,
) -> Path:
    """Verify once, then export outside the immutable proof without publishing."""
    proof = proof.expanduser().resolve()
    output = (output or publication_path(proof)).expanduser().resolve()
    if output.is_relative_to(proof) or proof.is_relative_to(output):
        raise ValueError("publication directory must be outside the immutable proof")
    if output.exists():
        raise FileExistsError(output)
    if proof_url and (
        urlsplit(proof_url).scheme not in {"https", "http"} or any(c in proof_url for c in "\n\r()")
    ):
        raise ValueError("proof URL must be an HTTP(S) archive or release link")
    verification = verify_bundle(proof)
    if not verification["passed"]:
        raise ValueError("proof verification failed: " + "; ".join(verification["errors"]))
    manifest = read_json(proof / "manifest.json")
    benchmark = proof / "benchmark" if (proof / "benchmark").is_dir() else proof
    if not (benchmark / "result.json").is_file():
        raise ValueError("expected a comparison benchmark or showcase proof")
    benchmark_manifest = read_json(benchmark / "manifest.json")
    result = read_json(benchmark / "result.json")
    profile = bundle_profile(benchmark, result)
    # The profile authority is the reference regardless of command-line order.
    # For pairs without that authority, retain the caller's left/right roles.
    candidate_side = (
        "left" if result["comparison"]["right"]["provider"] == profile.authority else "right"
    )
    if candidate_side == "left":
        result = copy.deepcopy(result)
        comparison = result["comparison"]
        comparison["left"], comparison["right"] = comparison["right"], comparison["left"]
        comparison["outcome"] = {
            "left_faster": "right_faster",
            "right_faster": "left_faster",
            "inconclusive": "inconclusive",
        }[comparison["outcome"]]
        for row in comparison["shapes"].values():
            row["statistics"] = reciprocal_statistics(row["statistics"])
    diagnostic = (
        result["claim"]["status"] != "official" or result["comparison"]["outcome"] == "inconclusive"
    )
    contract = (
        read_json(proof / "policy/policy-contract.json")
        if (proof / "policy/policy-contract.json").is_file()
        else None
    )
    if contract is None and (proof / "policy/manifest.json").is_file():
        contract = read_json(proof / "policy/manifest.json")["bindings"]["contract"]
    publication = {
        "format": "turbobench.publication/v1",
        "proof_id": manifest.get("proof_id", manifest.get("bundle_id")),
        "proof_manifest_sha256": sha256_file(proof / "manifest.json"),
        "proof_url": proof_url or quote(os.path.relpath(proof, output)),
        "source_candidate_side": candidate_side,
        "benchmark_id": benchmark_manifest.get("proof_id", benchmark_manifest.get("bundle_id")),
        "diagnostic": diagnostic,
        "verifier": _verifier(),
        "media": {},
        "earlier_proofs": [],
    }
    if previous:
        prior = read_json(previous / "publication.json")
        if prior.get("format") != "turbobench.publication/v1":
            raise ValueError("unsupported previous publication")
        for name, record in prior["files"].items():
            path = (previous / name).resolve()
            if not path.is_relative_to(previous.resolve()) or sha256_file(path) != record["sha256"]:
                raise ValueError("previous publication artifact hash mismatch")
        publication["earlier_proofs"] = [
            record
            for record in prior.get("earlier_proofs", [])
            if record["proof_id"] != publication["proof_id"]
        ]
        if prior["proof_id"] != publication["proof_id"]:
            publication["earlier_proofs"].append(
                {"proof_id": prior["proof_id"], "proof_url": prior["proof_url"]}
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".turbobench-publication-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        publication["charts"] = {
            name: write_publication_chart(
                benchmark,
                staging / name,
                benchmark_manifest,
                result,
                full=full,
                candidate_side=candidate_side,
            )
            for name, full in (("benchmark-readme.svg", False), ("benchmark.svg", True))
        }
        # Only the verified arcade showcase has the common 4x playback contract.
        if manifest["schema"].startswith("turbobench.showcase-proof/"):
            for name, source in (("demo.mp4", "comparison.mp4"), ("demo.webp", "comparison.webp")):
                shutil.copyfile(proof / "media" / source, staging / name)
                publication["media"][name] = sha256_file(staging / name)
        (staging / "benchmarks.md").write_text(
            render_benchmarks(result, profile, publication, contract)
        )
        media = (
            '<p align="center"><a href="demo.mp4"><img src="demo.webp" width="800" alt="Same Actions: verified environment throughput comparison"></a></p>\n\n'
            if publication["media"]
            else ""
        )
        (staging / "README-snippet.md").write_text(
            media
            + '<p align="center"><img src="benchmark-readme.svg" width="800" alt="Measured provider throughput and speedup by environment count"></p>\n\n[Benchmark results, method, and proof](benchmarks.md).\n'
        )
        publication["files"] = {
            path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(staging.iterdir())
        }
        write_json(staging / "publication.json", publication)
        staging.rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--proof-url")
    parser.add_argument("--previous-publication", type=Path)
    args = parser.parse_args()
    print(
        export_publication(
            args.proof, args.output, proof_url=args.proof_url, previous=args.previous_publication
        )
    )
