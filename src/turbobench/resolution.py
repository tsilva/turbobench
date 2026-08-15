"""Exact provider resolution, compatibility solving, and checkout provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from turbobench.model import Profile, ProviderDefinition, ProviderRef, ResolvedProvider
from turbobench.providers import lineage_version
from turbobench.util import canonical_json_hash, sha256_file

SEVEN_DAYS = timedelta(days=7)
EXACT_QUARANTINE_EXEMPT_PROVIDERS = frozenset(
    {"breakout-turbo-env", "supermariobrosnes-turbo"}
)
PYPI_URL = "https://pypi.org/pypi/{project}/json"


@dataclass(frozen=True)
class ReleaseCandidate:
    version: str
    uploaded: datetime
    requires_python: str | None
    files: tuple[dict[str, Any], ...]
    yanked: bool


@dataclass(frozen=True)
class ResolutionResult:
    left: ResolvedProvider
    right: ResolvedProvider
    excluded: dict[str, tuple[dict[str, str], ...]]


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": f"turbobench/{_tool_version()}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _tool_version() -> str:
    from turbobench import __version__

    return __version__


def pypi_candidates(
    definition: ProviderDefinition,
    *,
    python_minor: str,
    now: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[ReleaseCandidate], list[dict[str, str]]]:
    current = now or datetime.now(UTC)
    payload = metadata or _request_json(
        PYPI_URL.format(project=definition.pypi_project or definition.distribution)
    )
    candidates: list[ReleaseCandidate] = []
    excluded: list[dict[str, str]] = []
    for raw_version, raw_files in payload.get("releases", {}).items():
        try:
            version = Version(raw_version)
        except InvalidVersion:
            excluded.append({"version": raw_version, "reason": "invalid-version"})
            continue
        if version.is_prerelease:
            excluded.append({"version": raw_version, "reason": "prerelease"})
            continue
        files = [item for item in raw_files if isinstance(item, dict)]
        if not files:
            excluded.append({"version": raw_version, "reason": "no-artifacts"})
            continue
        non_yanked = [item for item in files if not bool(item.get("yanked"))]
        if not non_yanked:
            excluded.append({"version": raw_version, "reason": "yanked"})
            continue
        eligible_python: list[dict[str, Any]] = []
        for item in non_yanked:
            requires_python = item.get("requires_python") or payload.get("info", {}).get(
                "requires_python"
            )
            if requires_python:
                try:
                    if Version(f"{python_minor}.0") not in SpecifierSet(str(requires_python)):
                        continue
                except InvalidSpecifier:
                    continue
            eligible_python.append(item)
        if not eligible_python:
            excluded.append({"version": raw_version, "reason": f"python-{python_minor}-incompatible"})
            continue
        uploaded_values = [
            _parse_uploaded(item.get("upload_time_iso_8601") or item.get("upload_time"))
            for item in eligible_python
        ]
        known_uploads = [value for value in uploaded_values if value is not None]
        if not known_uploads:
            excluded.append({"version": raw_version, "reason": "missing-upload-time"})
            continue
        uploaded = min(known_uploads)
        requires_python = eligible_python[0].get("requires_python")
        normalized_files = tuple(
            {
                "filename": str(item.get("filename")),
                "packagetype": str(item.get("packagetype")),
                "python_version": str(item.get("python_version")),
                "requires_python": item.get("requires_python"),
                "sha256": str(item.get("digests", {}).get("sha256", "")),
                "size": int(item.get("size", 0)),
                "upload_time": str(item.get("upload_time_iso_8601", item.get("upload_time", ""))),
                "url": str(item.get("url", "")),
            }
            for item in eligible_python
        )
        candidate = ReleaseCandidate(
            version=str(version),
            uploaded=uploaded,
            requires_python=str(requires_python) if requires_python else None,
            files=normalized_files,
            yanked=False,
        )
        candidates.append(candidate)
        if current - uploaded < SEVEN_DAYS:
            excluded.append({"version": str(version), "reason": "seven-day-quarantine"})
    candidates.sort(key=lambda item: Version(item.version), reverse=True)
    return candidates, excluded


def _parse_uploaded(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _eligible_latest(candidates: list[ReleaseCandidate], now: datetime) -> list[ReleaseCandidate]:
    return [candidate for candidate in candidates if now - candidate.uploaded >= SEVEN_DAYS]


def _release_to_resolved(
    definition: ProviderDefinition,
    candidate: ReleaseCandidate,
    *,
    python_minor: str,
    diagnostic_reasons: tuple[str, ...] = (),
) -> ResolvedProvider:
    artifact_hash = canonical_json_hash(
        [{"filename": item["filename"], "sha256": item["sha256"]} for item in candidate.files]
    )
    lineage = (
        lineage_version(candidate.version)
        if definition.lineage in {"stable-retro", "vizdoom"}
        else definition.lineage
    )
    return ResolvedProvider(
        provider=definition.id,
        adapter=definition.adapter,
        distribution=definition.distribution,
        import_name=definition.import_name,
        version=candidate.version,
        source_kind="pypi",
        source_identity=f"pypi:{definition.distribution}=={candidate.version}",
        artifact_sha256=artifact_hash,
        python_minor=python_minor,
        build_root=definition.build_subdirectory,
        release_files=candidate.files,
        compatibility_lineage=lineage,
        diagnostic_reasons=diagnostic_reasons,
    )


def resolve_pair(
    profile: Profile,
    left_ref: ProviderRef,
    right_ref: ProviderRef,
    definitions: dict[str, ProviderDefinition],
    *,
    python_minor: str = "3.14",
    now: datetime | None = None,
    allow_dirty: bool = False,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> ResolutionResult:
    if not profile.compatible(left_ref.provider, right_ref.provider):
        raise ValueError(
            f"providers {left_ref.provider!r} and {right_ref.provider!r} are not a compatible pair "
            f"for {profile.id}"
        )
    current = now or datetime.now(UTC)
    all_candidates: dict[str, list[ReleaseCandidate]] = {}
    excluded: dict[str, tuple[dict[str, str], ...]] = {}
    resolved: dict[str, ResolvedProvider] = {}
    refs = {"left": left_ref, "right": right_ref}
    for side, reference in refs.items():
        definition = definitions[reference.provider]
        if reference.selector == "checkout":
            resolved[side] = resolve_checkout(
                definition,
                Path(reference.value),
                python_minor=python_minor,
                allow_dirty=allow_dirty,
            )
            excluded[reference.provider] = ()
            continue
        candidates, rejected = pypi_candidates(
            definition,
            python_minor=python_minor,
            now=current,
            metadata=(metadata or {}).get(reference.provider),
        )
        all_candidates[side] = candidates
        excluded[reference.provider] = tuple(rejected)
        if reference.selector == "version":
            selected = next((item for item in candidates if item.version == reference.value), None)
            if selected is None:
                raise ValueError(f"{reference.provider}@{reference.value} is unavailable or incompatible")
            reasons: tuple[str, ...] = ()
            if (
                current - selected.uploaded < SEVEN_DAYS
                and reference.provider not in EXACT_QUARANTINE_EXEMPT_PROVIDERS
            ):
                reasons = ("exact version is inside the seven-day quarantine",)
            resolved[side] = _release_to_resolved(
                definition,
                selected,
                python_minor=python_minor,
                diagnostic_reasons=reasons,
            )
    latest_sides = [side for side, reference in refs.items() if reference.selector == "latest"]
    eligible_by_side = {
        side: _eligible_latest(all_candidates[side], current) for side in latest_sides
    }
    for side, eligible in eligible_by_side.items():
        if not eligible:
            raise ValueError(f"no seven-day-eligible release for {refs[side].provider}")
    if len(latest_sides) == 2:
        compatible: list[tuple[ResolvedProvider, ResolvedProvider]] = []
        for left_candidate in eligible_by_side["left"]:
            left = _release_to_resolved(
                definitions[left_ref.provider], left_candidate, python_minor=python_minor
            )
            for right_candidate in eligible_by_side["right"]:
                right = _release_to_resolved(
                    definitions[right_ref.provider], right_candidate, python_minor=python_minor
                )
                if _lineages_match(left, right):
                    compatible.append((left, right))
        if not compatible:
            raise ValueError("no seven-day-eligible compatible provider release tuple")
        resolved["left"], resolved["right"] = max(compatible, key=_compatible_tuple_key)
    elif latest_sides:
        side = latest_sides[0]
        other = "right" if side == "left" else "left"
        definition = definitions[refs[side].provider]
        selected = next(
            (
                candidate
                for candidate in eligible_by_side[side]
                if _lineages_match(
                    _release_to_resolved(definition, candidate, python_minor=python_minor),
                    resolved[other],
                )
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"no seven-day-eligible {refs[side].provider} release is compatible with "
                f"{resolved[other].provider}@{resolved[other].version}"
            )
        resolved[side] = _release_to_resolved(
            definition, selected, python_minor=python_minor
        )
    for side in latest_sides:
        selected_version = Version(resolved[side].version)
        other = "right" if side == "left" else "left"
        extra = list(excluded[refs[side].provider])
        for candidate in eligible_by_side[side]:
            if Version(candidate.version) <= selected_version:
                continue
            candidate_provider = _release_to_resolved(
                definitions[refs[side].provider], candidate, python_minor=python_minor
            )
            if not _lineages_match(candidate_provider, resolved[other]):
                extra.append({"version": candidate.version, "reason": "lineage-incompatible"})
        excluded[refs[side].provider] = tuple(extra)
    _enforce_lineage(resolved["left"], resolved["right"])
    return ResolutionResult(resolved["left"], resolved["right"], excluded)


def _enforce_lineage(left: ResolvedProvider, right: ResolvedProvider) -> None:
    if _lineages_match(left, right):
        return
    pair = {left.provider, right.provider}
    if pair == {"vizdoom", "vizdoom-turbo"}:
        raise ValueError(
            f"ViZDoom base lineage mismatch: {left.version} versus {right.version}"
        )
    if pair == {"stable-retro", "stable-retro-turbo"}:
        raise ValueError(
            f"Stable Retro base lineage mismatch: {left.version} versus {right.version}"
        )


def _lineages_match(left: ResolvedProvider, right: ResolvedProvider) -> bool:
    pair = {left.provider, right.provider}
    if pair == {"vizdoom", "vizdoom-turbo"}:
        return lineage_version(left.version) == lineage_version(right.version)
    if pair == {"stable-retro", "stable-retro-turbo"}:
        return lineage_version(left.version) == lineage_version(right.version)
    return True


def _compatible_tuple_key(
    pair: tuple[ResolvedProvider, ResolvedProvider],
) -> tuple[Version, Version, Version]:
    left, right = pair
    if {left.provider, right.provider} in (
        {"vizdoom", "vizdoom-turbo"},
        {"stable-retro", "stable-retro-turbo"},
    ):
        common = Version(lineage_version(left.version))
    else:
        common = min(Version(left.version), Version(right.version))
    return common, Version(left.version), Version(right.version)


def resolve_checkout(
    definition: ProviderDefinition,
    checkout: Path,
    *,
    python_minor: str,
    allow_dirty: bool = False,
) -> ResolvedProvider:
    root = checkout.expanduser().resolve()
    if not (root / ".git").exists():
        raise ValueError(f"checkout is not a Git worktree: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty_lines = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    reasons: tuple[str, ...] = ()
    if dirty_lines and not allow_dirty:
        raise ValueError(f"checkout {root} is dirty; clean it or use a diagnostic override")
    if dirty_lines:
        reasons = ("dirty checkout override",)
        artifact_hash = _worktree_hash(root)
        source_identity = f"dirty-checkout:{commit}:{artifact_hash}"
    else:
        artifact_hash = hashlib.sha256(f"git-tree:{tree}".encode()).hexdigest()
        source_identity = f"git:{commit}:{tree}"
    build_root = root / definition.build_subdirectory if definition.build_subdirectory else root
    if not (build_root / "pyproject.toml").is_file() and not (build_root / "setup.py").is_file():
        raise ValueError(f"provider build root has no Python project: {build_root}")
    version = _checkout_version(build_root, root, definition)
    lineage = (
        lineage_version(version)
        if definition.lineage in {"stable-retro", "vizdoom"}
        else definition.lineage
    )
    return ResolvedProvider(
        provider=definition.id,
        adapter=definition.adapter,
        distribution=definition.distribution,
        import_name=definition.import_name,
        version=version,
        source_kind="checkout",
        source_identity=source_identity,
        artifact_sha256=artifact_hash,
        python_minor=python_minor,
        build_root=definition.build_subdirectory,
        commit=commit,
        tree=tree,
        compatibility_lineage=lineage,
        diagnostic_reasons=reasons,
    )


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise ValueError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _worktree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkout_version(build_root: Path, root: Path, definition: ProviderDefinition) -> str:
    if definition.id == "stable-retro-turbo":
        return (root / "stable_retro" / "VERSION.txt").read_text().strip()
    text = (build_root / "pyproject.toml").read_text(encoding="utf-8")
    import tomllib

    data = tomllib.loads(text)
    version = data.get("project", {}).get("version")
    if not version:
        raise ValueError(f"cannot resolve checkout version for {definition.id}")
    return str(Version(str(version)))


def with_runtime(provider: ResolvedProvider, **updates: Any) -> ResolvedProvider:
    return replace(provider, **updates)


def fake_resolved(provider: str, *, speed: float = 1.0, python_minor: str = "3.14") -> ResolvedProvider:
    identity = canonical_json_hash({"fake": provider, "speed": speed})
    return ResolvedProvider(
        provider=provider,
        adapter="fake",
        distribution="turbobench",
        import_name="turbobench",
        version=f"1.0+speed.{str(speed).replace('.', '')}",
        source_kind="fake",
        source_identity=f"fake:{provider}:{speed}",
        artifact_sha256=identity,
        python_minor=python_minor,
        compatibility_lineage="fake",
    )
