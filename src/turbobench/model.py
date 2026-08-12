"""Versioned public data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ClaimStatus = Literal["official", "diagnostic"]
Outcome = Literal["left_faster", "right_faster", "inconclusive"]


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    distribution: str
    import_name: str
    adapter: str
    build_subdirectory: str | None = None
    turbo_api: int | None = None
    pypi_project: str | None = None
    lineage: str | None = None


@dataclass(frozen=True)
class ProviderRef:
    provider: str
    selector: Literal["latest", "version", "checkout"]
    value: str

    def __str__(self) -> str:
        if self.selector == "latest":
            return self.provider
        if self.selector == "checkout":
            return f"{self.provider}@checkout:{self.value}"
        return f"{self.provider}@{self.value}"


@dataclass(frozen=True)
class ResolvedProvider:
    provider: str
    adapter: str
    distribution: str
    import_name: str
    version: str
    source_kind: Literal["pypi", "checkout", "fake"]
    source_identity: str
    artifact_sha256: str
    python_minor: str
    build_root: str | None = None
    commit: str | None = None
    tree: str | None = None
    release_files: tuple[dict[str, Any], ...] = ()
    selected_artifact: dict[str, Any] | None = None
    compatibility_lineage: str | None = None
    diagnostic_reasons: tuple[str, ...] = ()
    runtime_id: str | None = None
    runtime_python: str | None = None
    installed_lock: tuple[str, ...] = ()

    def portable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("runtime_python", None)
        payload["release_files"] = list(self.release_files)
        payload["installed_lock"] = [
            (
                f"{self.distribution}=={self.version}"
                if " @ file://" in line and line.split(" @ ", 1)[0].casefold().replace("_", "-")
                == self.distribution.casefold().replace("_", "-")
                else line
            )
            for line in self.installed_lock
        ]
        return payload


@dataclass(frozen=True)
class Profile:
    id: str
    logical_environment: str
    game: str
    providers: tuple[str, ...]
    shapes: tuple[int, ...]
    states: tuple[str, ...]
    semantic_actions: tuple[str, ...]
    action_table: dict[str, tuple[str, ...]]
    info_integer: tuple[str, ...]
    info_float: tuple[str, ...] = ()
    frame_skip: int = 4
    frame_stack: int = 4
    crop_top: int = 0
    crop_bottom: int = 0
    crop_mode: str = "remove"
    resize: tuple[int, int] = (84, 84)
    grayscale: bool = True
    layout: str = "chw"
    resize_algorithm: str = "area"
    maxpool_last_two: bool = False
    benchmark_steps: int = 250
    correctness_steps: int = 64
    promo_kind: str = "deterministic"
    promo_steps: int = 1_200
    completion: dict[str, Any] = field(default_factory=dict)
    asset_sha256: str | None = None
    semantic_authority: str | None = None
    semantic_authority_version: str | None = None
    native_transition_exact: bool = False
    oracle_shapes: tuple[int, ...] = (1, 4)
    oracle_steps: int = 4_096
    snapshot_prefix_steps: int = 128
    snapshot_suffix_steps: int = 128

    def compatible(self, left: str, right: str) -> bool:
        return left != right and left in self.providers and right in self.providers


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RuntimeLocation:
    root: Path
    python: Path
