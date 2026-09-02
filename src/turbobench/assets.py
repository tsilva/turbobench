"""Local asset discovery with canonical digest validation and portable identities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from turbobench.model import Profile
from turbobench.profiles import MARIO_ROM_SHA256
from turbobench.util import sha256_file

BREAKOUT_ROM_SHA256 = "376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5"
STATE_SHA256: dict[str, dict[str, str]] = {
    "supermario/world1-v1": {
        "Level1-1": "905a2e5d8a1bcc8b5955d132a77a8244025d205c2c6a9b07404758d3b84174b5",
        "Level1-2": "68d94ad097de8920a4ec5035be30cb6ec38b0bcdf48fcf65360c07b8e337900a",
        "Level1-3": "f83f72d6e46d8ebe580bde2ce473faa1aa736640c9e99b4358867ace6c5d64bb",
        "Level1-4": "d763572ad5ea3382b7ad901b3f4bfe991fc641cf5251f8223c32f532896ed8b6",
    },
    "breakout/start-v1": {
        "Start": "7020a72745c7e1df9284e8da0dd1ddae1f1cf2ac8ca24fbc51b743c001195b79",
    },
}


def discover_assets(profile: Profile) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return private runner paths and a separate path-free portable record."""
    if profile.logical_environment == "vizdoom-basic":
        return {}, {"required": False, "assets": []}
    expected = MARIO_ROM_SHA256 if profile.logical_environment == "supermario" else BREAKOUT_ROM_SHA256
    game_dirs = _find_game_dirs(profile)
    if not game_dirs:
        return {}, {
            "required": True,
            "available": False,
            "expected_sha256": expected,
            "detail": f"canonical {profile.game} payload was not found",
        }
    roms = sorted(
        {path.resolve() for game_dir in game_dirs for path in game_dir.glob("rom.*") if path.is_file()}
    )
    matching = next((path for path in roms if sha256_file(path) == expected), None)
    if matching is None:
        observed = [{"name": path.name, "sha256": sha256_file(path)} for path in roms]
        return {}, {
            "required": True,
            "available": False,
            "expected_sha256": expected,
            "observed": observed,
            "detail": "available game payload does not match the canonical digest",
        }
    state_paths: dict[str, str] = {}
    state_records: list[dict[str, str]] = []
    missing_states: list[str] = []
    for state in profile.states:
        expected_state = STATE_SHA256.get(profile.id, {}).get(state)
        candidates = [
            game_dir / f"{state}.state"
            for game_dir in game_dirs
            if (game_dir / f"{state}.state").is_file()
        ]
        path = next(
            (
                candidate
                for candidate in candidates
                if expected_state is None or sha256_file(candidate) == expected_state
            ),
            None,
        )
        if path is None:
            missing_states.append(state)
            continue
        state_paths[state] = str(path.resolve())
        state_records.append({"id": state, "sha256": expected_state or sha256_file(path)})
    private = {"rom_path": str(matching.resolve()), "state_paths": state_paths}
    metadata_records: list[dict[str, str]] = []
    info_path = next(
        (
            path
            for game_dir in game_dirs
            if (path := game_dir / "data.json").is_file()
            and _declares_infos(path, profile.info_integer + profile.info_float)
        ),
        None,
    )
    scenario_path = next(
        (path for game_dir in game_dirs if (path := game_dir / "scenario.json").is_file()),
        None,
    )
    for role, path in (("info-schema", info_path), ("scenario", scenario_path)):
        if path is not None:
            private[f"{role.replace('-', '_')}_path"] = str(path.resolve())
            metadata_records.append({"role": role, "sha256": sha256_file(path)})
    portable = {
        "required": True,
        "available": not missing_states,
        "detail": (
            "canonical payload and state catalog found"
            if not missing_states
            else f"missing canonical states: {', '.join(missing_states)}"
        ),
        "assets": [
            {"role": "game-payload", "sha256": expected},
            *({"role": "state", **record} for record in state_records),
            *metadata_records,
        ],
    }
    return private, portable


def _declares_infos(path: Path, names: tuple[str, ...]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    declared = {str(key).casefold() for key in payload.get("info", {})}
    return all(name.casefold() in declared for name in names)


def _find_game_dirs(profile: Profile) -> list[Path]:
    directories: list[Path] = []
    explicit_rom = os.environ.get("TURBOBENCH_ROM_PATH")
    if explicit_rom:
        path = Path(explicit_rom).expanduser()
        if path.is_file():
            directories.append(path.parent.resolve())
    roots: list[Path] = []
    for variable in ("TURBOBENCH_ASSET_ROOT", "RETRO_DATA_PATH"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(
        (
            Path.home() / "roms" / "stable_retro" / "data" / "stable",
            Path.home() / "roms" / "stable-retro" / "data" / "stable",
            Path(__file__).resolve().parents[3]
            / "env-StableRetro-turbo"
            / "env_stableretro_turbo"
            / "data"
            / "stable",
        )
    )
    for root in roots:
        candidates = (
            root / profile.game,
            root / "stable" / profile.game,
            root / "data" / "stable" / profile.game,
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if candidate.is_dir() and resolved not in directories:
                directories.append(resolved)
    return directories
