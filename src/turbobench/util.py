"""Deterministic serialization, hashing, redaction, and atomic I/O."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(r"(?:token|secret|password|passwd|api[_-]?key|authorization)", re.I)
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?:/[A-Za-z0-9_.+@%:,=-]+){2,}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    encoding = None if isinstance(content, bytes) else "utf-8"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, mode, encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def redact(value: Any, *, key: str = "") -> Any:
    """Remove secrets and non-portable host/cache/asset paths recursively."""
    if _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item, key=key) for item in value]
    if isinstance(value, Path):
        return f"<redacted-path>/{value.name}"
    if isinstance(value, str):
        lowered_key = key.casefold()
        if "hostname" in lowered_key or "host_name" in lowered_key:
            return "<redacted>"
        if any(part in lowered_key for part in ("path", "cache", "hostname", "host_name")):
            return f"<redacted-path>/{Path(value).name}" if value else value
        return _ABSOLUTE_PATH.sub("<redacted-path>", value)
    return value


def find_portability_violations(value: Any, *, key: str = "") -> list[str]:
    violations: list[str] = []
    if _SECRET_KEY.search(key) and value not in (None, "", "<redacted>"):
        violations.append(f"secret-like field {key!r} is not redacted")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            violations.extend(find_portability_violations(child, key=str(child_key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            violations.extend(find_portability_violations(child, key=key))
    elif (
        isinstance(value, str)
        and not value.startswith(("https://", "http://"))
        and _ABSOLUTE_PATH.search(value)
    ):
        violations.append(f"absolute path in field {key!r}")
    return violations


def relative_files(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
