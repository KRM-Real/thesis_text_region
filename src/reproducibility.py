"""Hashing and run-identity helpers used by every artifact-producing command."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prototype_config import canonical_json


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def source_revision(root: str | Path = ".") -> str:
    root_path = Path(root).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return f"git:{result.stdout.strip()}"
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        excluded = {
            ".venv",
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            ".codex",
            ".agents",
            ".codex-remote-attachments",
            "data",
            "dataset",
            "sources",
            "outputs",
            "artifacts",
            "easyocr-models",
            "uploads",
            "user-data",
        }
        for current, directories, filenames in os.walk(root_path, topdown=True):
            # Prune user data directories before walking into them. Besides
            # keeping run identity independent of private data, this avoids
            # enumerating or hashing any dataset files in the no-Git fallback.
            directories[:] = sorted(name for name in directories if name.casefold() not in excluded)
            current_path = Path(current)
            for filename in sorted(filenames):
                path = current_path / filename
                if path.is_symlink() or not path.is_file():
                    continue
                relative = str(path.relative_to(root_path)).replace(os.sep, "/")
                digest.update(relative.encode("utf-8"))
                digest.update(sha256_file(path).encode("ascii"))
        return f"tree:{digest.hexdigest()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_run_id(config_sha: str, source_rev: str, label: str | None = None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_short = source_rev.split(":", 1)[-1][:8]
    value = f"{timestamp}-{config_sha[:8]}-{source_short}"
    if label:
        value += f"-{label.replace(' ', '-').lower()}"
    return value


def runtime_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "Pillow", "PyYAML", "scikit-image", "scikit-learn", "streamlit", "easyocr", "torch"]
    versions: dict[str, str] = {"python": platform.python_version()}
    try:
        from importlib import metadata

        for name in names:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "unavailable"
    except Exception:  # pragma: no cover
        pass
    return versions


def hardware_context() -> dict[str, str]:
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor() or "unknown"}


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
