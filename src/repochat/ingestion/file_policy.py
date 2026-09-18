"""Classify every file in a clone as code / docs / config / test / excluded.

Every exclusion carries a reason -- nothing is silently dropped, because
`repochat status` needs to be able to explain the manifest after the fact.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "vendor",
    "venv",
    ".venv",
    "__pycache__",
}

CODE_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

DOCS_EXTENSIONS = {".md", ".txt"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}
CONFIG_BASENAMES = {"requirements.txt", "package.json", "pyproject.toml", "Pipfile"}

MAX_FILE_SIZE_BYTES = 500_000
MAX_FILES_PER_REVISION = 10_000
BINARY_SNIFF_BYTES = 8_000


@dataclass
class ManifestEntry:
    path: str
    language: str | None
    size_bytes: int
    status: str
    content_hash: str | None
    exclusion_reason: str | None


@dataclass
class Manifest:
    entries: list[ManifestEntry] = field(default_factory=list)
    truncated: bool = False

    @property
    def included(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.status != "excluded"]

    @property
    def excluded(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.status == "excluded"]


def _is_test_path(relative_path: str) -> bool:
    parts = relative_path.split("/")
    basename = parts[-1]
    if any(part in ("test", "tests") for part in parts[:-1]):
        return True
    stem = basename.rsplit(".", 1)[0]
    return stem.startswith("test_") or stem.endswith("_test")


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as fh:
        chunk = fh.read(BINARY_SNIFF_BYTES)
    return b"\x00" in chunk


def _classify(relative_path: str, basename: str, suffix: str) -> tuple[str, str | None]:
    """Returns (tentative_status, language) ignoring size/binary checks."""
    if suffix in CODE_LANGUAGE_BY_EXTENSION:
        language = CODE_LANGUAGE_BY_EXTENSION[suffix]
        status = "test" if _is_test_path(relative_path) else "code"
        return status, language
    if suffix in DOCS_EXTENSIONS or basename.upper().startswith("README"):
        return "docs", None
    if suffix in CONFIG_EXTENSIONS or basename in CONFIG_BASENAMES:
        return "config", None
    return "excluded", None


def build_manifest(root: Path) -> Manifest:
    manifest = Manifest()
    scanned = 0

    for dirpath, dirnames, filenames in _walk_sorted(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for filename in filenames:
            if scanned >= MAX_FILES_PER_REVISION:
                manifest.truncated = True
                return manifest
            scanned += 1

            file_path = Path(dirpath) / filename
            relative_path = file_path.relative_to(root).as_posix()
            suffix = file_path.suffix.lower()
            size_bytes = file_path.stat().st_size

            if size_bytes > MAX_FILE_SIZE_BYTES:
                manifest.entries.append(
                    ManifestEntry(
                        path=relative_path,
                        language=None,
                        size_bytes=size_bytes,
                        status="excluded",
                        content_hash=None,
                        exclusion_reason=f"exceeds size limit ({size_bytes} > {MAX_FILE_SIZE_BYTES} bytes)",
                    )
                )
                continue

            status, language = _classify(relative_path, filename, suffix)

            if status == "excluded":
                manifest.entries.append(
                    ManifestEntry(
                        path=relative_path,
                        language=None,
                        size_bytes=size_bytes,
                        status="excluded",
                        content_hash=None,
                        exclusion_reason="unsupported file type",
                    )
                )
                continue

            if ".min." in filename:
                manifest.entries.append(
                    ManifestEntry(
                        path=relative_path,
                        language=language,
                        size_bytes=size_bytes,
                        status="excluded",
                        content_hash=None,
                        exclusion_reason="minified filename pattern",
                    )
                )
                continue

            if _looks_binary(file_path):
                manifest.entries.append(
                    ManifestEntry(
                        path=relative_path,
                        language=None,
                        size_bytes=size_bytes,
                        status="excluded",
                        content_hash=None,
                        exclusion_reason="binary content detected",
                    )
                )
                continue

            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            manifest.entries.append(
                ManifestEntry(
                    path=relative_path,
                    language=language,
                    size_bytes=size_bytes,
                    status=status,
                    content_hash=content_hash,
                    exclusion_reason=None,
                )
            )

    return manifest


def _walk_sorted(root: Path):
    """os.walk, but deterministic -- same manifest hash for the same tree."""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        yield dirpath, dirnames, sorted(filenames)
