from pathlib import Path

from repochat.ingestion.file_policy import MAX_FILE_SIZE_BYTES, build_manifest


def _write(root: Path, relative_path: str, content: bytes) -> None:
    full_path = root / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(content)


def _status_by_path(manifest, path: str) -> str:
    return next(e.status for e in manifest.entries if e.path == path)


def test_classifies_python_module_as_code(tmp_path: Path):
    _write(tmp_path, "app/main.py", b"def handler():\n    pass\n")
    manifest = build_manifest(tmp_path)
    assert _status_by_path(manifest, "app/main.py") == "code"


def test_classifies_test_file_by_directory(tmp_path: Path):
    _write(tmp_path, "tests/test_main.py", b"def test_ok():\n    assert True\n")
    manifest = build_manifest(tmp_path)
    assert _status_by_path(manifest, "tests/test_main.py") == "test"


def test_classifies_test_file_by_filename_prefix(tmp_path: Path):
    _write(tmp_path, "src/test_utils.py", b"def test_ok():\n    assert True\n")
    manifest = build_manifest(tmp_path)
    assert _status_by_path(manifest, "src/test_utils.py") == "test"


def test_classifies_markdown_as_docs(tmp_path: Path):
    _write(tmp_path, "README.md", b"# Title\n")
    manifest = build_manifest(tmp_path)
    assert _status_by_path(manifest, "README.md") == "docs"


def test_classifies_package_json_as_config(tmp_path: Path):
    _write(tmp_path, "package.json", b"{}")
    manifest = build_manifest(tmp_path)
    assert _status_by_path(manifest, "package.json") == "config"


def test_excludes_ignored_directories(tmp_path: Path):
    _write(tmp_path, "node_modules/lib/index.js", b"module.exports = {};\n")
    _write(tmp_path, "app/index.js", b"console.log('hi');\n")
    manifest = build_manifest(tmp_path)
    paths = {e.path for e in manifest.entries}
    assert "node_modules/lib/index.js" not in paths
    assert "app/index.js" in paths


def test_excludes_oversized_file(tmp_path: Path):
    _write(tmp_path, "big.py", b"x" * (MAX_FILE_SIZE_BYTES + 1))
    manifest = build_manifest(tmp_path)
    entry = next(e for e in manifest.entries if e.path == "big.py")
    assert entry.status == "excluded"
    assert "size limit" in entry.exclusion_reason


def test_excludes_binary_content(tmp_path: Path):
    _write(tmp_path, "blob.py", b"\x00\x01\x02binary")
    manifest = build_manifest(tmp_path)
    entry = next(e for e in manifest.entries if e.path == "blob.py")
    assert entry.status == "excluded"
    assert "binary" in entry.exclusion_reason


def test_excludes_minified_filename(tmp_path: Path):
    _write(tmp_path, "app.min.js", b"console.log(1)")
    manifest = build_manifest(tmp_path)
    entry = next(e for e in manifest.entries if e.path == "app.min.js")
    assert entry.status == "excluded"
    assert "minified" in entry.exclusion_reason


def test_included_files_get_a_content_hash(tmp_path: Path):
    _write(tmp_path, "app/main.py", b"def handler():\n    pass\n")
    manifest = build_manifest(tmp_path)
    entry = next(e for e in manifest.entries if e.path == "app/main.py")
    assert entry.content_hash is not None
    assert len(entry.content_hash) == 64  # sha256 hex digest
