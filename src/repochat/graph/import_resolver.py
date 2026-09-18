"""Resolve raw import statements to local files within the same revision.

Imports are a retrieval aid, not a verifier of what actually executes
(per the design doc's framing). An import that can't be resolved to a file
already in this revision is left alone -- it's presumed external
(stdlib/package), never guessed at.
"""

import ast
import posixpath
import re
from pathlib import PurePosixPath

_QUOTED_SPECIFIER = re.compile(r"""['"]([^'"]+)['"]""")


def _lookup(path_index: dict[str, object], *candidates: str) -> object | None:
    for candidate in candidates:
        normalized = posixpath.normpath(candidate)
        if normalized in path_index:
            return path_index[normalized]
    return None


def resolve_python_import(import_stmt: str, file_path: str, path_index: dict[str, object]) -> object | None:
    try:
        node = ast.parse(import_stmt).body[0]
    except (SyntaxError, IndexError):
        return None

    file_dir = PurePosixPath(file_path).parent

    if isinstance(node, ast.ImportFrom) and node.level > 0:
        target_dir = file_dir
        for _ in range(node.level - 1):
            target_dir = target_dir.parent
        parts = node.module.split(".") if node.module else []
        joined = target_dir.joinpath(*parts) if parts else target_dir
        return _lookup(path_index, f"{joined}.py", str(joined / "__init__.py"))

    if isinstance(node, ast.ImportFrom):
        module = node.module
    elif isinstance(node, ast.Import) and node.names:
        module = node.names[0].name
    else:
        return None

    if not module:
        return None
    parts = module.split(".")
    return _lookup(path_index, posixpath.join(*parts) + ".py", posixpath.join(*parts, "__init__.py"))


def resolve_javascript_import(import_stmt: str, file_path: str, path_index: dict[str, object]) -> object | None:
    match = _QUOTED_SPECIFIER.search(import_stmt)
    if not match:
        return None
    specifier = match.group(1)
    if not (specifier.startswith(".") or specifier.startswith("/")):
        return None  # external package (node_modules) -- not expanded

    file_dir = PurePosixPath(file_path).parent
    base = posixpath.normpath(str(file_dir / specifier))
    return _lookup(path_index, base, f"{base}.js", f"{base}.jsx", f"{base}/index.js")
