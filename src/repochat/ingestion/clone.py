"""Safe, read-only shallow cloning.

The cloned tree is never executed: no hooks, no package installs, no tests.
It exists purely so the file-policy and parser stages have something to read.
"""

import subprocess
from pathlib import Path


class CloneError(RuntimeError):
    pass


def shallow_clone(clone_url: str, dest: Path, ref: str | None) -> str:
    """Shallow-clone `ref` (or the default branch) into `dest`.

    Returns the resolved commit SHA actually checked out, which is what gets
    pinned to the revision record -- never the branch name, which can move.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1", "--no-tags"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [clone_url, str(dest)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise CloneError(f"git clone failed: {result.stderr.strip()}")

    rev_parse = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if rev_parse.returncode != 0:
        raise CloneError(f"git rev-parse failed: {rev_parse.stderr.strip()}")

    return rev_parse.stdout.strip()
