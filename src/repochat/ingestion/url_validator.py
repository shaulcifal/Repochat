"""Validate and normalize GitHub repository URLs.

Only public github.com HTTPS URLs are accepted; embedded credentials are
rejected outright rather than silently stripped, so a leaked token in a URL
never gets a chance to be used or logged.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

_OWNER_OR_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class InvalidRepositoryURLError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedRepository:
    owner: str
    name: str
    canonical_url: str
    clone_url: str


def parse_github_url(url: str) -> ParsedRepository:
    parts = urlsplit(url)

    if parts.scheme != "https":
        raise InvalidRepositoryURLError("Only https:// GitHub URLs are supported.")
    if parts.username or parts.password:
        raise InvalidRepositoryURLError("URLs with embedded credentials are not allowed.")
    if parts.hostname != "github.com":
        raise InvalidRepositoryURLError("Only github.com repositories are supported.")
    if parts.port is not None:
        raise InvalidRepositoryURLError("Unexpected port in GitHub URL.")

    path = parts.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) != 2:
        raise InvalidRepositoryURLError("Expected a URL like https://github.com/<owner>/<repo>.")

    owner, name = segments
    if not (_OWNER_OR_REPO_RE.match(owner) and _OWNER_OR_REPO_RE.match(name)):
        raise InvalidRepositoryURLError("Owner or repository name contains unexpected characters.")

    canonical_url = f"https://github.com/{owner}/{name}"
    return ParsedRepository(owner=owner, name=name, canonical_url=canonical_url, clone_url=f"{canonical_url}.git")
