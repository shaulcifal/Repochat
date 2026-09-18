import pytest

from repochat.ingestion.url_validator import InvalidRepositoryURLError, parse_github_url


def test_parses_plain_url():
    result = parse_github_url("https://github.com/karpathy/nanochat")
    assert result.owner == "karpathy"
    assert result.name == "nanochat"
    assert result.canonical_url == "https://github.com/karpathy/nanochat"
    assert result.clone_url == "https://github.com/karpathy/nanochat.git"


def test_strips_dot_git_suffix():
    result = parse_github_url("https://github.com/karpathy/nanochat.git")
    assert result.canonical_url == "https://github.com/karpathy/nanochat"


def test_rejects_non_https_scheme():
    with pytest.raises(InvalidRepositoryURLError):
        parse_github_url("http://github.com/karpathy/nanochat")


def test_rejects_embedded_credentials():
    with pytest.raises(InvalidRepositoryURLError):
        parse_github_url("https://user:token@github.com/karpathy/nanochat")


def test_rejects_non_github_host():
    with pytest.raises(InvalidRepositoryURLError):
        parse_github_url("https://gitlab.com/karpathy/nanochat")


def test_rejects_wrong_path_shape():
    with pytest.raises(InvalidRepositoryURLError):
        parse_github_url("https://github.com/karpathy")


def test_rejects_unexpected_characters_in_owner():
    with pytest.raises(InvalidRepositoryURLError):
        parse_github_url("https://github.com/karp;athy/nanochat")
