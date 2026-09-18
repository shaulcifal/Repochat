"""Citation validity checking.

Extraction matches the bare token (S1, S2, ...) directly rather than trying
to match every bracket/emphasis style a model might wrap it in. Observed
live from gpt-oss, all for the same intended citation: [S1], full-width
brackets, a trailing span annotation inside them (full-width S2 + a dagger
+ "L60-L78"), backtick emphasis inside brackets ([`S6`]), and bare markdown
bold with no brackets at all (**S2**). Chasing each wrapping style one at a
time kept losing real citations to formatting; matching the token itself is
what turned out to actually be robust. Cleanup on removal is best-effort --
a stray leftover delimiter is a smaller problem than silently losing the
citation extraction outright.
"""

import re

_CITATION_TOKEN = re.compile(r"(?<![A-Za-z0-9])S(\d+)(?![0-9A-Za-z])")

_WRAPPING_CHARS = r"[\[【［`*]"
_WRAPPING_CHARS_CLOSE = r"[\]】］`*]"


def _sorted_labels(labels: set[str]) -> list[str]:
    return sorted(labels, key=lambda label: int(label[1:]))


def extract_cited_labels(answer_text: str) -> list[str]:
    return [f"S{n}" for n in _CITATION_TOKEN.findall(answer_text)]


def validate_citations(answer_text: str, valid_labels: set[str]) -> tuple[str, list[str]]:
    cited = set(extract_cited_labels(answer_text))
    invalid = _sorted_labels(cited - valid_labels)
    confirmed = _sorted_labels(cited & valid_labels)

    if not invalid:
        return answer_text, confirmed

    cleaned = answer_text
    for label in invalid:
        digits = label[1:]
        removal_pattern = re.compile(
            rf"{_WRAPPING_CHARS}*\s*S{digits}(?![0-9])[^\[\]【】［］]{{0,40}}?{_WRAPPING_CHARS_CLOSE}*"
        )
        cleaned = removal_pattern.sub("", cleaned)
    cleaned += f"\n\n_Note: removed unsupported citation(s) {', '.join(invalid)} not present in retrieved evidence._"
    return cleaned, confirmed
