"""Citation validity checking.

Phase 1 scope: does every [S#] the model used actually exist in the context
pack it was given? An invalid one is stripped and flagged in the answer text
rather than silently kept, but a full repair-then-abstain regeneration loop
(re-prompting the model to fix or abstain) is a Phase 5 deliverable, not
built here yet.
"""

import re

# Models sometimes render brackets as full-width (CJK) characters instead of
# the plain ASCII we asked for in the prompt -- matched live against gpt-oss
# output as [S1] rendered as full-width. Accept both so a citation isn't
# silently lost just because of bracket style.
_CITATION_PATTERN = re.compile(r"[\[【［]S(\d+)[\]】］]")


def _sorted_labels(labels: set[str]) -> list[str]:
    return sorted(labels, key=lambda label: int(label[1:]))


def extract_cited_labels(answer_text: str) -> list[str]:
    return [f"S{n}" for n in _CITATION_PATTERN.findall(answer_text)]


def validate_citations(answer_text: str, valid_labels: set[str]) -> tuple[str, list[str]]:
    cited = set(extract_cited_labels(answer_text))
    invalid = _sorted_labels(cited - valid_labels)
    confirmed = _sorted_labels(cited & valid_labels)

    if not invalid:
        return answer_text, confirmed

    def _strip_if_invalid(match: re.Match) -> str:
        label = f"S{match.group(1)}"
        return "" if label in invalid else match.group(0)

    cleaned = _CITATION_PATTERN.sub(_strip_if_invalid, answer_text)
    cleaned += f"\n\n_Note: removed unsupported citation(s) {', '.join(invalid)} not present in retrieved evidence._"
    return cleaned, confirmed
