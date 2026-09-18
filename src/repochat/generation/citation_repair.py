"""Repair-then-abstain citation validation (design doc Fix 9).

The previous flow (validate -> strip) was a silent failure mode: the reader
sees a claim with its citation quietly deleted, no indication the claim is
now unsupported. This flow gives the model one chance to correct itself
against the same context before falling back to strip-and-flag.
"""

from repochat.generation.citation_validator import extract_cited_labels, validate_citations
from repochat.generation.llm_provider import LLMProvider

_REPAIR_INSTRUCTION = (
    "\n\nYour previous answer cited {invalid_labels}, but those source tags do not exist "
    "in the SOURCES above. Valid source tags are: {valid_labels}. Revise your answer: fix any "
    "incorrect citation to point at a valid tag that actually supports the claim, or remove "
    "the claim and say plainly that evidence is missing for it. Do not invent a new tag.\n\n"
    "Previous answer:\n{previous_answer}"
)


def repair_then_abstain(
    llm_provider: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    raw_answer: str,
    valid_labels: set[str],
) -> tuple[str, list[str]]:
    invalid = set(extract_cited_labels(raw_answer)) - valid_labels
    if not invalid:
        return validate_citations(raw_answer, valid_labels)

    sorted_valid = sorted(valid_labels, key=lambda label: int(label[1:]))
    sorted_invalid = sorted(invalid, key=lambda label: int(label[1:]))
    repair_prompt = user_prompt + _REPAIR_INSTRUCTION.format(
        invalid_labels=", ".join(sorted_invalid),
        valid_labels=", ".join(sorted_valid),
        previous_answer=raw_answer,
    )

    repaired_answer = llm_provider.generate(system_prompt, repair_prompt)
    # Whatever's still invalid after this one repair attempt gets the
    # existing strip-and-flag treatment -- an explicit abstention, not a
    # silent edit.
    return validate_citations(repaired_answer, valid_labels)
