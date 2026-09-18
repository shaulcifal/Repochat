from repochat.generation.citation_validator import validate_citations


def test_keeps_valid_citations_unchanged():
    text = "Training starts in main() [S1] and calls Trainer.train() [S2]."
    cleaned, confirmed = validate_citations(text, valid_labels={"S1", "S2"})
    assert cleaned == text
    assert confirmed == ["S1", "S2"]


def test_strips_invalid_citation_and_flags_it():
    text = "Training starts in main() [S1] and uses a cache [S5]."
    cleaned, confirmed = validate_citations(text, valid_labels={"S1", "S2"})
    assert "[S5]" not in cleaned
    assert "[S1]" in cleaned
    assert "S5" in cleaned  # flagged in the trailing note
    assert confirmed == ["S1"]


def test_no_citations_at_all():
    cleaned, confirmed = validate_citations("No sources needed here.", valid_labels={"S1"})
    assert confirmed == []
    assert cleaned == "No sources needed here."


def test_recognizes_full_width_brackets():
    text = "Training starts in main() 【S1】."
    cleaned, confirmed = validate_citations(text, valid_labels={"S1"})
    assert confirmed == ["S1"]
    assert cleaned == text  # valid citation left untouched, whatever bracket style


def test_strips_invalid_full_width_citation():
    text = "Uses a cache 【S9】."
    cleaned, confirmed = validate_citations(text, valid_labels={"S1"})
    assert confirmed == []
    assert "【S9】" not in cleaned
    assert "S9" in cleaned  # still flagged in the trailing note


def test_recognizes_citation_with_trailing_span_annotation():
    # Observed live from gpt-oss: 【S2†L60-L78】 instead of a bare [S2].
    text = "Uses the KV cache 【S2†L60-L78】 for decoding."
    cleaned, confirmed = validate_citations(text, valid_labels={"S1", "S2"})
    assert confirmed == ["S2"]
    assert cleaned == text


def test_recognizes_citation_wrapped_in_markdown_emphasis():
    # Observed live from gpt-oss: [`S6`] instead of a bare [S6].
    text = "Handled per example [`S6`]."
    cleaned, confirmed = validate_citations(text, valid_labels={"S6"})
    assert confirmed == ["S6"]
    assert cleaned == text


def test_recognizes_bare_bold_citation_with_no_brackets_at_all():
    # Observed live from gpt-oss: **S2** with no brackets whatsoever.
    text = "Creates an Engine **S2** to run generation."
    cleaned, confirmed = validate_citations(text, valid_labels={"S2"})
    assert confirmed == ["S2"]
    assert cleaned == text


def test_does_not_confuse_s1_with_s10_or_s12():
    text = "See [S1], [S10], and [S12]."
    cleaned, confirmed = validate_citations(text, valid_labels={"S1"})
    assert confirmed == ["S1"]
    body, _, note = cleaned.partition("\n\n_Note:")
    assert "[S1]" in body
    assert "[S10]" not in body  # bracket form removed from the body...
    assert "[S12]" not in body
    assert "S10" in note  # ...but still named in the trailing note
