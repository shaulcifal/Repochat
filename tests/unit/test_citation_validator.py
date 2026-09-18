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
