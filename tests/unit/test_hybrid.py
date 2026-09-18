from repochat.retrieval.hybrid import reciprocal_rank_fusion


def test_item_ranked_first_in_both_lists_wins():
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["a", "c", "b"])
    assert fused[0][0] == "a"


def test_item_only_in_one_list_still_included():
    fused = reciprocal_rank_fusion(["a", "b"], ["c"])
    items = [item for item, _ in fused]
    assert set(items) == {"a", "b", "c"}


def test_item_appearing_in_both_lists_outranks_single_list_item():
    # "b" is #2 in both lists; "a" is #1 in only one list.
    fused = reciprocal_rank_fusion(["a", "b"], ["c", "b"])
    ranked_items = [item for item, _ in fused]
    assert ranked_items.index("b") < ranked_items.index("a")


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([], []) == []
