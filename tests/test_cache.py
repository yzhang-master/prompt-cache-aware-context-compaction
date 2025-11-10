import pytest

from prompt_cache_compaction.cache import ExactPrefixCache
from prompt_cache_compaction.tokens import Message, count_tokens, encode_messages, tokenize


def test_tokenizer_and_message_overhead_are_deterministic() -> None:
    assert tokenize("Project Apollo: 2027-03-14!") == (
        "Project",
        "Apollo",
        ":",
        "2027",
        "-",
        "03",
        "-",
        "14",
        "!",
    )
    message = Message("user", "hello world")
    assert encode_messages([message]) == ("<user>", "hello", "world", "<end>")
    assert count_tokens("hello world") == 2


def test_prefix_cache_counts_only_complete_chained_blocks() -> None:
    cache = ExactPrefixCache(block_size=2)
    original = ("a", "b", "c", "d", "e")
    assert cache.lookup(original) == 0
    cache.store(original)
    assert len(cache) == 2
    assert cache.lookup(original) == 4
    assert cache.lookup(("a", "b", "c", "d", "x", "y")) == 4
    assert cache.lookup(("a", "b", "x", "d")) == 2


def test_same_block_under_different_prefix_is_not_a_false_hit() -> None:
    cache = ExactPrefixCache(block_size=2)
    cache.store(("a", "b", "c", "d"))
    assert cache.lookup(("x", "y", "c", "d")) == 0
    with pytest.raises(ValueError):
        ExactPrefixCache(block_size=0)
