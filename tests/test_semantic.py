from prompt_cache_compaction.semantic import SemanticCompactor
from prompt_cache_compaction.tokens import Message, count_tokens


def test_semantic_compactor_prefers_specific_numeric_facts() -> None:
    messages = [
        Message("user", "We had a nice meeting and talked about many general things."),
        Message("user", "Project Apollo deadline is 2027-03-14."),
        Message("assistant", "Noted."),
    ]
    summary = SemanticCompactor().summarize(messages, max_tokens=13)
    assert "Project Apollo deadline is 2027-03-14." in summary
    assert "nice meeting" not in summary
    assert count_tokens(summary) <= 13


def test_semantic_compactor_is_stable_budgeted_and_deduplicated() -> None:
    messages = [
        Message("user", "Project Apollo deadline is 2027-03-14."),
        Message("user", "Project Apollo deadline is 2027-03-14."),
        Message("user", "Project Borealis deadline is 2027-04-19."),
    ]
    compactor = SemanticCompactor()
    first = compactor.summarize(messages, max_tokens=30)
    second = compactor.summarize(messages, max_tokens=30)
    assert first == second
    assert first.count("Apollo") == 1
    assert "Borealis" in first
    assert count_tokens(first) <= 30
